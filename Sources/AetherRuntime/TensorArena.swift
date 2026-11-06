import CoreAI
import Darwin
import Metal

/// Owns the storage of every ``TensorBuffer`` one model uses (DESIGN.md §6.1, §11).
///
/// Why an arena rather than one `NDArray` per buffer object: `NDArray` is a copy-on-write *value*, so the storage a
/// graph writes into must live in exactly one place, and Core AI's `~Escapable` views can only be bound to a run
/// from storage the binding code owns directly. Both hold when all of a model's arrays sit in this arena's
/// `arrays`, and ``TensorBuffer`` is a typed handle into it. See LifetimeEscape.swift for the run itself.
///
/// Two kinds of storage, chosen per arena:
/// - **host** (``init()``): `NDArray`s. Enough for `Graph.run` and for the fake decoder in tests.
/// - **Metal** (``init(device:)``, internal): shared `MTLBuffer`s, which both `run` (through views over the buffer)
///   and `InferenceFunction.encode` (through `AsyncValue(unsafeBuffer:)`) can bind. The pipelined decoder needs them:
///   its states, logits and token slots are buffers that Core AI and our own Metal kernels share.
///
/// Non-`Sendable`: an arena belongs to one engine or pipeline actor and never leaves it. While a graph runs, its
/// buffers are lent to Core AI; any host access to the arena in that window (from a re-entrant call on the owning
/// actor) fails with `.tensor(…, .unavailable)` instead of racing the graph. For encoded (asynchronous) work the
/// owner lends the arena with ``beginRun(_:)`` until the stream has drained.
public final class TensorArena {
    /// Host storage. For a Metal slot this holds a 1-element placeholder and `metal[slot]` is the storage.
    /// Mutated only through the methods below; views into it exist only inside ``run``/``encode``.
    var arrays: [NDArray] = []
    /// Metal storage per slot (nil for host slots).
    var metal: [(any MTLBuffer)?] = []
    private var generations: [Int] = []
    private var freeSlots: [Int] = []
    /// Non-nil while a graph run (or a stream of encoded work) holds views into the storage.
    private(set) var runningGraph: String?
    /// The device Metal storage is allocated on, or nil for a host arena.
    let device: (any MTLDevice)?

    /// A host arena: every buffer is an `NDArray`.
    public init() {
        device = nil
    }

    /// A Metal arena: every buffer is a shared `MTLBuffer` on `device`.
    init(device: any MTLDevice) {
        self.device = device
    }

    /// Whether buffers are Metal-backed (and so can be bound to `encode`).
    public var isMetalBacked: Bool { device != nil }

    /// Bytes held by live buffers.
    public private(set) var byteCount = 0

    // MARK: Allocation

    /// Allocates a zero-filled, contiguous row-major buffer.
    public func allocate(name: String, shape: [Int], scalarType: ScalarType) throws(AetherError) -> TensorBuffer {
        try allocate(name: name, shape: shape, scalarType: scalarType, strides: nil)
    }

    /// Allocates a zero-filled buffer with explicit (for example padded) strides, in elements.
    func allocate(name: String, shape: [Int], scalarType: ScalarType, strides: [Int]?) throws(AetherError) -> TensorBuffer {
        guard let coreType = scalarType.coreAI, let width = scalarType.byteWidth else {
            throw .tensor(name: name, reason: .unsupported("cannot allocate \(scalarType) tensors"))
        }
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw .tensor(name: name, reason: .shapeMismatch(expected: [], found: shape))
        }
        if let strides {
            guard strides.count == shape.count, strides.allSatisfy({ $0 > 0 }) else {
                throw .tensor(name: name, reason: .unsupported("strides \(strides) for shape \(shape)"))
            }
        }
        let layout = StridedLayout(shape: shape, strides: strides ?? StridedLayout.rowMajorStrides(shape))
        if device != nil {
            return try adoptMetal(byteCount: layout.spannedElements * width, name: name, scalarType: scalarType, layout: layout)
        }
        if let strides {
            return try adopt(NDArray(shape: shape, scalarType: coreType, strides: strides), name: name)
        }
        return try adopt(NDArray(shape: shape, scalarType: coreType), name: name)
    }

    /// Allocates from a graph descriptor already resolved to a concrete shape, keeping Core AI's preferred strides.
    func allocate(name: String, descriptor: NDArrayDescriptor) throws(AetherError) -> TensorBuffer {
        guard !descriptor.hasDynamicShape else {
            throw .tensor(name: name, reason: .shapeMismatch(expected: descriptor.shape, found: descriptor.shape))
        }
        guard descriptor.interleaveLayout == nil else {
            throw .tensor(name: name, reason: .unsupported("interleaved layouts are not supported yet"))
        }
        let scalarType = ScalarType(descriptor.scalarType)
        guard scalarType.byteWidth != nil else {
            throw .tensor(name: name, reason: .unsupported("cannot allocate \(scalarType) tensors"))
        }
        if device != nil {
            let layout = StridedLayout(shape: descriptor.shape, strides: descriptor.preferredStrides)
            return try adoptMetal(
                byteCount: max(descriptor.minimumByteCount, layout.spannedElements * scalarType.byteWidth!), name: name,
                scalarType: scalarType, layout: layout)
        }
        return try adopt(NDArray(descriptor: descriptor), name: name)
    }

    /// Allocates a token slot for the pipelined decoder: an int32 `[1, 1]` buffer in its **own small `MTLBuffer`**
    /// whose contents do not start on a page boundary.
    ///
    /// Measured in the lab (DESIGN.md §6.1, "Token slots"): Core AI orders its GPU read of a step's input behind an
    /// earlier same-queue kernel's write only when the input buffer is suballocated. A page-aligned buffer (as a
    /// token-history view at capacity ≥ 4096) raced and produced wrong tokens. So a page-aligned allocation is
    /// rejected and retried; ``isPageAligned(_:)`` lets the engine assert the invariant again at load.
    func allocateTokenSlot(name: String) throws(AetherError) -> TensorBuffer {
        guard let device else {
            throw .tensor(name: name, reason: .unsupported("token slots need a Metal arena"))
        }
        try ensureIdle(name)
        var rejected: [any MTLBuffer] = []  // held until a slot is found, so the allocator does not hand them back
        defer { rejected.removeAll() }
        while rejected.count < 64 {
            guard let buffer = device.makeBuffer(length: Self.tokenSlotBytes, options: .storageModeShared) else { break }
            if Self.isPageAligned(buffer) {
                rejected.append(buffer)
                continue
            }
            memset(buffer.contents(), 0, buffer.length)
            let layout = StridedLayout(shape: [1, 1], strides: [1, 1])
            return place(metal: buffer, name: name, scalarType: .int32, layout: layout)
        }
        throw .tensor(name: name, reason: .unsupported("could not allocate a sub-page token slot"))
    }

    /// 64 B: the fork's `minimumMPSNDArrayBufferSize`, comfortably below a page.
    static let tokenSlotBytes = 64

    static func isPageAligned(_ buffer: any MTLBuffer) -> Bool {
        UInt(bitPattern: buffer.contents()) % UInt(getpagesize()) == 0
    }

    private func adoptMetal(byteCount: Int, name: String, scalarType: ScalarType, layout: StridedLayout) throws(AetherError) -> TensorBuffer {
        try ensureIdle(name)
        guard let device, let buffer = device.makeBuffer(length: max(byteCount, 16), options: .storageModeShared) else {
            throw .capacity(what: "Metal buffer '\(name)'", needed: Int64(byteCount), available: 0)
        }
        memset(buffer.contents(), 0, buffer.length)
        return place(metal: buffer, name: name, scalarType: scalarType, layout: layout)
    }

    private func place(metal buffer: any MTLBuffer, name: String, scalarType: ScalarType, layout: StridedLayout) -> TensorBuffer {
        let slot = takeSlot(array: NDArray(shape: [1], scalarType: .int8), metal: buffer)
        let handle = TensorBuffer(
            arena: self, slot: slot, generation: generations[slot], name: name, scalarType: scalarType, layout: layout)
        byteCount += handle.storageByteCount
        return handle
    }

    private func adopt(_ array: NDArray, name: String) throws(AetherError) -> TensorBuffer {
        try ensureIdle(name)
        let layout = StridedLayout(shape: array.shape, strides: array.strides)
        let scalarType = ScalarType(array.scalarType)
        let slot = takeSlot(array: array, metal: nil)
        let buffer = TensorBuffer(
            arena: self, slot: slot, generation: generations[slot], name: name, scalarType: scalarType, layout: layout)
        byteCount += buffer.storageByteCount
        try buffer.zero()
        return buffer
    }

    private func takeSlot(array: NDArray, metal buffer: (any MTLBuffer)?) -> Int {
        if let reused = freeSlots.popLast() {
            arrays[reused] = array
            metal[reused] = buffer
            return reused
        }
        arrays.append(array)
        metal.append(buffer)
        generations.append(0)
        return arrays.count - 1
    }

    /// Returns the buffer's storage to the arena. The handle (and any copy of it) is invalid afterwards.
    public func release(_ buffer: TensorBuffer) throws(AetherError) {
        try check(buffer)
        arrays[buffer.slot] = NDArray(shape: [1], scalarType: .int8)
        metal[buffer.slot] = nil
        generations[buffer.slot] += 1
        freeSlots.append(buffer.slot)
        byteCount -= buffer.storageByteCount
    }

    // MARK: Access

    func check(_ buffer: TensorBuffer) throws(AetherError) {
        try checkOwned(buffer)
        try ensureIdle(buffer.name)
    }

    /// The handle is live and belongs to this arena (lending aside).
    func checkOwned(_ buffer: TensorBuffer) throws(AetherError) {
        guard buffer.arena === self else {
            throw .tensor(name: buffer.name, reason: .unavailable("the buffer belongs to another arena"))
        }
        guard buffer.slot < generations.count, generations[buffer.slot] == buffer.generation else {
            throw .tensor(name: buffer.name, reason: .unavailable("the buffer was released"))
        }
    }

    private func ensureIdle(_ name: String) throws(AetherError) {
        if let runningGraph {
            throw .tensor(name: name, reason: .unavailable("lent to the running graph '\(runningGraph)'"))
        }
    }

    /// The `MTLBuffer` behind a Metal-backed buffer (for our own kernels and `encode`). Does not check lending: the
    /// caller orders its GPU work itself.
    func metalBuffer(of buffer: TensorBuffer) throws(AetherError) -> any MTLBuffer {
        guard buffer.arena === self, buffer.slot < generations.count, generations[buffer.slot] == buffer.generation else {
            throw .tensor(name: buffer.name, reason: .unavailable("the buffer was released or belongs to another arena"))
        }
        guard let storage = metal[buffer.slot] else {
            throw .tensor(name: buffer.name, reason: .unsupported("not Metal-backed"))
        }
        return storage
    }

    /// Typed pointer access to a buffer's storage, for the duration of `body`.
    func withMutablePointer<T: TensorScalar, R>(
        _ buffer: TensorBuffer, as type: T.Type, _ body: (UnsafeMutablePointer<T>) throws(AetherError) -> R
    ) throws(AetherError) -> R {
        try check(buffer)
        guard buffer.scalarType == T.scalarType else {
            throw .tensor(name: buffer.name, reason: .typeMismatch(expected: buffer.scalarType.rawValue, found: T.scalarType.rawValue))
        }
        if let storage = metal[buffer.slot] {
            return try body(storage.contents().bindMemory(to: T.self, capacity: buffer.layout.spannedElements))
        }
        return try Self.withPointer(&arrays[buffer.slot], as: T.self, body)
    }

    /// Raw access to the bytes a buffer's elements span.
    func withMutableBytes<R>(
        _ buffer: TensorBuffer, _ body: (UnsafeMutableRawBufferPointer) throws(AetherError) -> R
    ) throws(AetherError) -> R {
        try check(buffer)
        let count = buffer.storageByteCount
        if let storage = metal[buffer.slot] {
            return try body(UnsafeMutableRawBufferPointer(start: storage.contents(), count: count))
        }
        return try Self.withBytes(&arrays[buffer.slot], count: count, body)
    }

    /// Copies `source`'s bytes over `destination`'s (same byte count; both checked). One pass, no staging copy.
    func copyBytes(from source: TensorBuffer, to destination: TensorBuffer) throws(AetherError) {
        try check(source)
        try check(destination)
        precondition(source.storageByteCount == destination.storageByteCount && source.slot != destination.slot)
        try withTwo(source, destination) { from, to in to.copyMemory(from: UnsafeRawBufferPointer(from)) }
    }

    /// Copies the leading region the two buffers share (the elementwise minimum of their shapes) from `source` into
    /// `destination`. Both must have the same rank and scalar type; strides may differ. Used to move a KV cache into
    /// a larger (or smaller) capacity along its sequence axis.
    func copyRegion(from source: TensorBuffer, to destination: TensorBuffer) throws(AetherError) {
        try check(source)
        try check(destination)
        guard source.scalarType == destination.scalarType, source.shape.count == destination.shape.count,
              let width = source.scalarType.byteWidth, source.slot != destination.slot
        else {
            throw .tensor(name: destination.name, reason: .shapeMismatch(expected: destination.shape, found: source.shape))
        }
        let region = zip(source.shape, destination.shape).map { min($0, $1) }
        let rank = region.count
        let innerContiguous = source.strides[rank - 1] == 1 && destination.strides[rank - 1] == 1
        let run = innerContiguous ? region[rank - 1] : 1
        let outer = innerContiguous ? rank - 1 : rank
        try withTwo(source, destination) { from, to in
            var index = [Int](repeating: 0, count: max(outer, 1))
            let total = region.prefix(outer).reduce(1, *)
            guard total > 0, run > 0 else { return }
            for _ in 0..<total {
                var src = 0, dst = 0
                for axis in 0..<outer {
                    src += index[axis] * source.strides[axis]
                    dst += index[axis] * destination.strides[axis]
                }
                (to.baseAddress! + dst * width).copyMemory(from: from.baseAddress! + src * width, byteCount: run * width)
                var axis = outer - 1
                while axis >= 0 {
                    index[axis] += 1
                    if index[axis] < region[axis] { break }
                    index[axis] = 0
                    axis -= 1
                }
            }
        }
    }

    /// Byte access to two distinct buffers at once.
    private func withTwo(
        _ a: TensorBuffer, _ b: TensorBuffer,
        _ body: (UnsafeMutableRawBufferPointer, UnsafeMutableRawBufferPointer) -> Void
    ) throws(AetherError) {
        let countA = a.storageByteCount, countB = b.storageByteCount
        switch (metal[a.slot], metal[b.slot]) {
        case let (x?, y?):
            body(UnsafeMutableRawBufferPointer(start: x.contents(), count: countA),
                 UnsafeMutableRawBufferPointer(start: y.contents(), count: countB))
        case let (x?, nil):
            try Self.withBytes(&arrays[b.slot], count: countB) { to in
                body(UnsafeMutableRawBufferPointer(start: x.contents(), count: countA), to)
            }
        case let (nil, y?):
            try Self.withBytes(&arrays[a.slot], count: countA) { from in
                body(from, UnsafeMutableRawBufferPointer(start: y.contents(), count: countB))
            }
        case (nil, nil):
            // Two elements of `arrays` are accessed at once, so go through the buffer pointer: overlapping accesses
            // to the array itself would violate exclusivity.
            arrays.withUnsafeMutableBufferPointer { slots in
                try! Self.withBytes(&slots[a.slot], count: countA) { from throws(AetherError) in
                    try Self.withBytes(&slots[b.slot], count: countB) { to throws(AetherError) in body(from, to) }
                }
            }
        }
    }

    // Views must be taken from an `inout` binding and used within the statement's scope; a chained call on
    // `arrays[i]` does not satisfy the lifetime checker.
    private static func withPointer<T: TensorScalar, R>(
        _ array: inout NDArray, as type: T.Type, _ body: (UnsafeMutablePointer<T>) throws(AetherError) -> R
    ) throws(AetherError) -> R {
        let view = array.mutableView(as: T.self)
        let result: Result<R, AetherError> = view.withUnsafeMutablePointer { pointer, _, _ in
            do throws(AetherError) { return .success(try body(pointer)) } catch { return .failure(error) }
        }
        return try result.get()
    }

    private static func withBytes<R>(
        _ array: inout NDArray, count: Int, _ body: (UnsafeMutableRawBufferPointer) throws(AetherError) -> R
    ) throws(AetherError) -> R {
        let view = array.mutableRawView()
        let result: Result<R, AetherError> = view.withUnsafeMutableBytes { pointer, _, _ in
            do throws(AetherError) {
                return .success(try body(UnsafeMutableRawBufferPointer(start: pointer, count: count)))
            } catch {
                return .failure(error)
            }
        }
        return try result.get()
    }

    /// Marks the arena as lent to `graph`: for one `run`, or by an engine for as long as encoded work it submitted
    /// may still read or write the storage. Every host access fails until ``endRun()``.
    func beginRun(_ graph: String) throws(AetherError) {
        try ensureIdle(graph)
        runningGraph = graph
    }

    func endRun() {
        runningGraph = nil
    }
}

/// Element types with typed access. The conformance names the matching ``ScalarType``.
public protocol TensorScalar: BitwiseCopyable {
    static var scalarType: ScalarType { get }
}

extension Float: TensorScalar { public static var scalarType: ScalarType { .float32 } }
extension Float16: TensorScalar { public static var scalarType: ScalarType { .float16 } }
extension Double: TensorScalar { public static var scalarType: ScalarType { .float64 } }
extension Int8: TensorScalar { public static var scalarType: ScalarType { .int8 } }
extension Int16: TensorScalar { public static var scalarType: ScalarType { .int16 } }
extension Int32: TensorScalar { public static var scalarType: ScalarType { .int32 } }
extension Int64: TensorScalar { public static var scalarType: ScalarType { .int64 } }
extension UInt8: TensorScalar { public static var scalarType: ScalarType { .uint8 } }
extension UInt16: TensorScalar { public static var scalarType: ScalarType { .uint16 } }
extension UInt32: TensorScalar { public static var scalarType: ScalarType { .uint32 } }
extension UInt64: TensorScalar { public static var scalarType: ScalarType { .uint64 } }
extension Bool: TensorScalar { public static var scalarType: ScalarType { .bool } }
