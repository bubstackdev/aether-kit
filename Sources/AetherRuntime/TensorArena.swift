import CoreAI

/// Owns the storage of every ``TensorBuffer`` one model uses (DESIGN.md §6.1, §11).
///
/// Why an arena rather than one `NDArray` per buffer object: `NDArray` is a copy-on-write *value*, so the storage a
/// graph writes into must live in exactly one place, and Core AI's `~Escapable` views can only be bound to a run
/// from storage the binding code owns directly. Both hold when all of a model's arrays sit in this arena's
/// `arrays`, and ``TensorBuffer`` is a typed handle into it. See LifetimeEscape.swift for the run itself.
///
/// Non-`Sendable`: an arena belongs to one engine or pipeline actor and never leaves it. While a graph runs, its
/// buffers are lent to Core AI; any host access to the arena in that window (from a re-entrant call on the owning
/// actor) fails with `.tensor(…, .unavailable)` instead of racing the graph.
public final class TensorArena {
    /// The storage. Mutated only through the methods below; views into it exist only inside ``run``.
    var arrays: [NDArray] = []
    private var generations: [Int] = []
    private var freeSlots: [Int] = []
    /// Non-nil while a graph run holds views into `arrays`.
    private(set) var runningGraph: String?

    public init() {}

    /// Bytes held by live buffers.
    public private(set) var byteCount = 0

    // MARK: Allocation

    /// Allocates a zero-filled, contiguous row-major buffer.
    public func allocate(name: String, shape: [Int], scalarType: ScalarType) throws(AetherError) -> TensorBuffer {
        try allocate(name: name, shape: shape, scalarType: scalarType, strides: nil)
    }

    /// Allocates a zero-filled buffer with explicit (for example padded) strides, in elements.
    func allocate(name: String, shape: [Int], scalarType: ScalarType, strides: [Int]?) throws(AetherError) -> TensorBuffer {
        guard let coreType = scalarType.coreAI, scalarType.byteWidth != nil else {
            throw .tensor(name: name, reason: .unsupported("cannot allocate \(scalarType) tensors"))
        }
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw .tensor(name: name, reason: .shapeMismatch(expected: [], found: shape))
        }
        if let strides {
            guard strides.count == shape.count, strides.allSatisfy({ $0 > 0 }) else {
                throw .tensor(name: name, reason: .unsupported("strides \(strides) for shape \(shape)"))
            }
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
        guard ScalarType(descriptor.scalarType).byteWidth != nil else {
            throw .tensor(name: name, reason: .unsupported("cannot allocate \(ScalarType(descriptor.scalarType)) tensors"))
        }
        return try adopt(NDArray(descriptor: descriptor), name: name)
    }

    private func adopt(_ array: NDArray, name: String) throws(AetherError) -> TensorBuffer {
        try ensureIdle(name)
        let layout = StridedLayout(shape: array.shape, strides: array.strides)
        let scalarType = ScalarType(array.scalarType)
        let slot: Int
        if let reused = freeSlots.popLast() {
            slot = reused
            arrays[slot] = array
        } else {
            slot = arrays.count
            arrays.append(array)
            generations.append(0)
        }
        let buffer = TensorBuffer(
            arena: self, slot: slot, generation: generations[slot], name: name, scalarType: scalarType, layout: layout)
        byteCount += buffer.storageByteCount
        try buffer.zero()
        return buffer
    }

    /// Returns the buffer's storage to the arena. The handle (and any copy of it) is invalid afterwards.
    public func release(_ buffer: TensorBuffer) throws(AetherError) {
        try check(buffer)
        arrays[buffer.slot] = NDArray(shape: [1], scalarType: .int8)
        generations[buffer.slot] += 1
        freeSlots.append(buffer.slot)
        byteCount -= buffer.storageByteCount
    }

    // MARK: Access

    func check(_ buffer: TensorBuffer) throws(AetherError) {
        guard buffer.arena === self else {
            throw .tensor(name: buffer.name, reason: .unavailable("the buffer belongs to another arena"))
        }
        guard buffer.slot < generations.count, generations[buffer.slot] == buffer.generation else {
            throw .tensor(name: buffer.name, reason: .unavailable("the buffer was released"))
        }
        try ensureIdle(buffer.name)
    }

    private func ensureIdle(_ name: String) throws(AetherError) {
        if let runningGraph {
            throw .tensor(name: name, reason: .unavailable("lent to the running graph '\(runningGraph)'"))
        }
    }

    /// Typed pointer access to a buffer's storage, for the duration of `body`.
    func withMutablePointer<T: TensorScalar, R>(
        _ buffer: TensorBuffer, as type: T.Type, _ body: (UnsafeMutablePointer<T>) throws(AetherError) -> R
    ) throws(AetherError) -> R {
        try check(buffer)
        guard buffer.scalarType == T.scalarType else {
            throw .tensor(name: buffer.name, reason: .typeMismatch(expected: buffer.scalarType.rawValue, found: T.scalarType.rawValue))
        }
        return try Self.withPointer(&arrays[buffer.slot], as: T.self, body)
    }

    /// Raw access to the bytes a buffer's elements span.
    func withMutableBytes<R>(
        _ buffer: TensorBuffer, _ body: (UnsafeMutableRawBufferPointer) throws(AetherError) -> R
    ) throws(AetherError) -> R {
        try check(buffer)
        let count = buffer.storageByteCount
        return try Self.withBytes(&arrays[buffer.slot], count: count, body)
    }

    /// Copies `source`'s bytes over `destination`'s (same byte count; both checked). One pass, no staging copy.
    func copyBytes(from source: TensorBuffer, to destination: TensorBuffer) throws(AetherError) {
        try check(source)
        try check(destination)
        precondition(source.storageByteCount == destination.storageByteCount && source.slot != destination.slot)
        let count = source.storageByteCount
        // Two elements of `arrays` are accessed at once, so go through the buffer pointer: overlapping accesses to
        // the array itself would violate exclusivity.
        arrays.withUnsafeMutableBufferPointer { slots in
            try! Self.withBytes(&slots[source.slot], count: count) { from throws(AetherError) in
                try Self.withBytes(&slots[destination.slot], count: count) { to throws(AetherError) in
                    to.copyMemory(from: UnsafeRawBufferPointer(from))
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

    /// Marks the arena as lent to `graph` for the duration of a run. Used only by ``run``.
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
