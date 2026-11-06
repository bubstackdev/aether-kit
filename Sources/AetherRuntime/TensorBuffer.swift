import Accelerate

/// A typed handle to one tensor's storage in a ``TensorArena`` (DESIGN.md §6.1).
///
/// Allocated once, from a graph descriptor or an explicit shape, and reused across steps. All access is
/// stride-aware: indices are *logical* row-major element indices, mapped through the storage's strides, so padded
/// layouts read and write correctly. Non-`Sendable`, like its arena.
public struct TensorBuffer {
    let arena: TensorArena
    let slot: Int
    let generation: Int
    public let name: String
    public let scalarType: ScalarType
    let layout: StridedLayout

    init(arena: TensorArena, slot: Int, generation: Int, name: String, scalarType: ScalarType, layout: StridedLayout) {
        self.arena = arena
        self.slot = slot
        self.generation = generation
        self.name = name
        self.scalarType = scalarType
        self.layout = layout
    }

    public var shape: [Int] { layout.shape }
    /// Strides in elements.
    public var strides: [Int] { layout.strides }
    /// Logical element count.
    public var count: Int { layout.count }
    public var isContiguous: Bool { layout.isContiguous }
    /// Bytes from the first to one past the last element, given the strides.
    public var storageByteCount: Int { layout.spannedElements * (scalarType.byteWidth ?? 0) }

    /// Whether two handles name the same storage.
    public func isSameStorage(as other: TensorBuffer) -> Bool {
        arena === other.arena && slot == other.slot && generation == other.generation
    }

    // MARK: Typed access

    /// Writes `values` at logical indices `start ..< start + values.count`.
    public func write<T: TensorScalar>(_ values: some Collection<T>, startingAt start: Int = 0) throws(AetherError) {
        try checkRange(start, values.count)
        try arena.withMutablePointer(self, as: T.self) { pointer throws(AetherError) in
            var iterator = values.makeIterator()
            layout.forEachOffset(from: start, count: values.count) { _, offset in
                pointer[offset] = iterator.next()!
            }
        }
    }

    /// Writes `count` values produced by `value(i)` (logical index `start + i`), without an intermediate array.
    public func write<T: TensorScalar>(
        _ type: T.Type, startingAt start: Int = 0, count: Int, _ value: (Int) -> T
    ) throws(AetherError) {
        try checkRange(start, count)
        try arena.withMutablePointer(self, as: T.self) { pointer throws(AetherError) in
            layout.forEachOffset(from: start, count: count) { i, offset in pointer[offset] = value(i) }
        }
    }

    /// Sets every element to `value`.
    public func fill<T: TensorScalar>(_ value: T) throws(AetherError) {
        try write(T.self, count: count) { _ in value }
    }

    /// Reads logical elements `range`.
    public func read<T: TensorScalar>(_ type: T.Type, _ range: Range<Int>? = nil) throws(AetherError) -> [T] {
        let range = range ?? 0..<count
        try checkRange(range.lowerBound, range.count)
        return try arena.withMutablePointer(self, as: T.self) { pointer throws(AetherError) in
            [T](unsafeUninitializedCapacity: range.count) { out, initialized in
                layout.forEachOffset(from: range.lowerBound, count: range.count) { i, offset in
                    (out.baseAddress! + i).initialize(to: pointer[offset])
                }
                initialized = range.count
            }
        }
    }

    /// Reads logical elements `range` of a floating-point buffer (fp16, bf16, fp32) as `Float` into `out`,
    /// replacing its contents. `out`'s storage is reused when it is large enough, so a caller that keeps one array
    /// (for logits) reads every step without allocating.
    public func readFloats(_ range: Range<Int>, into out: inout [Float]) throws(AetherError) {
        try checkRange(range.lowerBound, range.count)
        out.removeAll(keepingCapacity: true)
        out.reserveCapacity(range.count)
        switch scalarType {
        case .float32:
            try arena.withMutablePointer(self, as: Float.self) { pointer throws(AetherError) in
                layout.forEachOffset(from: range.lowerBound, count: range.count) { _, offset in out.append(pointer[offset]) }
            }
        case .float16 where layout.isContiguous:
            // A logits row is ~150k values per token: convert with Accelerate, not element by element.
            out = [Float](repeating: 0, count: range.count)
            try arena.withMutablePointer(self, as: Float16.self) { pointer throws(AetherError) in
                out.withUnsafeMutableBufferPointer { dst in
                    var source = vImage_Buffer(
                        data: UnsafeMutableRawPointer(pointer + range.lowerBound), height: 1, width: vImagePixelCount(range.count),
                        rowBytes: range.count * 2)
                    var destination = vImage_Buffer(
                        data: UnsafeMutableRawPointer(dst.baseAddress!), height: 1, width: vImagePixelCount(range.count),
                        rowBytes: range.count * 4)
                    _ = vImageConvert_Planar16FtoPlanarF(&source, &destination, vImage_Flags(kvImageNoFlags))
                }
            }
        case .float32 where layout.isContiguous:
            out = [Float](repeating: 0, count: range.count)
            try arena.withMutablePointer(self, as: Float.self) { pointer throws(AetherError) in
                out.withUnsafeMutableBufferPointer { dst in
                    dst.baseAddress!.update(from: pointer + range.lowerBound, count: range.count)
                }
            }
        case .float16:
            try arena.withMutablePointer(self, as: Float16.self) { pointer throws(AetherError) in
                layout.forEachOffset(from: range.lowerBound, count: range.count) { _, offset in
                    out.append(Float(pointer[offset]))
                }
            }
        case .bfloat16:
            try arena.withMutableBytes(self) { bytes throws(AetherError) in
                let words = bytes.bindMemory(to: UInt16.self)
                layout.forEachOffset(from: range.lowerBound, count: range.count) { _, offset in
                    out.append(Float(bitPattern: UInt32(words[offset]) << 16))
                }
            }
        default:
            throw .tensor(name: name, reason: .typeMismatch(expected: "float16, bfloat16 or float32", found: scalarType.rawValue))
        }
    }

    /// Reads logical elements `range` as `Float`.
    public func readFloats(_ range: Range<Int>? = nil) throws(AetherError) -> [Float] {
        var out: [Float] = []
        try readFloats(range ?? 0..<count, into: &out)
        return out
    }

    /// Sets every byte the buffer spans to zero.
    public func zero() throws(AetherError) {
        try arena.withMutableBytes(self) { bytes in
            if let base = bytes.baseAddress { base.initializeMemory(as: UInt8.self, repeating: 0, count: bytes.count) }
        }
    }

    /// Copies the bytes of `source`, which must have the same scalar type, shape and strides.
    public func copy(from source: TensorBuffer) throws(AetherError) {
        guard source.scalarType == scalarType, source.layout == layout else {
            throw .tensor(name: name, reason: .shapeMismatch(expected: shape, found: source.shape))
        }
        if source.isSameStorage(as: self) { return }
        try arena.copyBytes(from: source, to: self)
    }

    /// Copies the leading region this buffer shares with `source` (the elementwise minimum of the two shapes). Same
    /// rank and scalar type; strides may differ. A KV cache moves to a new capacity this way.
    public func copyRegion(from source: TensorBuffer) throws(AetherError) {
        try arena.copyRegion(from: source, to: self)
    }

    private func checkRange(_ start: Int, _ length: Int) throws(AetherError) {
        guard start >= 0, length >= 0, start + length <= count else {
            throw .tensor(name: name, reason: .outOfBounds(index: start + length, count: count))
        }
    }
}

/// Maps logical row-major element indices to strided storage offsets.
struct StridedLayout: Hashable {
    let shape: [Int]
    let strides: [Int]
    let count: Int
    /// Elements from offset 0 to one past the largest offset.
    let spannedElements: Int
    let isContiguous: Bool

    /// Contiguous row-major strides, in elements.
    static func rowMajorStrides(_ shape: [Int]) -> [Int] {
        var strides = [Int](repeating: 1, count: shape.count)
        var running = 1
        for axis in shape.indices.reversed() {
            strides[axis] = running
            running *= shape[axis]
        }
        return strides
    }

    init(shape: [Int], strides: [Int]) {
        self.shape = shape
        self.strides = strides
        count = shape.reduce(1, *)
        spannedElements = count == 0 ? 0 : zip(shape, strides).reduce(1) { $0 + ($1.0 - 1) * $1.1 }
        var expected = 1
        var contiguous = true
        for axis in shape.indices.reversed() {
            if shape[axis] != 1 && strides[axis] != expected { contiguous = false; break }
            expected *= shape[axis]
        }
        isContiguous = contiguous
    }

    /// The storage offset of logical index `index`.
    func offset(of index: Int) -> Int {
        if isContiguous { return index }
        var remainder = index
        var offset = 0
        for axis in shape.indices.reversed() {
            offset += (remainder % shape[axis]) * strides[axis]
            remainder /= shape[axis]
        }
        return offset
    }

    /// Calls `body(i, offset)` for logical indices `start ..< start + count`, `i` counting from 0.
    @inline(__always)
    func forEachOffset(from start: Int, count: Int, _ body: (Int, Int) -> Void) {
        guard count > 0 else { return }
        if isContiguous {
            for i in 0..<count { body(i, start + i) }
            return
        }
        // Odometer over the multi-index, starting at `start`'s position.
        let rank = shape.count
        withUnsafeTemporaryAllocation(of: Int.self, capacity: rank) { index in
            var remainder = start
            var offset = 0
            for axis in (0..<rank).reversed() {
                index[axis] = remainder % shape[axis]
                remainder /= shape[axis]
                offset += index[axis] * strides[axis]
            }
            for i in 0..<count {
                body(i, offset)
                var axis = rank - 1
                while axis >= 0 {
                    index[axis] += 1
                    offset += strides[axis]
                    if index[axis] < shape[axis] { break }
                    offset -= index[axis] * strides[axis]
                    index[axis] = 0
                    axis -= 1
                }
            }
        }
    }
}
