/// A named group of state buffers: a decoder's KV caches and recurrent states (DESIGN.md §6.1).
///
/// The same set is bound into every ``Bindings`` of an engine (prefill-chunk and single-step shapes share state).
/// `snapshot()` is a host copy; on Qwen3.5 the 35 MB of hybrid state copied in 4 ms in M0, cheap enough to take one
/// at the end of every prompt.
public struct StateSet {
    public let arena: TensorArena
    /// In binding order.
    public private(set) var entries: [(name: String, buffer: TensorBuffer)]

    public init(arena: TensorArena, entries: [(name: String, buffer: TensorBuffer)]) throws(AetherError) {
        var seen = Set<String>()
        for (name, buffer) in entries {
            try arena.check(buffer)
            guard seen.insert(name).inserted else {
                throw .tensor(name: name, reason: .unsupported("the state is listed twice"))
            }
        }
        self.arena = arena
        self.entries = entries
    }

    public var names: [String] { entries.map(\.name) }

    public subscript(name: String) -> TensorBuffer? {
        entries.first { $0.name == name }?.buffer
    }

    /// Bytes held by the states.
    public var byteCount: Int { entries.reduce(0) { $0 + $1.buffer.storageByteCount } }

    /// Zeros every state (a fresh sequence).
    public func zero() throws(AetherError) {
        for (_, buffer) in entries { try buffer.zero() }
    }

    /// Copies every state to host memory.
    public func snapshot() throws(AetherError) -> StateSnapshot {
        var result = StateSnapshot(entries: [])
        try self.snapshot(into: &result)
        return result
    }

    /// Copies only the leading `regions[name]` of the named states (the rest whole): a KV cache up to the position
    /// rather than its whole capacity. Restoring writes those regions back and leaves the rest of the buffer as is.
    public func snapshot(regions: [String: [Int]]) throws(AetherError) -> StateSnapshot {
        var result = StateSnapshot(entries: [])
        for (name, buffer) in entries {
            guard let region = regions[name] else {
                var whole = StateSnapshot(entries: [])
                try StateSet(arena: arena, entries: [(name, buffer)]).snapshot(into: &whole)
                result.entries += whole.entries
                continue
            }
            guard region.count == buffer.shape.count, zip(region, buffer.shape).allSatisfy({ $0 >= 0 && $0 <= $1 }),
                  let width = buffer.scalarType.byteWidth
            else { throw .tensor(name: name, reason: .shapeMismatch(expected: buffer.shape, found: region)) }
            var bytes = [UInt8](repeating: 0, count: region.reduce(1, *) * width)
            try arena.withMutableBytes(buffer) { source in
                bytes.withUnsafeMutableBytes { packed in
                    Self.forEachRun(region: region, strides: buffer.strides, width: width) { strided, offset, count in
                        (packed.baseAddress! + offset).copyMemory(from: source.baseAddress! + strided, byteCount: count)
                    }
                }
            }
            result.entries.append(StateSnapshot.Entry(
                name: name, scalarType: buffer.scalarType, shape: buffer.shape, strides: buffer.strides, bytes: bytes,
                region: region))
        }
        return result
    }

    /// Calls `body(stridedByteOffset, packedByteOffset, byteCount)` for each contiguous run of `region`.
    static func forEachRun(region: [Int], strides: [Int], width: Int, _ body: (Int, Int, Int) -> Void) {
        let rank = region.count
        guard rank > 0, region.allSatisfy({ $0 > 0 }) else { return }
        let inner = strides[rank - 1] == 1
        let run = inner ? region[rank - 1] : 1
        let outer = inner ? rank - 1 : rank
        var index = [Int](repeating: 0, count: max(outer, 1))
        var packed = 0
        for _ in 0..<region.prefix(outer).reduce(1, *) {
            var offset = 0
            for axis in 0..<outer { offset += index[axis] * strides[axis] }
            body(offset * width, packed, run * width)
            packed += run * width
            var axis = outer - 1
            while axis >= 0 {
                index[axis] += 1
                if index[axis] < region[axis] { break }
                index[axis] = 0
                axis -= 1
            }
        }
    }

    /// Copies every state into `snapshot`, reusing its storage when it already has this set's layout (so taking a
    /// snapshot after every prompt does not allocate).
    public func snapshot(into snapshot: inout StateSnapshot) throws(AetherError) {
        let reusable = snapshot.entries.count == entries.count
            && zip(snapshot.entries, entries).allSatisfy { $0.region == nil && $0.matches($1.name, $1.buffer) }
        if !reusable {
            snapshot.entries = entries.map { name, buffer in
                StateSnapshot.Entry(
                    name: name, scalarType: buffer.scalarType, shape: buffer.shape, strides: buffer.strides,
                    bytes: [UInt8](repeating: 0, count: buffer.storageByteCount))
            }
        }
        for i in entries.indices {
            let buffer = entries[i].buffer
            try arena.withMutableBytes(buffer) { source in
                snapshot.entries[i].bytes.withUnsafeMutableBytes { destination in
                    destination.copyMemory(from: UnsafeRawBufferPointer(source))
                }
            }
        }
    }

    /// Copies `snapshot` back into the states. The snapshot must come from a set with the same names and layouts.
    public func restore(_ snapshot: StateSnapshot) throws(AetherError) {
        guard snapshot.entries.count == entries.count else {
            throw .tensor(
                name: names.joined(separator: ","),
                reason: .unsupported("the snapshot holds \(snapshot.entries.count) states, the set \(entries.count)"))
        }
        for (entry, (name, buffer)) in zip(snapshot.entries, entries) {
            guard entry.matches(name, buffer) else {
                throw .tensor(name: name, reason: .shapeMismatch(expected: buffer.shape, found: entry.shape))
            }
        }
        for (entry, (_, buffer)) in zip(snapshot.entries, entries) {
            try arena.withMutableBytes(buffer) { destination in
                entry.bytes.withUnsafeBytes { source in
                    if let region = entry.region {
                        Self.forEachRun(region: region, strides: buffer.strides, width: buffer.scalarType.byteWidth!) { strided, offset, count in
                            (destination.baseAddress! + strided).copyMemory(from: source.baseAddress! + offset, byteCount: count)
                        }
                    } else {
                        destination.copyMemory(from: source)
                    }
                }
            }
        }
    }
}

/// Host copies of a ``StateSet``'s buffers. A `Sendable` value, so an engine can hand it out.
public struct StateSnapshot: Sendable {
    public struct Entry: Sendable {
        public var name: String
        public var scalarType: ScalarType
        public var shape: [Int]
        public var strides: [Int]
        var bytes: [UInt8]
        /// The leading region stored, or nil for the whole buffer.
        public var region: [Int]?

        init(name: String, scalarType: ScalarType, shape: [Int], strides: [Int], bytes: [UInt8], region: [Int]? = nil) {
            self.name = name
            self.scalarType = scalarType
            self.shape = shape
            self.strides = strides
            self.bytes = bytes
            self.region = region
        }

        func matches(_ name: String, _ buffer: TensorBuffer) -> Bool {
            guard self.name == name, scalarType == buffer.scalarType, shape == buffer.shape, strides == buffer.strides else {
                return false
            }
            if let region { return bytes.count == region.reduce(1, *) * (scalarType.byteWidth ?? 0) }
            return bytes.count == buffer.storageByteCount
        }
    }

    public internal(set) var entries: [Entry]

    public var byteCount: Int { entries.reduce(0) { $0 + $1.bytes.count } }
}
