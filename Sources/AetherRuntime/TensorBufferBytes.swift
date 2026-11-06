// Raw bytes in and out of a buffer, for scalar types the typed `write`/`read` don't cover (bf16 travels as its bit
// pattern) and for moving a tensor between a host arena (`Graph.run`) and a Metal arena (a `GraphChain`).

extension TensorBuffer {
    /// The bytes the buffer's elements span (row-major for a contiguous buffer).
    public func readBytes() throws(AetherError) -> [UInt8] {
        try arena.withMutableBytes(self) { bytes throws(AetherError) in Array(UnsafeRawBufferPointer(bytes)) }
    }

    /// Overwrites the buffer's bytes with `bytes`, which must be exactly ``storageByteCount`` long.
    public func writeBytes(_ bytes: UnsafeRawBufferPointer) throws(AetherError) {
        guard bytes.count == storageByteCount else {
            throw .tensor(name: name, reason: .unsupported("\(bytes.count) bytes for a buffer of \(storageByteCount)"))
        }
        try arena.withMutableBytes(self) { target throws(AetherError) in target.copyMemory(from: bytes) }
    }

    /// Overwrites the buffer's bytes with `bytes`, which must be exactly ``storageByteCount`` long.
    public func writeBytes(_ bytes: [UInt8]) throws(AetherError) {
        guard bytes.count == storageByteCount else {
            throw .tensor(name: name, reason: .unsupported("\(bytes.count) bytes for a buffer of \(storageByteCount)"))
        }
        try arena.withMutableBytes(self) { target throws(AetherError) in bytes.withUnsafeBytes { target.copyMemory(from: $0) } }
    }

    /// Copies `source`'s bytes in: same scalar type and shape, any arena (host to Metal, say).
    public func copyBytes(from source: TensorBuffer) throws(AetherError) {
        guard source.scalarType == scalarType, source.shape == shape, source.isContiguous, isContiguous else {
            throw .tensor(name: name, reason: .shapeMismatch(expected: shape, found: source.shape))
        }
        if source.isSameStorage(as: self) { return }
        // Two buffers of one arena: nested `withMutableBytes` would overlap two mutable accesses to its host storage
        // (an exclusivity trap); the arena takes both at once.
        if source.arena === arena { return try arena.copyBytes(from: source, to: self) }
        try source.arena.withMutableBytes(source) { from throws(AetherError) in
            try arena.withMutableBytes(self) { to throws(AetherError) in to.copyMemory(from: UnsafeRawBufferPointer(from)) }
        }
    }
}

extension WeightBanks {
    /// Reads block `block` into bank `bank` outside any chain, while the arena isn't lent (a one-block weights-as-inputs
    /// graph run with `Graph.run`, say). Nothing may be running on the bank.
    public func load(block: Int, into bank: Int) throws(AetherError) {
        guard table.blocks.indices.contains(block), banks.indices.contains(bank) else {
            throw .tensor(name: "weights", reason: .outOfBounds(index: block, count: table.blockCount))
        }
        try fill(block: block, bank: bank) { buffer, body throws(AetherError) in try buffer.arena.withMutableBytes(buffer, body) }
    }
}
