public import AetherCore
public import Foundation
import Darwin

/// Where each block's weights live in an installed bundle (MANIFEST.md §5.8, proposed): per block, for each weight
/// input of the block graph, a byte extent of a bundle file. One weights-as-inputs graph then serves every block
/// (lab/h3-spike §2: baking a graph per block costs about 2.5× the weight bytes in Metal per resident graph).
///
/// JSON form: `{"inputs": ["w0", …], "blocks": [{"w0": {"file": "main:weights/block-00.bin", "offset": 0,
/// "bytes": 16384}, …}, …]}`.
public struct WeightTable: Sendable, Hashable, Codable {
    public struct Extent: Sendable, Hashable, Codable {
        /// `"<asset>:<path>"` of a bundle file.
        public var file: String
        public var offset: Int64
        public var bytes: Int64

        public init(file: String, offset: Int64 = 0, bytes: Int64) {
            self.file = file
            self.offset = offset
            self.bytes = bytes
        }
    }

    /// The graph inputs bound from weights, in the order the exporter lifted them.
    public var inputs: [String]
    /// Per block, each input's extent.
    public var blocks: [[String: Extent]]

    public init(inputs: [String], blocks: [[String: Extent]]) throws(AetherError) {
        self.inputs = inputs
        self.blocks = blocks
        try validate()
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        inputs = try container.decode([String].self, forKey: .inputs)
        blocks = try container.decode([[String: Extent]].self, forKey: .blocks)
        try validate()
    }

    /// Reads a table document (a bundle file the manifest names).
    public init(contentsOf url: URL) throws(AetherError) {
        do {
            self = try JSONDecoder().decode(WeightTable.self, from: Data(contentsOf: url))
        } catch let error as AetherError {
            throw error
        } catch {
            throw .manifestInvalid(field: "weights.table", reason: "\(url.lastPathComponent): \(error)")
        }
    }

    public var blockCount: Int { blocks.count }

    func validate() throws(AetherError) {
        guard !inputs.isEmpty, Set(inputs).count == inputs.count else {
            throw .manifestInvalid(field: "weights.inputs", reason: "must name distinct inputs")
        }
        guard !blocks.isEmpty else { throw .manifestInvalid(field: "weights.blocks", reason: "no blocks") }
        for (b, block) in blocks.enumerated() {
            guard Set(block.keys) == Set(inputs) else {
                throw .manifestInvalid(field: "weights.blocks[\(b)]", reason: "must give exactly the inputs \(inputs)")
            }
            for (name, extent) in block {
                guard extent.offset >= 0, extent.bytes > 0, (try? AssetPath(parsing: extent.file)) != nil else {
                    throw .manifestInvalid(field: "weights.blocks[\(b)].\(name)", reason: "a bad extent \(extent)")
                }
            }
        }
    }
}

/// Two banks of a block graph's weight inputs, refilled from bundle files while the GPU runs the other bank's block
/// (DESIGN.md §6.6). The 0.0.x prototype's H3 weight streaming, reading the installed bundle instead of the Hugging
/// Face cache.
///
/// - **No re-hashing:** the store verified every file at install. The caller holds the store's reader lease for the
///   bundle while the banks read it.
/// - **Sync rule:** ``load(block:into:chain:)`` refuses a bank whose buffers a call since the chain's last
///   ``GraphChain/synchronize()`` bound (the GPU may still read them). The usual order: encode block *i* from bank
///   *i* mod 2, load block *i + 1* into the other bank (overlapping the GPU), then synchronize.
/// - Each bank buffer is allocated from the graph's own descriptor, and an extent must hold exactly its bytes.
///
/// Non-`Sendable`: owned with the chain.
public final class WeightBanks {
    public let table: WeightTable
    /// Per bank, per input name, its buffer.
    public let banks: [[String: TensorBuffer]]
    /// The block each bank holds, if any.
    public private(set) var loaded: [Int?] = [nil, nil]
    private let urls: [String: URL]
    private var files: [URL: FileHandle] = [:]

    /// Banks for `graph`'s weight inputs in `arena`, reading files that `resolve` maps to URLs (an installed
    /// bundle's ``InstalledBundle/url(of:)``).
    public init(
        table: WeightTable, graph: any GraphBackend, arena: TensorArena, resolve: (AssetPath) -> URL?
    ) throws(AetherError) {
        self.table = table
        var banks: [[String: TensorBuffer]] = [[:], [:]]
        for k in 0..<2 {
            for name in table.inputs {
                banks[k][name] = try graph.makeBuffer(for: name, role: .input, dimensions: [], in: arena)
            }
        }
        var urls: [String: URL] = [:]
        for (b, block) in table.blocks.enumerated() {
            for (name, extent) in block {
                let buffer = banks[0][name]!
                guard buffer.isContiguous, Int64(buffer.storageByteCount) == extent.bytes else {
                    throw .manifestMismatch(
                        field: "weights.blocks[\(b)].\(name).bytes", expected: "\(buffer.storageByteCount) (\(buffer.shape))",
                        found: "\(extent.bytes)")
                }
                guard urls[extent.file] == nil else { continue }
                guard let url = resolve(try AssetPath(parsing: extent.file)) else {
                    throw .manifestInvalid(field: "weights.blocks[\(b)].\(name).file", reason: "'\(extent.file)' is not in the bundle")
                }
                urls[extent.file] = url
            }
        }
        self.banks = banks
        self.urls = urls
    }

    /// Banks for an installed bundle.
    public convenience init(
        table: WeightTable, graph: any GraphBackend, arena: TensorArena, bundle: InstalledBundle
    ) throws(AetherError) {
        try self.init(table: table, graph: graph, arena: arena, resolve: { bundle.url(of: $0) })
    }

    /// Reads block `block`'s weights into bank `bank` (0 or 1), unless it already holds them.
    public func load(block: Int, into bank: Int, chain: GraphChain) throws(AetherError) {
        guard table.blocks.indices.contains(block), banks.indices.contains(bank) else {
            throw .tensor(name: "weights", reason: .outOfBounds(index: block, count: table.blockCount))
        }
        guard loaded[bank] != block else { return }
        for buffer in banks[bank].values where chain.isInFlight(buffer) {
            throw .tensor(
                name: buffer.name,
                reason: .unavailable("bank \(bank) is bound by a call since the last synchronize(); the GPU may still read it"))
        }
        try fill(block: block, bank: bank) { buffer, body throws(AetherError) in
            try chain.arena.withLentMutableBytes(buffer, body)
        }
    }

    /// Reads block `block`'s extents into bank `bank` through `access` (lent or idle arena access).
    func fill(
        block: Int, bank: Int,
        access: (TensorBuffer, (UnsafeMutableRawBufferPointer) throws(AetherError) -> Void) throws(AetherError) -> Void
    ) throws(AetherError) {
        guard loaded[bank] != block else { return }
        loaded[bank] = nil
        for name in table.inputs {
            let extent = table.blocks[block][name]!
            let handle = try handle(for: urls[extent.file]!)
            try access(banks[bank][name]!) { bytes throws(AetherError) in
                try Self.read(handle, offset: extent.offset, into: bytes, name: extent.file)
            }
        }
        loaded[bank] = block
    }

    /// Binds bank `bank`'s buffers as the weight inputs.
    public func bind(bank: Int, into bindings: inout Bindings) throws(AetherError) {
        for (name, buffer) in banks[bank] { try bindings.bindInput(name, to: buffer) }
    }

    private func handle(for url: URL) throws(AetherError) -> FileHandle {
        if let open = files[url] { return open }
        do {
            let handle = try FileHandle(forReadingFrom: url)
            files[url] = handle
            return handle
        } catch {
            throw .runtime(underlying: error)
        }
    }

    /// `pread` until `bytes` is full; a short file is an error.
    private static func read(
        _ handle: FileHandle, offset: Int64, into bytes: UnsafeMutableRawBufferPointer, name: String
    ) throws(AetherError) {
        var done = 0
        while done < bytes.count {
            let n = pread(handle.fileDescriptor, bytes.baseAddress! + done, bytes.count - done, off_t(offset) + off_t(done))
            if n < 0 {
                if errno == EINTR { continue }
                throw .runtime(underlying: POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO))
            }
            guard n > 0 else {
                throw .integrity(file: name, expected: "\(bytes.count) bytes at \(offset)", actual: "\(done)")
            }
            done += n
        }
    }

    deinit {
        for handle in files.values { try? handle.close() }
    }
}
