// Block weights as graph inputs (MANIFEST.md §5.8): a graph that serves many blocks (H3's transformer body, its
// conditioner's decoder layer) takes its weights as runtime inputs, bound per block from a weight table — a bundle
// file listing, per block, each weight input's byte extent in a bundle file. The table itself is read by the runtime
// (`WeightTable`, AetherRuntime); the manifest names it and the inputs it binds, so a loader can check the graph's
// signature and that no role binds a weight input.

extension Manifest {
    public struct GraphWeights: Sendable, Hashable, Codable {
        /// `"<asset>:<path>"` of the table file.
        public var table: AssetPath
        /// The graph inputs the table binds, in the order the exporter lifted them.
        public var inputs: [String]
        /// How many blocks the table holds (a loader's cross-check; the table is the authority).
        public var blocks: Int?

        public init(table: AssetPath, inputs: [String], blocks: Int? = nil) {
            self.table = table
            self.inputs = inputs
            self.blocks = blocks
        }

        func validate(at prefix: String, graph: Graph, manifest: Manifest) throws(AetherError) {
            try manifest.checkAssetPath(table, "\(prefix).table")
            try require(!inputs.isEmpty, "\(prefix).inputs", "must name at least one weight input")
            var seen = Set<String>()
            for name in inputs {
                try require(!name.isEmpty && seen.insert(name).inserted, "\(prefix).inputs", "'\(name)' is empty or listed twice")
            }
            let bound = Set(graph.inputs.values.map(\.name) + (graph.inputsByFunction ?? [:]).values.flatMap { $0.values.map(\.name) })
            if let clash = inputs.first(where: bound.contains) {
                throw .manifestInvalid(field: "\(prefix).inputs", reason: "'\(clash)' is also bound by a role; weight inputs come from the table only")
            }
            if let blocks { try require(blocks > 0, "\(prefix).blocks", "must be > 0") }
        }
    }
}
