public import Foundation

/// The per-entry override table, `catalog/zoo-overrides.json` (MANIFEST.md §2.9): the only home of family
/// knowledge the zoo adapter cannot infer. Compiled in by `scripts/gen-catalog` and pinned by the goldens.
public struct ZooOverrides: Sendable, Hashable, Codable {
    public struct Entry: Sendable, Hashable, Codable {
        /// Must equal the catalog pin: an override is valid for exactly one revision.
        public var revision: String
        /// Variant key, or `*` for every variant without its own entry.
        public var variants: [String: Override]
    }

    /// What one variant overrides. Every field is optional; absent means "infer".
    public struct Override: Sendable, Hashable, Codable {
        public var driver: DriverID?
        public var task: String?
        public var tables: [String: Manifest.Table]?
        public var graphs: [String: GraphOverride]?
        public var tokenizer: TokenizerOverride?

        public init(
            driver: DriverID? = nil, task: String? = nil, tables: [String: Manifest.Table]? = nil,
            graphs: [String: GraphOverride]? = nil, tokenizer: TokenizerOverride? = nil
        ) {
            self.driver = driver
            self.task = task
            self.tables = tables
            self.graphs = graphs
            self.tokenizer = tokenizer
        }
    }

    public struct GraphOverride: Sendable, Hashable, Codable {
        public var file: AssetPath?
        public var functions: [String: String]?
        /// Merged over the inferred inputs (adds static inputs such as tables).
        public var inputs: [String: Manifest.Binding]?
        /// Replaces the inferred state classification.
        public var states: Manifest.States?
        public var compute: Manifest.ComputeUnit?
        public var warmup: Manifest.Warmup?
        public var kvCapacityCap: [String: Int]?

        public init(
            file: AssetPath? = nil, functions: [String: String]? = nil, inputs: [String: Manifest.Binding]? = nil,
            states: Manifest.States? = nil, compute: Manifest.ComputeUnit? = nil, warmup: Manifest.Warmup? = nil,
            kvCapacityCap: [String: Int]? = nil
        ) {
            self.file = file
            self.functions = functions
            self.inputs = inputs
            self.states = states
            self.compute = compute
            self.warmup = warmup
            self.kvCapacityCap = kvCapacityCap
        }
    }

    public struct TokenizerOverride: Sendable, Hashable, Codable {
        /// The pinned upstream stop list.
        public struct Stop: Sendable, Hashable, Codable {
            /// Where `eosTokenIds` was read: `<repo>@<revision>:<file>#<pointer>`.
            public var source: String
            public var eosTokenIds: [Int]
            /// The same ids as token strings, in the same order.
            public var tokens: [String]
        }

        /// Tri-state: absent (infer), explicit `null` (no template), or a path.
        public enum ChatTemplate: Sendable, Hashable {
            case none
            case path(AssetPath)
        }

        public var source: Manifest.TokenizerSource?
        public var chatTemplate: ChatTemplate?
        public var renderer: String?
        public var turnFormat: JSONValue?
        public var addBos: Manifest.AddBos?
        public var thinking: ThinkingOverride?
        public var toolDialect: String?
        public var stop: Stop?

        enum CodingKeys: String, CodingKey {
            case source, chatTemplate, renderer, turnFormat, addBos, thinking, toolDialect, stop
        }

        public init(
            source: Manifest.TokenizerSource? = nil, chatTemplate: ChatTemplate? = nil, renderer: String? = nil,
            turnFormat: JSONValue? = nil, addBos: Manifest.AddBos? = nil, thinking: ThinkingOverride? = nil,
            toolDialect: String? = nil, stop: Stop? = nil
        ) {
            self.source = source
            self.chatTemplate = chatTemplate
            self.renderer = renderer
            self.turnFormat = turnFormat
            self.addBos = addBos
            self.thinking = thinking
            self.toolDialect = toolDialect
            self.stop = stop
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            source = try c.decodeIfPresent(Manifest.TokenizerSource.self, forKey: .source)
            if c.contains(.chatTemplate) {
                chatTemplate = try c.decodeNil(forKey: .chatTemplate)
                    ? ChatTemplate.none : .path(try c.decode(AssetPath.self, forKey: .chatTemplate))
            }
            renderer = try c.decodeIfPresent(String.self, forKey: .renderer)
            turnFormat = try c.decodeIfPresent(JSONValue.self, forKey: .turnFormat)
            addBos = try c.decodeIfPresent(Manifest.AddBos.self, forKey: .addBos)
            thinking = try c.decodeIfPresent(ThinkingOverride.self, forKey: .thinking)
            toolDialect = try c.decodeIfPresent(String.self, forKey: .toolDialect)
            stop = try c.decodeIfPresent(Stop.self, forKey: .stop)
        }

        public func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encodeIfPresent(source, forKey: .source)
            switch chatTemplate {
            case .none?: try c.encodeNil(forKey: .chatTemplate)
            case .path(let p)?: try c.encode(p, forKey: .chatTemplate)
            case nil: break
            }
            try c.encodeIfPresent(renderer, forKey: .renderer)
            try c.encodeIfPresent(turnFormat, forKey: .turnFormat)
            try c.encodeIfPresent(addBos, forKey: .addBos)
            try c.encodeIfPresent(thinking, forKey: .thinking)
            try c.encodeIfPresent(toolDialect, forKey: .toolDialect)
            try c.encodeIfPresent(stop, forKey: .stop)
        }
    }

    /// Fields replace the inferred ones; a full `open`/`close` pair declares thinking where none was inferred.
    public struct ThinkingOverride: Sendable, Hashable, Codable {
        public var open: String?
        public var close: String?
        public var toggle: Manifest.Thinking.Toggle?
        public var defaultOn: Bool?
    }

    public var schema: Int
    public var entries: [String: Entry]

    public init(schema: Int = 1, entries: [String: Entry]) {
        self.schema = schema
        self.entries = entries
    }

    /// The override for `variant` of `entry`, checked against the catalog pin. Nil when the table has none.
    public func override(for entry: CatalogEntry, variant: String) throws(AetherError) -> Override? {
        guard let item = entries[entry.id] else { return nil }
        guard item.revision == entry.revision else {
            throw .manifestInvalid(
                field: "zoo-overrides.\(entry.id).revision",
                reason: "pinned at \(item.revision), but the catalog is at \(entry.revision); re-review the overrides")
        }
        return item.variants[variant] ?? item.variants["*"]
    }

    public static func decode(_ data: Data) throws(AetherError) -> ZooOverrides {
        let table = try decodeSanitized(ZooOverrides.self, from: data)
        guard table.schema == 1 else {
            throw .manifestInvalid(field: "zoo-overrides.schema", reason: "unsupported schema \(table.schema)")
        }
        return table
    }

    /// The compiled-in table. `scripts/gen-catalog check` and the tests keep it decodable.
    public static let builtin: ZooOverrides = {
        do { return try decode(Data(builtinJSON.utf8)) } catch { preconditionFailure("zoo-overrides.json: \(error)") }
    }()
}
