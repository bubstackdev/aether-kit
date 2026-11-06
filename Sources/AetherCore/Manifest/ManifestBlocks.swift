// MANIFEST.md §2.2 (variant, assets, tables, provenance), §2.4 (tokenizer) and the driver-owned blocks
// §2.5–§2.8. The driver blocks are typed but loose: their key fields are typed, nested parameters are JSONValue,
// and validation is structural only, until their drivers arrive (M4/M5).

extension Manifest {
    /// `{platform: macos|ios, architectures?}`: the catalog variant this manifest describes.
    public struct Variant: Sendable, Hashable, Codable {
        public var platform: Device.Platform
        public var architectures: [String]?

        public init(platform: Device.Platform, architectures: [String]? = nil) {
            self.platform = platform
            self.architectures = architectures
        }

        func validate() throws(AetherError) {
            if let architectures {
                try require(!architectures.isEmpty && architectures.allSatisfy { !$0.isEmpty }, "variant.architectures",
                            "must be a non-empty list of architecture names")
            }
        }
    }

    /// One subtree of a repo. `repo`/`revision` default to the catalog entry's.
    public struct Asset: Sendable, Hashable, Codable {
        public struct File: Sendable, Hashable, Codable {
            public var path: String
            public var bytes: Int64
            public var sha256: String
        }

        public var repo: String?
        public var revision: String?
        /// Directory in the repo that is the asset root (`""` is the repo root). May name an `.aimodel` package.
        public var subtree: String
        public var files: [File]?

        public init(repo: String? = nil, revision: String? = nil, subtree: String, files: [File]? = nil) {
            self.repo = repo
            self.revision = revision
            self.subtree = subtree
            self.files = files
        }

        func validate(at prefix: String) throws(AetherError) {
            try require(subtree.isEmpty || isSafeRelativePath(subtree), "\(prefix).subtree", "'\(subtree)' is not a safe relative path")
            try require(repo == nil || revision != nil, "\(prefix).revision",
                        "a foreign repo needs a pinned revision")
            for (i, file) in (files ?? []).enumerated() {
                try require(isSafeRelativePath(file.path), "\(prefix).files.\(i).path", "'\(file.path)' is not a safe relative path")
            }
        }
    }

    /// A non-graph data file (PLE tables, embedding rows, head weights).
    public struct Table: Sendable, Hashable, Codable {
        public var asset: String
        public var file: String
        public var dtype: ScalarType
        public var shape: [Int]
        public var layout: String?
        public var sha256: String?

        public init(asset: String, file: String, dtype: ScalarType, shape: [Int], layout: String? = nil, sha256: String? = nil) {
            self.asset = asset
            self.file = file
            self.dtype = dtype
            self.shape = shape
            self.layout = layout
            self.sha256 = sha256
        }

        func validate(at prefix: String, assets: [String: Asset]) throws(AetherError) {
            try require(assets[asset] != nil, "\(prefix).asset", "unknown asset '\(asset)'")
            try require(isSafeRelativePath(file), "\(prefix).file", "'\(file)' is not a safe relative path")
            try require(!shape.isEmpty && shape.allSatisfy { $0 > 0 }, "\(prefix).shape", "must be positive dimensions")
        }

        /// Bytes the table occupies, when its dtype has a byte width.
        public var byteCount: Int64? {
            dtype.byteWidth.map { Int64($0) * shape.reduce(Int64(1)) { $0 * Int64($1) } }
        }
    }

    // MARK: Tokenizer (§2.4)

    public struct Tokenizer: Sendable, Hashable, Codable {
        public var source: TokenizerSource
        /// Absent or null: no jinja template (Gemma 4, decision formats, encoders).
        public var chatTemplate: AssetPath?
        public var chatTemplateSha256: String?
        /// Default `jinja`.
        public var renderer: String?
        public var turnFormat: JSONValue?
        /// Default `template`.
        public var addBos: AddBos?
        public var addSpecialTokens: Bool?
        public var stop: [StopToken]?
        public var thinking: Thinking?
        public var toolDialect: String?
        public var text: JSONValue?
        public var escapes: [Escape]?
        public var preTokenize: String?
        public var `class`: String?
        public var markerIds: JSONValue?

        public init(
            source: TokenizerSource, chatTemplate: AssetPath? = nil, chatTemplateSha256: String? = nil,
            renderer: String? = nil, turnFormat: JSONValue? = nil, addBos: AddBos? = nil,
            addSpecialTokens: Bool? = nil, stop: [StopToken]? = nil, thinking: Thinking? = nil,
            toolDialect: String? = nil, text: JSONValue? = nil, escapes: [Escape]? = nil,
            preTokenize: String? = nil, class tokenizerClass: String? = nil, markerIds: JSONValue? = nil
        ) {
            self.source = source
            self.chatTemplate = chatTemplate
            self.chatTemplateSha256 = chatTemplateSha256
            self.renderer = renderer
            self.turnFormat = turnFormat
            self.addBos = addBos
            self.addSpecialTokens = addSpecialTokens
            self.stop = stop
            self.thinking = thinking
            self.toolDialect = toolDialect
            self.text = text
            self.escapes = escapes
            self.preTokenize = preTokenize
            self.class = tokenizerClass
            self.markerIds = markerIds
        }

        func validate(assets: [String: Asset]) throws(AetherError) {
            switch source {
            case .dir(let path), .file(let path):
                try require(assets[path.asset] != nil, "tokenizer.source", "names asset '\(path.asset)', which is not in assets")
            case .hf(let repo, let revision):
                try require(!repo.isEmpty, "tokenizer.source.hf", "must not be empty")
                try require(revision.count == 40 && revision.allSatisfy(\.isHexDigit), "tokenizer.source.revision",
                            "must be a full commit SHA: a tokenizer is never fetched unpinned")
            case .embedded:
                break
            }
            if let chatTemplate {
                try require(assets[chatTemplate.asset] != nil, "tokenizer.chatTemplate", "unknown asset '\(chatTemplate.asset)'")
            }
            for (i, stop) in (stop ?? []).enumerated() {
                if case .text(let s) = stop { try require(!s.isEmpty, "tokenizer.stop.\(i)", "must not be empty") }
                if case .id(let id) = stop { try require(id >= 0, "tokenizer.stop.\(i)", "must be >= 0") }
            }
            if let thinking {
                try require(!thinking.open.isEmpty, "tokenizer.thinking.open", "must not be empty")
                try require(!thinking.close.isEmpty, "tokenizer.thinking.close", "must not be empty")
            }
        }
    }

    public enum TokenizerSource: Sendable, Hashable, Codable {
        /// `{dir: "<asset>:tokenizer"}`
        case dir(AssetPath)
        /// `{file: "<asset>:tokenizer.json"}`
        case file(AssetPath)
        /// `{hf: "owner/name", revision}`: installed from a pinned, hashed catalog asset.
        case hf(String, revision: String)
        /// `{embedded: true}`
        case embedded

        enum CodingKeys: String, CodingKey { case dir, file, hf, revision, embedded }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            if let dir = try c.decodeIfPresent(AssetPath.self, forKey: .dir) { self = .dir(dir) }
            else if let file = try c.decodeIfPresent(AssetPath.self, forKey: .file) { self = .file(file) }
            else if let hf = try c.decodeIfPresent(String.self, forKey: .hf) {
                self = .hf(hf, revision: try c.decode(String.self, forKey: .revision))
            } else if try c.decodeIfPresent(Bool.self, forKey: .embedded) == true {
                self = .embedded
            } else {
                throw DecodingError.dataCorrupted(.init(
                    codingPath: decoder.codingPath, debugDescription: "source must be one of dir, file, hf, embedded"))
            }
        }

        public func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .dir(let p): try c.encode(p, forKey: .dir)
            case .file(let p): try c.encode(p, forKey: .file)
            case .hf(let repo, let revision):
                try c.encode(repo, forKey: .hf)
                try c.encode(revision, forKey: .revision)
            case .embedded: try c.encode(true, forKey: .embedded)
            }
        }
    }

    public enum AddBos: Sendable, Hashable, Codable {
        /// The chat template owns BOS; the runtime adds none.
        case template
        case renderer
        case always
        case never

        public init(from decoder: any Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let flag = try? c.decode(Bool.self) { self = flag ? .always : .never; return }
            switch try c.decode(String.self) {
            case "template": self = .template
            case "renderer": self = .renderer
            default: throw DecodingError.dataCorruptedError(in: c, debugDescription: "addBos is template, renderer, true or false")
            }
        }

        public func encode(to encoder: any Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .template: try c.encode("template")
            case .renderer: try c.encode("renderer")
            case .always: try c.encode(true)
            case .never: try c.encode(false)
            }
        }
    }

    /// A stop marker, as a token string or a token id.
    public enum StopToken: Sendable, Hashable, Codable, ExpressibleByStringLiteral {
        case text(String)
        case id(Int)

        public init(stringLiteral value: String) { self = .text(value) }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let id = try? c.decode(Int.self) { self = .id(id) } else { self = .text(try c.decode(String.self)) }
        }

        public func encode(to encoder: any Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .text(let s): try c.encode(s)
            case .id(let i): try c.encode(i)
            }
        }
    }

    public struct Thinking: Sendable, Hashable, Codable {
        public enum Toggle: Sendable, Hashable, Codable {
            /// `{templateVar: "enable_thinking"}`
            case templateVar(String)
            /// `{emitEmpty: "<|channel>thought\n<channel|>"}`: disable by emitting an empty block.
            case emitEmpty(String)
            /// `"none"`: thinking cannot be switched off.
            case none

            enum CodingKeys: String, CodingKey { case templateVar, emitEmpty }

            public init(from decoder: any Decoder) throws {
                if let word = try? decoder.singleValueContainer().decode(String.self) {
                    guard word == "none" else {
                        throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "toggle string must be none"))
                    }
                    self = .none
                    return
                }
                let c = try decoder.container(keyedBy: CodingKeys.self)
                if let v = try c.decodeIfPresent(String.self, forKey: .templateVar) { self = .templateVar(v) }
                else { self = .emitEmpty(try c.decode(String.self, forKey: .emitEmpty)) }
            }

            public func encode(to encoder: any Encoder) throws {
                switch self {
                case .none:
                    var c = encoder.singleValueContainer()
                    try c.encode("none")
                case .templateVar(let v):
                    var c = encoder.container(keyedBy: CodingKeys.self)
                    try c.encode(v, forKey: .templateVar)
                case .emitEmpty(let v):
                    var c = encoder.container(keyedBy: CodingKeys.self)
                    try c.encode(v, forKey: .emitEmpty)
                }
            }
        }

        public var open: String
        public var close: String
        public var toggle: Toggle?
        public var defaultOn: Bool?

        public init(open: String, close: String, toggle: Toggle? = nil, defaultOn: Bool? = nil) {
            self.open = open
            self.close = close
            self.toggle = toggle
            self.defaultOn = defaultOn
        }
    }

    public struct Escape: Sendable, Hashable, Codable {
        public var from: String
        public var to: String
    }

    // MARK: Driver-owned blocks (typed but loose)

    /// §2.5: `{image?, audio?}`; parameters are interpreted by the image and audio pipelines (M4).
    public struct Preprocess: Sendable, Hashable, Codable {
        public var image: [String: JSONValue]?
        public var audio: [String: JSONValue]?

        func validate() throws(AetherError) {
            try require(image != nil || audio != nil, "preprocess", "must contain image or audio")
        }
    }

    /// §2.6: VLM and vision-decision image injection.
    public struct ImageBlock: Sendable, Hashable, Codable {
        public struct Feed: Sendable, Hashable, Codable {
            public var from: String
            public var to: String
        }
        public var feed: [Feed]
        public var tokens: [String: JSONValue]
        public var grid: [Int]?
        public var rope: JSONValue?
        public var deepstack: JSONValue?
        public var maxImages: Int?
        public var arrange: JSONValue?

        func validate() throws(AetherError) {
            try require(!feed.isEmpty, "image.feed", "must list at least one feed")
            for (i, f) in feed.enumerated() {
                try require(f.from.contains(".") && f.to.contains("."), "image.feed.\(i)", "from/to are <graph>.<role>")
            }
            if let maxImages { try require(maxImages > 0, "image.maxImages", "must be > 0") }
        }
    }

    /// §2.7. `promptFormat` and `maxOptions` are required; the rest is interpreted by the decision drivers (M5).
    public struct Decision: Sendable, Hashable, Codable {
        public var promptFormat: String
        public var maxOptions: Int
        public var layout: JSONValue?
        public var labels: JSONValue?
        public var slot: JSONValue?
        public var nSlots: Int?
        public var abstainSlot: JSONValue?
        public var multiSlotRows: JSONValue?
        public var rowPerOption: JSONValue?
        public var state: String?
        public var noulOrder: JSONValue?
        public var questionTypeOrder: JSONValue?
        public var readoutMarkers: JSONValue?
        public var head: JSONValue?

        func validate() throws(AetherError) {
            try require(!promptFormat.isEmpty, "decision.promptFormat", "must not be empty")
            try require(maxOptions > 0, "decision.maxOptions", "must be > 0")
        }
    }

    /// §2.8. `kind` is required; the rest is interpreted by each postprocess (M4).
    public struct Postprocess: Sendable, Hashable, Codable {
        public var kind: String
        public var parameters: [String: JSONValue]

        public init(from decoder: any Decoder) throws {
            var all = try decoder.singleValueContainer().decode([String: JSONValue].self)
            guard let kind = all.removeValue(forKey: "kind")?.string else {
                throw DecodingError.keyNotFound(
                    AnyKey("kind"), .init(codingPath: decoder.codingPath, debugDescription: "postprocess.kind is required"))
            }
            self.kind = kind
            parameters = all
        }

        public func encode(to encoder: any Encoder) throws {
            var all = parameters
            all["kind"] = .string(kind)
            var c = encoder.singleValueContainer()
            try c.encode(all)
        }

        func validate() throws(AetherError) {
            try require(!kind.isEmpty, "postprocess.kind", "must not be empty")
        }
    }

    /// `{set: "coco-91"}` or `{table}`.
    public struct Labels: Sendable, Hashable, Codable {
        public var set: String?
        public var table: String?

        func validate() throws(AetherError) {
            try require((set == nil) != (table == nil), "labels", "exactly one of set, table")
        }
    }

    public struct Generation: Sendable, Hashable, Codable {
        public var noRepeatNgram: Int?
        public var runCap: Int?
        public var prompts: [String: String]?
        public var temperature: Double?
    }

    /// `{temperature} | {byType} | {byTypeAndOptionCount}`, plus `source`.
    public struct Calibration: Sendable, Hashable, Codable {
        public var temperature: Double?
        public var byType: [String: Double]?
        public var byTypeAndOptionCount: [String: Double]?
        public var source: String?

        func validate() throws(AetherError) {
            let set = [temperature != nil, byType != nil, byTypeAndOptionCount != nil].filter { $0 }.count
            try require(set == 1, "calibration", "exactly one of temperature, byType, byTypeAndOptionCount")
        }
    }

    public struct Provenance: Sendable, Hashable, Codable {
        public var source: String?
        public var sourceRevision: String?
        public var base: String?
        /// The adapter that synthesized this manifest, e.g. `zoo-bundle-adapter@2`.
        public var adapter: String?
        public var zooRepo: String?
        public var zooRevision: String?
        /// Asset name → subtree.
        public var subtrees: [String: String]?
        public var zooCompression: String?
        public var producer: String?
        public var exportScript: String?
        /// Recorded, never drives behaviour.
        public var zooEngineHint: String?
        public var metadataKind: String?
        public var gate: String?
        /// Manifest field → where its value came from (an inlined `$ref`, or a pinned upstream file).
        public var refs: [String: String]?

        public init(
            source: String? = nil, sourceRevision: String? = nil, base: String? = nil, adapter: String? = nil,
            zooRepo: String? = nil, zooRevision: String? = nil, subtrees: [String: String]? = nil,
            zooCompression: String? = nil, producer: String? = nil, exportScript: String? = nil,
            zooEngineHint: String? = nil, metadataKind: String? = nil, gate: String? = nil,
            refs: [String: String]? = nil
        ) {
            self.source = source
            self.sourceRevision = sourceRevision
            self.base = base
            self.adapter = adapter
            self.zooRepo = zooRepo
            self.zooRevision = zooRevision
            self.subtrees = subtrees
            self.zooCompression = zooCompression
            self.producer = producer
            self.exportScript = exportScript
            self.zooEngineHint = zooEngineHint
            self.metadataKind = metadataKind
            self.gate = gate
            self.refs = refs
        }
    }
}

struct AnyKey: CodingKey {
    var stringValue: String
    var intValue: Int? { nil }
    init(_ string: String) { stringValue = string }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { nil }
}
