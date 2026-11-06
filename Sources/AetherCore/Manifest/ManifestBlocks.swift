// MANIFEST.md §2.2 (variant, assets, tables, provenance), §2.4 (tokenizer) and the driver-owned blocks §2.5–§2.7.
// `image`, `decision` and `preprocess.audio` are typed but loose: their key fields are typed, nested parameters are
// JSONValue, and validation is structural only, until their drivers arrive (M4 VLM slices, M5). `preprocess.image`,
// `postprocess` and `labels` are fully typed in ManifestVision.swift.

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
        /// How an encoder's plain text becomes its fixed-length ids (§2.4 `tokenizer.text`).
        public var text: Text?
        public var escapes: [Escape]?
        public var preTokenize: String?
        public var `class`: String?
        public var markerIds: JSONValue?

        public init(
            source: TokenizerSource, chatTemplate: AssetPath? = nil, chatTemplateSha256: String? = nil,
            renderer: String? = nil, turnFormat: JSONValue? = nil, addBos: AddBos? = nil,
            addSpecialTokens: Bool? = nil, stop: [StopToken]? = nil, thinking: Thinking? = nil,
            toolDialect: String? = nil, text: Text? = nil, escapes: [Escape]? = nil,
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
                try require(isSafeRelativePath(path.path, allowDot: true), "tokenizer.source", "'\(path.path)' is not a safe relative path")
            case .hf(let repo, let revision):
                try require(!repo.isEmpty, "tokenizer.source.hf", "must not be empty")
                try require(revision.count == 40 && revision.allSatisfy(\.isHexDigit), "tokenizer.source.revision",
                            "must be a full commit SHA: a tokenizer is never fetched unpinned")
            case .embedded:
                break
            }
            if let chatTemplate {
                try require(assets[chatTemplate.asset] != nil, "tokenizer.chatTemplate", "unknown asset '\(chatTemplate.asset)'")
                try require(isSafeRelativePath(chatTemplate.path), "tokenizer.chatTemplate", "'\(chatTemplate.path)' is not a safe relative path")
            }
            for (i, stop) in (stop ?? []).enumerated() {
                if case .text(let s) = stop { try require(!s.isEmpty, "tokenizer.stop.\(i)", "must not be empty") }
                // Token ids are Int32 at run time.
                if case .id(let id) = stop { try require((0...Int(Int32.max)).contains(id), "tokenizer.stop.\(i)", "must be in 0...\(Int32.max)") }
            }
            if let thinking {
                try require(!thinking.open.isEmpty, "tokenizer.thinking.open", "must not be empty")
                try require(!thinking.close.isEmpty, "tokenizer.thinking.close", "must not be empty")
            }
            try text?.validate()
        }

        /// `tokenizer.text` (§2.4): an encoder's plain-text rules. The text is the role's prompt followed by the input,
        /// stripped of leading and trailing whitespace and encoded with the tokenizer's special tokens
        /// (`addSpecialTokens`); a longer one is truncated, then padded to exactly `padTo` ids.
        public struct Text: Sendable, Hashable, Codable {
            public enum Side: String, Sendable, Hashable, Codable { case left, right }

            public enum Truncation: String, Sendable, Hashable, Codable {
                /// Keep the first content ids and every special token the tokenizer adds (HF `longest_first` for one
                /// sequence).
                case right
                /// Cut at a word boundary.
                case word
                /// Truncate only a template's body (`template`).
                case bodyOnly
            }

            public enum MaskRule: String, Sendable, Hashable, Codable {
                /// 1 for every real id (prompt and special tokens included), 0 for padding.
                case realTokens
            }

            /// The padding id.
            public var padId: Int?
            /// Default `right`.
            public var padSide: Side?
            /// The fixed length every text is padded to (the graph's static sequence).
            public var padTo: Int?
            /// Default `right`.
            public var truncation: Truncation?
            /// Role → prompt prepended to the text (EmbeddingGemma: `query`, `document`).
            public var prompts: [String: String]?
            /// Default `realTokens`.
            public var maskRule: MaskRule?

            public init(
                padId: Int? = nil, padSide: Side? = nil, padTo: Int? = nil, truncation: Truncation? = nil,
                prompts: [String: String]? = nil, maskRule: MaskRule? = nil
            ) {
                self.padId = padId
                self.padSide = padSide
                self.padTo = padTo
                self.truncation = truncation
                self.prompts = prompts
                self.maskRule = maskRule
            }

            func validate() throws(AetherError) {
                if let padId { try require((0...Int(Int32.max)).contains(padId), "tokenizer.text.padId", "must be in 0...\(Int32.max)") }
                if let padTo { try require(padTo > 0, "tokenizer.text.padTo", "must be positive") }
                for key in (prompts ?? [:]).keys.sorted() {
                    try require(!key.isEmpty, "tokenizer.text.prompts", "a role must not be empty")
                }
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

    // MARK: Driver-owned blocks

    /// §2.5: `{image?, audio?}`. `image` is typed (ManifestVision.swift); `audio`'s parameters are interpreted by the
    /// audio encoder's driver (M5).
    public struct Preprocess: Sendable, Hashable, Codable {
        public var image: ImagePreprocess?
        public var audio: [String: JSONValue]?

        public init(image: ImagePreprocess? = nil, audio: [String: JSONValue]? = nil) {
            self.image = image
            self.audio = audio
        }

        func validate() throws(AetherError) {
            try require(image != nil || audio != nil, "preprocess", "must contain image or audio")
            try image?.validate(at: "preprocess.image")
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
        /// An adapter merged into the weights before export (MANIFEST.md §5.3), e.g. Krea 2's realism LoKr.
        public struct MergedAdapter: Sendable, Hashable, Codable {
            public var file: String
            public var sha256: String
            public var format: String
            /// How it was merged, e.g. `W + multiplier * scale * kron(w1, w2), float32, before export`.
            public var merge: String
            public var multiplier: Double
            /// The adapter's scale per module group: one number, or a list (ai-toolkit records `[1.0]`).
            public var scale: [Double]

            public init(file: String, sha256: String, format: String, merge: String, multiplier: Double, scale: [Double]) {
                self.file = file
                self.sha256 = sha256
                self.format = format
                self.merge = merge
                self.multiplier = multiplier
                self.scale = scale
            }

            enum CodingKeys: String, CodingKey { case file, sha256, format, merge, multiplier, scale }

            public init(from decoder: any Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                file = try c.decode(String.self, forKey: .file)
                sha256 = try c.decode(String.self, forKey: .sha256)
                format = try c.decode(String.self, forKey: .format)
                merge = try c.decode(String.self, forKey: .merge)
                multiplier = try c.decode(Double.self, forKey: .multiplier)
                if let one = try? c.decode(Double.self, forKey: .scale) {
                    scale = [one]
                } else {
                    scale = try c.decode([Double].self, forKey: .scale)
                }
            }

            /// A single scale encodes as a number.
            public func encode(to encoder: any Encoder) throws {
                var c = encoder.container(keyedBy: CodingKeys.self)
                try c.encode(file, forKey: .file)
                try c.encode(sha256, forKey: .sha256)
                try c.encode(format, forKey: .format)
                try c.encode(merge, forKey: .merge)
                try c.encode(multiplier, forKey: .multiplier)
                if scale.count == 1 { try c.encode(scale[0], forKey: .scale) } else { try c.encode(scale, forKey: .scale) }
            }
        }

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
        public var mergedAdapter: MergedAdapter?
        /// Manifest field → where its value came from (an inlined `$ref`, or a pinned upstream file).
        public var refs: [String: String]?

        public init(
            source: String? = nil, sourceRevision: String? = nil, base: String? = nil, adapter: String? = nil,
            zooRepo: String? = nil, zooRevision: String? = nil, subtrees: [String: String]? = nil,
            zooCompression: String? = nil, producer: String? = nil, exportScript: String? = nil,
            zooEngineHint: String? = nil, metadataKind: String? = nil, gate: String? = nil,
            refs: [String: String]? = nil, mergedAdapter: MergedAdapter? = nil
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
            self.mergedAdapter = mergedAdapter
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
