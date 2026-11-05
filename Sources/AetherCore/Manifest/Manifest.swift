public import Foundation

// The `aether.json` schema, version 1 (DESIGN.md §4).
//
// Every schema type lives in this one file so that the schema can be revised in one place (the contract survey in
// lab/contracts may still move fields). Rules:
// - Unknown fields are ignored on decode, so newer bundles still load on older builds when they only add fields.
// - Optional fields are omitted when encoding, so goldens stay minimal.
// - Structural validation (`validate()`) is here too. Driver role rules live in DriverContract.swift; checks against
//   a real graph live in AetherRuntime.

/// A bundle's self-description: its graphs, tensor roles, tokenizer, preprocessing and provenance.
public struct Manifest: Sendable, Hashable, Codable {
    /// The schema version this build reads and writes.
    public static let currentSchema = 1
    /// The manifest's file name at the bundle root.
    public static let fileName = "aether.json"

    public var schema: Int
    /// The catalog id this bundle belongs to.
    public var id: String
    /// What the bundle does, e.g. `text-generation`.
    public var task: String
    /// Who loads it: the registry key.
    public var driver: DriverID
    /// Graphs by graph role (for `decoder-lm@1`, `decoder`).
    public var graphs: [String: Graph]
    public var tokenizer: Tokenizer?
    public var preprocess: Preprocess?
    public var precision: Precision?
    public var provenance: Provenance?

    public init(
        schema: Int = Manifest.currentSchema, id: String, task: String, driver: DriverID, graphs: [String: Graph],
        tokenizer: Tokenizer? = nil, preprocess: Preprocess? = nil, precision: Precision? = nil,
        provenance: Provenance? = nil
    ) {
        self.schema = schema
        self.id = id
        self.task = task
        self.driver = driver
        self.graphs = graphs
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.precision = precision
        self.provenance = provenance
    }

    // MARK: Graph

    /// One graph function and the roles of its tensors.
    public struct Graph: Sendable, Hashable, Codable {
        /// The `.aimodel` / `.aimodelc`, relative to the bundle root, possibly per device architecture.
        public var file: GraphFile
        public var function: String
        /// Input role → graph input name.
        public var inputs: [String: String]
        /// Output role → graph output name.
        public var outputs: [String: String]
        public var states: States
        public var limits: Limits
        /// How Core AI specializes the graph. Required to load: Aether never falls back to `.default` (§6.1).
        public var specialization: Specialization?

        public init(
            file: GraphFile, function: String, inputs: [String: String], outputs: [String: String],
            states: States = States(), limits: Limits = Limits(), specialization: Specialization? = nil
        ) {
            self.file = file
            self.function = function
            self.inputs = inputs
            self.outputs = outputs
            self.states = states
            self.limits = limits
            self.specialization = specialization
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            file = try c.decode(GraphFile.self, forKey: .file)
            function = try c.decode(String.self, forKey: .function)
            inputs = try c.decodeIfPresent([String: String].self, forKey: .inputs) ?? [:]
            outputs = try c.decodeIfPresent([String: String].self, forKey: .outputs) ?? [:]
            states = try c.decodeIfPresent(States.self, forKey: .states) ?? States()
            limits = try c.decodeIfPresent(Limits.self, forKey: .limits) ?? Limits()
            specialization = try c.decodeIfPresent(Specialization.self, forKey: .specialization)
        }
    }

    /// A graph file name, either one for every device or one per Core AI architecture with `*` as the fallback:
    /// `"x.aimodel"` or `{"h18p": "x.h18p.aimodelc", "*": "x.aimodel"}`.
    public enum GraphFile: Sendable, Hashable, Codable {
        case single(String)
        case perArchitecture([String: String])

        public static let anyArchitecture = "*"

        /// The file for `architecture`, or nil when the map has neither that architecture nor `*`.
        public func resolve(architecture: String) -> String? {
            switch self {
            case .single(let file): file
            case .perArchitecture(let map): map[architecture] ?? map[Self.anyArchitecture]
            }
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let single = try? c.decode(String.self) {
                self = .single(single)
            } else {
                self = .perArchitecture(try c.decode([String: String].self))
            }
        }

        public func encode(to encoder: any Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .single(let file): try c.encode(file)
            case .perArchitecture(let map): try c.encode(map)
            }
        }
    }

    /// State tensors by role.
    public struct States: Sendable, Hashable, Codable {
        /// Attention KV caches, as graph state names. A pair per cache (`[keys, values]`); fused caches are one pair.
        public var kv: [String]
        /// Recurrent / conv / SSM states: fixed size, rewound only by snapshot/restore.
        public var recurrent: [String]
        /// The axis of every `kv` state that indexes sequence positions.
        public var sequenceAxis: Int?

        public init(kv: [String] = [], recurrent: [String] = [], sequenceAxis: Int? = nil) {
            self.kv = kv
            self.recurrent = recurrent
            self.sequenceAxis = sequenceAxis
        }

        public var all: [String] { kv + recurrent }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            kv = try c.decodeIfPresent([String].self, forKey: .kv) ?? []
            recurrent = try c.decodeIfPresent([String].self, forKey: .recurrent) ?? []
            sequenceAxis = try c.decodeIfPresent(Int.self, forKey: .sequenceAxis)
        }
    }

    /// Execution limits. Each of them is a correctness limit, not a tuning knob.
    public struct Limits: Sendable, Hashable, Codable {
        /// Most tokens the graph may hold in state.
        public var maxContext: Int?
        /// Most tokens per prefill call (M0 showed single batches above ~800 tokens give wrong logits).
        public var prefillChunk: Int?
        /// The token axis is static with this length (decode-only graphs: 1).
        public var staticSequence: Int?

        public init(maxContext: Int? = nil, prefillChunk: Int? = nil, staticSequence: Int? = nil) {
            self.maxContext = maxContext
            self.prefillChunk = prefillChunk
            self.staticSequence = staticSequence
        }
    }

    /// Core AI specialization, stated per graph (§6.1). M0: a dynamic-shape decoder ran at 4 tok/s under
    /// `SpecializationOptions.default` and at 274 tok/s on the GPU with `expectFrequentReshapes`.
    public struct Specialization: Sendable, Hashable, Codable {
        public var computeUnit: ComputeUnit
        public var expectFrequentReshapes: Bool

        public init(computeUnit: ComputeUnit, expectFrequentReshapes: Bool) {
            self.computeUnit = computeUnit
            self.expectFrequentReshapes = expectFrequentReshapes
        }
    }

    public enum ComputeUnit: String, Sendable, Hashable, Codable, CaseIterable {
        case cpu
        case gpu
        case neuralEngine
    }

    // MARK: Tokenizer, preprocessing, precision, provenance

    public struct Tokenizer: Sendable, Hashable, Codable {
        /// Directory holding the tokenizer files, relative to the bundle root.
        public var dir: String
        public var chatTemplate: String?
        /// Stop markers, as token strings.
        public var stop: [String]
        public var thinking: Thinking?
        /// Tool-call dialect id (`hermes`, …), chosen by the manifest and never probed at runtime (§7).
        public var toolDialect: String?

        public init(
            dir: String, chatTemplate: String? = nil, stop: [String] = [], thinking: Thinking? = nil,
            toolDialect: String? = nil
        ) {
            self.dir = dir
            self.chatTemplate = chatTemplate
            self.stop = stop
            self.thinking = thinking
            self.toolDialect = toolDialect
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            dir = try c.decode(String.self, forKey: .dir)
            chatTemplate = try c.decodeIfPresent(String.self, forKey: .chatTemplate)
            stop = try c.decodeIfPresent([String].self, forKey: .stop) ?? []
            thinking = try c.decodeIfPresent(Thinking.self, forKey: .thinking)
            toolDialect = try c.decodeIfPresent(String.self, forKey: .toolDialect)
        }
    }

    public struct Thinking: Sendable, Hashable, Codable {
        public var open: String
        public var close: String

        public init(open: String, close: String) {
            self.open = open
            self.close = close
        }
    }

    /// Vision preprocessing (§8). Every parameter of the image pipeline comes from here; M4 fills in the semantics.
    public struct Preprocess: Sendable, Hashable, Codable {
        /// Target size `[height, width]`.
        public var resize: [Int]?
        /// `stretch`, `letterbox` or `centerCrop`.
        public var mode: String?
        public var mean: [Double]?
        public var std: [Double]?
        /// Tensor layout, e.g. `NCHW`.
        public var layout: String?
        /// Whether EXIF orientation is applied (`exif`) or ignored (`none`).
        public var orientation: String?

        public init(
            resize: [Int]? = nil, mode: String? = nil, mean: [Double]? = nil, std: [Double]? = nil,
            layout: String? = nil, orientation: String? = nil
        ) {
            self.resize = resize
            self.mode = mode
            self.mean = mean
            self.std = std
            self.layout = layout
            self.orientation = orientation
        }
    }

    public struct Precision: Sendable, Hashable, Codable {
        public var weights: String?
        public var activations: String?

        public init(weights: String? = nil, activations: String? = nil) {
            self.weights = weights
            self.activations = activations
        }
    }

    public struct Provenance: Sendable, Hashable, Codable {
        /// Upstream model, e.g. `Qwen/Qwen3-0.6B`.
        public var source: String?
        /// The parity gate this bundle passed.
        public var gate: String?
        public var zooRepo: String?
        public var zooRevision: String?
        /// Set when the manifest was synthesized, e.g. `zoo-bundle-adapter@1`.
        public var synthesizedBy: String?

        public init(
            source: String? = nil, gate: String? = nil, zooRepo: String? = nil, zooRevision: String? = nil,
            synthesizedBy: String? = nil
        ) {
            self.source = source
            self.gate = gate
            self.zooRepo = zooRepo
            self.zooRevision = zooRevision
            self.synthesizedBy = synthesizedBy
        }
    }
}

// MARK: - Coding

extension Manifest {
    /// Decodes and structurally validates a manifest. Errors name the offending field.
    public static func decode(_ data: Data) throws(AetherError) -> Manifest {
        let manifest: Manifest
        do {
            manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch let error as DecodingError {
            throw AetherError(decoding: error)
        } catch {
            throw AetherError.manifestInvalid(field: "", reason: "\(error)")
        }
        try manifest.validate()
        return manifest
    }

    /// Reads `aether.json` from a bundle directory.
    public static func load(fromBundle bundle: URL) throws(AetherError) -> Manifest {
        let url = bundle.appending(path: fileName)
        let data: Data
        do { data = try Data(contentsOf: url) } catch { throw AetherError.runtime(underlying: error) }
        return try decode(data)
    }

    /// Canonical JSON: pretty-printed, sorted keys, no escaped slashes. Goldens are compared in this form.
    public func encoded() throws(AetherError) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        do {
            var data = try encoder.encode(self)
            data.append(0x0A)
            return data
        } catch {
            throw AetherError.runtime(underlying: error)
        }
    }

    /// Writes `aether.json` into `bundle` atomically.
    public func write(toBundle bundle: URL) throws(AetherError) {
        do {
            try encoded().write(to: bundle.appending(path: Self.fileName), options: .atomic)
        } catch let error as AetherError {
            throw error
        } catch {
            throw AetherError.runtime(underlying: error)
        }
    }
}

extension AetherError {
    /// Maps a decoding failure onto the field it happened at.
    init(decoding error: DecodingError) {
        func path(_ codingPath: [any CodingKey], _ key: (any CodingKey)? = nil) -> String {
            (codingPath + (key.map { [$0] } ?? [])).map { $0.intValue.map(String.init) ?? $0.stringValue }
                .joined(separator: ".")
        }
        switch error {
        case .keyNotFound(let key, let context):
            self = .manifestInvalid(field: path(context.codingPath, key), reason: "required field is missing")
        case .typeMismatch(let type, let context):
            self = .manifestInvalid(field: path(context.codingPath), reason: "expected \(type)")
        case .valueNotFound(let type, let context):
            self = .manifestInvalid(field: path(context.codingPath), reason: "expected \(type), found null")
        case .dataCorrupted(let context):
            self = .manifestInvalid(field: path(context.codingPath), reason: context.debugDescription)
        @unknown default:
            self = .manifestInvalid(field: "", reason: "\(error)")
        }
    }
}

// MARK: - Structural validation

extension Manifest {
    /// Checks everything that can be checked without a driver or a graph. Errors name the offending field.
    public func validate() throws(AetherError) {
        guard schema == Self.currentSchema else {
            throw .manifestInvalid(
                field: "schema", reason: "unsupported schema \(schema); this build reads \(Self.currentSchema)")
        }
        try Self.require(!id.isEmpty, "id", "must not be empty")
        try Self.require(!task.isEmpty, "task", "must not be empty")
        try Self.require(!graphs.isEmpty, "graphs", "at least one graph is required")
        for key in graphs.keys.sorted() {
            try graphs[key]!.validate(at: "graphs.\(key)")
        }
        if let tokenizer {
            try Self.require(!tokenizer.dir.isEmpty, "tokenizer.dir", "must not be empty")
            for (i, stop) in tokenizer.stop.enumerated() {
                try Self.require(!stop.isEmpty, "tokenizer.stop.\(i)", "must not be empty")
            }
            if let thinking = tokenizer.thinking {
                try Self.require(!thinking.open.isEmpty, "tokenizer.thinking.open", "must not be empty")
                try Self.require(!thinking.close.isEmpty, "tokenizer.thinking.close", "must not be empty")
            }
        }
        if let preprocess {
            if let resize = preprocess.resize {
                try Self.require(resize.count == 2 && resize.allSatisfy { $0 > 0 }, "preprocess.resize",
                                 "must be [height, width] with positive values")
            }
            if let mean = preprocess.mean, let std = preprocess.std {
                try Self.require(mean.count == std.count, "preprocess.std", "must have as many values as mean")
            }
            if let std = preprocess.std {
                try Self.require(std.allSatisfy { $0 != 0 }, "preprocess.std", "must not contain 0")
            }
        }
    }

    static func require(_ condition: Bool, _ field: String, _ reason: String) throws(AetherError) {
        if !condition { throw .manifestInvalid(field: field, reason: reason) }
    }
}

extension Manifest.Graph {
    func validate(at prefix: String) throws(AetherError) {
        switch file {
        case .single(let name):
            try Manifest.require(!name.isEmpty, "\(prefix).file", "must not be empty")
        case .perArchitecture(let map):
            try Manifest.require(!map.isEmpty, "\(prefix).file", "the per-architecture map must not be empty")
            for (arch, name) in map {
                try Manifest.require(!name.isEmpty, "\(prefix).file.\(arch)", "must not be empty")
            }
        }
        try Manifest.require(!function.isEmpty, "\(prefix).function", "must not be empty")
        for (role, name) in inputs.sorted(by: { $0.key < $1.key }) {
            try Manifest.require(!name.isEmpty, "\(prefix).inputs.\(role)", "must name a graph input")
        }
        for (role, name) in outputs.sorted(by: { $0.key < $1.key }) {
            try Manifest.require(!name.isEmpty, "\(prefix).outputs.\(role)", "must name a graph output")
        }
        let inputNames = Array(inputs.values)
        try Manifest.require(Set(inputNames).count == inputNames.count, "\(prefix).inputs",
                             "two roles name the same graph input")
        let all = states.all
        for (i, name) in states.kv.enumerated() {
            try Manifest.require(!name.isEmpty, "\(prefix).states.kv.\(i)", "must name a graph state")
        }
        for (i, name) in states.recurrent.enumerated() {
            try Manifest.require(!name.isEmpty, "\(prefix).states.recurrent.\(i)", "must name a graph state")
        }
        try Manifest.require(Set(all).count == all.count, "\(prefix).states",
                             "a state is listed twice (kv and recurrent must be disjoint)")
        if let axis = states.sequenceAxis {
            try Manifest.require(axis >= 0, "\(prefix).states.sequenceAxis", "must be >= 0")
            try Manifest.require(!states.kv.isEmpty, "\(prefix).states.sequenceAxis", "is set but there are no kv states")
        }
        if let maxContext = limits.maxContext {
            try Manifest.require(maxContext > 0, "\(prefix).limits.maxContext", "must be > 0")
        }
        if let chunk = limits.prefillChunk {
            try Manifest.require(chunk > 0, "\(prefix).limits.prefillChunk", "must be > 0")
            if let maxContext = limits.maxContext {
                try Manifest.require(chunk <= maxContext, "\(prefix).limits.prefillChunk", "must be <= maxContext")
            }
        }
        if let s = limits.staticSequence {
            try Manifest.require(s > 0, "\(prefix).limits.staticSequence", "must be > 0")
            if let chunk = limits.prefillChunk {
                try Manifest.require(chunk % s == 0, "\(prefix).limits.prefillChunk",
                                     "must be a multiple of staticSequence (\(s))")
            }
        }
    }
}
