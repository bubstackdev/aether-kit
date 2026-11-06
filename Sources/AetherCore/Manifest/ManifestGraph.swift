import Foundation

// MANIFEST.md §2.3: `graphs.<role>`.

/// `"<asset>:<path>"`: a file inside one of the manifest's assets. `"."` as the path is the asset itself.
public struct AssetPath: Sendable, Hashable, Codable, CustomStringConvertible {
    public var asset: String
    public var path: String

    public init(asset: String, path: String) {
        self.asset = asset
        self.path = path
    }

    public init(parsing text: String) throws(AetherError) {
        guard let colon = text.firstIndex(of: ":"), colon != text.startIndex else {
            throw .manifestInvalid(field: "", reason: "'\(text)' is not of the form <asset>:<path>")
        }
        asset = String(text[..<colon])
        path = String(text[text.index(after: colon)...])
    }

    public var description: String { "\(asset):\(path)" }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        let text = try c.decode(String.self)
        do { self = try AssetPath(parsing: text) } catch {
            throw DecodingError.dataCorruptedError(in: c, debugDescription: "'\(text)' is not of the form <asset>:<path>")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(description)
    }
}

extension Manifest {
    public enum ComputeUnit: String, Sendable, Hashable, Codable, CaseIterable {
        case cpu
        case gpu
        case neuralEngine
    }

    /// One graph file and the roles of its tensors.
    public struct Graph: Sendable, Hashable, Codable {
        public var file: GraphFile
        /// The single-function case. Exactly one of `function`, `functions`, `functionFamily` is set.
        public var function: String?
        /// Phase → function name (`step`, `prefill`, `decode`, `act`, …). All phases share the graph's states.
        public var functions: [String: String]?
        public var functionFamily: FunctionFamily?
        public var buckets: Buckets?
        public var inputs: [String: Binding]
        /// Phase → role → binding, when phases take different inputs. Absent: `inputs` applies to every phase.
        public var inputsByFunction: [String: [String: Binding]]?
        public var inputEncodings: [String: InputEncoding]?
        public var outputs: [String: OutputBinding]
        public var unusedOutputs: [String]?
        public var states: States?
        public var limits: Limits?
        /// Required: nothing falls back to Core AI's defaults.
        public var compute: ComputeUnit
        /// Default: `compute == .gpu` and any input or state has a dynamic dimension.
        public var expectFrequentReshapes: Bool?
        public var warmup: Warmup?
        /// §5.3: when a generative driver loads and releases the graph. Absent: resident for the model's lifetime.
        public var residency: Residency?

        public init(
            file: GraphFile, function: String? = nil, functions: [String: String]? = nil,
            functionFamily: FunctionFamily? = nil, buckets: Buckets? = nil, inputs: [String: Binding],
            inputsByFunction: [String: [String: Binding]]? = nil, inputEncodings: [String: InputEncoding]? = nil,
            outputs: [String: OutputBinding], unusedOutputs: [String]? = nil, states: States? = nil,
            limits: Limits? = nil, compute: ComputeUnit, expectFrequentReshapes: Bool? = nil, warmup: Warmup? = nil,
            residency: Residency? = nil
        ) {
            self.file = file
            self.function = function
            self.functions = functions
            self.functionFamily = functionFamily
            self.buckets = buckets
            self.inputs = inputs
            self.inputsByFunction = inputsByFunction
            self.inputEncodings = inputEncodings
            self.outputs = outputs
            self.unusedOutputs = unusedOutputs
            self.states = states
            self.limits = limits
            self.compute = compute
            self.expectFrequentReshapes = expectFrequentReshapes
            self.warmup = warmup
            self.residency = residency
        }

        /// Phase → function name: `function` is the `step` phase.
        public var phases: [String: String] {
            if let function { return ["step": function] }
            return functions ?? [:]
        }

        /// The input bindings of `phase` (its own set if `inputsByFunction` lists it).
        public func inputs(forPhase phase: String) -> [String: Binding] {
            inputsByFunction?[phase] ?? inputs
        }

        /// The effective reshape hint, given whether the graph has a dynamic input or state.
        public func expectFrequentReshapes(hasDynamicDimension: Bool) -> Bool {
            expectFrequentReshapes ?? (compute == .gpu && hasDynamicDimension)
        }
    }

    /// `"main:x.aimodel"`, or per architecture `{"h18p": "main:x.h18p.aimodelc", "*": "main:x.aimodel"}`.
    public enum GraphFile: Sendable, Hashable, Codable {
        case single(AssetPath)
        case perArchitecture([String: AssetPath])

        public static let anyArchitecture = "*"

        public func resolve(architecture: String) -> AssetPath? {
            switch self {
            case .single(let path): path
            case .perArchitecture(let map): map[architecture] ?? map[Self.anyArchitecture]
            }
        }

        var all: [(String, AssetPath)] {
            switch self {
            case .single(let path): [("", path)]
            case .perArchitecture(let map): map.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
            }
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let single = try? c.decode(AssetPath.self) {
                self = .single(single)
            } else {
                self = .perArchitecture(try c.decode([String: AssetPath].self))
            }
        }

        public func encode(to encoder: any Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .single(let path): try c.encode(path)
            case .perArchitecture(let map): try c.encode(map)
            }
        }
    }

    /// `{pattern, axes: {capacity: [...], chunk: [...]}, phases: {extend: pattern, promptOnly?: pattern}}`.
    public struct FunctionFamily: Sendable, Hashable, Codable {
        public var pattern: String
        public var axes: [String: [Int]]
        public var phases: [String: String]?

        public init(pattern: String, axes: [String: [Int]], phases: [String: String]? = nil) {
            self.pattern = pattern
            self.axes = axes
            self.phases = phases
        }

        /// Every function name the family declares (each pattern × the cartesian product of its axes).
        public var expandedNames: [String] {
            var names: Set<String> = []
            for pattern in [pattern] + (phases ?? [:]).values.sorted() {
                var partial = [pattern]
                for (axis, values) in axes {
                    guard pattern.contains("{\(axis)}") else { continue }
                    partial = partial.flatMap { p in values.map { p.replacingOccurrences(of: "{\(axis)}", with: String($0)) } }
                }
                names.formUnion(partial)
            }
            return names.sorted()
        }
    }

    /// Separate files, functions or assets per shape bucket.
    public struct Buckets: Sendable, Hashable, Codable {
        public struct Entry: Sendable, Hashable, Codable {
            public var value: Int
            public var file: AssetPath?
            public var function: String?
            public var asset: String?
        }
        public var axis: String
        public var rule: String
        public var entries: [Entry]
    }

    /// An input role's binding: a graph name, plus how and from where it is bound (§2.3 "Binding").
    /// Encodes as a bare string when only the name is set.
    public struct Binding: Sendable, Hashable, Codable, ExpressibleByStringLiteral {
        public enum Kind: String, Sendable, Hashable, Codable {
            /// Rewritten every call (the default).
            case perCall
            /// Bound once, rewritten only on events (image change, load).
            case `static`
        }

        public var name: String
        public var binding: Kind?
        public var source: Source?

        public init(name: String, binding: Kind? = nil, source: Source? = nil) {
            self.name = name
            self.binding = binding
            self.source = source
        }

        public init(stringLiteral value: String) { self.init(name: value) }

        public var isStatic: Bool { binding == .static }

        enum CodingKeys: String, CodingKey { case name, binding, source }

        public init(from decoder: any Decoder) throws {
            if let name = try? decoder.singleValueContainer().decode(String.self) {
                self.init(name: name)
                return
            }
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                name: try c.decode(String.self, forKey: .name),
                binding: try c.decodeIfPresent(Kind.self, forKey: .binding),
                source: try c.decodeIfPresent(Source.self, forKey: .source))
        }

        public func encode(to encoder: any Encoder) throws {
            if binding == nil && source == nil {
                var c = encoder.singleValueContainer()
                try c.encode(name)
                return
            }
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(name, forKey: .name)
            try c.encodeIfPresent(binding, forKey: .binding)
            try c.encodeIfPresent(source, forKey: .source)
        }
    }

    /// Where a bound input's values come from.
    public enum Source: Sendable, Hashable, Codable {
        public enum Storage: String, Sendable, Hashable, Codable { case owned, mapped, auto }
        public enum Fill: Sendable, Hashable, Codable {
            case zeros, ones
            case value(Int)

            public init(from decoder: any Decoder) throws {
                let c = try decoder.singleValueContainer()
                if let word = try? c.decode(String.self) {
                    switch word {
                    case "zeros": self = .zeros
                    case "ones": self = .ones
                    default: throw DecodingError.dataCorruptedError(in: c, debugDescription: "fill must be zeros, ones or {value}")
                    }
                } else {
                    struct V: Decodable { var value: Int }
                    self = .value(try c.decode(V.self).value)
                }
            }

            public func encode(to encoder: any Encoder) throws {
                var c = encoder.singleValueContainer()
                switch self {
                case .zeros: try c.encode("zeros")
                case .ones: try c.encode("ones")
                case .value(let v): try c.encode(["value": v])
                }
            }
        }

        /// A `tables` entry, owned or copy-on-write mapped.
        case table(String, storage: Storage?)
        /// Another graph's output, `"<graph>.<role>"`.
        case from(String)
        /// Generated once.
        case fill(Fill)

        enum CodingKeys: String, CodingKey { case table, storage, from, fill }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            if let table = try c.decodeIfPresent(String.self, forKey: .table) {
                self = .table(table, storage: try c.decodeIfPresent(Storage.self, forKey: .storage))
            } else if let from = try c.decodeIfPresent(String.self, forKey: .from) {
                self = .from(from)
            } else if let fill = try c.decodeIfPresent(Fill.self, forKey: .fill) {
                self = .fill(fill)
            } else {
                throw DecodingError.dataCorrupted(.init(
                    codingPath: decoder.codingPath, debugDescription: "source must be one of table, from, fill"))
            }
        }

        public func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .table(let table, let storage):
                try c.encode(table, forKey: .table)
                try c.encodeIfPresent(storage, forKey: .storage)
            case .from(let from): try c.encode(from, forKey: .from)
            case .fill(let fill): try c.encode(fill, forKey: .fill)
            }
        }
    }

    /// Host encoding of an input when it differs from the descriptor's dtype or the default convention.
    public struct InputEncoding: Sendable, Hashable, Codable {
        public struct ExtensionIDs: Sendable, Hashable, Codable {
            public var base: Int
            public var count: Int
        }
        public var dtype: ScalarType?
        public var range: String?
        public var pad: Int?
        public var extensionIds: ExtensionIDs?
    }

    /// An output role: a graph name plus which rows it returns and in which space. Bare string when only the name.
    public struct OutputBinding: Sendable, Hashable, Codable, ExpressibleByStringLiteral {
        public enum Rows: String, Sendable, Hashable, Codable { case all, last, none }
        public enum Space: String, Sendable, Hashable, Codable { case vocab, slots, options, scalar }

        public var name: String
        public var rows: Rows?
        public var space: Space?

        public init(name: String, rows: Rows? = nil, space: Space? = nil) {
            self.name = name
            self.rows = rows
            self.space = space
        }

        public init(stringLiteral value: String) { self.init(name: value) }

        enum CodingKeys: String, CodingKey { case name, rows, space }

        public init(from decoder: any Decoder) throws {
            if let name = try? decoder.singleValueContainer().decode(String.self) {
                self.init(name: name)
                return
            }
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                name: try c.decode(String.self, forKey: .name),
                rows: try c.decodeIfPresent(Rows.self, forKey: .rows),
                space: try c.decodeIfPresent(Space.self, forKey: .space))
        }

        public func encode(to encoder: any Encoder) throws {
            if rows == nil && space == nil {
                var c = encoder.singleValueContainer()
                try c.encode(name)
                return
            }
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(name, forKey: .name)
            try c.encodeIfPresent(rows, forKey: .rows)
            try c.encodeIfPresent(space, forKey: .space)
        }
    }

    /// §2.3.1.
    public struct States: Sendable, Hashable, Codable {
        public var sequence: [SequenceState]
        public var recurrent: [RecurrentState]
        /// Reserved for `decoder-hostcache@1`; rejected by v1 drivers.
        public var host: [JSONValue]?

        public init(sequence: [SequenceState] = [], recurrent: [RecurrentState] = [], host: [JSONValue]? = nil) {
            self.sequence = sequence
            self.recurrent = recurrent
            self.host = host
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            sequence = try c.decodeIfPresent([SequenceState].self, forKey: .sequence) ?? []
            recurrent = try c.decodeIfPresent([RecurrentState].self, forKey: .recurrent) ?? []
            host = try c.decodeIfPresent([JSONValue].self, forKey: .host)
        }

        enum CodingKeys: String, CodingKey { case sequence, recurrent, host }

        public func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(sequence, forKey: .sequence)
            if !recurrent.isEmpty { try c.encode(recurrent, forKey: .recurrent) }
            try c.encodeIfPresent(host, forKey: .host)
        }

        /// Every state name, sequence tensors first.
        public var allNames: [String] { sequence.flatMap(\.tensors) + recurrent.map(\.name) }
    }

    /// Tensors indexed by sequence position (KV pairs, MLA latents).
    public struct SequenceState: Sendable, Hashable, Codable {
        public enum Kind: String, Sendable, Hashable, Codable { case kv, latent }

        public var tensors: [String]
        /// Default `kv`.
        public var kind: Kind?
        public var seqAxis: Int
        public var layers: [Int]?
        public var window: JSONValue?
        public var carriesNonAttention: Bool?

        public init(
            tensors: [String], kind: Kind? = nil, seqAxis: Int, layers: [Int]? = nil, window: JSONValue? = nil,
            carriesNonAttention: Bool? = nil
        ) {
            self.tensors = tensors
            self.kind = kind
            self.seqAxis = seqAxis
            self.layers = layers
            self.window = window
            self.carriesNonAttention = carriesNonAttention
        }
    }

    /// A fixed-size state (conv, SSM). Its presence forces rewind by snapshot/restore.
    public struct RecurrentState: Sendable, Hashable, Codable {
        public enum Kind: String, Sendable, Hashable, Codable { case conv, ssm }

        public var name: String
        public var kind: Kind?
        /// Default `zeros`.
        public var initial: String?
        public var layers: [Int]?

        public init(name: String, kind: Kind? = nil, initial: String? = nil, layers: [Int]? = nil) {
            self.name = name
            self.kind = kind
            self.initial = initial
            self.layers = layers
        }

        enum CodingKeys: String, CodingKey {
            case name, kind, layers
            case initial = "init"
        }
    }

    /// §2.3.2. `staticSequence` is an int, a set of ints, or per phase.
    public struct Limits: Sendable, Hashable, Codable {
        /// The model's declared maximum, not a capacity (capacity is derived from memory at load).
        public var maxContext: Int?
        /// Bug caps per platform (`{"ios": 1024}`).
        public var kvCapacityCap: [String: Int]?
        public var maxPosition: Int?
        public var staticSequence: StaticSequence?
        public var prefillChunk: Int?
        public var padTail: Bool?
        public var padTokenId: Int?

        public init(
            maxContext: Int? = nil, kvCapacityCap: [String: Int]? = nil, maxPosition: Int? = nil,
            staticSequence: StaticSequence? = nil, prefillChunk: Int? = nil, padTail: Bool? = nil,
            padTokenId: Int? = nil
        ) {
            self.maxContext = maxContext
            self.kvCapacityCap = kvCapacityCap
            self.maxPosition = maxPosition
            self.staticSequence = staticSequence
            self.prefillChunk = prefillChunk
            self.padTail = padTail
            self.padTokenId = padTokenId
        }
    }

    public enum StaticSequence: Sendable, Hashable, Codable {
        case one(Int)
        case set([Int])
        case perPhase([String: StaticSequence])

        /// The value for `phase` (a flat value applies to every phase).
        public func values(forPhase phase: String) -> [Int]? {
            switch self {
            case .one(let n): [n]
            case .set(let s): s
            case .perPhase(let map): map[phase]?.values(forPhase: phase)
            }
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let n = try? c.decode(Int.self) { self = .one(n) }
            else if let s = try? c.decode([Int].self) { self = .set(s) }
            else { self = .perPhase(try c.decode([String: StaticSequence].self)) }
        }

        public func encode(to encoder: any Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .one(let n): try c.encode(n)
            case .set(let s): try c.encode(s)
            case .perPhase(let map): try c.encode(map)
            }
        }
    }

    /// `false` (never warm up) or `{shapes}`.
    public enum Warmup: Sendable, Hashable, Codable {
        case disabled
        case shapes(JSONValue)

        public init(from decoder: any Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let flag = try? c.decode(Bool.self) {
                guard !flag else {
                    throw DecodingError.dataCorruptedError(in: c, debugDescription: "warmup is false or {shapes}")
                }
                self = .disabled
            } else {
                struct S: Decodable { var shapes: JSONValue }
                self = .shapes(try c.decode(S.self).shapes)
            }
        }

        public func encode(to encoder: any Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .disabled: try c.encode(false)
            case .shapes(let shapes): try c.encode(["shapes": shapes])
            }
        }
    }
}

// MARK: - Validation

extension Manifest.Graph {
    func validate(at prefix: String, manifest: Manifest) throws(AetherError) {
        for (arch, path) in file.all {
            try manifest.checkAssetPath(path, arch.isEmpty ? "\(prefix).file" : "\(prefix).file.\(arch)")
        }
        let modes = [function != nil, functions != nil, functionFamily != nil].filter { $0 }.count
        try require(modes == 1, "\(prefix).function", "exactly one of function, functions, functionFamily is required")
        if let function { try require(!function.isEmpty, "\(prefix).function", "must not be empty") }
        if let functions {
            try require(!functions.isEmpty, "\(prefix).functions", "must name at least one phase")
            for (phase, name) in functions.sorted(by: { $0.key < $1.key }) {
                try require(!name.isEmpty, "\(prefix).functions.\(phase)", "must not be empty")
            }
        }
        if let family = functionFamily {
            try require(!family.pattern.isEmpty, "\(prefix).functionFamily.pattern", "must not be empty")
            for (axis, values) in family.axes.sorted(by: { $0.key < $1.key }) {
                try require(!values.isEmpty, "\(prefix).functionFamily.axes.\(axis)", "must list at least one value")
            }
        }
        if let buckets {
            try require(!buckets.entries.isEmpty, "\(prefix).buckets.entries", "must not be empty")
            for (i, entry) in buckets.entries.enumerated() {
                let set = [entry.file != nil, entry.function != nil, entry.asset != nil].filter { $0 }.count
                try require(set == 1, "\(prefix).buckets.entries.\(i)", "exactly one of file, function, asset")
                if let file = entry.file { try manifest.checkAssetPath(file, "\(prefix).buckets.entries.\(i).file") }
                if let asset = entry.asset {
                    try require(manifest.assets[asset] != nil, "\(prefix).buckets.entries.\(i).asset", "unknown asset '\(asset)'")
                }
            }
        }
        try Self.validate(inputs, at: "\(prefix).inputs", manifest: manifest)
        if let byPhase = inputsByFunction {
            for (phase, bindings) in byPhase.sorted(by: { $0.key < $1.key }) {
                try require(functions?[phase] != nil, "\(prefix).inputsByFunction.\(phase)",
                            "names a phase that functions does not declare")
                try Self.validate(bindings, at: "\(prefix).inputsByFunction.\(phase)", manifest: manifest)
            }
        }
        for (role, output) in outputs.sorted(by: { $0.key < $1.key }) {
            try require(!output.name.isEmpty, "\(prefix).outputs.\(role)", "must name a graph output")
        }
        if let states {
            var seen = Set<String>()
            for (i, entry) in states.sequence.enumerated() {
                let field = "\(prefix).states.sequence.\(i)"
                try require(!entry.tensors.isEmpty, "\(field).tensors", "must name at least one state")
                try require(entry.seqAxis >= 0, "\(field).seqAxis", "must be >= 0")
                for name in entry.tensors {
                    try require(!name.isEmpty && seen.insert(name).inserted, "\(field).tensors", "'\(name)' is empty or listed twice")
                }
            }
            for (i, entry) in states.recurrent.enumerated() {
                try require(!entry.name.isEmpty && seen.insert(entry.name).inserted, "\(prefix).states.recurrent.\(i).name",
                            "'\(entry.name)' is empty or listed twice")
                if let initial = entry.initial {
                    try require(initial == "zeros", "\(prefix).states.recurrent.\(i).init", "only zeros is defined")
                }
            }
        }
        if let limits {
            let field = "\(prefix).limits"
            for (key, value) in [("maxContext", limits.maxContext), ("maxPosition", limits.maxPosition),
                                 ("prefillChunk", limits.prefillChunk)] {
                if let value { try require(value > 0, "\(field).\(key)", "must be > 0") }
            }
            if let max = limits.maxContext, let chunk = limits.prefillChunk {
                try require(chunk <= max, "\(field).prefillChunk", "must be <= maxContext")
            }
            if let max = limits.maxContext, let position = limits.maxPosition {
                try require(position < max, "\(field).maxPosition", "must be < maxContext")
            }
            for (platform, cap) in (limits.kvCapacityCap ?? [:]).sorted(by: { $0.key < $1.key }) {
                try require(Device.Platform(rawValue: platform) != nil, "\(field).kvCapacityCap.\(platform)", "unknown platform")
                try require(cap > 0, "\(field).kvCapacityCap.\(platform)", "must be > 0")
            }
            if let s = limits.staticSequence { try Self.validate(s, at: "\(field).staticSequence", phases: phases) }
        }
    }

    static func validate(_ s: Manifest.StaticSequence, at field: String, phases: [String: String]) throws(AetherError) {
        switch s {
        case .one(let n): try require(n > 0, field, "must be > 0")
        case .set(let set): try require(!set.isEmpty && set.allSatisfy { $0 > 0 }, field, "must be a non-empty set of positive ints")
        case .perPhase(let map):
            for (phase, value) in map.sorted(by: { $0.key < $1.key }) {
                try require(phases[phase] != nil, "\(field).\(phase)", "names a phase the graph does not declare")
                if case .perPhase = value { throw .manifestInvalid(field: "\(field).\(phase)", reason: "cannot nest phases") }
                try validate(value, at: "\(field).\(phase)", phases: phases)
            }
        }
    }

    static func validate(_ bindings: [String: Manifest.Binding], at prefix: String, manifest: Manifest) throws(AetherError) {
        var names = Set<String>()
        for (role, binding) in bindings.sorted(by: { $0.key < $1.key }) {
            let field = "\(prefix).\(role)"
            try require(!binding.name.isEmpty, field, "must name a graph input")
            try require(names.insert(binding.name).inserted, field, "another role already binds '\(binding.name)'")
            switch binding.source {
            case .table(let table, _)?:
                try require(manifest.tables?[table] != nil, "\(field).source.table", "unknown table '\(table)'")
            case .from(let reference)?:
                let parts = reference.split(separator: ".", omittingEmptySubsequences: false)
                try require(parts.count == 2 && manifest.graphs[String(parts[0])] != nil && !parts[1].isEmpty,
                            "\(field).source.from", "'\(reference)' is not <graph>.<role> of a declared graph")
            case .fill?, nil:
                break
            }
        }
    }
}
