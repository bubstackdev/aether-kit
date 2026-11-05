public import Foundation

/// Synthesizes an `aether.json` for a zoo bundle (DESIGN.md §4, "Zoo adapter").
///
/// **Every heuristic in Aether-Kit lives in this file and nowhere else.** The adapter runs once per install; its
/// output is written next to the bundle, and from then on code binds to the manifest's roles. Each heuristic is
/// listed in ``Heuristic`` and covered by a golden `aether.json` per catalog entry.
///
/// Inputs are plain data. The graph signature is passed in (AetherRuntime reads it from Core AI), so this type
/// never touches Core AI.
public enum ZooBundleAdapter {
    /// Recorded in `provenance.synthesizedBy`. Bump it when a heuristic changes, so stale sidecars can be found.
    public static let version = "zoo-bundle-adapter@1"

    /// The default prefill chunk for a dynamic token axis: M0's correctness limit (≤ 512 tokens per call).
    public static let defaultPrefillChunk = 512

    /// The heuristics, named so tests and reviews can refer to them.
    public enum Heuristic: String, CaseIterable, Sendable {
        /// `kind: "llm"` → `text-generation` served by `decoder-lm@1`.
        case llmKindMeansDecoderLM
        /// The decoder function is `main` if `function_map` lists it, otherwise the first one listed.
        case decoderFunction
        /// Tokens: the input whose name contains `input_ids`/`token`, else the first input. Positions: the input
        /// whose name contains `position`, else the second. Exactly two inputs are supported.
        case tokensThenPositions
        /// Logits: the output named `logits`, else the first output. An output whose name contains `hidden` is the
        /// optional `hidden` role.
        case logitsOutput
        /// A state is KV if its name says key/value (`key`, `value`, `k_cache`, `v_cache`); otherwise one with a
        /// dynamic axis is KV and a fully static one is recurrent.
        case stateRoles
        /// The KV sequence axis is the single dynamic axis shared by every KV state.
        case kvSequenceAxis
        /// A static token axis of length N means `staticSequence = N`; a dynamic one gets the default prefill chunk.
        case tokenAxisLimits
        /// Any dynamic input or state → GPU with `expectFrequentReshapes`; a fully static graph → Neural Engine.
        case specializationFromShapeKind
        /// Stop markers: `eos_token` from `tokenizer_config.json`.
        case stopFromEOS
        /// Thinking tags: `<think>`/`</think>` when the chat template mentions both.
        case thinkingFromTemplate
        /// Tool dialect: `hermes` when the chat template emits `<tool_call>`.
        case hermesFromTemplate
        /// Weights precision from `compression` (`4bit` → `int4`); activations from the logits scalar type.
        case precisionFromCompression
    }

    public struct Input: Sendable {
        /// The bundle's `metadata.json`.
        public var metadata: Data
        /// The signature of every function the adapter may bind, keyed by function name.
        public var signatures: [String: GraphSignature]
        /// Every file in the bundle, relative to its root (`tokenizer/vocab.json`, …).
        public var files: [String]
        /// `tokenizer/tokenizer_config.json`, if the bundle has one.
        public var tokenizerConfig: Data?
        /// `tokenizer/chat_template.jinja`, if the bundle has one.
        public var chatTemplate: String?
        /// The catalog id, repo and revision the bundle was installed from.
        public var catalogID: String
        public var repo: String?
        public var revision: String?

        public init(
            metadata: Data, signatures: [String: GraphSignature], files: [String], tokenizerConfig: Data? = nil,
            chatTemplate: String? = nil, catalogID: String, repo: String? = nil, revision: String? = nil
        ) {
            self.metadata = metadata
            self.signatures = signatures
            self.files = files
            self.tokenizerConfig = tokenizerConfig
            self.chatTemplate = chatTemplate
            self.catalogID = catalogID
            self.repo = repo
            self.revision = revision
        }
    }

    /// The subset of the zoo's `metadata.json` (format 0.2) the adapter reads.
    public struct ZooMetadata: Sendable, Hashable, Decodable {
        public struct Assets: Sendable, Hashable, Decodable {
            public var main: String
        }
        public struct Language: Sendable, Hashable, Decodable {
            public var tokenizer: String?
            public var vocabSize: Int?
            public var maxContextLength: Int?
            public var functionMap: [String: [String]]?

            enum CodingKeys: String, CodingKey {
                case tokenizer
                case vocabSize = "vocab_size"
                case maxContextLength = "max_context_length"
                case functionMap = "function_map"
            }
        }
        public struct Source: Sendable, Hashable, Decodable {
            public var hfModelID: String?
            enum CodingKeys: String, CodingKey { case hfModelID = "hf_model_id" }
        }

        public var metadataVersion: String?
        public var kind: String
        public var name: String?
        public var assets: Assets
        public var language: Language?
        public var source: Source?
        public var compression: String?

        enum CodingKeys: String, CodingKey {
            case metadataVersion = "metadata_version"
            case kind, name, assets, language, source, compression
        }
    }

    /// The function names the adapter will want signatures for, read from `metadata.json` alone. AetherRuntime
    /// calls this first, loads those functions, and passes their signatures to ``manifest(from:)``.
    public static func graphRequests(metadata data: Data) throws(AetherError) -> (file: String, functions: [String]) {
        let meta = try decodeMetadata(data)
        return (meta.assets.main, [try decoderFunction(meta)])
    }

    /// Produces a validated manifest. Errors name the offending field: `metadata.*` for the zoo document,
    /// `graphs.decoder.*` for anything bound from the signature.
    public static func manifest(from input: Input) throws(AetherError) -> Manifest {
        let meta = try decodeMetadata(input.metadata)
        // Heuristic.llmKindMeansDecoderLM
        guard meta.kind == "llm" else {
            throw .manifestInvalid(field: "metadata.kind", reason: "zoo kind '\(meta.kind)' has no adapter yet")
        }
        let function = try decoderFunction(meta)
        guard let signature = input.signatures[function] else {
            throw .manifestInvalid(field: "graphs.decoder.function", reason: "no signature given for '\(function)'")
        }
        let files = Set(input.files)
        guard files.contains(where: { $0 == meta.assets.main || $0.hasPrefix(meta.assets.main + "/") }) else {
            throw .manifestInvalid(field: "metadata.assets.main", reason: "'\(meta.assets.main)' is not in the bundle")
        }

        let graph = try decoderGraph(file: meta.assets.main, function: function, signature: signature, meta: meta)
        let manifest = Manifest(
            id: input.catalogID,
            task: "text-generation",
            driver: .decoderLM,
            graphs: ["decoder": graph],
            tokenizer: try tokenizer(files: files, config: input.tokenizerConfig, template: input.chatTemplate),
            precision: precision(meta: meta, signature: signature, logits: graph.outputs["logits"]),
            provenance: Manifest.Provenance(
                source: meta.source?.hfModelID ?? meta.language?.tokenizer,
                zooRepo: input.repo,
                zooRevision: input.revision,
                synthesizedBy: version))
        try manifest.validate()
        try DriverContract.decoderLM.validate(manifest)
        return manifest
    }

    // MARK: - Pieces

    static func decodeMetadata(_ data: Data) throws(AetherError) -> ZooMetadata {
        do {
            return try JSONDecoder().decode(ZooMetadata.self, from: data)
        } catch let error as DecodingError {
            let mapped = AetherError(decoding: error)
            if case .manifestInvalid(let field, let reason) = mapped {
                throw .manifestInvalid(field: field.isEmpty ? "metadata" : "metadata.\(field)", reason: reason)
            }
            throw mapped
        } catch {
            throw .manifestInvalid(field: "metadata", reason: "\(error)")
        }
    }

    /// Heuristic.decoderFunction
    static func decoderFunction(_ meta: ZooMetadata) throws(AetherError) -> String {
        let listed = (meta.language?.functionMap ?? [:]).sorted { $0.key < $1.key }.flatMap(\.value)
        if listed.isEmpty || listed.contains("main") { return "main" }
        return listed[0]
    }

    static func decoderGraph(
        file: String, function: String, signature: GraphSignature, meta: ZooMetadata
    ) throws(AetherError) -> Manifest.Graph {
        // Heuristic.tokensThenPositions
        guard signature.inputs.count == 2 else {
            throw .manifestInvalid(
                field: "graphs.decoder.inputs",
                reason: "expected 2 graph inputs (tokens, positions), found \(signature.inputs.map(\.name))")
        }
        let names = signature.inputs.map(\.name)
        let tokens = names.first { $0.lowercased().contains("input_ids") || $0.lowercased().contains("token") }
            ?? names[0]
        let positions = names.first { $0 != tokens && $0.lowercased().contains("position") }
            ?? names.first { $0 != tokens }!

        // Heuristic.logitsOutput
        guard let logits = signature.outputs.first(where: { $0.name == "logits" }) ?? signature.outputs.first else {
            throw .manifestInvalid(field: "graphs.decoder.outputs.logits", reason: "the graph has no outputs")
        }
        var outputs = ["logits": logits.name]
        if let hidden = signature.outputs.first(where: { $0.name != logits.name && $0.name.lowercased().contains("hidden") }) {
            outputs["hidden"] = hidden.name
        }

        // Heuristic.stateRoles
        var kv: [TensorSpec] = []
        var recurrent: [TensorSpec] = []
        for state in signature.states {
            switch nameRole(state.name) ?? (state.isDynamic ? .kv : .recurrent) {
            case .kv: kv.append(state)
            case .recurrent: recurrent.append(state)
            }
        }
        // Keys before values within the pair, so `kv` reads [keys…, values…] whatever the graph's order.
        kv.sort { kvRank($0.name) < kvRank($1.name) }

        // Heuristic.kvSequenceAxis
        var sequenceAxis: Int?
        if !kv.isEmpty {
            let axes = Set(kv.map(\.dynamicAxes))
            guard axes.count == 1, let only = axes.first, only.count == 1 else {
                throw .manifestInvalid(
                    field: "graphs.decoder.states.sequenceAxis",
                    reason: "cannot infer the KV sequence axis from \(kv.map(\.summary)); ship an aether.json")
            }
            sequenceAxis = only[0]
        }

        // Heuristic.tokenAxisLimits
        let tokenSpec = signature.input(tokens)!
        var limits = Manifest.Limits(maxContext: meta.language?.maxContextLength)
        if tokenSpec.rank == 2, tokenSpec.shape[1] != TensorSpec.dynamic {
            limits.staticSequence = tokenSpec.shape[1]
        } else {
            limits.prefillChunk = min(defaultPrefillChunk, meta.language?.maxContextLength ?? defaultPrefillChunk)
        }

        // Heuristic.specializationFromShapeKind
        let dynamic = signature.hasDynamicInputsOrStates
        let specialization = Manifest.Specialization(
            computeUnit: dynamic ? .gpu : .neuralEngine, expectFrequentReshapes: dynamic)

        return Manifest.Graph(
            file: .single(file),
            function: function,
            inputs: ["tokens": tokens, "positions": positions],
            outputs: outputs,
            states: Manifest.States(kv: kv.map(\.name), recurrent: recurrent.map(\.name), sequenceAxis: sequenceAxis),
            limits: limits,
            specialization: specialization)
    }

    enum StateRole { case kv, recurrent }

    static func nameRole(_ name: String) -> StateRole? {
        let n = name.lowercased()
        if kvRank(name) < 2 { return .kv }
        if n.contains("conv") || n.contains("rec") || n.contains("ssm") { return .recurrent }
        return nil
    }

    /// 0 for keys, 1 for values, 2 for neither.
    static func kvRank(_ name: String) -> Int {
        let n = name.lowercased()
        if n.contains("key") || n.hasPrefix("k_") || n.contains("_k_") || n.hasSuffix("_k") { return 0 }
        if n.contains("value") || n.hasPrefix("v_") || n.contains("_v_") || n.hasSuffix("_v") { return 1 }
        return 2
    }

    static func tokenizer(files: Set<String>, config: Data?, template: String?) throws(AetherError) -> Manifest.Tokenizer? {
        let dir = "tokenizer"
        guard files.contains(where: { $0.hasPrefix(dir + "/") }) else { return nil }
        let templatePath = "\(dir)/chat_template.jinja"

        // Heuristic.stopFromEOS
        var stop: [String] = []
        if let config {
            struct Config: Decodable {
                var eosToken: String?
                enum CodingKeys: String, CodingKey { case eosToken = "eos_token" }
            }
            do {
                if let eos = try JSONDecoder().decode(Config.self, from: config).eosToken, !eos.isEmpty {
                    stop.append(eos)
                }
            } catch {
                throw .manifestInvalid(field: "tokenizer_config.eos_token", reason: "\(error)")
            }
        }

        // Heuristic.thinkingFromTemplate, Heuristic.hermesFromTemplate
        var thinking: Manifest.Thinking?
        var toolDialect: String?
        if let template {
            if template.contains("<think>") && template.contains("</think>") {
                thinking = Manifest.Thinking(open: "<think>", close: "</think>")
            }
            if template.contains("<tool_call>") { toolDialect = "hermes" }
        }
        return Manifest.Tokenizer(
            dir: dir, chatTemplate: files.contains(templatePath) ? templatePath : nil, stop: stop,
            thinking: thinking, toolDialect: toolDialect)
    }

    /// Heuristic.precisionFromCompression
    static func precision(meta: ZooMetadata, signature: GraphSignature, logits: String?) -> Manifest.Precision {
        var weights: String?
        if let compression = meta.compression?.lowercased() {
            if compression.hasSuffix("bit"), let bits = Int(compression.dropLast(3)) {
                weights = "int\(bits)"
            } else {
                weights = compression
            }
        }
        let activations = logits.flatMap { signature.output($0) }.map(\.scalarType.precisionName)
        return Manifest.Precision(weights: weights, activations: activations)
    }
}
