public import Foundation

/// Synthesizes the `aether.json` of a zoo catalog variant (DESIGN.md §4, MANIFEST.md §2.9).
///
/// Inputs: the catalog entry and variant (assets, file lists and hashes), the main asset's `metadata.json`, the
/// graph signatures (read from Core AI by AetherRuntime and passed in as plain data), the tokenizer config and
/// chat template, and the override table (`catalog/zoo-overrides.json`), which is the only home of family
/// knowledge. **Every inference rule lives in this file**; each is named in ``Rule`` and pinned by a golden.
public enum ZooBundleAdapter {
    /// Recorded in `provenance.adapter`. Bump it when a rule changes, so stale sidecars can be found.
    public static let version = "zoo-bundle-adapter@2"

    /// The dynamic-axis prefill chunk: M0's correctness limit (≤ 512 tokens per call).
    public static let defaultPrefillChunk = 512

    /// The inference rules (MANIFEST.md §2.9, "the adapter infers these").
    public enum Rule: String, CaseIterable, Sendable {
        /// `kind: llm` → `decoder-lm@1` unless the override names a driver; it must agree with the catalog variant.
        case driverFromKind
        /// `language.function_map`: `[main]` → `function: main`; `[main, prefill]` → `functions {step, prefill}`.
        case functionsFromFunctionMap
        /// Tokens and positions by the standard zoo names (`input_ids`, `position_ids`); every other graph input
        /// must be bound by the override.
        case standardInputNames
        /// Logits: the output named `logits`; `hidden` by name. `rows` from the shape: rank 0 → none, dim 1
        /// static 1 → last, else all.
        case logitsRowsFromShape
        /// A state with one dynamic axis is a sequence state on that axis (states sharing an axis form one entry,
        /// keys first); a fully static state is recurrent. Exceptions come from the override.
        case statesFromDynamicAxis
        /// `maxContext` from `language.max_context_length`; `staticSequence` from static token axes (per phase);
        /// `prefillChunk` 512 for a dynamic token axis.
        case limitsFromMetadataAndShapes
        /// `compute: gpu` unless overridden (measured); `expectFrequentReshapes` written explicitly.
        case gpuCompute
        /// Tokenizer source: the `tokenizer/` directory when the bundle has one (never `embedded_tokenizer`),
        /// else the override's pinned source.
        case tokenizerSourceFromDirectory
        /// Chat template path and sha256 from the catalog listing; `addBos` follows the template or renderer.
        case chatTemplateFromListing
        /// Stops: tokenizer eos ∪ the override's pinned generation_config eos list.
        case stopsFromEOSAndGenerationConfig
        /// Thinking tags when the template uses `<think>`, `</think>` and `enable_thinking`.
        case thinkingFromTemplate
    }

    public struct Input: Sendable {
        public var entry: CatalogEntry
        /// The catalog variant key (`macos`, `ios`, `ios-h18p`).
        public var variant: String
        /// The main asset's `metadata.json`.
        public var metadata: Data
        /// Signatures of the decoder graph's functions, keyed by function name.
        public var signatures: [String: GraphSignature]
        public var tokenizerConfig: Data?
        public var chatTemplate: String?
        public var overrides: ZooOverrides

        public init(
            entry: CatalogEntry, variant: String, metadata: Data, signatures: [String: GraphSignature],
            tokenizerConfig: Data? = nil, chatTemplate: String? = nil, overrides: ZooOverrides = .builtin
        ) {
            self.entry = entry
            self.variant = variant
            self.metadata = metadata
            self.signatures = signatures
            self.tokenizerConfig = tokenizerConfig
            self.chatTemplate = chatTemplate
            self.overrides = overrides
        }
    }

    /// The subset of the zoo's `metadata.json` (format 0.2) the adapter reads.
    public struct ZooMetadata: Sendable, Hashable, Decodable {
        public struct Assets: Sendable, Hashable, Decodable { public var main: String }
        public struct Language: Sendable, Hashable, Decodable {
            public var tokenizer: String?
            public var maxContextLength: Int?
            public var functionMap: [String: [String]]?
            public var chatTemplateSha256: String?

            enum CodingKeys: String, CodingKey {
                case tokenizer
                case maxContextLength = "max_context_length"
                case functionMap = "function_map"
                case chatTemplateSha256 = "chat_template_sha256"
            }
        }
        public struct Source: Sendable, Hashable, Decodable {
            public var hfModelID: String?
            public var hfRevision: String?
            enum CodingKeys: String, CodingKey {
                case hfModelID = "hf_model_id"
                case hfRevision = "hf_revision"
            }
        }

        public var kind: String
        public var assets: Assets
        public var language: Language?
        public var source: Source?
        /// A string (`4bit`) or an object (`{"scheme": "int8hu"}`) depending on the exporter.
        public var compression: JSONValue?
    }

    /// The decoder file and functions to inspect, from the catalog, metadata and overrides alone. AetherRuntime
    /// loads these and passes their signatures to ``manifest(from:)``.
    public static func graphRequest(
        entry: CatalogEntry, variant: String, metadata: Data, overrides: ZooOverrides = .builtin
    ) throws(AetherError) -> (file: AssetPath, functions: [String]) {
        let meta = try decodeMetadata(metadata)
        let override = try overrides.override(for: entry, variant: variant)?.graphs?["decoder"]
        let file = override?.file ?? AssetPath(asset: "main", path: meta.assets.main)
        let phases = try phases(meta, override: override)
        return (file, Array(Set(phases.values)).sorted())
    }

    /// Produces a validated manifest. Errors name the offending field: `metadata.*` for the zoo document,
    /// `zoo-overrides.*` for the table, manifest paths for everything else.
    public static func manifest(from input: Input) throws(AetherError) -> Manifest {
        let entry = input.entry
        let meta = try decodeMetadata(input.metadata)
        guard let variant = entry.variants[input.variant] else {
            throw .manifestInvalid(field: "variant", reason: "\(entry.id) has no catalog variant '\(input.variant)'")
        }
        guard let main = variant.assets["main"] else {
            throw .manifestInvalid(field: "assets.main", reason: "the catalog variant has no main asset")
        }
        let override = try input.overrides.override(for: entry, variant: input.variant)

        // Rule.driverFromKind
        let driver: DriverID
        if let named = override?.driver {
            driver = named
        } else if meta.kind == "llm" {
            driver = .decoderLM
        } else {
            throw .manifestInvalid(field: "metadata.kind", reason: "zoo kind '\(meta.kind)' needs a driver override")
        }
        guard driver == variant.driver else {
            throw .manifestMismatch(field: "driver", expected: variant.driver.description, found: driver.description)
        }
        guard driver == .decoderLM else {
            throw .unsupported(driver: driver)  // the adapter synthesizes decoder-lm@1 only so far
        }

        let platformName = String(input.variant.split(separator: "-").first ?? "")
        guard let platform = Device.Platform(rawValue: platformName) else {
            throw .manifestInvalid(field: "variant.platform", reason: "'\(input.variant)' does not start with macos or ios")
        }
        let mainFiles = Set(main.files.map(\.path))
        var assets: [String: Manifest.Asset] = [:]
        for (name, asset) in variant.assets {
            assets[name] = Manifest.Asset(repo: asset.repo, revision: asset.revision, subtree: asset.subtree)
        }

        let graphOverride = override?.graphs?["decoder"]
        let graph = try decoderGraph(
            meta: meta, signatures: input.signatures, mainFiles: mainFiles, override: graphOverride)
        let tables = try tables(override?.tables, variant: variant)
        let (tokenizer, stopSource) = try tokenizer(
            input: input, meta: meta, variant: variant, mainFiles: mainFiles, override: override?.tokenizer)

        var refs: [String: String] = [:]
        if let stopSource { refs["tokenizer.stop"] = stopSource }
        let compression: String? = switch meta.compression {
        case .string(let s)?: s
        case nil, .null?: nil
        case let value?: value.compactText
        }
        let manifest = Manifest(
            id: entry.id,
            task: override?.task ?? entry.task,
            driver: driver,
            variant: Manifest.Variant(platform: platform, architectures: variant.architectures),
            assets: assets,
            graphs: ["decoder": graph],
            tables: tables,
            tokenizer: tokenizer,
            provenance: Manifest.Provenance(
                source: meta.source?.hfModelID ?? meta.language?.tokenizer,
                sourceRevision: meta.source?.hfRevision,
                adapter: version,
                zooRepo: entry.repo,
                zooRevision: entry.revision,
                subtrees: assets.mapValues(\.subtree),
                zooCompression: compression,
                metadataKind: meta.kind,
                refs: refs.isEmpty ? nil : refs),
            license: entry.license)
        try manifest.validate()
        try DriverContract.decoderLM.validate(manifest)
        return manifest
    }

    // MARK: - Graph

    static func decodeMetadata(_ data: Data) throws(AetherError) -> ZooMetadata {
        do {
            return try JSONDecoder().decode(ZooMetadata.self, from: data)
        } catch let error as DecodingError {
            if case .manifestInvalid(let field, let reason) = AetherError(decoding: error) {
                throw .manifestInvalid(field: field.isEmpty ? "metadata" : "metadata.\(field)", reason: reason)
            }
            throw .manifestInvalid(field: "metadata", reason: "\(error)")
        } catch {
            throw .manifestInvalid(field: "metadata", reason: "\(error)")
        }
    }

    /// Rule.functionsFromFunctionMap
    static func phases(_ meta: ZooMetadata, override: ZooOverrides.GraphOverride?) throws(AetherError) -> [String: String] {
        if let functions = override?.functions { return functions }
        let listed = Set((meta.language?.functionMap ?? [:]).values.flatMap { $0 })
        switch listed {
        case [], ["main"]: return ["step": "main"]
        case ["main", "prefill"]: return ["step": "main", "prefill": "prefill"]
        default:
            throw .manifestInvalid(
                field: "graphs.decoder.functions",
                reason: "cannot infer phases from function_map \(listed.sorted()); add a functions override")
        }
    }

    static func decoderGraph(
        meta: ZooMetadata, signatures: [String: GraphSignature], mainFiles: Set<String>,
        override: ZooOverrides.GraphOverride?
    ) throws(AetherError) -> Manifest.Graph {
        let file = override?.file ?? AssetPath(asset: "main", path: meta.assets.main)
        if file.asset == "main" {
            guard mainFiles.contains(where: { $0 == file.path || $0.hasPrefix(file.path + "/") }) else {
                throw .manifestInvalid(field: "graphs.decoder.file", reason: "'\(file.path)' is not in the main asset")
            }
        }
        let phases = try phases(meta, override: override)
        var phaseSignatures: [String: GraphSignature] = [:]
        for (phase, function) in phases {
            guard let signature = signatures[function] else {
                throw .manifestInvalid(field: "graphs.decoder.functions.\(phase)", reason: "no signature given for '\(function)'")
            }
            phaseSignatures[phase] = signature
        }
        guard let step = phaseSignatures["step"] else {
            throw .manifestInvalid(field: "graphs.decoder.functions.step", reason: "a decoder needs a step phase")
        }

        // Rule.standardInputNames
        let staticNames = Set((override?.inputs ?? [:]).values.map(\.name))
        let names = step.inputs.map(\.name).filter { !staticNames.contains($0) }
        guard let tokens = names.first(where: { $0 == "input_ids" }),
              let positions = names.first(where: { $0 == "position_ids" })
        else {
            throw .manifestInvalid(
                field: "graphs.decoder.inputs",
                reason: "expected input_ids and position_ids, found \(names); bind non-standard inputs in the override")
        }
        if let extra = names.first(where: { $0 != tokens && $0 != positions }) {
            throw .manifestInvalid(
                field: "graphs.decoder.inputs", reason: "graph input '\(extra)' has no role; bind it in the override")
        }
        var inputs: [String: Manifest.Binding] = ["tokens": .init(name: tokens), "positions": .init(name: positions)]
        for (role, binding) in override?.inputs ?? [:] { inputs[role] = binding }

        // Rule.logitsRowsFromShape
        guard let logits = step.output("logits") else {
            throw .manifestInvalid(field: "graphs.decoder.outputs.logits", reason: "the graph has no output named logits")
        }
        var outputs = ["logits": Manifest.OutputBinding(name: logits.name, rows: rows(of: logits))]
        if let hidden = step.outputs.first(where: { $0.name.lowercased().contains("hidden") }) {
            outputs["hidden"] = Manifest.OutputBinding(name: hidden.name, rows: rows(of: hidden))
        }

        // Rule.statesFromDynamicAxis
        let states: Manifest.States
        if let replaced = override?.states {
            states = replaced
        } else {
            var byAxis: [Int: [String]] = [:]
            var recurrent: [Manifest.RecurrentState] = []
            for state in step.states {
                switch state.dynamicAxes.count {
                case 0: recurrent.append(Manifest.RecurrentState(name: state.name))
                case 1: byAxis[state.dynamicAxes[0], default: []].append(state.name)
                default:
                    throw .manifestInvalid(
                        field: "graphs.decoder.states",
                        reason: "'\(state.name)' \(state.summary) has several dynamic axes; classify it in the override")
                }
            }
            let sequence = byAxis.keys.sorted().map { axis in
                Manifest.SequenceState(tensors: byAxis[axis]!.sorted { keysFirst($0, $1) }, seqAxis: axis)
            }
            states = Manifest.States(sequence: sequence, recurrent: recurrent)
        }

        // Rule.limitsFromMetadataAndShapes
        guard let maxContext = meta.language?.maxContextLength else {
            throw .manifestInvalid(field: "metadata.language.max_context_length", reason: "a decoder needs its declared maximum")
        }
        var staticByPhase: [String: Int] = [:]
        for (phase, signature) in phaseSignatures {
            if let spec = signature.input(tokens), spec.rank == 2, spec.shape[1] != TensorSpec.dynamic {
                staticByPhase[phase] = spec.shape[1]
            }
        }
        var limits = Manifest.Limits(maxContext: maxContext, kvCapacityCap: override?.kvCapacityCap)
        if phases.count == 1, let s = staticByPhase["step"] {
            limits.staticSequence = .one(s)
        } else if !staticByPhase.isEmpty {
            limits.staticSequence = .perPhase(staticByPhase.mapValues { .one($0) })
        }
        if staticByPhase["step"] == nil || staticByPhase["prefill"] != nil {
            limits.prefillChunk = min(defaultPrefillChunk, staticByPhase["prefill"] ?? maxContext)
        }

        // Rule.gpuCompute
        let compute = override?.compute ?? .gpu
        let dynamic = phaseSignatures.values.contains { $0.hasDynamicInputsOrStates }

        return Manifest.Graph(
            file: .single(file),
            function: phases.count == 1 ? phases["step"] : nil,
            functions: phases.count == 1 ? nil : phases,
            inputs: inputs,
            outputs: outputs,
            states: states,
            limits: limits,
            compute: compute,
            expectFrequentReshapes: compute == .gpu && dynamic,
            warmup: override?.warmup)
    }

    static func rows(of spec: TensorSpec) -> Manifest.OutputBinding.Rows {
        if spec.rank == 0 { return .none }
        if spec.rank >= 2 && spec.shape[1] == 1 { return .last }
        return .all
    }

    /// Keys before values, then by name.
    static func keysFirst(_ a: String, _ b: String) -> Bool {
        func rank(_ n: String) -> Int {
            let n = n.lowercased()
            if n.contains("key") || n.hasPrefix("k_") { return 0 }
            if n.contains("value") || n.hasPrefix("v_") { return 1 }
            return 2
        }
        return (rank(a), a) < (rank(b), b)
    }

    static func tables(_ tables: [String: Manifest.Table]?, variant: CatalogVariant) throws(AetherError) -> [String: Manifest.Table]? {
        guard var tables else { return nil }
        for name in tables.keys.sorted() {
            let table = tables[name]!
            guard let asset = variant.assets[table.asset] else {
                throw .manifestInvalid(field: "tables.\(name).asset", reason: "the catalog variant has no asset '\(table.asset)'")
            }
            guard let file = asset.files.first(where: { $0.path == table.file }) else {
                throw .manifestInvalid(field: "tables.\(name).file", reason: "'\(table.file)' is not in asset '\(table.asset)'")
            }
            if let bytes = table.byteCount, bytes != file.size {
                throw .manifestMismatch(
                    field: "tables.\(name).shape", expected: "\(file.size) bytes (catalog)", found: "\(bytes) bytes from dtype × shape")
            }
            tables[name]!.sha256 = file.sha256
        }
        return tables
    }

    // MARK: - Tokenizer

    static func tokenizer(
        input: Input, meta: ZooMetadata, variant: CatalogVariant, mainFiles: Set<String>,
        override: ZooOverrides.TokenizerOverride?
    ) throws(AetherError) -> (Manifest.Tokenizer, String?) {
        // Rule.tokenizerSourceFromDirectory: `embedded_tokenizer` is never trusted.
        let hasDirectory = mainFiles.contains { $0.hasPrefix("tokenizer/") }
        let source: Manifest.TokenizerSource
        if let pinned = override?.source {
            source = pinned
        } else if hasDirectory {
            source = .dir(AssetPath(asset: "main", path: "tokenizer"))
        } else {
            throw .manifestInvalid(
                field: "tokenizer.source",
                reason: "the bundle has no tokenizer/ directory (embedded_tokenizer is not trusted); pin a source in the override")
        }
        if case .hf(let repo, let revision) = source {
            guard variant.assets.values.contains(where: { $0.repo == repo && $0.revision == revision }) else {
                throw .manifestInvalid(
                    field: "tokenizer.source", reason: "no catalog asset installs \(repo)@\(revision)")
            }
        }

        // Rule.chatTemplateFromListing
        let chatTemplate: AssetPath?
        switch override?.chatTemplate {
        case .none?: chatTemplate = nil
        case .path(let path)?: chatTemplate = path
        case nil:
            chatTemplate = mainFiles.contains("tokenizer/chat_template.jinja")
                ? AssetPath(asset: "main", path: "tokenizer/chat_template.jinja") : nil
        }
        var templateSha: String?
        if let chatTemplate {
            templateSha = variant.assets[chatTemplate.asset]?.files.first { $0.path == chatTemplate.path }?.sha256
            if let declared = meta.language?.chatTemplateSha256, let templateSha, declared != templateSha {
                throw .manifestMismatch(
                    field: "tokenizer.chatTemplateSha256", expected: "\(declared) (metadata)", found: "\(templateSha) (catalog)")
            }
        }
        let renderer = override?.renderer
        let addBos: Manifest.AddBos? = override?.addBos ?? (chatTemplate != nil ? .template : renderer != nil ? .renderer : nil)

        // Rule.stopsFromEOSAndGenerationConfig
        let config = try tokenizerConfig(input.tokenizerConfig)
        var stop: [Manifest.StopToken] = []
        if let eos = config.eos { stop.append(.text(eos)) }
        if let pinned = override?.stop {
            guard pinned.tokens.count == pinned.eosTokenIds.count else {
                throw .manifestInvalid(
                    field: "zoo-overrides.\(input.entry.id).tokenizer.stop", reason: "tokens and eosTokenIds differ in length")
            }
            for (id, token) in zip(pinned.eosTokenIds, pinned.tokens) {
                if let known = config.addedTokens[id], known != token {
                    throw .manifestMismatch(
                        field: "tokenizer.stop", expected: "id \(id) = '\(token)' (override)", found: "'\(known)' (tokenizer_config)")
                }
                if !stop.contains(.text(token)) { stop.append(.text(token)) }
            }
        }

        // Rule.thinkingFromTemplate
        var thinking: Manifest.Thinking?
        if let template = input.chatTemplate, chatTemplate != nil,
           template.contains("<think>"), template.contains("</think>"), template.contains("enable_thinking")
        {
            thinking = Manifest.Thinking(open: "<think>", close: "</think>", toggle: .templateVar("enable_thinking"))
        }
        if let t = override?.thinking {
            if thinking == nil {
                guard let open = t.open, let close = t.close else {
                    throw .manifestInvalid(
                        field: "zoo-overrides.\(input.entry.id).tokenizer.thinking",
                        reason: "no thinking tags were inferred, so the override must give open and close")
                }
                thinking = Manifest.Thinking(open: open, close: close)
            }
            if let open = t.open { thinking!.open = open }
            if let close = t.close { thinking!.close = close }
            if let toggle = t.toggle { thinking!.toggle = toggle }
            if let defaultOn = t.defaultOn { thinking!.defaultOn = defaultOn }
        }

        let tokenizer = Manifest.Tokenizer(
            source: source, chatTemplate: chatTemplate, chatTemplateSha256: templateSha, renderer: renderer,
            turnFormat: override?.turnFormat, addBos: addBos, stop: stop.isEmpty ? nil : stop, thinking: thinking,
            toolDialect: override?.toolDialect)
        return (tokenizer, override?.stop?.source)
    }

    /// `eos_token` (string or `{content}`) and `added_tokens_decoder` id → content.
    static func tokenizerConfig(_ data: Data?) throws(AetherError) -> (eos: String?, addedTokens: [Int: String]) {
        guard let data else { return (nil, [:]) }
        let value: JSONValue
        do { value = try JSONDecoder().decode(JSONValue.self, from: data) } catch {
            throw .manifestInvalid(field: "tokenizer_config", reason: "\(error)")
        }
        let eos = value["eos_token"]?.string ?? value["eos_token"]?["content"]?.string
        var added: [Int: String] = [:]
        for (key, token) in value["added_tokens_decoder"]?.object ?? [:] {
            if let id = Int(key), let content = token["content"]?.string { added[id] = content }
        }
        return (eos?.isEmpty == false ? eos : nil, added)
    }
}
