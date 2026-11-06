/// Every driver id and graph kind MANIFEST.md §1 defines. A known id that no registered driver implements is
/// `unsupported`; an id outside this set is `unknownDriver`.
public enum DriverCatalog {
    public static let drivers: Set<DriverID> = Set([
        "decoder-lm@1", "static-chunked-lm@1", "decoder-hostcache@1",
        "vlm-static-inject@1", "vlm-embeds@1",
        "detector-dense@1", "detector-detr@1", "depth-estimator@1", "clip-joint@1", "image-to-image-tiled@1",
        "text-embedder@1", "late-interaction-retriever@1", "gliner-classifier@1", "yes-no-classifier@1",
        "decision-lm@1", "decision-hidden-head@1", "decision-joint-head@1", "encoder-decider@1",
        "latent-diffusion@1", "flux2-klein@1",
    ].map(DriverID.init(validated:)))

    /// Reusable graph contracts a driver embeds. Never a top-level `driver`.
    public static let graphKinds: Set<DriverID> = [DriverID(validated: "vision-tower@1"), DriverID(validated: "audio-encoder@1")]

    /// The role contracts implemented so far. The other drivers' contracts arrive with their milestones.
    public static let contracts: [DriverID: DriverContract] = [
        DriverContract.decoderLM.driver: .decoderLM, DriverContract.latentDiffusion.driver: .latentDiffusion,
        DriverContract.flux2Klein.driver: .flux2Klein,
    ]

    /// `.unknownDriver` unless `id` is a driver this build knows.
    public static func requireKnown(_ id: DriverID) throws(AetherError) {
        guard drivers.contains(id) else { throw .unknownDriver(id) }
    }
}

/// The closed set of roles a driver understands (MANIFEST.md §1). Pure data: AetherCore checks a manifest's
/// roles against it, and AetherRuntime checks the same roles against the graph signatures.
public struct DriverContract: Sendable, Hashable {
    public enum ScalarKind: String, Sendable, Hashable {
        case integer
        case floatingPoint

        public func admits(_ type: ScalarType) -> Bool {
            switch self {
            case .integer: type.isInteger
            case .floatingPoint: type.isFloatingPoint
            }
        }
    }

    /// What a role requires of the graph tensor it names.
    public struct RoleSpec: Sendable, Hashable {
        public var required: Bool
        /// The exact rank, or nil for any rank.
        public var rank: Int?
        public var scalar: ScalarKind?

        public init(required: Bool, rank: Int?, scalar: ScalarKind?) {
            self.required = required
            self.rank = rank
            self.scalar = scalar
        }
    }

    public struct GraphContract: Sendable, Hashable {
        public var inputs: [String: RoleSpec]
        /// Inputs outside `inputs` are accepted only as `binding: static` with a `source` (tables, fills, chaining).
        public var allowsStaticSourcedInputs: Bool
        public var outputs: [String: RoleSpec]
        public var minimumSequenceStates: Int
        public var sequenceScalar: ScalarKind
        public var allowsRecurrent: Bool
        public var recurrentScalar: ScalarKind
        /// Phases allowed in `functions` (`function` is the `step` phase).
        public var phases: Set<String>
        public var allowsFunctionFamily: Bool
        public var requiresMaxContext: Bool

        public init(
            inputs: [String: RoleSpec], allowsStaticSourcedInputs: Bool, outputs: [String: RoleSpec],
            minimumSequenceStates: Int, sequenceScalar: ScalarKind = .floatingPoint, allowsRecurrent: Bool,
            recurrentScalar: ScalarKind = .floatingPoint, phases: Set<String>, allowsFunctionFamily: Bool,
            requiresMaxContext: Bool
        ) {
            self.inputs = inputs
            self.allowsStaticSourcedInputs = allowsStaticSourcedInputs
            self.outputs = outputs
            self.minimumSequenceStates = minimumSequenceStates
            self.sequenceScalar = sequenceScalar
            self.allowsRecurrent = allowsRecurrent
            self.recurrentScalar = recurrentScalar
            self.phases = phases
            self.allowsFunctionFamily = allowsFunctionFamily
            self.requiresMaxContext = requiresMaxContext
        }
    }

    public var driver: DriverID
    /// Graph role → contract. Every key is required; no other graph role is accepted.
    public var graphs: [String: GraphContract]
    /// Graph roles a bundle may add (FLUX.2's VAE encoder), each checked against its contract when present.
    public var optionalGraphs: [String: GraphContract]
    public var requiresTokenizer: Bool

    public init(
        driver: DriverID, graphs: [String: GraphContract], optionalGraphs: [String: GraphContract] = [:],
        requiresTokenizer: Bool
    ) {
        self.driver = driver
        self.graphs = graphs
        self.optionalGraphs = optionalGraphs
        self.requiresTokenizer = requiresTokenizer
    }

    /// The contract of graph `role`, required or optional.
    public func graph(_ role: String) -> GraphContract? { graphs[role] ?? optionalGraphs[role] }

    /// Checks the manifest's roles against the contract. Errors name the offending field.
    public func validate(_ manifest: Manifest) throws(AetherError) {
        guard manifest.driver == driver else {
            throw .manifestMismatch(field: "driver", expected: driver.description, found: manifest.driver.description)
        }
        if requiresTokenizer {
            guard let tokenizer = manifest.tokenizer else {
                throw .manifestInvalid(field: "tokenizer", reason: "\(driver) requires a tokenizer")
            }
            try require(!(tokenizer.stop ?? []).isEmpty, "tokenizer.stop", "\(driver) requires at least one stop marker")
        }
        for key in manifest.graphs.keys.sorted() where graph(key) == nil {
            let known = (Array(graphs.keys) + Array(optionalGraphs.keys)).sorted()
            throw .manifestInvalid(
                field: "graphs.\(key)", reason: "\(driver) has no graph role '\(key)'; known: \(known.joined(separator: ", "))")
        }
        for (key, contract) in optionalGraphs.sorted(by: { $0.key < $1.key }) {
            if let graph = manifest.graphs[key] { try contract.validate(graph, at: "graphs.\(key)", driver: driver) }
        }
        for (key, contract) in graphs.sorted(by: { $0.key < $1.key }) {
            guard let graph = manifest.graphs[key] else {
                throw .manifestInvalid(field: "graphs.\(key)", reason: "required by \(driver)")
            }
            try contract.validate(graph, at: "graphs.\(key)", driver: driver)
        }
    }
}

extension DriverContract.GraphContract {
    func validate(_ graph: Manifest.Graph, at prefix: String, driver: DriverID) throws(AetherError) {
        if graph.functionFamily != nil, !allowsFunctionFamily {
            throw .manifestInvalid(field: "\(prefix).functionFamily", reason: "\(driver) does not take function families")
        }
        for phase in graph.phases.keys.sorted() where !phases.contains(phase) {
            throw .manifestInvalid(
                field: "\(prefix).functions.\(phase)",
                reason: "\(driver) has no phase '\(phase)'; known: \(phases.sorted().joined(separator: ", "))")
        }
        for phase in graph.phases.keys.sorted() {
            let field = graph.inputsByFunction?[phase] != nil ? "\(prefix).inputsByFunction.\(phase)" : "\(prefix).inputs"
            try checkInputs(graph.inputs(forPhase: phase), at: field, driver: driver)
        }
        for role in graph.outputs.keys.sorted() where outputs[role] == nil {
            throw .manifestInvalid(
                field: "\(prefix).outputs.\(role)",
                reason: "\(driver) has no output role '\(role)'; known: \(outputs.keys.sorted().joined(separator: ", "))")
        }
        for (role, spec) in outputs.sorted(by: { $0.key < $1.key }) where spec.required && graph.outputs[role] == nil {
            throw .manifestInvalid(field: "\(prefix).outputs.\(role)", reason: "required by \(driver)")
        }
        let states = graph.states ?? Manifest.States()
        try require(states.sequence.count >= minimumSequenceStates, "\(prefix).states.sequence",
                    "\(driver) requires at least \(minimumSequenceStates) sequence state(s)")
        if !allowsRecurrent {
            try require(states.recurrent.isEmpty, "\(prefix).states.recurrent", "\(driver) does not take recurrent states")
        }
        try require((states.host ?? []).isEmpty, "\(prefix).states.host", "host-held state is reserved for decoder-hostcache@1")
        if requiresMaxContext {
            try require(graph.limits?.maxContext != nil, "\(prefix).limits.maxContext", "required by \(driver)")
        }
    }

    func checkInputs(_ bindings: [String: Manifest.Binding], at prefix: String, driver: DriverID) throws(AetherError) {
        for (role, binding) in bindings.sorted(by: { $0.key < $1.key }) where inputs[role] == nil {
            guard allowsStaticSourcedInputs, binding.isStatic, binding.source != nil else {
                throw .manifestInvalid(
                    field: "\(prefix).\(role)",
                    reason: "\(driver) has no input role '\(role)' (known: \(inputs.keys.sorted().joined(separator: ", "))); "
                        + "other inputs must be binding: static with a source")
            }
        }
        for (role, spec) in inputs.sorted(by: { $0.key < $1.key }) where spec.required && bindings[role] == nil {
            throw .manifestInvalid(field: "\(prefix).\(role)", reason: "required by \(driver)")
        }
    }
}

extension DriverContract {
    /// `decoder-lm@1` (MANIFEST.md §1.1): in `tokens`, `positions`; out `logits`, optional `hidden`; at least one
    /// sequence state; optional recurrent states; static table inputs; phases `step` and `prefill`.
    public static let decoderLM = DriverContract(
        driver: .decoderLM,
        graphs: [
            "decoder": GraphContract(
                inputs: [
                    "tokens": RoleSpec(required: true, rank: 2, scalar: .integer),
                    "positions": RoleSpec(required: true, rank: 2, scalar: .integer),
                ],
                allowsStaticSourcedInputs: true,
                outputs: [
                    "logits": RoleSpec(required: true, rank: 3, scalar: .floatingPoint),
                    "hidden": RoleSpec(required: false, rank: 3, scalar: .floatingPoint),
                ],
                minimumSequenceStates: 1, allowsRecurrent: true, phases: ["step", "prefill"],
                allowsFunctionFamily: false, requiresMaxContext: true),
        ],
        requiresTokenizer: true)
}

extension DriverContract {
    /// `latent-diffusion@1` (MANIFEST.md §1.1, §5): a text-conditioned latent diffusion model with a U-Net denoiser.
    /// `textEncoder` in `tokens`, out `conditioning`; `denoiser` in `latents`, `timestep`, `conditioning` (chained from
    /// the text encoder), out `epsilon`; `vaeDecoder` in `latents`, out `image`. Stateless, one function each. The
    /// driver also requires a tokenizer and the `schedule` and `latent` blocks (checked when it loads).
    public static let latentDiffusion = DriverContract(
        driver: .latentDiffusion,
        graphs: [
            "textEncoder": stateless(
                inputs: ["tokens": RoleSpec(required: true, rank: 2, scalar: .integer)],
                outputs: ["conditioning": RoleSpec(required: true, rank: 3, scalar: .floatingPoint)]),
            "denoiser": stateless(
                inputs: [
                    "latents": RoleSpec(required: true, rank: 4, scalar: .floatingPoint),
                    "timestep": RoleSpec(required: true, rank: 1, scalar: .floatingPoint),
                    "conditioning": RoleSpec(required: true, rank: 3, scalar: .floatingPoint),
                ],
                outputs: ["epsilon": RoleSpec(required: true, rank: 4, scalar: .floatingPoint)]),
            "vaeDecoder": stateless(
                inputs: ["latents": RoleSpec(required: true, rank: 4, scalar: .floatingPoint)],
                outputs: ["image": RoleSpec(required: true, rank: 3, scalar: .floatingPoint)]),
        ],
        requiresTokenizer: false)

    private static func stateless(inputs: [String: RoleSpec], outputs: [String: RoleSpec]) -> GraphContract {
        GraphContract(
            inputs: inputs, allowsStaticSourcedInputs: false, outputs: outputs, minimumSequenceStates: 0,
            allowsRecurrent: false, phases: ["step"], allowsFunctionFamily: false, requiresMaxContext: false)
    }
}

extension DriverContract {
    /// `flux2-klein@1` (MANIFEST.md §5.7): FLUX.2 [klein]. `textEncoder` in `tokens`, `positions`, `tokenMask`, out
    /// `conditioning`; three denoiser segments chained by their carries (`denoiser0` in `latents`, `conditioning`
    /// from the text encoder, `sigma`, `textRotary`, `imageRotary`; out `carryImage`, `carryText`, `condition`;
    /// `denoiser1` and `denoiser2` in the previous carries, `condition` and the rotary tables; `denoiser1` out the
    /// carries, `denoiser2` out `velocity`); `vaeDecoder` in `latents`, out `image`; an optional `vaeEncoder` (reference
    /// images) in `referencePixels`, out `referenceLatents`. The driver also requires a tokenizer with a chat template
    /// and the `schedule`, `latent` and `flux2` blocks (checked when it loads).
    public static let flux2Klein = DriverContract(
        driver: .flux2Klein,
        graphs: [
            "textEncoder": stateless(
                inputs: [
                    "tokens": RoleSpec(required: true, rank: 2, scalar: .integer),
                    "positions": RoleSpec(required: true, rank: 2, scalar: .integer),
                    "tokenMask": RoleSpec(required: true, rank: 2, scalar: .integer),
                ],
                outputs: ["conditioning": RoleSpec(required: true, rank: 2, scalar: .floatingPoint)]),
            "denoiser0": stateless(
                inputs: [
                    "latents": RoleSpec(required: true, rank: 2, scalar: .floatingPoint),
                    "conditioning": RoleSpec(required: true, rank: 2, scalar: .floatingPoint),
                    "sigma": RoleSpec(required: true, rank: 1, scalar: .floatingPoint),
                ].merging(rotaryInputs) { a, _ in a },
                outputs: carryOutputs.merging(["condition": RoleSpec(required: true, rank: 2, scalar: .floatingPoint)]) { a, _ in a }),
            "denoiser1": stateless(inputs: segmentInputs, outputs: carryOutputs),
            "denoiser2": stateless(
                inputs: segmentInputs, outputs: ["velocity": RoleSpec(required: true, rank: 2, scalar: .floatingPoint)]),
            "vaeDecoder": stateless(
                inputs: ["latents": RoleSpec(required: true, rank: 3, scalar: .floatingPoint)],
                outputs: ["image": RoleSpec(required: true, rank: 3, scalar: .floatingPoint)]),
        ],
        optionalGraphs: [
            "vaeEncoder": stateless(
                inputs: ["referencePixels": RoleSpec(required: true, rank: 3, scalar: .floatingPoint)],
                outputs: ["referenceLatents": RoleSpec(required: true, rank: 3, scalar: .floatingPoint)]),
        ],
        requiresTokenizer: false)

    private static let rotaryInputs: [String: RoleSpec] = [
        "textRotary": RoleSpec(required: true, rank: 3, scalar: .floatingPoint),
        "imageRotary": RoleSpec(required: true, rank: 3, scalar: .floatingPoint),
    ]
    private static let carryOutputs: [String: RoleSpec] = [
        "carryImage": RoleSpec(required: true, rank: 2, scalar: .floatingPoint),
        "carryText": RoleSpec(required: true, rank: 2, scalar: .floatingPoint),
    ]
    private static let segmentInputs: [String: RoleSpec] = carryOutputs.merging([
        "condition": RoleSpec(required: true, rank: 2, scalar: .floatingPoint),
    ]) { a, _ in a }.merging(rotaryInputs) { a, _ in a }
}
