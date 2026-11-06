import Foundation
import CoreAI
import Metal

/// The owned decode engine for `decoder-lm@1` (DESIGN.md §6.2): one decoder graph (with an optional `prefill`
/// phase), its KV and recurrent states, a position counter, rewind and snapshot, the prefill schedule, and greedy
/// decoding pipelined on the GPU (§6.1).
///
/// The Core AI ``LanguageEngine``. An actor: the graphs, the arena and every buffer live inside it and never leave
/// (§11). Results are `Sendable` values (``Logits``, ``EngineSnapshot``). One request at a time: a greedy decode in
/// progress makes every other call fail with `unsupportedOperation` until ``endGreedy()``. Beyond the protocol it
/// exposes its Core AI specifics: ``Options`` (capacity, pipeline depth, the canary), ``canary``, ``decodeMode``,
/// ``description`` and ``sizing``.
public actor DecoderEngine: LanguageEngine {
    public struct Options: Sendable, Hashable {
        public enum Execution: Sendable, Hashable {
            /// `encode` on a Metal-backed arena for real graphs; `run` for other backends (test fakes).
            case automatic
            /// `Graph.run` on host `NDArray`s, one call at a time, CPU argmax. The reference path.
            case run
        }

        /// An explicit capacity (tokens). Above the derived capacity it throws `capacity`.
        public var capacity: Int?
        /// The first allocation of the sequence axis; it grows geometrically from there. Default min(capacity, 1024).
        public var initialCapacity: Int?
        public var memoryBudget: MemoryBudget
        public var execution: Execution
        /// Greedy steps in flight once the canary passes (DESIGN.md §6.1: 3).
        public var pipelineDepth: Int
        /// Run the delayed-argmax canary at load. Without it (or if it fails) greedy decoding uses depth 1.
        public var canary: Bool
        /// Canary length, argmax delay (spin iterations) and the capacity it runs at (default ≥ 4096: the lab's
        /// race appeared at capacity ≥ 4096).
        public var canarySteps: Int
        public var canaryDelay: UInt32
        public var canaryCapacity: Int
        public var cachePolicy: GraphOptions.CachePolicy
        public var cache: GraphOptions.Cache
        /// Diagnostics only: hand each greedy token to the next step through a view into the page-aligned token
        /// history (the lab's `--handoff offset`), the hand-off that races. The canary's negative control must catch
        /// it. Never for production use.
        @_spi(Diagnostics) public var racingTokenHandoff = false
        /// Diagnostics only: the argmax writes a scratch slot instead of the next step's, so every pipelined step reads
        /// a stale token, as an unordered read would. Proves the canary's detection and fallback deterministically.
        @_spi(Diagnostics) public var staleTokenHandoff = false
        /// Diagnostics only: use `pipelineDepth` without the canary. With `canary` off (and without this), greedy
        /// decoding uses depth 1, as documented.
        @_spi(Diagnostics) public var uncheckedPipelineDepth = false

        public init(
            capacity: Int? = nil, initialCapacity: Int? = nil, memoryBudget: MemoryBudget = .platformDefault,
            execution: Execution = .automatic, pipelineDepth: Int = 3, canary: Bool = true, canarySteps: Int = 64,
            canaryDelay: UInt32 = 200_000, canaryCapacity: Int = 4096, cachePolicy: GraphOptions.CachePolicy = .default,
            cache: GraphOptions.Cache = .default
        ) {
            self.capacity = capacity
            self.initialCapacity = initialCapacity
            self.memoryBudget = memoryBudget
            self.execution = execution
            self.pipelineDepth = pipelineDepth
            self.canary = canary
            self.canarySteps = canarySteps
            self.canaryDelay = canaryDelay
            self.canaryCapacity = canaryCapacity
            self.cachePolicy = cachePolicy
            self.cache = cache
        }
    }

    /// How greedy tokens are produced (diagnostics).
    public enum DecodeMode: Sendable, Hashable, CustomStringConvertible {
        /// `run` + CPU argmax, one step at a time.
        case host
        /// `encode` + a same-queue GPU argmax with `depth` steps in flight (1 = no overlap).
        case gpu(depth: Int)

        public var description: String {
            switch self {
            case .host: "host run, depth 1"
            case .gpu(let depth): "encode + GPU argmax, depth \(depth)"
            }
        }
    }

    public struct CanaryReport: Sendable, Hashable {
        public var passed: Bool
        public var depth: Int
        public var capacity: Int
        public var steps: Int
        /// The first token index where the pipelined run differed from depth 1.
        public var firstMismatch: Int?
        public var seconds: Double
    }

    /// Host time spent per greedy phase, summed (diagnostics).
    public struct GreedyTimings: Sendable, Hashable {
        public var steps = 0
        public var bindings = 0.0
        public var encode = 0.0
        public var commit = 0.0
        public var wait = 0.0
    }

    /// Host time per greedy phase since load (diagnostics).
    @_spi(Diagnostics) public var greedyTimings: GreedyTimings { core.timings }

    /// The engine state at one position: a host copy of every state, carried as an ``EngineSnapshot``'s payload.
    struct Snapshot: Sendable {
        var position: Int
        var capacity: Int
        var states: StateSnapshot
        var byteCount: Int { states.byteCount }
    }

    /// Fixed facts about the loaded model.
    public struct Description: Sendable, Hashable {
        public var vocab: Int
        /// The capacity the sequence axis may grow to.
        public var maxCapacity: Int
        /// Recurrent states force snapshot/restore instead of a partial rewind.
        public var hasRecurrentState: Bool
        /// The prefill phase's static chunk, when the graph has one.
        public var prefillChunk: Int?
        public var schedule: PrefillPlan.Shape
        public var encodes: Bool
    }

    private let core: DecoderCore
    /// The sizing inputs the capacity was derived from (diagnostics).
    public nonisolated let sizing: KVSizing
    /// The model's measured Metal allocation delta across load and its files' size (diagnostics; zero for fakes).
    public private(set) var loadedModelBytes: (gpu: Int64, disk: Int64) = (0, 0)
    public nonisolated let description: Description
    public nonisolated var vocab: Int { description.vocab }
    public nonisolated var maxCapacity: Int { description.maxCapacity }
    public nonisolated var hasRecurrentState: Bool { description.hasRecurrentState }
    public private(set) var canary: CanaryReport?
    public private(set) var decodeMode: DecodeMode
    public var decodeModeDescription: String { decodeMode.description }

    /// Loads `manifest`'s decoder graph from an installed bundle. Compute and the reshape hint come from the
    /// manifest; capacity from the memory budget (``KVSizing``).
    public init(bundle: InstalledBundle, manifest: Manifest, device: Device, options: Options = Options()) async throws(AetherError) {
        guard manifest.driver == .decoderLM else {
            throw .manifestMismatch(field: "driver", expected: DriverID.decoderLM.description, found: manifest.driver.description)
        }
        try DriverContract.decoderLM.validate(manifest)
        guard let graph = manifest.graphs["decoder"] else {
            throw .manifestInvalid(field: "graphs.decoder", reason: "required by decoder-lm@1")
        }
        guard let path = graph.file.resolve(architecture: device.architecture), let url = bundle.url(of: path) else {
            throw .manifestInvalid(field: "graphs.decoder.file", reason: "no file for architecture \(device.architecture)")
        }
        let graphOptions = try GraphOptions(
            manifest: graph, at: "graphs.decoder", cachePolicy: options.cachePolicy, cache: options.cache)
        let phases = graph.phases
        guard let stepName = phases["step"] else {
            throw .manifestInvalid(field: "graphs.decoder.functions.step", reason: "decoder-lm@1 needs a step phase")
        }
        let gpu = MTLCreateSystemDefaultDevice()
        let gpuBefore = gpu?.currentAllocatedSize ?? 0
        let loaded = try await Graph.load(url: url, functions: Array(Set(phases.values)).sorted(), options: graphOptions)
        // The model's own memory (MemoryBudget): its Metal allocations across the load, at least its files' size.
        let gpuDelta = Int64(max(0, (gpu?.currentAllocatedSize ?? 0) - gpuBefore))
        let modelBytes = max(gpuDelta, diskBytes(url))
        var signatures: [String: GraphSignature] = [:]
        for (name, g) in loaded { signatures[name] = g.signature }
        try SignatureValidator.validate(manifest, signatures: ["decoder": signatures])
        let device = device
        let metal: (any MTLDevice)? = options.execution == .automatic ? gpu : nil
        let pair = Backends(step: loaded[stepName]!, prefill: phases["prefill"].flatMap { loaded[$0] })
        try await self.init(backends: pair, graph: graph, device: device, options: options, metal: metal, modelBytes: modelBytes)
        loadedModelBytes = (gpu: gpuDelta, disk: diskBytes(url))
    }

    /// An engine over any ``GraphBackend`` pair (tests drive the fake decoder through this). `graph` supplies the
    /// roles, states and limits; `run` is the execution path unless the backends are real graphs and `metal` is set.
    public init(
        step: sending any GraphBackend, prefill: sending (any GraphBackend)? = nil, graph: Manifest.Graph,
        device: Device, options: Options = Options()
    ) async throws(AetherError) {
        try await self.init(backends: Backends(step: step, prefill: prefill), graph: graph, device: device, options: options, metal: nil, modelBytes: 0)
    }

    /// The step and prefill backends, sent into the actor as one region.
    private struct Backends {
        var step: any GraphBackend
        var prefill: (any GraphBackend)?
    }

    private init(
        backends: sending Backends, graph: Manifest.Graph, device: Device, options: Options, metal: (any MTLDevice)?,
        modelBytes: Int64
    ) async throws(AetherError) {
        let step = backends.step, prefill = backends.prefill
        // DecoderCore keeps 8 token slots and needs ring > depth + 1; no option may reach a trap.
        guard (1...(DecoderCore.ring - 2)).contains(options.pipelineDepth) else {
            throw .unsupportedOperation("pipelineDepth \(options.pipelineDepth); supported: 1...\(DecoderCore.ring - 2)")
        }
        let encodable = metal != nil && step is Graph && (prefill == nil || prefill is Graph)
        let arena = encodable ? TensorArena(device: metal!) : TensorArena()
        let kernels = encodable ? try await GreedyKernels(device: metal!) : nil
        let limits = graph.limits ?? Manifest.Limits()
        guard let maxContext = limits.maxContext else {
            throw .manifestInvalid(field: "graphs.decoder.limits.maxContext", reason: "required by decoder-lm@1")
        }
        var sizing = try Self.sizing(step.signature, graph: graph, maxContext: maxContext, platform: device.platform)
        sizing.modelBytes = modelBytes
        self.sizing = sizing
        let maxCapacity = try sizing.capacity(budget: options.memoryBudget.bytes(on: device), requested: options.capacity)
        let initial = options.initialCapacity ?? min(maxCapacity, 1024)
        guard initial <= maxCapacity else {
            throw .capacity(what: "KV initial capacity (tokens)", needed: Int64(initial), available: Int64(maxCapacity))
        }
        let core = try DecoderCore(
            step: step, prefill: prefill, graph: graph, arena: arena, kernels: kernels, capacityLimit: maxCapacity,
            initialCapacity: initial, racingHandoff: options.racingTokenHandoff,
            staleHandoff: options.staleTokenHandoff)
        self.core = core
        description = Description(
            vocab: core.vocab, maxCapacity: core.maxCapacity, hasRecurrentState: core.hasRecurrentState,
            prefillChunk: core.shape.prefillChunk, schedule: core.shape, encodes: core.isEncoding)
        decodeMode = core.isEncoding ? .gpu(depth: 1) : .host
        guard options.canarySteps > 0, options.canaryCapacity > 0 else {
            throw .generation(.invalidParameters("the canary needs steps > 0 and capacity > 0"))
        }
        if core.isEncoding, options.pipelineDepth > 1, options.canary {
            let report = try await core.runCanary(
                depth: options.pipelineDepth, steps: options.canarySteps, delay: options.canaryDelay,
                capacity: options.canaryCapacity)
            canary = report
            if report.passed { decodeMode = .gpu(depth: options.pipelineDepth) }
        } else if core.isEncoding, options.pipelineDepth > 1, options.uncheckedPipelineDepth {
            // Without the canary, depth one unless a diagnostic asks otherwise.
            decodeMode = .gpu(depth: options.pipelineDepth)
        }
    }

    /// The device's current Metal allocations (diagnostics: Core AI's shape-change retention shows up here).
    public nonisolated static func metalAllocatedBytes() -> Int64 {
        Int64(MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0)
    }

    /// Sizes KV from the graph signature: bytes per sequence slot over every sequence state, and the owned bytes
    /// that do not scale (recurrent states, logits and token buffers).
    static func sizing(
        _ signature: GraphSignature, graph: Manifest.Graph, maxContext: Int, platform: Device.Platform
    ) throws(AetherError) -> KVSizing {
        let states = graph.states ?? Manifest.States()
        var perToken: Int64 = 0
        var staticAxis: Int?
        for entry in states.sequence {
            for name in entry.tensors {
                guard let spec = signature.state(name), entry.seqAxis < spec.rank, let width = spec.scalarType.byteWidth else {
                    throw .manifestMismatch(field: "graphs.decoder.states.sequence", expected: "state '\(name)'", found: "\(signature.states.map(\.name))")
                }
                let others = spec.shape.enumerated().filter { $0.offset != entry.seqAxis }.map { max($0.element, 1) }
                perToken += Int64(others.reduce(1, *) * width)
                if spec.shape[entry.seqAxis] != TensorSpec.dynamic { staticAxis = spec.shape[entry.seqAxis] }
            }
        }
        var fixed: Int64 = 0
        for entry in states.recurrent {
            if let spec = signature.state(entry.name), let width = spec.scalarType.byteWidth {
                fixed += Int64(spec.shape.map { max($0, 1) }.reduce(1, *) * width)
            }
        }
        if let logits = graph.outputs["logits"], let spec = signature.output(logits.name), let v = spec.shape.last,
           let width = spec.scalarType.byteWidth {
            let rows = spec.shape.count > 1 && spec.shape[1] == TensorSpec.dynamic ? (graph.limits?.prefillChunk ?? 512) : 1
            fixed += Int64((rows + DecoderCore.ring) * v * width)
        }
        let limits = graph.limits ?? Manifest.Limits()
        return KVSizing(
            maxContext: maxContext, platformCap: limits.kvCapacityCap?[platform.rawValue], maxPosition: limits.maxPosition,
            // + 8: the int32 token history and positions buffers, sized with the capacity.
            bytesPerToken: perToken + 8, fixedBytes: fixed, staticAxis: staticAxis)
    }

    // MARK: - State

    public var position: Int { core.position }
    /// The current allocation of the sequence axis.
    public var capacity: Int { core.capacity }
    /// True when pipelined steps ran recurrent state past ``position``: restore a snapshot or reset first.
    public var stateRanAhead: Bool { core.stateAhead }

    /// Feeds `tokens` at the current position on the prefill schedule (§6.2) and returns the rows asked for.
    public func prefill(_ tokens: [Int32], logits: LogitsSelection = .last) async throws(AetherError) -> Logits? {
        try await core.feed(tokens, select: logits)
    }

    /// Feeds one token and returns the next-token logits (the depth-1 host path for CPU sampling).
    public func step(_ token: Int32) async throws(AetherError) -> Logits {
        try await core.feed([token], select: .last)!
    }

    /// Moves the position back. O(1) on attention-only models (stale slots are overwritten later). With recurrent
    /// states only `rewind(to: 0)` (a reset) is possible; use ``snapshot()`` and ``restore(_:)``.
    public func rewind(to position: Int) throws(AetherError) { try core.rewind(to: position) }

    public func reset() throws(AetherError) { try core.reset() }

    /// A host copy of the state: sequence states up to the position, recurrent states whole.
    public func snapshot() throws(AetherError) -> EngineSnapshot {
        let snapshot = try core.snapshot()
        return EngineSnapshot(position: snapshot.position, byteCount: snapshot.byteCount, payload: snapshot)
    }

    /// Restores a snapshot of this engine's model, validated before any state is touched.
    public func restore(_ snapshot: EngineSnapshot) throws(AetherError) {
        guard let state = snapshot.payload as? Snapshot else {
            throw .unsupportedOperation("restore of a snapshot another engine type took (\(type(of: snapshot.payload)))")
        }
        try core.restore(state)
    }

    // MARK: - Greedy decoding

    /// Starts greedy decoding after `first` (the caller's pick from the prompt's logits, not yet fed), for at most
    /// `limit` more tokens, in ``decodeMode``. Then call ``nextGreedy()`` until it returns nil or the caller stops,
    /// and always ``endGreedy()``.
    public func beginGreedy(first: Int32, limit: Int) throws(AetherError) {
        let depth = switch decodeMode {
        case .host: 0
        case .gpu(let depth): depth
        }
        try core.beginGreedy(first: first, limit: limit, depth: depth)
    }

    /// The next token: delivered by the GPU argmax's completion handler, with the pipeline refilled before this
    /// returns. Nil at the limit or when the calling task is cancelled.
    public func nextGreedy() async throws(AetherError) -> Int32? { try await core.nextGreedy() }

    /// Ends greedy decoding: waits for in-flight GPU work and sets the position to the tokens fed (every token
    /// returned except the last). Returns how many were fed and how many steps ran past the last returned token.
    /// Refused (`unsupportedOperation`) while a ``nextGreedy()`` is in progress.
    @discardableResult
    public func endGreedy() async throws(AetherError) -> (fed: Int, overshoot: Int) {
        let end = try await core.endGreedy()
        return (end.fed, end.overshoot)
    }
}

/// The `decoder-lm@1` driver (DESIGN.md §6.3): loads an installed bundle into a ``DecoderEngine``.
public enum DecoderLMDriver: ModelDriver {
    public static let id: DriverID = .decoderLM

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> DecoderEngine {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        return try await DecoderEngine(bundle: bundle, manifest: manifest, device: options.device, options: options.decoder)
    }
}
