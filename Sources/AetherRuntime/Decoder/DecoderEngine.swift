import Foundation
import CoreAI
import Metal

/// The owned decode engine for `decoder-lm@1` (DESIGN.md §6.2): one decoder graph (with an optional `prefill`
/// phase), its KV and recurrent states, a position counter, rewind and snapshot, the prefill schedule, and greedy
/// decoding pipelined on the GPU (§6.1). For `vlm-static-inject@1` it also owns the vision tower and the decoder's
/// static image inputs (``ImageConditionedEngine``, docs/VLM.md).
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

        /// An explicit capacity (tokens): the most the state holds. Above the derived capacity (or a graph's static
        /// sequence axis) it throws `capacity`; below a static axis it limits the tokens, the allocation staying static.
        public var capacity: Int?
        /// The first allocation of the sequence axis; it grows geometrically from there. Default min(capacity, 1024),
        /// or the whole capacity when the budget can't hold the growth copy (``KVSizing/initialAllocation(capacity:budget:)``).
        public var initialCapacity: Int?
        public var memoryBudget: MemoryBudget
        public var execution: Execution
        /// Greedy steps in flight once the canary passes (DESIGN.md §6.1: 3).
        public var pipelineDepth: Int
        /// Qualify ``pipelineDepth`` with the delayed-argmax canary (DESIGN.md §6.1) before greedy decoding first
        /// uses it: before the first ``beginGreedy(first:limit:)``, or at load with ``canaryAtLoad``. Without it (or if
        /// it fails) greedy decoding uses depth 1.
        public var canary: Bool
        /// Run the canary during load instead of before the first greedy decode. Off by default: the canary costs a
        /// load about 7 s on a 27B, and a client that never decodes greedily on the pipelined path (a repetition
        /// penalty, a constraint, sampling) never needs it. Either way no greedy decode runs deeper than depth 1 before
        /// the canary has passed on this engine.
        public var canaryAtLoad = false
        /// Host snapshots at full capacity that the memory budget keeps room for, on a model with recurrent state
        /// (where prefix reuse goes through ``snapshot()`` and ``restore(_:)``; attention-only models are charged
        /// none). The default, 4, is the Generator's most (`Generator.maxSnapshots`): the system turn's, the last image
        /// turn's and the last text prompt's two. 0 for a client that never snapshots.
        public var reservedSnapshots: Int
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
        /// The most tokens one call of a dynamic token axis feeds (the prefill's chunks), at most the manifest's
        /// `limits.prefillChunk`. Nil (the default): ``DecoderEngine/reshapeSafePrefillWidth`` on a Core AI graph
        /// specialized with `expectFrequentReshapes`, the manifest's chunk otherwise. Wider calls prefill faster, but
        /// Core AI then retains Metal on every request (see ``DecoderEngine/reshapeSafePrefillWidth``).
        public var prefillWidth: Int? = nil
        /// How much Metal the model may retain past its level after the first request before the engine reloads its
        /// decoder (``reloadIfRetainedPastBudget()``; docs/VLM.md, V6). Nil (the default):
        /// ``DecoderEngine/defaultReloadBudgetBytes`` for a bundle-loaded Core AI decoder specialized with
        /// `expectFrequentReshapes` and without recurrent state, no reloads otherwise. 0 never reloads.
        public var reloadBudgetBytes: Int64? = nil

        /// What load charges for the model's own memory against the memory budget (``ModelCharge``). Loading
        /// allocates almost nothing (Core AI's memory arrives with the first run), so what load can measure is the
        /// graph files' size, below what the model reaches: a 4B `vlm-static-inject@1` export of 8.28 GiB on disk read
        /// 9.4 GiB of Metal after load and 13.6 GiB over a session (process-wide, its KV included). Default
        /// ``ModelCharge/qualified``.
        public var modelCharge: ModelCharge = .qualified

        /// What load charges for the model (``Options/modelCharge``). Every choice charges at least what load
        /// measures: the graph files' size, or the Metal delta across the loads.
        public enum ModelCharge: Sendable, Hashable {
            /// The default: a bundle load charges the kit's qualified figure for the export on this chip
            /// (``DecoderEngine/qualifiedModelBytes(for:device:)``) when it is larger than what load measures, and
            /// what load measures when the kit has no figure.
            case qualified
            /// What load measures, without the kit's qualified figure.
            case measured
            /// A declared figure: what a measurement of this model took (``Memory/measuredModelBytes`` from an earlier
            /// session on this device, or a gate's). Load charges it when it is larger than what load measures, and
            /// refuses with `capacity` when the KV no longer fits beside it. At least 0.
            case declared(bytes: Int64)
        }

        public init(
            capacity: Int? = nil, initialCapacity: Int? = nil, memoryBudget: MemoryBudget = .platformDefault,
            execution: Execution = .automatic, pipelineDepth: Int = 3, canary: Bool = true, canarySteps: Int = 64,
            canaryDelay: UInt32 = 200_000, canaryCapacity: Int = 4096, cachePolicy: GraphOptions.CachePolicy = .default,
            cache: GraphOptions.Cache = .default, reservedSnapshots: Int = 4
        ) {
            self.reservedSnapshots = reservedSnapshots
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

    /// The engine's memory, read on its probe (`GPUMemoryProbe.system`: the larger of the Metal device's allocation
    /// and the process footprint, so Core AI's memory is counted). The probe reads the whole process, so memory
    /// anything else in the process allocates while the engine is loaded counts here too.
    public struct Memory: Sendable, Hashable {
        /// The level just before the engine loaded its graphs.
        public var baselineBytes: Int64
        /// The level at the call.
        public var currentBytes: Int64
        /// The highest level read since load: after load (and the canary), and after each prefill, step, greedy decode
        /// and image bind.
        public var peakBytes: Int64
        /// The engine's own buffers at the call: KV and recurrent states at their allocation, logits and token
        /// buffers, the image statics and the tower's buffers.
        public var ownedBytes: Int64
        /// What load charged for the model against the memory budget (``KVSizing/modelBytes``).
        public var chargedModelBytes: Int64
        /// The model's own memory as measured: the most the level stood above ``baselineBytes`` at any of those
        /// reads, less the engine's own buffers at that read. Pass it as ``Options/ModelCharge/declared(bytes:)`` to charge it at the
        /// next load.
        public var measuredModelBytes: Int64

        /// What the engine holds now, its buffers included: ``currentBytes`` above ``baselineBytes`` (at least 0). The
        /// same name as the pipelines' live reading (`Krea2Pipeline.liveResidentBytes`).
        public var liveResidentBytes: Int64 { max(0, currentBytes - baselineBytes) }
    }

    /// The engine's memory now (``Memory``): live, read at the call.
    public var memory: Memory {
        sampleMemory()
        return Memory(
            baselineBytes: memoryBaseline, currentBytes: lastLevel, peakBytes: peakLevel, ownedBytes: ownedBytes,
            chargedModelBytes: sizing.modelBytes, measuredModelBytes: peakModel)
    }

    /// The engine's buffers: its arena and the tower's.
    private var ownedBytes: Int64 { Int64(core.arena.byteCount + (imageBinder?.arena.byteCount ?? 0)) }

    /// Reads the probe into ``memory``'s peak and measured model bytes.
    func sampleMemory() {
        let level = probe.allocatedBytes()
        lastLevel = level
        peakLevel = max(peakLevel, level)
        peakModel = max(peakModel, level - memoryBaseline - ownedBytes)
    }

    /// Host time per greedy phase since load (diagnostics).
    @_spi(Diagnostics) public var greedyTimings: GreedyTimings { core.timings }

    /// The engine state at one position: a host copy of every state, carried as an ``EngineSnapshot``'s payload.
    struct Snapshot: Sendable {
        var position: Int
        var capacity: Int
        var states: StateSnapshot
        /// The images the statics held (a snapshot records them, not their rows).
        var images: [ImagePromptPlan.Image] = []
        var byteCount: Int { states.byteCount }
    }

    /// Fixed facts about the loaded model.
    public struct Description: Sendable, Hashable {
        public var vocab: Int
        /// The most tokens the state can hold: what the sequence axis may grow to, or, on a graph with a static
        /// sequence axis, at most that axis (``DecoderEngine/capacity`` is the allocation).
        public var maxCapacity: Int
        /// Recurrent states force snapshot/restore instead of a partial rewind.
        public var hasRecurrentState: Bool
        /// The prefill phase's static chunk, when the graph has one.
        public var prefillChunk: Int?
        public var schedule: PrefillPlan.Shape
        public var encodes: Bool
    }

    let core: DecoderCore
    /// The vision tower and the image record of a `vlm-static-inject@1` model; nil for a text model.
    let imageBinder: ImageBinder?
    /// The image contract (nil: the model takes no images).
    public nonisolated let imageInput: ImageInputSpec?
    /// The sizing inputs the capacity was derived from (diagnostics).
    public nonisolated let sizing: KVSizing
    /// The model's measured Metal allocation delta across load and its files' size (diagnostics; zero for fakes).
    public private(set) var loadedModelBytes: (gpu: Int64, disk: Int64) = (0, 0)
    /// The model memory load was given as declared: ``Options/ModelCharge/declared(bytes:)``, or, for a bundle load
    /// with ``Options/ModelCharge/qualified``, the kit's qualified figure for the export on this chip
    /// (``qualifiedModelBytes(for:device:)``). Nil when neither; the charge is then what load measured. Load charges the larger of this and what it measured (``Memory/chargedModelBytes``).
    public nonisolated let declaredModelBytes: Int64?
    public nonisolated let description: Description
    public nonisolated var vocab: Int { description.vocab }
    public nonisolated var maxCapacity: Int { description.maxCapacity }
    public nonisolated var hasRecurrentState: Bool { description.hasRecurrentState }
    /// The canary's report: nil until it has run (at load with ``Options/canaryAtLoad``, else before the first greedy
    /// decode), and nil without ``Options/canary``.
    public private(set) var canary: CanaryReport?
    /// The canary still to run before the first greedy decode (its depth, length, delay and capacity).
    private var pendingCanary: (depth: Int, steps: Int, delay: UInt32, capacity: Int)?
    /// The bundle's reader lease, held until the engine is released.
    private var bundleLease: BundleReadLease?
    private let probe: GPUMemoryProbe
    private let memoryBaseline: Int64
    private var lastLevel: Int64
    private var peakLevel: Int64
    private var peakModel: Int64 = 0
    /// The greedy mode: depth 1 until the canary has passed, then ``Options/pipelineDepth``.
    public private(set) var decodeMode: DecodeMode
    /// Reloads since load (``reloadIfRetainedPastBudget()``).
    public private(set) var reloads = Reloads()
    private let reloader: Reloader?
    /// The level reloads are measured from (the first ``reloadIfRetainedPastBudget()`` call's), and its probe.
    private var retentionFloor: Int64?
    private let levelProbe: @Sendable () -> Int64
    private var reloading = false
    public var decodeModeDescription: String { decodeMode.description }

    /// Loads `manifest`'s decoder graph from an installed bundle. Compute and the reshape hint come from the
    /// manifest; capacity from the memory budget (``KVSizing``). A bundle from a `ModelStore` stays leased
    /// (``InstalledBundle/readLease()``) for the engine's lifetime, so the store refuses to remove it meanwhile.
    /// With ``Options/ModelCharge/qualified`` (the default), the model's charge is the kit's qualified figure for this
    /// export on this chip when it has one (``qualifiedModelBytes(for:device:)``), else what load measures.
    public init(bundle: InstalledBundle, manifest: Manifest, device: Device, options: Options = Options()) async throws(AetherError) {
        let options = Self.declaringQualified(options, manifest: manifest, device: device)
        let lease = try bundle.readLease()
        let vlm = manifest.driver == .vlmStaticInject
        guard manifest.driver == .decoderLM || vlm else {
            throw .manifestMismatch(field: "driver", expected: DriverID.decoderLM.description, found: manifest.driver.description)
        }
        try (vlm ? DriverContract.vlmStaticInject : .decoderLM).validate(manifest)
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
        let probe = GPUMemoryProbe.system
        let baseline = probe.allocatedBytes()
        let gpuBefore = gpu?.currentAllocatedSize ?? 0
        let loaded = try await Graph.load(url: url, functions: Array(Set(phases.values)).sorted(), options: graphOptions)
        var signatures: [String: GraphSignature] = [:]
        for (name, g) in loaded { signatures[name] = g.signature }
        // A VLM's tower: one static function, its own specialization, resident for the engine's lifetime.
        var tower: Graph?
        var towerDisk: Int64 = 0
        var image: ImageConfiguration?
        var all = ["decoder": signatures]
        if vlm {
            guard let vision = manifest.graphs["vision"], let function = vision.function,
                  let towerPath = vision.file.resolve(architecture: device.architecture), let towerURL = bundle.url(of: towerPath) else {
                throw .manifestInvalid(field: "graphs.vision", reason: "vlm-static-inject@1 needs a one-function vision graph for \(device.architecture)")
            }
            guard let block = manifest.image, let preprocess = manifest.preprocess?.image else {
                throw .manifestInvalid(field: "image", reason: "vlm-static-inject@1 requires the image block and preprocess.image")
            }
            let towerOptions = try GraphOptions(manifest: vision, at: "graphs.vision", cachePolicy: options.cachePolicy, cache: options.cache)
            let loadedTower = try await Graph.load(url: towerURL, functions: [function], options: towerOptions)[function]!
            all["vision"] = [function: loadedTower.signature]
            tower = loadedTower
            towerDisk = diskBytes(towerURL)
            image = ImageConfiguration(vision: vision, image: block, preprocess: preprocess)
        }
        try SignatureValidator.validate(manifest, signatures: all)
        // The model's own memory (MemoryBudget): its Metal allocations across the loads (decoder and tower), at least
        // its files' size.
        let gpuTotal = Int64(max(0, (gpu?.currentAllocatedSize ?? 0) - gpuBefore))
        let device = device
        let metal: (any MTLDevice)? = options.execution == .automatic ? gpu : nil
        // A reload loads the same functions again (through the compile cache) once the old ones are released.
        let functions = Array(Set(phases.values)).sorted(), prefillName = phases["prefill"]
        let reloader: Reloader = { () async throws(AetherError) -> ReloadedGraphs in
            let graphs = try await Graph.load(url: url, functions: functions, options: graphOptions)
            return ReloadedGraphs(step: graphs[stepName]!, prefill: prefillName.flatMap { graphs[$0] })
        }
        let pair = Backends(step: loaded[stepName]!, prefill: prefillName.flatMap { loaded[$0] }, tower: tower,
                            reloader: reloader)
        try await self.init(backends: pair, graph: graph, image: image, device: device, options: options, metal: metal,
                            modelBytes: max(gpuTotal, diskBytes(url) + towerDisk), probe: probe, baseline: baseline)
        loadedModelBytes = (gpu: gpuTotal, disk: diskBytes(url) + towerDisk)
        bundleLease = lease
    }

    /// An engine over any ``GraphBackend`` pair (tests drive the fake decoder through this). `graph` supplies the
    /// roles, states and limits; `run` is the execution path unless the backends are real graphs and `metal` is set.
    public init(
        step: sending any GraphBackend, prefill: sending (any GraphBackend)? = nil, graph: Manifest.Graph,
        device: Device, options: Options = Options()
    ) async throws(AetherError) {
        try await self.init(backends: Backends(step: step, prefill: prefill), graph: graph, image: nil, device: device,
                            options: options, metal: nil, modelBytes: 0, probe: .system, baseline: nil)
    }

    /// A `vlm-static-inject@1` engine over any ``GraphBackend``s (tests drive a fake tower and decoder through this):
    /// `graph` and `visionGraph` supply the roles, `image` and `preprocess` the image contract, all checked against
    /// the backends' signatures. Package-only: hosts load a bundle (``init(bundle:manifest:device:options:)``).
    package init(
        step: sending any GraphBackend, prefill: sending (any GraphBackend)? = nil, graph: Manifest.Graph,
        vision: sending any GraphBackend, visionGraph: Manifest.Graph, image: Manifest.ImageBlock,
        preprocess: Manifest.ImagePreprocess, device: Device, options: Options = Options()
    ) async throws(AetherError) {
        try await self.init(
            backends: Backends(step: step, prefill: prefill, tower: vision), graph: graph,
            image: ImageConfiguration(vision: visionGraph, image: image, preprocess: preprocess), device: device,
            options: options, metal: nil, modelBytes: 0, probe: .system, baseline: nil)
    }

    /// An engine over any ``GraphBackend``s with its memory read on `probe` and `modelBytes` as what its load measured
    /// (tests: a hand-moved level, and a load's charge without a bundle). A `vision` backend makes it a
    /// `vlm-static-inject@1` engine, with `visionGraph`, `image` and `preprocess`.
    package init(
        step: sending any GraphBackend, prefill: sending (any GraphBackend)? = nil, graph: Manifest.Graph,
        vision: sending (any GraphBackend)? = nil, visionGraph: Manifest.Graph? = nil, image: Manifest.ImageBlock? = nil,
        preprocess: Manifest.ImagePreprocess? = nil, device: Device, options: Options = Options(), probe: GPUMemoryProbe,
        modelBytes: Int64 = 0
    ) async throws(AetherError) {
        var configuration: ImageConfiguration?
        if let visionGraph, let image, let preprocess {
            configuration = ImageConfiguration(vision: visionGraph, image: image, preprocess: preprocess)
        }
        try await self.init(
            backends: Backends(step: step, prefill: prefill, tower: vision), graph: graph, image: configuration,
            device: device, options: options, metal: nil, modelBytes: modelBytes, probe: probe, baseline: nil)
    }

    /// An engine over any ``GraphBackend`` pair that reloads through `reloader` and reads its level on `levelProbe`
    /// (tests: a fake reload budget).
    init(
        step: sending any GraphBackend, prefill: sending (any GraphBackend)? = nil, graph: Manifest.Graph,
        device: Device, options: Options = Options(),
        reloader: sending @escaping () async throws(AetherError) -> (step: any GraphBackend, prefill: (any GraphBackend)?),
        levelProbe: @escaping @Sendable () -> Int64
    ) async throws(AetherError) {
        let reload: Reloader = { () async throws(AetherError) -> ReloadedGraphs in
            let graphs = try await reloader()
            return ReloadedGraphs(step: graphs.step, prefill: graphs.prefill)
        }
        try await self.init(backends: Backends(step: step, prefill: prefill, reloader: reload), graph: graph, image: nil,
                            device: device, options: options, metal: nil, modelBytes: 0, probe: .system, baseline: nil,
                            levelProbe: levelProbe)
    }

    /// The step, prefill and tower backends, sent into the actor as one region.
    private struct Backends {
        var step: any GraphBackend
        var prefill: (any GraphBackend)?
        var tower: (any GraphBackend)?
        var reloader: Reloader?
    }

    /// A VLM's manifest blocks for its tower and statics.
    struct ImageConfiguration: Sendable {
        var vision: Manifest.Graph
        var image: Manifest.ImageBlock
        var preprocess: Manifest.ImagePreprocess
    }

    private init(
        backends: sending Backends, graph: Manifest.Graph, image: ImageConfiguration?, device: Device, options: Options,
        metal: (any MTLDevice)?, modelBytes: Int64, probe: GPUMemoryProbe, baseline: Int64?,
        levelProbe: @escaping @Sendable () -> Int64 = DecoderEngine.metalAllocatedBytes
    ) async throws(AetherError) {
        // The level before the graphs loaded: the bundle init reads it before its loads; otherwise it is now.
        let memoryBaseline = baseline ?? probe.allocatedBytes()
        let step = backends.step, prefill = backends.prefill
        // DecoderCore keeps 8 token slots and needs ring > depth + 1; no option may reach a trap.
        guard (1...(DecoderCore.ring - 2)).contains(options.pipelineDepth) else {
            throw .unsupportedOperation("pipelineDepth \(options.pipelineDepth); supported: 1...\(DecoderCore.ring - 2)")
        }
        let prefillWidth = try Self.prefillWidth(requested: options.prefillWidth, step: step)
        let encodable = metal != nil && step is Graph && (prefill == nil || prefill is Graph)
        let arena = encodable ? TensorArena(device: metal!) : TensorArena()
        let kernels = encodable ? try await GreedyKernels(device: metal!) : nil
        let limits = graph.limits ?? Manifest.Limits()
        guard let maxContext = limits.maxContext else {
            throw .manifestInvalid(field: "graphs.decoder.limits.maxContext", reason: "required by decoder-lm@1")
        }
        guard options.reservedSnapshots >= 0 else {
            throw .generation(.invalidParameters("reservedSnapshots \(options.reservedSnapshots); at least 0"))
        }
        var binder: ImageBinder?
        if let image {
            guard let tower = backends.tower else {
                throw .manifestInvalid(field: "graphs.vision", reason: "vlm-static-inject@1 needs a vision tower")
            }
            binder = try ImageBinder(tower: tower, vision: image.vision, decoder: graph, decoderSignature: step.signature,
                                     image: image.image, preprocess: image.preprocess)
        }
        var sizing = try Self.sizing(
            step.signature, prefill: prefill?.signature, graph: graph, maxContext: maxContext, platform: device.platform,
            options: options)
        var declared: Int64?
        if case .declared(let bytes) = options.modelCharge {
            guard bytes >= 0 else { throw .generation(.invalidParameters("modelCharge declared \(bytes) bytes; at least 0")) }
            declared = bytes
        }
        // The larger of what load measured (the files, or the Metal delta across the loads) and the declared figure.
        sizing.modelBytes = max(modelBytes, declared ?? 0)
        // The static inputs (image rows, rotary inputs) and the tower's buffers.
        for (_, binding) in graph.inputs(forPhase: "step") where binding.isStatic && binding.source == nil {
            if let spec = step.signature.input(binding.name), let width = spec.scalarType.byteWidth {
                sizing.fixedBytes += Int64(spec.shape.map { max($0, 1) }.reduce(1, *) * width)
            }
        }
        sizing.fixedBytes += binder?.bufferBytes ?? 0
        self.sizing = sizing
        let budget = options.memoryBudget.bytes(on: device)
        let maxCapacity = try sizing.capacity(budget: budget, requested: options.capacity)
        // Grow on demand when the growth copy fits, else allocate the whole capacity upfront.
        let initial = sizing.initialAllocation(capacity: maxCapacity, budget: budget)
        guard initial <= maxCapacity else {
            throw .capacity(what: "KV initial capacity (tokens)", needed: Int64(initial), available: Int64(maxCapacity))
        }
        let core = try DecoderCore(
            step: step, prefill: prefill, graph: graph, arena: arena, kernels: kernels, capacityLimit: maxCapacity,
            initialCapacity: initial, prefillWidth: prefillWidth, racingHandoff: options.racingTokenHandoff,
            staleHandoff: options.staleTokenHandoff)
        self.core = core
        reloadBudgetBytes = try Self.reloadBudget(
            requested: options.reloadBudgetBytes, canReload: backends.reloader != nil, step: step,
            hasRecurrentState: core.hasRecurrentState)
        reloader = backends.reloader
        self.levelProbe = levelProbe
        declaredModelBytes = declared
        self.probe = probe
        self.memoryBaseline = memoryBaseline
        lastLevel = memoryBaseline
        peakLevel = memoryBaseline
        imageBinder = binder
        imageInput = binder?.spec
        if let binder { try core.withStatics { statics throws(AetherError) in try binder.clear(statics) } }
        description = Description(
            vocab: core.vocab, maxCapacity: core.maxCapacity, hasRecurrentState: core.hasRecurrentState,
            prefillChunk: core.shape.prefillChunk, schedule: core.shape, encodes: core.isEncoding)
        decodeMode = core.isEncoding ? .gpu(depth: 1) : .host
        guard options.canarySteps > 0, options.canaryCapacity > 0 else {
            throw .generation(.invalidParameters("the canary needs steps > 0 and capacity > 0"))
        }
        if core.isEncoding, options.pipelineDepth > 1, options.canary {
            pendingCanary = (options.pipelineDepth, options.canarySteps, options.canaryDelay, options.canaryCapacity)
            if options.canaryAtLoad { try await runPendingCanary() }
        } else if core.isEncoding, options.pipelineDepth > 1, options.uncheckedPipelineDepth {
            // Without the canary, depth one unless a diagnostic asks otherwise.
            decodeMode = .gpu(depth: options.pipelineDepth)
        }
        // After load (and the canary, when it ran at load).
        sampleMemory()
    }

    /// Runs the pending canary, keeping the state: the position, every state and the bound images are as they were
    /// after it. Sets ``canary`` and, when it passes, ``decodeMode``. A canary that throws stays pending.
    private func runPendingCanary() async throws(AetherError) {
        guard let pending = pendingCanary else { return }
        let report = try await core.runCanary(
            depth: pending.depth, steps: pending.steps, delay: pending.delay, capacity: pending.capacity)
        pendingCanary = nil
        canary = report
        if report.passed { decodeMode = .gpu(depth: pending.depth) }
    }

    /// The widest dynamic call a Core AI decoder specialized with `expectFrequentReshapes` is fed by default
    /// (``Options/prefillWidth``).
    ///
    /// Core AI retains Metal allocations, until the model is released, on every call wider than this and on every
    /// decode step after one, even when each shape repeats one it already ran (the same bug 0.0.x worked around with
    /// its reload budget). Measured on the 4B Qwen3-VL export (M5 Max, 2026-11-10; text-only prompts alike, so not the
    /// image path): a 513-token prompt fed in calls of 40, 48, 64, 128, 256 or 512 tokens retained 123–180 MiB in
    /// its prefill and 255–263 MiB over 60 decode steps, every time it was repeated; in calls of 8, 16, 32 or 33 it
    /// retained nothing once its shapes had run once. The cost is the prefill's speed: 0.40 s instead of 0.11 s for
    /// 513 tokens, 1.1 s instead of 0.33 s for 1,390.
    ///
    /// It does not bound retention on its own: a narrow call whose shape (tokens, positions) has not run for a while
    /// can still grow again, about 11–26 MiB for one of 32 tokens, the more the longer its context. Over G5's 20 mixed
    /// requests the growth after warm-up fell from 5,577 to 1,139 MiB (bar 1,024), nearly all of it from two of the
    /// eight cases, each growing again by the same amount every time it came round. The reload budget releases that
    /// (``Options/reloadBudgetBytes``, ``reloadIfRetainedPastBudget()``).
    public static let reshapeSafePrefillWidth = 32

    /// The prefill width an engine over `step` uses: `requested` when given (at least 1), else
    /// ``reshapeSafePrefillWidth`` when `step` is a Core AI graph specialized with `expectFrequentReshapes`, else nil
    /// (the manifest's chunk).
    static func prefillWidth(requested: Int?, step: any GraphBackend) throws(AetherError) -> Int? {
        if let requested {
            guard requested >= 1 else { throw .generation(.invalidParameters("prefillWidth \(requested); at least 1")) }
            return requested
        }
        guard let graph = step as? Graph, graph.options.expectFrequentReshapes else { return nil }
        return reshapeSafePrefillWidth
    }

    // MARK: - Reload budget (docs/VLM.md, V6)

    /// The default ``Options/reloadBudgetBytes``: 2 GiB, as 0.0.x's.
    public static let defaultReloadBudgetBytes: Int64 = 2 << 30

    /// The engine's reloads since load.
    public struct Reloads: Sendable, Hashable {
        public var count = 0
        /// Wall time of the last reload and of all of them: dropping the graphs, loading them again (through the
        /// compile cache) and one warming step.
        public var lastSeconds = 0.0
        public var totalSeconds = 0.0
        /// The retained bytes that decided the last reload.
        public var lastRetainedBytes: Int64 = 0
    }

    /// The decoder's graphs, loaded again from the bundle (a reload).
    struct ReloadedGraphs {
        var step: any GraphBackend
        var prefill: (any GraphBackend)?
    }

    typealias Reloader = () async throws(AetherError) -> ReloadedGraphs

    /// The budget this engine reloads at: ``Options/reloadBudgetBytes`` resolved (nil: it never reloads).
    public nonisolated let reloadBudgetBytes: Int64?

    /// Reloads the decoder when the model retains more than ``reloadBudgetBytes`` past its level at the first call.
    /// Returns whether it reloaded.
    ///
    /// Core AI retains Metal on a reshaping decoder's calls until the model is released (``reshapeSafePrefillWidth``);
    /// a reload releases it. The retained bytes are the device's Metal allocation less the engine's own buffers,
    /// above that level at the first call (so the first request's working set is the reference, and anything else
    /// the process allocates meanwhile counts too). Call it between requests: the ``Generator`` does at the end of
    /// each, inside its turn and GPU grant, so the next request waits for the reload rather than racing it.
    ///
    /// The reload drops the decoder's graphs, loads them again from the bundle and runs one warming step, so the
    /// weights are back before the next request. The KV state, position, images and statics stay: the next request
    /// still reuses its prefix. The vision tower stays loaded (one static shape, nothing retained). Refused while a
    /// feed, decode or bind is in progress; a failed load leaves the graphs unloaded, and the next feed loads them.
    @discardableResult
    public func reloadIfRetainedPastBudget() async throws(AetherError) -> Bool {
        guard let budget = reloadBudgetBytes, reloader != nil, !reloading else { return false }
        let level = levelProbe() - ownedBufferBytes
        let floor = retentionFloor ?? level
        retentionFloor = floor
        let retained = level - floor
        guard retained > budget || !core.graphsLoaded else { return false }
        try await reload(retained: max(0, retained))
        return true
    }

    /// The engine's own buffers: its arena and the tower's.
    private var ownedBufferBytes: Int64 { Int64(core.arena.byteCount + (imageBinder?.arena.byteCount ?? 0)) }

    private func reload(retained: Int64) async throws(AetherError) {
        guard let reloader else { throw .unsupportedOperation("this engine cannot reload its graphs") }
        guard !reloading else { throw .unsupportedOperation("a reload while one is in progress") }
        reloading = true
        defer { reloading = false }
        let clock = ContinuousClock(), began = clock.now
        try core.unloadGraphs()
        let graphs = try await reloader()
        try core.installGraphs(step: graphs.step, prefill: graphs.prefill)
        try? await core.warmAfterReload()
        let seconds = (clock.now - began) / .seconds(1)
        reloads.count += 1
        reloads.lastSeconds = seconds
        reloads.totalSeconds += seconds
        reloads.lastRetainedBytes = retained
    }

    /// Before a feed or decode: loads the graphs a failed reload left unloaded.
    private func ensureGraphs() async throws(AetherError) {
        if !core.graphsLoaded, !reloading { try await reload(retained: 0) }
    }

    /// The budget an engine uses: `requested` (0: none), else the default for a reloadable Core AI decoder specialized
    /// with `expectFrequentReshapes` and without recurrent state; none when the engine cannot reload.
    static func reloadBudget(
        requested: Int64?, canReload: Bool, step: any GraphBackend, hasRecurrentState: Bool
    ) throws(AetherError) -> Int64? {
        if let requested, requested < 0 {
            throw .generation(.invalidParameters("reloadBudgetBytes \(requested); at least 0"))
        }
        guard canReload, !hasRecurrentState else { return nil }
        if let requested { return requested == 0 ? nil : requested }
        guard let graph = step as? Graph, graph.options.expectFrequentReshapes else { return nil }
        return defaultReloadBudgetBytes
    }

    /// The device's current Metal allocations (diagnostics: Core AI's shape-change retention shows up here).
    public nonisolated static func metalAllocatedBytes() -> Int64 {
        Int64(MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0)
    }

    /// Sizes KV from the graph signatures: bytes per sequence slot over every sequence state; the owned bytes that do
    /// not scale (recurrent states, the step's and the prefill phase's logits buffers, the greedy ring, token
    /// buffers); and what one host snapshot costs, reserved `options.reservedSnapshots` times on a model with
    /// recurrent state (attention-only models rewind instead).
    static func sizing(
        _ signature: GraphSignature, prefill: GraphSignature?, graph: Manifest.Graph, maxContext: Int,
        platform: Device.Platform, options: Options
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
                if spec.shape[entry.seqAxis] != TensorSpec.dynamic { staticAxis = min(staticAxis ?? .max, spec.shape[entry.seqAxis]) }
            }
        }
        var recurrent: Int64 = 0
        for entry in states.recurrent {
            if let spec = signature.state(entry.name), let width = spec.scalarType.byteWidth {
                recurrent += Int64(spec.shape.map { max($0, 1) }.reduce(1, *) * width)
            }
        }
        // The logits buffers DecoderCore allocates: the step's for its largest call, one row per greedy ring slot, and
        // the prefill phase's own for its static chunk.
        var logitsBytes: Int64 = 0
        var vocab: Int64 = 0
        let limits = graph.limits ?? Manifest.Limits()
        if let logits = graph.outputs["logits"], let spec = signature.output(logits.name), spec.rank == 3,
           let v = spec.shape.last, let width = spec.scalarType.byteWidth {
            vocab = Int64(v)
            let tokens = graph.inputs(forPhase: "step")["tokens"].flatMap { signature.input($0.name) }
            let stepTokens = tokens.flatMap { $0.rank == 2 && $0.shape[1] != TensorSpec.dynamic ? $0.shape[1] : nil }
            let staticRows = spec.shape[1] == TensorSpec.dynamic ? nil : spec.shape[1]
            let stepRows = staticRows ?? stepTokens ?? max(1, limits.prefillChunk ?? 512)
            var rows = stepRows + DecoderCore.ring * (staticRows ?? 1)
            if let prefill, let p = prefill.output(logits.name), p.rank == 3,
               let chunk = graph.inputs(forPhase: "prefill")["tokens"].flatMap({ prefill.input($0.name) }), chunk.rank == 2 {
                rows += p.shape[1] == TensorSpec.dynamic ? chunk.shape[1] : p.shape[1]
            }
            logitsBytes = Int64(rows) * vocab * Int64(width)
        }
        return KVSizing(
            maxContext: maxContext, platformCap: limits.kvCapacityCap?[platform.rawValue], maxPosition: limits.maxPosition,
            // + 8: the int32 token history and positions buffers, sized with the capacity.
            bytesPerToken: perToken + 8, fixedBytes: recurrent + logitsBytes, staticAxis: staticAxis,
            initialCapacity: options.initialCapacity,
            // A snapshot: the sequence states up to its position, the recurrent states, and the float logits row a
            // client keeps with it (the Generator does, to regenerate).
            snapshotBytesPerToken: perToken, snapshotFixedBytes: recurrent + 4 * vocab,
            reservedSnapshots: states.recurrent.isEmpty ? 0 : options.reservedSnapshots)
    }

    // MARK: - State

    public var position: Int { core.position }
    /// The current allocation of the sequence axis.
    public var capacity: Int { core.capacity }
    /// True when pipelined steps ran recurrent state past ``position``: restore a snapshot or reset first.
    public var stateRanAhead: Bool { core.stateAhead }

    /// Feeds `tokens` at the current position on the prefill schedule (§6.2) and returns the rows asked for.
    public func prefill(_ tokens: [Int32], logits: LogitsSelection = .last) async throws(AetherError) -> Logits? {
        defer { sampleMemory() }
        try await ensureGraphs()
        return try await core.feed(tokens, select: logits)
    }

    /// Feeds one token and returns the next-token logits (the depth-1 host path for CPU sampling).
    public func step(_ token: Int32) async throws(AetherError) -> Logits {
        defer { sampleMemory() }
        try await ensureGraphs()
        return try await core.feed([token], select: .last)!
    }

    /// Moves the position back. O(1) on attention-only models (stale slots are overwritten later). With recurrent
    /// states only `rewind(to: 0)` (a reset) is possible; use ``snapshot()`` and ``restore(_:)``.
    public func rewind(to position: Int) throws(AetherError) {
        try core.rewind(to: position)
        if let imageBinder { try core.withStatics { statics throws(AetherError) in try imageBinder.truncate(to: position, statics) } }
    }

    /// Moves the position back to 0 and clears every image the static inputs hold.
    public func reset() throws(AetherError) {
        try core.reset()
        if let imageBinder { try core.withStatics { statics throws(AetherError) in try imageBinder.clear(statics) } }
    }

    /// A host copy of the state: sequence states up to the position, recurrent states whole.
    public func snapshot() throws(AetherError) -> EngineSnapshot {
        var snapshot = try core.snapshot()
        snapshot.images = boundImages
        return EngineSnapshot(position: snapshot.position, byteCount: snapshot.byteCount, payload: snapshot)
    }

    /// Restores a snapshot of this engine's model, validated before any state is touched. Every image that starts
    /// before the snapshot's position must be bound as it was when the snapshot was taken (the same picture at the same
    /// place and slots): the state up to there depends on those images alone, so the images bound after it may differ
    /// (a prompt that shares the snapshot's prefix and goes on with other pictures binds them first).
    public func restore(_ snapshot: EngineSnapshot) throws(AetherError) {
        guard let state = snapshot.payload as? Snapshot else {
            throw .unsupportedOperation("restore of a snapshot another engine type took (\(type(of: snapshot.payload)))")
        }
        let fed = { (images: [ImagePromptPlan.Image]) in images.filter { $0.start < state.position } }
        guard fed(state.images) == fed(boundImages) else {
            throw .unsupportedOperation("restore of a snapshot taken with other images bound before its position; bind its images first")
        }
        try core.restore(state)
    }

    // MARK: - Greedy decoding

    /// Starts greedy decoding after `first` (the caller's pick from the prompt's logits, not yet fed), for at most
    /// `limit` more tokens, in ``decodeMode``. Then call ``nextGreedy()`` until it returns nil or the caller stops,
    /// and always ``endGreedy()``. The first call on an engine whose canary is pending runs it first (``canary``),
    /// keeping the state, so the decode continues the prompt in the mode the canary qualified.
    public func beginGreedy(first: Int32, limit: Int) async throws(AetherError) {
        if pendingCanary != nil {
            defer { sampleMemory() }
            try await runPendingCanary()
        }
        let depth = switch decodeMode {
        case .host: 0
        case .gpu(let depth): depth
        }
        try core.beginGreedy(first: first, limit: limit, depth: depth)
    }

    /// The next token: delivered by the GPU argmax's completion handler, with the pipeline refilled before this
    /// returns. Nil at the limit or when the calling task is cancelled. A step whose command buffer failed throws
    /// `runtime` (Metal's error), and a row with no token `invalidLogits`; the state is then undefined past the
    /// tokens returned: end the decode, then ``reset()`` or ``restore(_:)`` (the Generator does).
    public func nextGreedy() async throws(AetherError) -> Int32? { try await core.nextGreedy() }

    /// Ends greedy decoding: waits for in-flight GPU work and sets the position to the tokens fed (every token
    /// returned except the last). Returns how many were fed and how many steps ran past the last returned token.
    /// Refused (`unsupportedOperation`) while a ``nextGreedy()`` is in progress. A GPU failure seen while draining
    /// is thrown after the decode has ended.
    @discardableResult
    public func endGreedy() async throws(AetherError) -> (fed: Int, overshoot: Int) {
        defer { sampleMemory() }
        let end = try await core.endGreedy()
        return (end.fed, end.overshoot)
    }

    /// Tests only: make every command buffer the engine commits from now on report a GPU failure (no-op on `run`).
    func injectGPUFailure(_ on: Bool) { core.kernels?.injectedFailure = on }
}

/// The `decoder-lm@1` driver (DESIGN.md §6.3): loads an installed bundle into a ``DecoderEngine``.
public enum DecoderLMDriver: ModelDriver {
    public static let id: DriverID = .decoderLM

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> DecoderEngine {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        return try await DecoderEngine(bundle: bundle, manifest: manifest, device: options.device, options: options.decoder)
    }
}
