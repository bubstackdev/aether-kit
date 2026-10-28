import CoreAI
import Foundation
import AetherCoreAI
import AetherKit
// The backend-neutral language types (`LanguageBackend`, sampling, `LanguageGenerationError`) live in the
// Core-AI-free runtime target; importing this module keeps them in scope for existing consumers.
@_exported import AetherLanguageRuntime
#if os(iOS)
import os
#endif
import Metal

/// Sequential correctness reference. Core AI's unannotated stream stays inside
/// this actor. Admission rejects overlap, and every submission is joined before
/// buffers are read, reset, or released, including after task cancellation.
public actor LanguageGenerationSession {
    public struct Reply: Sendable {
        public let tokens: [Int]
        public let timeToFirstTokenSeconds: Double
        public let totalSeconds: Double
        public let stopReason: String
    }
    public enum NativeEvent: Sendable, Equatable {
        case submitted(position: Int, count: Int)
        case completed(position: Int, count: Int)
    }
    public typealias Observer = @Sendable (NativeEvent) -> Void
    public enum State: String, Sendable { case ready, running, closing, closed }

    private var model: AIModel?
    private var function: InferenceFunction?
    private var stream: SerialComputeStream?
    private var keys: (any MTLBuffer)?
    private var values: (any MTLBuffer)?
    private let keyTemplate: NDArrayDescriptor
    private let valueTemplate: NDArrayDescriptor
    private var keyDescriptor: NDArrayDescriptor
    private var valueDescriptor: NDArrayDescriptor
    /// Rows of the logits head (``LanguageBackend``).
    public nonisolated let vocabularySize: Int
    /// The registered end-of-sequence IDs (``LanguageBackend``).
    public nonisolated let stopTokenIDs: Set<Int>
    private struct ReopenConfiguration: Sendable {
        let url: URL
        let compute: ModelComputePolicy
        let expectFrequentReshapes: Bool
        /// nil: never recycle (bounded-shape contracts).
        let maximumSubmissions: Int?
    }
    private var reopenConfiguration: ReopenConfiguration?
    private var submissionsSinceReload = 0
    private var modelReloadCount = 0
    private var memoryBudget: LanguageMemoryBudget
    /// Functions and shapes submitted since the model was (re)loaded: a request that adds one is warming.
    private struct WarmShape: Hashable { let image: Bool; let count: Int; let capacity: Int }
    private var warmShapes: Set<WarmShape> = []
    /// The current request's memory window: Metal's allocated size and the KV cache bytes at admission,
    /// then whether it warmed a function or shape and how many tokens it submitted.
    private var memoryWindow: (allocated: Int, cacheBytes: Int, warming: Bool, tokens: Int)?
    /// The default memory budget: reload once warm requests have retained 2 GiB of Metal
    /// allocations since the model loaded.
    public static let defaultMemoryGrowthBudgetBytes = 2 << 30
    /// The smallest budget `open` accepts (256 MiB): a writer request retains about 137 MiB, so a
    /// smaller budget would reload before nearly every request.
    public static let minimumMemoryGrowthBudgetBytes = LanguageMemoryBudget.minimumLimitBytes
    /// The asset's KV addressing, detected from the `main` function signature.
    public nonisolated let contract: LanguageStepContract
    private struct StepShape: Hashable { let count: Int; let capacity: Int }
    private var stepShapes: Set<StepShape> = []
    private let queue: any MTLCommandQueue
    private let device: any MTLDevice
    private var allocatedCapacity: Int
    private var materializedTokens = 0
    private var growthCount = 0
    private let capacity: Int
    /// The registered context: prompt plus output (``LanguageBackend``).
    public nonisolated var maximumContextLength: Int { capacity }
    private let initialCapacity: Int
    /// Largest prefill submission (the declared export bound), nil when undeclared.
    private let prefillLimit: Int?
    /// Largest image submission (the declared export bound), nil when undeclared.
    private let imageLimit: Int?
    /// Qwen3-VL decoders take 3-axis rotary positions (`rope_positions`) beside the cache positions.
    public nonisolated let takesRopePositions: Bool
    /// The asset takes images, through `main` itself or a 0.0.2 `withImage` entry point.
    public nonisolated let acceptsImages: Bool
    /// Image steps go through `main` (Qwen3-VL's one-function decoder, 0.0.3): one copy of the weights is
    /// wired instead of one per function. False for text-only assets and 0.0.2 two-function assets.
    public nonisolated let imagesThroughMain: Bool
    /// The 0.0.2 image entry point, loaded only for two-function assets.
    private var imageFunction: InferenceFunction?
    /// A one-function asset's image row layout: `image_embeds` [q, hiddenSize] and `deepstack`
    /// [levels, q, hiddenSize] on every step.
    private struct ImageRowLayout: Sendable { let hiddenSize: Int; let levels: Int }
    private let imageRows: ImageRowLayout?
    /// Zero image rows for text steps of a one-function asset, by token count (read-only, reused).
    private var zeroImageRows: [Int: (rows: NDArray, deepstack: NDArray)] = [:]
    /// The current request's prompt rotary positions (rows: time, height, width); tokens past the
    /// prompt take `position + ropeDelta` on every axis.
    private var promptRope: [[Int32]] = []
    private var ropeDelta = 0
    private let lifecycle = LanguageSessionLifecycle()
    private let gpu: GPUAdmissionOwner?
    /// The model's estimates under a host's ``LanguageResourceAdmission``, and its residency reservation while loaded.
    private var resourceBudget: LanguageModelBudget?
    private var residency: ResourceReservation?
    private let admission = LanguageRequestAdmission()

    /// For an owning session's close (`VisionLanguageSession.unload()`): withdraws a request
    /// waiting for GPU admission and refuses later requests, so the owner never waits behind the
    /// GPU queue before it unloads this session.
    package func withdrawPendingAdmission() { admission.withdraw() }
    public var state: State { lifecycle.state }

    /// Opens a registered local model through Aether Kit. No download or fallback.
    /// `gpu` opts into a host-shared ``GPUAdmission``: opening (the model's load and its functions' specialization)
    /// waits for a grant, and each request waits for (or, in fail-fast mode, requires) one after validation and holds
    /// it through a memory-budget reload and until its native submissions have joined.
    /// `memoryGrowthBudgetBytes` bounds the native memory requests retain (see the raw-asset entry point).
    /// `resources` opts into reservations against a host's ``ResourceCoordinator`` (``LanguageResourceAdmission``):
    /// the model's residency while it is loaded and each request's scratch.
    public static func open(bundle directory: URL, platform: LanguageModelManifest.Platform,
                            memoryGrowthBudgetBytes: Int? = defaultMemoryGrowthBudgetBytes,
                            gpu: GPUAdmissionOwner? = nil, resources: LanguageResourceAdmission? = nil) async throws
        -> LanguageGenerationSession {
        let manifest = try LanguageModelManifest.load(from: directory)
        let modelURL = try manifest.verify(in: directory, platform: platform)
        let budget = try resources?.budget(modelSHA256: manifest.modelSHA256, contextLength: manifest.maximumContextLength)
        return try await open(at: modelURL, maximumContextLength: manifest.maximumContextLength,
            stopTokenIDs: Set(manifest.stopTokenIDs), memoryGrowthBudgetBytes: memoryGrowthBudgetBytes,
            exportBounds: manifest.exportBounds, gpu: gpu, budget: budget)
    }

    /// Explicit raw-asset entry point for qualification before bundle registration.
    /// `maximumSubmissionsPerModel` nil selects the contract default: recycle every 128
    /// submissions for `runningPositions`, never for `bucketedPositions`. `exportBounds` are the
    /// asset's exported shape maxima: required for a bucketed asset, optional for a running one,
    /// and enforced (context and prefill chunks) when given.
    ///
    /// `memoryGrowthBudgetBytes`: Core AI keeps Metal allocations on every input-shape change until
    /// the model is released (reshape-enabled specialization), so warm requests keep adding memory.
    /// Once the growth retained since the model loaded passes this budget, the session releases and
    /// reloads the model between requests: at idle, under a GPU grant, right after the request that
    /// spent it, or else in front of the next request (KV state is Kit-owned and unaffected).
    /// Reloads are counted in ``CacheStatistics``. nil never reloads for memory; a budget below
    /// ``minimumMemoryGrowthBudgetBytes`` is refused.
    public static func open(at modelURL: URL, maximumContextLength: Int, initialCacheCapacity: Int = 256,
                            stopTokenIDs: Set<Int> = [151645], compute: ModelComputePolicy = .preferGPU,
                            expectFrequentReshapes: Bool = true, maximumSubmissionsPerModel: Int? = nil,
                            memoryGrowthBudgetBytes: Int? = defaultMemoryGrowthBudgetBytes,
                            exportBounds: LanguageExportBounds? = nil,
                            gpu: GPUAdmissionOwner? = nil) async throws -> LanguageGenerationSession {
        try await open(at: modelURL, maximumContextLength: maximumContextLength, initialCacheCapacity: initialCacheCapacity,
                       stopTokenIDs: stopTokenIDs, compute: compute, expectFrequentReshapes: expectFrequentReshapes,
                       maximumSubmissionsPerModel: maximumSubmissionsPerModel, memoryGrowthBudgetBytes: memoryGrowthBudgetBytes,
                       exportBounds: exportBounds, gpu: gpu, budget: nil)
    }

    /// The raw-asset opening with an optional resource budget (the bundle entry points look it up by the registered
    /// model hash). With `gpu`, the load and its specialization run under a grant; the residency is reserved after the
    /// grant, before the load, and released if the load fails.
    static func open(at modelURL: URL, maximumContextLength: Int, initialCacheCapacity: Int = 256,
                     stopTokenIDs: Set<Int> = [151645], compute: ModelComputePolicy = .preferGPU,
                     expectFrequentReshapes: Bool = true, maximumSubmissionsPerModel: Int? = nil,
                     memoryGrowthBudgetBytes: Int? = defaultMemoryGrowthBudgetBytes,
                     exportBounds: LanguageExportBounds? = nil,
                     gpu: GPUAdmissionOwner? = nil, budget: LanguageModelBudget?) async throws -> LanguageGenerationSession {
        try Task.checkCancellation()
        guard (maximumSubmissionsPerModel ?? 1) > 0, LanguageMemoryBudget.admits(limitBytes: memoryGrowthBudgetBytes)
        else { throw LanguageGenerationError.invalid("Invalid model residency policy") }
        var options: SpecializationOptions
        switch compute {
        case .automatic: options = .default
        case .cpuOnly: options = .cpuOnly
        case .preferGPU: options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        }
        options.expectFrequentReshapes = expectFrequentReshapes
        let (session, residency) = try await withOpenAdmission(gpu) { () async throws -> (LanguageGenerationSession, ResourceReservation?) in
            let residency = try budget?.reserveResidency()
            do {
                let model = try await AIModel(contentsOf: modelURL, options: options)
                try Task.checkCancellation()
                let session = try LanguageGenerationSession(model: model, capacity: maximumContextLength,
                    initialCacheCapacity: initialCacheCapacity, stopTokenIDs: stopTokenIDs, exportBounds: exportBounds, gpu: gpu)
                return (session, residency)
            } catch {
                residency?.release()
                throw error
            }
        }
        await session.configureReopening(ReopenConfiguration(url: modelURL, compute: compute,
            expectFrequentReshapes: expectFrequentReshapes,
            maximumSubmissions: maximumSubmissionsPerModel ?? session.contract.defaultMaximumSubmissionsPerModel),
            memoryGrowthBudgetBytes: memoryGrowthBudgetBytes, budget: budget, residency: residency)
        return session
    }
    public static var deviceArchitectureName: String { AIModel.deviceArchitectureName }

    /// Wraps an already loaded model. It cannot be reloaded, so it has no memory budget.
    public init(model: AIModel, capacity: Int = 512, initialCacheCapacity: Int = 256, stopTokenIDs: Set<Int> = [151645],
                exportBounds: LanguageExportBounds? = nil, gpu: GPUAdmissionOwner? = nil) throws {
        guard capacity > 0, capacity <= 16384, initialCacheCapacity > 0,
              let d = model.functionDescriptor(for: "main"),
              case .ndArray(let logits)? = d.outputDescriptor(of: "logits"),
              let contract = LanguageStepContract.detect(inputNames: Set(d.inputNames), outputNames: Set(d.outputNames),
                                                         stateNames: Set(d.stateNames), logitsShape: logits.shape),
              case .ndArray(let key)? = contract.cachesAreStates ? d.stateDescriptor(of: "keyCache") : d.inputDescriptor(of: "keyCache"),
              case .ndArray(let value)? = contract.cachesAreStates ? d.stateDescriptor(of: "valueCache") : d.inputDescriptor(of: "valueCache"),
              key.shape.count == 5, key.shape[3] == -1, value.shape == key.shape,
              key.shape.enumerated().allSatisfy({ $0.offset == 3 || $0.element > 0 }),
              key.scalarType == .float16, value.scalarType == .float16,
              logits.scalarType == .float16, !stopTokenIDs.isEmpty,
              stopTokenIDs.allSatisfy({ $0 >= 0 && $0 < logits.shape[2] }),
              let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue()
        else { throw LanguageGenerationError.invalid("Unsupported language probe contract") }
        for name in ["input_ids", contract.positionInputName] {
            guard case .ndArray(let input)? = d.inputDescriptor(of: name),
                  input.scalarType == .int32, input.shape == [1, -1]
            else { throw LanguageGenerationError.invalid("Unsupported token/position contract") }
        }
        if !contract.cachesAreStates {
            // Update rows: the cache layout with the token axis sized by the submission.
            for name in ["keyUpdate", "valueUpdate"] {
                guard case .ndArray(let update)? = d.outputDescriptor(of: name), update.scalarType == .float16,
                      update.shape.count == 5, update.shape[3] == -1,
                      update.shape.enumerated().allSatisfy({ $0.offset == 3 || $0.element == key.shape[$0.offset] })
                else { throw LanguageGenerationError.invalid("Unsupported cache update contract") }
            }
        }
        let prefillLimit = try contract.admit(bounds: exportBounds, maximumContextLength: capacity)
        let takesRope = LanguageStepContract.takesRopePositions(Set(d.inputNames))
        if takesRope {
            guard case .ndArray(let rope)? = d.inputDescriptor(of: "rope_positions"), rope.scalarType == .int32, rope.shape == [3, -1]
            else { throw LanguageGenerationError.invalid("Unsupported rotary position contract") }
        }
        var imageRows: ImageRowLayout?
        if LanguageStepContract.takesImageRows(Set(d.inputNames)) {
            // One function for every step: image rows aligned with the tokens, zero on text steps.
            guard takesRope, model.functionDescriptor(for: "withImage") == nil,
                  case .ndArray(let rows)? = d.inputDescriptor(of: "image_embeds"), rows.scalarType == .float16,
                  rows.shape.count == 2, rows.shape[0] == -1, rows.shape[1] > 0,
                  case .ndArray(let deep)? = d.inputDescriptor(of: "deepstack"), deep.scalarType == .float16,
                  deep.shape.count == 3, deep.shape[0] > 0, deep.shape[1] == -1, deep.shape[2] == rows.shape[1]
            else { throw LanguageGenerationError.invalid("Unsupported image row contract") }
            imageRows = ImageRowLayout(hiddenSize: rows.shape[1], levels: deep.shape[0])
        }
        var imageFunction: InferenceFunction?
        if imageRows == nil, let image = model.functionDescriptor(for: "withImage") {
            guard takesRope, LanguageStepContract.isImageStep(inputNames: Set(image.inputNames), outputNames: Set(image.outputNames),
                                                              stateNames: Set(image.stateNames)),
                  let loaded = try model.loadFunction(named: "withImage")
            else { throw LanguageGenerationError.invalid("Unsupported image step contract") }
            imageFunction = loaded
        }
        var shape = key.shape
        let initial = min(initialCacheCapacity, capacity)
        shape[3] = initial
        let kd = key.resolvingDynamicDimensions(shape)
        let vd = value.resolvingDynamicDimensions(shape)
        guard let kb = device.makeBuffer(length: kd.minimumByteCount, options: .storageModeShared),
              let vb = device.makeBuffer(length: vd.minimumByteCount, options: .storageModeShared)
        else { throw LanguageGenerationError.invalid("Could not allocate bounded KV caches") }
        self.model = model
        guard let function = try model.loadFunction(named: "main") else {
            throw LanguageGenerationError.invalid("Missing main inference function")
        }
        self.function = function
        self.stream = SerialComputeStream(commandQueue: queue)
        self.keys = kb
        self.values = vb
        self.keyTemplate = key
        self.valueTemplate = value
        self.keyDescriptor = kd
        self.valueDescriptor = vd
        self.vocabularySize = logits.shape[2]
        self.stopTokenIDs = stopTokenIDs
        self.queue = queue
        self.device = device
        self.allocatedCapacity = initial
        self.capacity = capacity
        self.gpu = gpu
        self.initialCapacity = initial
        self.contract = contract
        self.prefillLimit = prefillLimit
        self.imageLimit = LanguageStepContract.imageChunkLimit(acceptsImages: imageFunction != nil || imageRows != nil,
                                                               bounds: exportBounds)
        self.takesRopePositions = takesRope
        self.acceptsImages = imageFunction != nil || imageRows != nil
        self.imagesThroughMain = imageRows != nil
        self.imageRows = imageRows
        self.imageFunction = imageFunction
        self.memoryBudget = LanguageMemoryBudget(limitBytes: nil)
    }

    /// Greedy generation from a rendered prompt with images (Qwen3-VL assets): the image request below with
    /// `.greedy` sampling and no constraint, its replies token for token those of 0.0.2.
    public func generate(prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens, maximumTokens: Int = 48,
                         prefillChunkSize: Int? = nil, onToken: (@Sendable (Int) async throws -> Void)? = nil,
                         observer: Observer? = nil) async throws -> Reply {
        var delivery: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?
        if let onToken {
            delivery = { token in
                try await onToken(token.id)
                return .continue
            }
        }
        let result = try await generate(prompt: prompt, imageTokens: imageTokens, maximumTokens: maximumTokens, sampling: .greedy,
                                        prefillChunkSize: prefillChunkSize, observer: observer, onToken: delivery)
        return Reply(tokens: result.tokens, timeToFirstTokenSeconds: result.timeToFirstTokenSeconds, totalSeconds: result.totalSeconds,
                     stopReason: result.stopReason.rawValue)
    }

    /// Generation from a rendered prompt with images (Qwen3-VL assets), with sampling and a constraint as a text
    /// request has them (``LanguageGenerationRequest``). Each image's placeholder expands to its tokens. A one-function
    /// asset prefills the whole prompt in the bucketed chunks, image tokens carrying their rows; a 0.0.2 asset sends each
    /// image in one `withImage` submission and text runs in the bucketed chunks. Rotary positions follow Qwen3-VL's
    /// M-RoPE rule. The sampler's repetition penalty and the processor's `begin(prompt:)` see the expanded prompt (each
    /// image's pad token repeated to its token count), as Transformers' `input_ids` do. Sampling runs on the host after
    /// each step joins, so it adds no input, shape or submission.
    /// Not grid-restricted: any image span up to the exported image chunk is accepted. Only
    /// ``VisionLanguageSession`` limits images to the registration's fixed grids.
    public func generate(prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens, maximumTokens: Int = 48,
                         sampling: LanguageSampling, constraint: (any LanguageLogitsProcessor)? = nil,
                         prefillChunkSize: Int? = nil, observer: Observer? = nil,
                         onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)? = nil) async throws
        -> LanguageGenerationResult {
        guard maximumTokens > 0 else { throw LanguageGenerationError.invalid("Output limit must be positive") }
        try sampling.validate()
        let plan = try imagePlan(prompt, imageTokens: imageTokens)
        try begin(prompt: plan.ids, reserve: maximumTokens, chunkSize: prefillChunkSize)
        defer { lifecycle.finish() }
        promptRope = plan.rope
        ropeDelta = plan.ropeDelta
        let grant = try await admission.begin(gpu: gpu, reloadNeeded: { self.memoryReloadNeeded },
                                              reload: { try await self.reloadForMemory() })
        defer { grant?.release() }
        let scratch = try reserveScratch()
        defer { scratch?.release() }
        openMemoryWindow()
        defer { closeMemoryWindow() }
        let output = try await LanguageDecodeLoop.generate(prompt: plan.ids, maximumTokens: maximumTokens,
            stopTokenIDs: stopTokenIDs, vocabularySize: vocabularySize, sampling: sampling, processor: constraint,
            prefill: { try await self.prefill(plan, images: prompt.images, chunkSize: prefillChunkSize, observer: observer) },
            step: { token, position in try await self.step(tokens: [Int32(token)], position: position, observer: observer) },
            onToken: onToken)
        return LanguageGenerationResult(tokens: output.tokens, stopReason: output.stopReason, seed: output.seed,
                                        timeToFirstTokenSeconds: output.timeToFirstTokenSeconds, totalSeconds: output.totalSeconds)
    }

    /// Teacher-forced last-row logits for an image prompt: after the prefill, then after each
    /// continuation token (qualification against an independent reference).
    public func teacherForcedLogits(prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens,
                                    continuation: [Int]) async throws -> [[Float]] {
        let plan = try imagePlan(prompt, imageTokens: imageTokens)
        try begin(prompt: plan.ids + continuation, reserve: 0, chunkSize: nil)
        defer { lifecycle.finish() }
        promptRope = plan.rope
        ropeDelta = plan.ropeDelta
        let grant = try await admission.begin(gpu: gpu, reloadNeeded: { self.memoryReloadNeeded },
                                              reload: { try await self.reloadForMemory() })
        defer { grant?.release() }
        let scratch = try reserveScratch()
        defer { scratch?.release() }
        openMemoryWindow()
        defer { closeMemoryWindow() }
        var rows = [try await prefill(plan, images: prompt.images, chunkSize: nil, observer: nil)]
        for (index, token) in continuation.enumerated() {
            rows.append(try await step(tokens: [Int32(token)], position: plan.ids.count + index, observer: nil))
        }
        return rows
    }

    private func imagePlan(_ prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens) throws -> LanguagePromptPlan {
        guard acceptsImages else { throw LanguageGenerationError.invalid("This asset has no image input") }
        let plan = try LanguagePromptPlan.expand(prompt, tokens: imageTokens)
        try plan.requireImageSpans(within: imageLimit)
        try resourceBudget?.check(imageTokens: plan.spans.reduce(0) { $0 + $1.count })
        return plan
    }

    private func prefill(_ plan: LanguagePromptPlan, images: [LanguageImageFeatures], chunkSize: Int?,
                         observer: Observer?) async throws -> [Float] {
        var last: [Float] = []
        if let imageRows {
            // One function: uniform chunks over the whole prompt, each carrying its image tokens' rows.
            guard images.allSatisfy({ $0.hiddenSize == imageRows.hiddenSize && $0.deepstackLevels == imageRows.levels })
            else { throw LanguageGenerationError.invalid("Image features do not match the decoder") }
            for chunk in plan.uniformSubmissions(contract: contract, requestedChunk: chunkSize, maximumChunk: prefillLimit) {
                let tokens = plan.ids[chunk.offset..<(chunk.offset + chunk.count)].map(Int32.init)
                let rows = plan.alignedImageRows(offset: chunk.offset, count: chunk.count, images: images,
                                                 hiddenSize: imageRows.hiddenSize, levels: imageRows.levels)
                last = try await step(tokens: tokens, position: chunk.offset, observer: observer, rows: rows)
            }
            return last
        }
        for submission in plan.submissions(contract: contract, requestedChunk: chunkSize, maximumChunk: prefillLimit) {
            let tokens = plan.ids[submission.start..<(submission.start + submission.count)].map(Int32.init)
            last = try await step(tokens: tokens, position: submission.start, observer: observer,
                                  image: submission.image.map { images[$0] })
        }
        return last
    }

    /// Greedy generation from a rendered prompt: the default ``LanguageGenerationRequest`` through
    /// ``generate(_:prefillChunkSize:observer:onToken:)``.
    public func generate(prompt: [Int], maximumTokens: Int = 48, prefillChunkSize: Int? = nil,
                  onToken: (@Sendable (Int) async throws -> Void)? = nil,
                  observer: Observer? = nil) async throws -> Reply {
        var delivery: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?
        if let onToken {
            delivery = { token in
                try await onToken(token.id)
                return .continue
            }
        }
        let result = try await generate(LanguageGenerationRequest(prompt: prompt, maximumTokens: maximumTokens),
                                        prefillChunkSize: prefillChunkSize, observer: observer, onToken: delivery)
        return Reply(tokens: result.tokens, timeToFirstTokenSeconds: result.timeToFirstTokenSeconds, totalSeconds: result.totalSeconds,
                     stopReason: result.stopReason.rawValue)
    }

    /// ``LanguageBackend``: one request with the default prefill plan.
    public func generate(_ request: LanguageGenerationRequest,
                         onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?) async throws
        -> LanguageGenerationResult {
        try await generate(request, prefillChunkSize: nil, observer: nil, onToken: onToken)
    }

    /// One request with this backend's own options: the prefill chunk size and an observer of native submissions.
    /// Sampling and the request's constraint run on the host after each step joins (``LanguageDecodeLoop``); they add
    /// no input, shape or submission, so step shapes, the memory budget and GPU admission are those of greedy decoding.
    public func generate(_ request: LanguageGenerationRequest, prefillChunkSize: Int?, observer: Observer?,
                         onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?) async throws
        -> LanguageGenerationResult {
        guard request.maximumTokens > 0 else { throw LanguageGenerationError.invalid("Output limit must be positive") }
        try request.sampling.validate()
        try begin(prompt: request.prompt, reserve: request.maximumTokens, chunkSize: prefillChunkSize)
        defer { lifecycle.finish() }
        let grant = try await admission.begin(gpu: gpu, reloadNeeded: { self.memoryReloadNeeded },
                                              reload: { try await self.reloadForMemory() })
        defer { grant?.release() }
        let scratch = try reserveScratch()
        defer { scratch?.release() }
        openMemoryWindow()
        defer { closeMemoryWindow() }
        // Token delivery and the constraint overlap the following native step instead of idling the GPU.
        let output = try await LanguageDecodeLoop.generate(prompt: request.prompt, maximumTokens: request.maximumTokens,
            stopTokenIDs: stopTokenIDs, vocabularySize: vocabularySize, sampling: request.sampling, processor: request.constraint,
            prefill: { try await self.prefill(request.prompt, chunkSize: prefillChunkSize, observer: observer) },
            step: { token, position in try await self.step(tokens: [Int32(token)], position: position, observer: observer) },
            onToken: onToken)
        return LanguageGenerationResult(tokens: output.tokens, stopReason: output.stopReason, seed: output.seed,
                                        timeToFirstTokenSeconds: output.timeToFirstTokenSeconds, totalSeconds: output.totalSeconds)
    }

    /// Last-row logits after a fresh-cache prefill. Different chunk sizes exercise
    /// cache continuity without conflating near-tied logits with answer quality.
    public func prefillLogits(prompt: [Int], chunkSize: Int? = nil) async throws -> [Float] {
        try begin(prompt: prompt, reserve: 0, chunkSize: chunkSize)
        defer { lifecycle.finish() }
        let grant = try await admission.begin(gpu: gpu, reloadNeeded: { self.memoryReloadNeeded },
                                              reload: { try await self.reloadForMemory() })
        defer { grant?.release() }
        let scratch = try reserveScratch()
        defer { scratch?.release() }
        openMemoryWindow()
        defer { closeMemoryWindow() }
        return try await prefill(prompt, chunkSize: chunkSize, observer: nil)
    }

    public func teacherForcedLogits(prompt: [Int], continuation: [Int]) async throws -> [[Float]] {
        guard !prompt.isEmpty else { throw LanguageGenerationError.invalid("Empty oracle prompt") }
        try begin(prompt: prompt + continuation, reserve: 0, chunkSize: nil)
        defer { lifecycle.finish() }
        let grant = try await admission.begin(gpu: gpu, reloadNeeded: { self.memoryReloadNeeded },
                                              reload: { try await self.reloadForMemory() })
        defer { grant?.release() }
        let scratch = try reserveScratch()
        defer { scratch?.release() }
        openMemoryWindow()
        defer { closeMemoryWindow() }
        var rows = [try await prefill(prompt, chunkSize: nil, observer: nil)]
        for (index, token) in continuation.enumerated() {
            rows.append(try await step(tokens: [Int32(token)], position: prompt.count + index, observer: nil))
        }
        return rows
    }

    private func begin(prompt: [Int], reserve: Int, chunkSize: Int?) throws {
        try Task.checkCancellation()
        try lifecycle.admit()
        guard !prompt.isEmpty, prompt.allSatisfy({ $0 >= 0 && $0 < vocabularySize }),
              chunkSize == nil || chunkSize! > 0 else {
            lifecycle.finish()
            throw LanguageGenerationError.invalid("Invalid prompt or chunk size")
        }
        // Subtraction avoids overflow for hostile Int.max output limits.
        guard prompt.count <= capacity, reserve <= capacity - prompt.count else {
            lifecycle.finish()
            throw LanguageGenerationError.contextLimit(limit: capacity)
        }
        materializedTokens = 0
        promptRope = []
        ropeDelta = 0
        // A bucketed request restarts at the initial capacity rather than keeping a previous long
        // request's caches (memory, and the zeroing below, follow the allocation).
        let start = LanguageCachePlan.requestCapacity(allocated: allocatedCapacity, initial: initialCapacity,
                                                      resets: !contract.cachesAreStates)
        if start != allocatedCapacity {
            var shape = keyDescriptor.shape
            shape[3] = start
            let kd = keyTemplate.resolvingDynamicDimensions(shape)
            let vd = valueTemplate.resolvingDynamicDimensions(shape)
            guard let kb = device.makeBuffer(length: kd.minimumByteCount, options: .storageModeShared),
                  let vb = device.makeBuffer(length: vd.minimumByteCount, options: .storageModeShared) else {
                lifecycle.finish()
                throw LanguageGenerationError.invalid("Could not allocate bounded KV caches")
            }
            keys = kb; values = vb; keyDescriptor = kd; valueDescriptor = vd
            allocatedCapacity = start
        }
        for buffer in [keys, values].compactMap({ $0 }) {
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        }
    }

    /// The contract chooses the submission sizes: uniform chunks for the running contract,
    /// one fixed width for the bucketed one, its last chunk stepping back over written rows
    /// (`LanguageStepContract.prefillChunks`).
    private func prefill(_ prompt: [Int], chunkSize: Int?, observer: Observer?) async throws -> [Float] {
        var last: [Float] = []
        for chunk in contract.prefillChunks(promptCount: prompt.count, requestedChunk: chunkSize, maximumChunk: prefillLimit) {
            last = try await step(tokens: prompt[chunk.offset..<(chunk.offset + chunk.count)].map(Int32.init), position: chunk.offset,
                                  observer: observer)
        }
        return last
    }

    /// [3, count] rotary positions, row-major: the prompt's planned rows, then `position + ropeDelta`
    /// on every axis (text-only requests have no planned rows and a zero delta).
    /// Zero image rows for a text step of a one-function asset: the graph uses rows only at image tokens. Up to
    /// four counts are kept (the prefill width and the 1-token step recur every request); others are made per step.
    private func zeroRows(count: Int, layout: ImageRowLayout) -> (rows: NDArray, deepstack: NDArray) {
        if let cached = zeroImageRows[count] { return cached }
        let zeros = (rows: NDArray(scalars: [Float16](repeating: 0, count: count * layout.hiddenSize), shape: [count, layout.hiddenSize]),
                     deepstack: NDArray(scalars: [Float16](repeating: 0, count: layout.levels * count * layout.hiddenSize),
                                        shape: [layout.levels, count, layout.hiddenSize]))
        if zeroImageRows.count < 4 {
            zeroImageRows[count] = zeros
        }
        return zeros
    }

    private func ropeRows(at position: Int, count: Int) -> [Int32] {
        var out = [Int32](repeating: 0, count: 3 * count)
        for axis in 0..<3 {
            for i in 0..<count {
                let p = position + i
                out[axis * count + i] = p < promptRope.first?.count ?? 0 ? promptRope[axis][p] : Int32(p + ropeDelta)
            }
        }
        return out
    }

    /// `image`: a 0.0.2 `withImage` submission (exactly one image's span). `rows`: a one-function asset's image rows
    /// aligned with `tokens`; nil sends zero rows.
    private func step(tokens: [Int32], position: Int, observer: Observer?, image: LanguageImageFeatures? = nil,
                      rows: (embeddings: [Float16], deepstack: [Float16])? = nil) async throws -> [Float] {
        try Task.checkCancellation()
        // A bucketed prefill's last chunk may step back over rows it rewrites (the graph attends only to rows before
        // the chunk), but must reach the materialized end.
        guard position == materializedTokens
              || (!contract.cachesAreStates && position < materializedTokens && position + tokens.count >= materializedTokens)
        else { throw LanguageGenerationError.invalid("Noncontiguous cache position") }
        try await recycleModelIfNeeded()
        try Task.checkCancellation()
        guard let function = image == nil || imageRows != nil ? self.function : imageFunction, let stream
        else { throw LanguageGenerationError.invalid("Missing native model handle") }
        try growCache(to: position + tokens.count)
        guard let keys, let values else { throw LanguageGenerationError.invalid("Missing KV caches") }
        // Running: the full index 0..<position+count. Bucketed: only the new tokens' positions.
        let positions = contract.positions(at: position, count: tokens.count)
        var inputs = [
            "input_ids": InferenceFunction.AsyncValue(NDArray(scalars: tokens, shape: [1, tokens.count])),
            contract.positionInputName: InferenceFunction.AsyncValue(NDArray(scalars: positions, shape: [1, positions.count]))
        ]
        if !contract.cachesAreStates {
            // Read-only, zero-copy contiguous views of the Kit-owned caches (their allocation tracks the
            // real length in 256-row steps); nothing writes them until the join.
            inputs["keyCache"] = InferenceFunction.AsyncValue(unsafeBuffer: keys, scalarType: .float16,
                shape: keyDescriptor.shape, strides: keyDescriptor.preferredStrides, interleaveLayout: keyDescriptor.interleaveLayout)
            inputs["valueCache"] = InferenceFunction.AsyncValue(unsafeBuffer: values, scalarType: .float16,
                shape: valueDescriptor.shape, strides: valueDescriptor.preferredStrides, interleaveLayout: valueDescriptor.interleaveLayout)
        }
        if takesRopePositions {
            inputs["rope_positions"] = InferenceFunction.AsyncValue(NDArray(scalars: ropeRows(at: position, count: tokens.count),
                                                                             shape: [3, tokens.count]))
        }
        if let image {
            guard imageRows == nil, image.tokenCount == tokens.count
            else { throw LanguageGenerationError.invalid("Image span does not match its features") }
            inputs["image_embeds"] = InferenceFunction.AsyncValue(NDArray(scalars: image.embeddings, shape: [image.tokenCount, image.hiddenSize]))
            inputs["deepstack"] = InferenceFunction.AsyncValue(NDArray(scalars: image.deepstack,
                                                                      shape: [image.deepstackLevels, image.tokenCount, image.hiddenSize]))
        } else if let imageRows, let rows {
            guard rows.embeddings.count == tokens.count * imageRows.hiddenSize,
                  rows.deepstack.count == imageRows.levels * tokens.count * imageRows.hiddenSize
            else { throw LanguageGenerationError.invalid("Image rows do not match the submission") }
            inputs["image_embeds"] = InferenceFunction.AsyncValue(NDArray(scalars: rows.embeddings, shape: [tokens.count, imageRows.hiddenSize]))
            inputs["deepstack"] = InferenceFunction.AsyncValue(NDArray(scalars: rows.deepstack,
                                                                      shape: [imageRows.levels, tokens.count, imageRows.hiddenSize]))
        } else if let imageRows {
            let zeros = zeroRows(count: tokens.count, layout: imageRows)
            inputs["image_embeds"] = InferenceFunction.AsyncValue(zeros.rows)
            inputs["deepstack"] = InferenceFunction.AsyncValue(zeros.deepstack)
        }
        stepShapes.insert(StepShape(count: tokens.count, capacity: allocatedCapacity))
        if warmShapes.insert(WarmShape(image: image != nil, count: tokens.count, capacity: allocatedCapacity)).inserted {
            memoryWindow?.warming = true
        }
        memoryWindow?.tokens += tokens.count
        let outputs: [String: InferenceFunction.AsyncValue]
        do {
            if contract.cachesAreStates {
                var k = InferenceFunction.AsyncMutableValue(unsafeBuffer: keys, scalarType: .float16,
                    shape: keyDescriptor.shape, strides: keyDescriptor.preferredStrides, interleaveLayout: keyDescriptor.interleaveLayout)
                var v = InferenceFunction.AsyncMutableValue(unsafeBuffer: values, scalarType: .float16,
                    shape: valueDescriptor.shape, strides: valueDescriptor.preferredStrides, interleaveLayout: valueDescriptor.interleaveLayout)
                var states = InferenceFunction.AsyncMutableViews()
                states.insert(&k, for: "keyCache")
                states.insert(&v, for: "valueCache")
                outputs = try stream.encode(function, inputs: inputs, states: states)
            } else {
                outputs = try stream.encode(function, inputs: inputs)
            }
        } catch {
            await stream.currentWorkCompleted()
            throw error
        }
        observer?(.submitted(position: position, count: tokens.count))
        await stream.currentWorkCompleted()
        observer?(.completed(position: position, count: tokens.count))
        submissionsSinceReload += 1
        materializedTokens = try await LanguageCachePlan.commit(position: position, count: tokens.count) {
            if !contract.cachesAreStates { try await writeUpdates(outputs, position: position, count: tokens.count) }
        }
        try Task.checkCancellation()
        guard let result = outputs["logits"], let logits = try await result.ndArray,
              logits.scalarType == .float16, logits.shape == [1, contract.logitsRows(count: tokens.count), vocabularySize]
        else { throw LanguageGenerationError.invalid("Unexpected logits") }
        return logits.view(as: Float16.self).withUnsafePointer { pointer, shape, strides in
            LanguageLogitsRow.lastRow(pointer, rows: shape[1], rowStride: strides[1], count: shape[2], stride: strides[2])
        }
    }

    /// Bucketed contract: copies the joined step's update rows into the caches at
    /// `position..<position + count`. The caches stay row-major `[layers·1·heads, capacity, dim]`.
    private func writeUpdates(_ outputs: [String: InferenceFunction.AsyncValue], position: Int, count: Int) async throws {
        guard let keys, let values else { throw LanguageGenerationError.invalid("Missing KV caches") }
        let shape = keyDescriptor.shape
        guard keyDescriptor.preferredStrides == LanguageCachePlan.strides(shape),
              valueDescriptor.preferredStrides == LanguageCachePlan.strides(valueDescriptor.shape)
        else { throw LanguageGenerationError.invalid("Cache updates require contiguous row-major caches") }
        // Validate both outputs before writing either, so a malformed step leaves the caches untouched.
        var updates: [(NDArray, any MTLBuffer)] = []
        for (name, buffer) in [("keyUpdate", keys), ("valueUpdate", values)] {
            guard let value = outputs[name], let update = try await value.ndArray, update.scalarType == .float16,
                  update.shape == [shape[0], shape[1], shape[2], count, shape[4]]
            else { throw LanguageGenerationError.invalid("Unexpected cache update") }
            updates.append((update, buffer))
        }
        for (update, buffer) in updates {
            let cache = buffer.contents().bindMemory(to: Float16.self, capacity: buffer.length / MemoryLayout<Float16>.size)
            update.view(as: Float16.self).withUnsafePointer { pointer, updateShape, strides in
                LanguageCachePlan.writeUpdateTensor(pointer, shape: (0..<updateShape.count).map { updateShape[$0] },
                                                    strides: (0..<strides.count).map { strides[$0] }, into: cache,
                                                    capacity: shape[3], start: position)
            }
        }
    }

    /// The first index of the largest finite logit (``LanguageSampling/greedy``); infinities are skipped and NaN throws.
    public static func greedy(_ logits: [Float]) throws -> Int { try LanguageSampler.greedy(logits) }

    public struct CacheStatistics: Sendable {
        public let maximumContext: Int
        public let allocatedCapacity: Int
        public let materializedTokens: Int
        public let allocatedBytes: Int
        public let growthCount: Int
        /// Model reloads: submission-count recycling (running contract) and memory-budget reloads.
        public let modelReloadCount: Int
        /// Memory-budget reloads done at idle, after the request that spent the budget.
        public let idleMemoryReloads: Int
        /// Memory-budget reloads done in front of a request (the idle reload was cancelled, failed or overtaken).
        public let requestMemoryReloads: Int
        /// Metal growth retained by warm requests since the model last loaded (the budget's measure).
        public let retainedMemoryBytes: Int
        public let memoryGrowthBudgetBytes: Int?
        /// Requests whose measured change was set aside as another allocator's (a net decrease, or
        /// growth far above this model's per-token retention), and the size of those changes.
        public let foreignMemoryRequests: Int
        public let foreignMemoryBytes: Int
        /// Distinct submitted (token count, cache capacity) pairs over the session's life: input
        /// shapes for the bucketed contract. The running contract adds a position dimension per
        /// step on top of these, so this undercounts its native shapes.
        public let distinctStepShapes: Int
    }
    public func cacheStatistics() -> CacheStatistics {
        CacheStatistics(maximumContext: capacity, allocatedCapacity: allocatedCapacity,
            materializedTokens: materializedTokens, allocatedBytes: (keys?.length ?? 0) + (values?.length ?? 0), growthCount: growthCount,
            modelReloadCount: modelReloadCount, idleMemoryReloads: admission.idleReloads,
            requestMemoryReloads: admission.requestReloads, retainedMemoryBytes: memoryBudget.retainedBytes,
            memoryGrowthBudgetBytes: memoryBudget.limitBytes, foreignMemoryRequests: memoryBudget.foreignRequests,
            foreignMemoryBytes: memoryBudget.foreignBytes, distinctStepShapes: stepShapes.count)
    }

    private func configureReopening(_ configuration: ReopenConfiguration, memoryGrowthBudgetBytes: Int?,
                                    budget: LanguageModelBudget?, residency: ResourceReservation?) {
        reopenConfiguration = configuration
        memoryBudget = LanguageMemoryBudget(limitBytes: memoryGrowthBudgetBytes)
        resourceBudget = budget
        self.residency = residency
    }

    /// A request's scratch under the resource budget, reserved once it holds its grant (nil without a budget).
    private func reserveScratch() throws -> ResourceReservation? { try resourceBudget?.reserveScratch() }

    /// Starts a request's measurement once it is admitted. Metal's allocated size is process-wide: a
    /// shared ``GPUAdmission`` keeps other admitted GPU work out of the window, and
    /// ``LanguageMemoryBudget`` sets aside changes that are not this model's retention.
    private func openMemoryWindow() {
        memoryWindow = (device.currentAllocatedSize, (keys?.length ?? 0) + (values?.length ?? 0), false, 0)
    }

    /// Ends it, including after a failure: the growth net of the request's own KV cache changes.
    private func closeMemoryWindow() {
        guard let window = memoryWindow else { return }
        memoryWindow = nil
        let cacheBytes = (keys?.length ?? 0) + (values?.length ?? 0)
        memoryBudget.record(requestGrowthBytes: device.currentAllocatedSize - window.allocated - (cacheBytes - window.cacheBytes),
                            submittedTokens: window.tokens, warming: window.warming)
        startIdleReloadIfNeeded()
    }

    /// The budget is spent and the model can be reloaded.
    private var memoryReloadNeeded: Bool { memoryBudget.exhausted && reopenConfiguration != nil }

    /// Between requests only (at idle, or in front of a request): the model is released and loaded again.
    private func reloadForMemory() async throws {
        guard let configuration = reopenConfiguration else { return }
        try await reloadModel(configuration)
    }

    /// After a request that left the budget spent: reload now, in the background under a GPU grant,
    /// rather than in front of the next request (``LanguageRequestAdmission``).
    private func startIdleReloadIfNeeded() {
        guard admission.shouldStartIdleReload(reloadNeeded: memoryReloadNeeded) else { return }
        admission.idleReloadStarted(Task {
            await self.admission.runIdleReload(gpu: self.gpu, reloadNeeded: { self.memoryReloadNeeded },
                                               reload: { try await self.reloadForMemory() })
        })
    }

    /// Terminal close. New requests fail with `closed`; an active request finishes and
    /// joins its native work first, so cancel its task to end a long generation sooner.
    /// Then the model, function, stream and KV caches are released. Concurrent calls
    /// return once the release completes. Open a new session to load the model again.
    public func unload() async {
        // A request or idle reload still waiting for GPU admission is withdrawn and later admissions are refused, so
        // close never waits behind the GPU queue; an idle reload is cancelled (a load in progress finishes and is
        // dropped), and a request reloading under its grant stops when the reload returns.
        await admission.close()
        guard await lifecycle.beginClose() else { return }
        function = nil
        imageFunction = nil
        zeroImageRows = [:]
        stream = nil
        model = nil
        reopenConfiguration = nil
        keys = nil
        values = nil
        materializedTokens = 0
        promptRope = []
        ropeDelta = 0
        residency?.release()
        residency = nil
        lifecycle.didClose()
    }

    /// Previous submissions have all joined. State remains in Kit-owned buffers;
    /// releasing the function, stream, and model bounds native shape-cache residency.
    /// Bounded-shape contracts have no submission limit; a handle lost to a failed
    /// reload is still reopened.
    private func recycleModelIfNeeded() async throws {
        guard let configuration = reopenConfiguration else { return }
        let exhausted = configuration.maximumSubmissions.map { submissionsSinceReload >= $0 } ?? false
        guard function == nil || exhausted else { return }
        try await reloadModel(configuration)
    }

    /// Releases the model, its functions and stream (returning every allocation Core AI retained
    /// for them), then loads it again. A failed load leaves the handles nil, and the next step retries.
    private func reloadModel(_ configuration: ReopenConfiguration) async throws {
        function = nil
        imageFunction = nil
        stream = nil
        model = nil
        warmShapes = []
        memoryBudget.reset()
        // The released model's memory left inside the window must not count against the new one.
        if memoryWindow != nil { openMemoryWindow(); memoryWindow?.warming = true }
        try Task.checkCancellation()
        var options: SpecializationOptions
        switch configuration.compute {
        case .automatic: options = .default
        case .cpuOnly: options = .cpuOnly
        case .preferGPU: options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        }
        options.expectFrequentReshapes = configuration.expectFrequentReshapes
        let loaded = try await AIModel(contentsOf: configuration.url, options: options)
        try Task.checkCancellation()
        guard let loadedFunction = try loaded.loadFunction(named: "main") else {
            throw LanguageGenerationError.invalid("Missing function after model reload")
        }
        if acceptsImages && imageRows == nil {
            guard let loadedImage = try loaded.loadFunction(named: "withImage") else {
                throw LanguageGenerationError.invalid("Missing image function after model reload")
            }
            imageFunction = loadedImage
        }
        model = loaded
        function = loadedFunction
        stream = SerialComputeStream(commandQueue: queue)
        submissionsSinceReload = 0
        modelReloadCount += 1
    }

    private func growCache(to needed: Int) throws {
        guard needed <= capacity else { throw LanguageGenerationError.contextLimit(limit: capacity) }
        guard needed > allocatedCapacity else { return }
        guard let oldKeys = keys, let oldValues = values else { throw LanguageGenerationError.invalid("Missing KV caches") }
        let next = contract.cachesAreStates
            ? try LanguageCachePlan.nextCapacity(current: allocatedCapacity, needed: needed, maximum: capacity)
            : try LanguageCachePlan.bucketedCapacity(needed: needed, maximum: capacity)
        var shape = keyDescriptor.shape
        shape[3] = next
        let kd = keyTemplate.resolvingDynamicDimensions(shape)
        let vd = valueTemplate.resolvingDynamicDimensions(shape)
        guard keyDescriptor.preferredStrides == LanguageCachePlan.strides(keyDescriptor.shape),
              valueDescriptor.preferredStrides == LanguageCachePlan.strides(valueDescriptor.shape),
              kd.preferredStrides == LanguageCachePlan.strides(shape), vd.preferredStrides == LanguageCachePlan.strides(shape)
        else { throw LanguageGenerationError.invalid("Growing KV caches require contiguous row-major storage") }
        #if os(iOS)
        let available = UInt64(max(0, os_proc_available_memory()))
        let required = UInt64(kd.minimumByteCount) + UInt64(vd.minimumByteCount) + 64 * 1024 * 1024
        guard required <= available else {
            throw LanguageGenerationError.memoryBudget(requiredBytes: required, availableBytes: available)
        }
        #endif
        guard let newKeys = device.makeBuffer(length: kd.minimumByteCount, options: .storageModeShared),
              let newValues = device.makeBuffer(length: vd.minimumByteCount, options: .storageModeShared)
        else { throw LanguageGenerationError.invalid("Could not grow KV caches") }
        newKeys.contents().initializeMemory(as: UInt8.self, repeating: 0, count: newKeys.length)
        newValues.contents().initializeMemory(as: UInt8.self, repeating: 0, count: newValues.length)
        let rows = shape[0] * shape[1] * shape[2]
        let rowBytesPerToken = shape[4] * MemoryLayout<Float16>.size
        LanguageCachePlan.copyPrefix(from: oldKeys.contents(), to: newKeys.contents(), rows: rows,
            oldCapacity: allocatedCapacity, newCapacity: next, validTokens: materializedTokens, tokenBytes: rowBytesPerToken)
        LanguageCachePlan.copyPrefix(from: oldValues.contents(), to: newValues.contents(), rows: rows,
            oldCapacity: allocatedCapacity, newCapacity: next, validTokens: materializedTokens, tokenBytes: rowBytesPerToken)
        keys = newKeys; values = newValues; keyDescriptor = kd; valueDescriptor = vd
        allocatedCapacity = next
        growthCount += 1
    }

    private func elapsed(_ start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}
