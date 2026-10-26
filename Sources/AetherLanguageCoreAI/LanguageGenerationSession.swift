@preconcurrency import CoreAI
import Foundation
import AetherCoreAI
import AetherKit
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
    private var stream: ComputeStream?
    private var keys: (any MTLBuffer)?
    private var values: (any MTLBuffer)?
    private let keyTemplate: NDArrayDescriptor
    private let valueTemplate: NDArrayDescriptor
    private var keyDescriptor: NDArrayDescriptor
    private var valueDescriptor: NDArrayDescriptor
    private let vocabularySize: Int
    private let stopTokenIDs: Set<Int>
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
    private let initialCapacity: Int
    /// Largest prefill submission (the declared export bound), nil when undeclared.
    private let prefillLimit: Int?
    /// Largest image submission (the declared export bound), nil when undeclared.
    private let imageLimit: Int?
    /// Qwen3-VL decoders take 3-axis rotary positions (`rope_positions`) beside the cache positions.
    public nonisolated let takesRopePositions: Bool
    /// The asset has a `withImage` entry point, so image prompts are accepted.
    public nonisolated let acceptsImages: Bool
    private var imageFunction: InferenceFunction?
    /// The current request's prompt rotary positions (rows: time, height, width); tokens past the
    /// prompt take `position + ropeDelta` on every axis.
    private var promptRope: [[Int32]] = []
    private var ropeDelta = 0
    private let lifecycle = LanguageSessionLifecycle()
    private let gpu: GPUAdmissionOwner?
    private var pendingGPU: GPUAdmissionAttempt?
    public var state: State { lifecycle.state }

    /// Opens a registered local model through Aether Kit. No download or fallback.
    /// `gpu` opts into a host-shared ``GPUAdmission``: each request waits for (or, in fail-fast mode, requires) a
    /// grant after validation and holds it until its native submissions have joined. Loading is not admitted.
    public static func open(bundle directory: URL, platform: LanguageModelManifest.Platform,
                            gpu: GPUAdmissionOwner? = nil) async throws -> LanguageGenerationSession {
        let manifest = try LanguageModelManifest.load(from: directory)
        let modelURL = try manifest.verify(in: directory, platform: platform)
        return try await open(at: modelURL, maximumContextLength: manifest.maximumContextLength,
            stopTokenIDs: Set(manifest.stopTokenIDs), exportBounds: manifest.exportBounds, gpu: gpu)
    }

    /// Explicit raw-asset entry point for qualification before bundle registration.
    /// `maximumSubmissionsPerModel` nil selects the contract default: recycle every 128
    /// submissions for `runningPositions`, never for `bucketedPositions`. `exportBounds` are the
    /// asset's exported shape maxima: required for a bucketed asset, optional for a running one,
    /// and enforced (context and prefill chunks) when given.
    public static func open(at modelURL: URL, maximumContextLength: Int, initialCacheCapacity: Int = 256,
                            stopTokenIDs: Set<Int> = [151645], compute: ModelComputePolicy = .preferGPU,
                            expectFrequentReshapes: Bool = true, maximumSubmissionsPerModel: Int? = nil,
                            exportBounds: LanguageExportBounds? = nil,
                            gpu: GPUAdmissionOwner? = nil) async throws -> LanguageGenerationSession {
        try Task.checkCancellation()
        guard (maximumSubmissionsPerModel ?? 1) > 0 else { throw LanguageGenerationError.invalid("Invalid model residency policy") }
        var options: SpecializationOptions
        switch compute {
        case .automatic: options = .default
        case .cpuOnly: options = .cpuOnly
        case .preferGPU: options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        }
        options.expectFrequentReshapes = expectFrequentReshapes
        let model = try await AIModel(contentsOf: modelURL, options: options)
        try Task.checkCancellation()
        let session = try LanguageGenerationSession(model: model, capacity: maximumContextLength,
            initialCacheCapacity: initialCacheCapacity, stopTokenIDs: stopTokenIDs, exportBounds: exportBounds, gpu: gpu)
        await session.configureReopening(ReopenConfiguration(url: modelURL, compute: compute,
            expectFrequentReshapes: expectFrequentReshapes,
            maximumSubmissions: maximumSubmissionsPerModel ?? session.contract.defaultMaximumSubmissionsPerModel))
        return session
    }
    public static var deviceArchitectureName: String { AIModel.deviceArchitectureName }

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
        var imageFunction: InferenceFunction?
        if let image = model.functionDescriptor(for: "withImage") {
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
        self.stream = ComputeStream(commandQueue: queue)
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
        self.imageLimit = exportBounds?.maximumImageChunk
        self.takesRopePositions = takesRope
        self.acceptsImages = imageFunction != nil
        self.imageFunction = imageFunction
    }

    /// Greedy generation from a rendered prompt with images (Qwen3-VL assets). Each image's
    /// placeholder expands to its tokens, which go to the decoder in one `withImage` submission;
    /// text runs use the bucketed chunks. Rotary positions follow Qwen3-VL's M-RoPE rule.
    public func generate(prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens, maximumTokens: Int = 48,
                         prefillChunkSize: Int? = nil, onToken: (@Sendable (Int) async throws -> Void)? = nil,
                         observer: Observer? = nil) async throws -> Reply {
        guard maximumTokens > 0 else { throw LanguageGenerationError.invalid("Output limit must be positive") }
        let plan = try imagePlan(prompt, imageTokens: imageTokens)
        try begin(prompt: plan.ids, reserve: maximumTokens, chunkSize: prefillChunkSize)
        defer { lifecycle.finish() }
        promptRope = plan.rope
        ropeDelta = plan.ropeDelta
        let grant = try await admitGPU()
        defer { grant?.release() }
        let start = ContinuousClock.now
        let next = try Self.greedy(await prefill(plan, images: prompt.images, chunkSize: prefillChunkSize, observer: observer))
        let first = elapsed(start)
        let decoded = try await LanguageDecodeLoop.run(first: next, prompt: plan.ids.count, maximumTokens: maximumTokens,
            stopTokenIDs: stopTokenIDs, step: { token, position in
                try Self.greedy(await self.step(tokens: [Int32(token)], position: position, observer: observer))
            }, onToken: onToken)
        return Reply(tokens: decoded.tokens, timeToFirstTokenSeconds: first, totalSeconds: elapsed(start), stopReason: decoded.reason)
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
        let grant = try await admitGPU()
        defer { grant?.release() }
        var rows = [try await prefill(plan, images: prompt.images, chunkSize: nil, observer: nil)]
        for (index, token) in continuation.enumerated() {
            rows.append(try await step(tokens: [Int32(token)], position: plan.ids.count + index, observer: nil))
        }
        return rows
    }

    private func imagePlan(_ prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens) throws -> LanguagePromptPlan {
        guard acceptsImages else { throw LanguageGenerationError.invalid("This asset has no image input") }
        let plan = try LanguagePromptPlan.expand(prompt, tokens: imageTokens)
        if let imageLimit, let largest = plan.spans.map(\.count).max(), largest > imageLimit {
            throw LanguageGenerationError.invalid("An image has more tokens than the exported image chunk (\(imageLimit))")
        }
        return plan
    }

    private func prefill(_ plan: LanguagePromptPlan, images: [LanguageImageFeatures], chunkSize: Int?,
                         observer: Observer?) async throws -> [Float] {
        var last: [Float] = []
        for submission in plan.submissions(contract: contract, requestedChunk: chunkSize, maximumChunk: prefillLimit) {
            let tokens = plan.ids[submission.start..<(submission.start + submission.count)].map(Int32.init)
            last = try await step(tokens: tokens, position: submission.start, observer: observer,
                                  image: submission.image.map { images[$0] })
        }
        return last
    }

    public func generate(prompt: [Int], maximumTokens: Int = 48, prefillChunkSize: Int? = nil,
                  onToken: (@Sendable (Int) async throws -> Void)? = nil,
                  observer: Observer? = nil) async throws -> Reply {
        guard maximumTokens > 0 else { throw LanguageGenerationError.invalid("Output limit must be positive") }
        try begin(prompt: prompt, reserve: maximumTokens, chunkSize: prefillChunkSize)
        defer { lifecycle.finish() }
        let grant = try await admitGPU()
        defer { grant?.release() }
        let start = ContinuousClock.now
        let next = try Self.greedy(await prefill(prompt, chunkSize: prefillChunkSize, observer: observer))
        let first = elapsed(start)
        // Token delivery overlaps the following native step instead of idling the GPU.
        let decoded = try await LanguageDecodeLoop.run(first: next, prompt: prompt.count, maximumTokens: maximumTokens,
            stopTokenIDs: stopTokenIDs, step: { token, position in
                try Self.greedy(await self.step(tokens: [Int32(token)], position: position, observer: observer))
            }, onToken: onToken)
        return Reply(tokens: decoded.tokens, timeToFirstTokenSeconds: first, totalSeconds: elapsed(start), stopReason: decoded.reason)
    }

    /// Last-row logits after a fresh-cache prefill. Different chunk sizes exercise
    /// cache continuity without conflating near-tied logits with answer quality.
    public func prefillLogits(prompt: [Int], chunkSize: Int? = nil) async throws -> [Float] {
        try begin(prompt: prompt, reserve: 0, chunkSize: chunkSize)
        defer { lifecycle.finish() }
        let grant = try await admitGPU()
        defer { grant?.release() }
        return try await prefill(prompt, chunkSize: chunkSize, observer: nil)
    }

    public func teacherForcedLogits(prompt: [Int], continuation: [Int]) async throws -> [[Float]] {
        guard !prompt.isEmpty else { throw LanguageGenerationError.invalid("Empty oracle prompt") }
        try begin(prompt: prompt + continuation, reserve: 0, chunkSize: nil)
        defer { lifecycle.finish() }
        let grant = try await admitGPU()
        defer { grant?.release() }
        var rows = [try await prefill(prompt, chunkSize: nil, observer: nil)]
        for (index, token) in continuation.enumerated() {
            rows.append(try await step(tokens: [Int32(token)], position: prompt.count + index, observer: nil))
        }
        return rows
    }

    /// Waits for GPU admission when the session has one. `unload()` withdraws a request still waiting.
    private func admitGPU() async throws -> GPUAdmissionGrant? {
        guard let gpu else { return nil }
        let attempt = GPUAdmissionAttempt(gpu)
        pendingGPU = attempt
        defer { pendingGPU = nil }
        return try await attempt.admit(orThrow: { LanguageGenerationError.closed })
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
    /// power-of-two buckets for the bucketed one (`LanguageStepContract.prefillChunks`).
    private func prefill(_ prompt: [Int], chunkSize: Int?, observer: Observer?) async throws -> [Float] {
        var offset = 0
        var last: [Float] = []
        for count in contract.prefillChunks(promptCount: prompt.count, requestedChunk: chunkSize, maximumChunk: prefillLimit) {
            last = try await step(tokens: prompt[offset..<(offset + count)].map(Int32.init), position: offset, observer: observer)
            offset += count
        }
        return last
    }

    /// [3, count] rotary positions, row-major: the prompt's planned rows, then `position + ropeDelta`
    /// on every axis (text-only requests have no planned rows and a zero delta).
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

    private func step(tokens: [Int32], position: Int, observer: Observer?, image: LanguageImageFeatures? = nil) async throws -> [Float] {
        try Task.checkCancellation()
        guard position == materializedTokens else { throw LanguageGenerationError.invalid("Noncontiguous cache position") }
        try await recycleModelIfNeeded()
        try Task.checkCancellation()
        guard let function = image == nil ? self.function : imageFunction, let stream
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
            guard image.tokenCount == tokens.count else { throw LanguageGenerationError.invalid("Image span does not match its features") }
            inputs["image_embeds"] = InferenceFunction.AsyncValue(NDArray(scalars: image.embeddings, shape: [image.tokenCount, image.hiddenSize]))
            inputs["deepstack"] = InferenceFunction.AsyncValue(NDArray(scalars: image.deepstack,
                                                                      shape: [image.deepstackLevels, image.tokenCount, image.hiddenSize]))
        }
        stepShapes.insert(StepShape(count: tokens.count, capacity: allocatedCapacity))
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
                outputs = try function.encode(inputs: inputs, states: states, to: stream)
            } else {
                outputs = try function.encode(inputs: inputs, states: InferenceFunction.AsyncMutableViews(), to: stream)
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
            let row = (shape[1] - 1) * strides[1]
            return (0..<shape[2]).map { Float(pointer[row + $0 * strides[2]]) }
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

    public static func greedy(_ logits: [Float]) throws -> Int {
        var best = -Float.infinity
        var token: Int?
        for (i, score) in logits.enumerated() {
            guard !score.isNaN else { throw LanguageGenerationError.invalid("NaN logits") }
            if score.isFinite && score > best { best = score; token = i }
        }
        guard let token else { throw LanguageGenerationError.invalid("No finite logits") }
        return token
    }

    public struct CacheStatistics: Sendable {
        public let maximumContext: Int
        public let allocatedCapacity: Int
        public let materializedTokens: Int
        public let allocatedBytes: Int
        public let growthCount: Int
        public let modelReloadCount: Int
        /// Distinct submitted (token count, cache capacity) pairs over the session's life: input
        /// shapes for the bucketed contract. The running contract adds a position dimension per
        /// step on top of these, so this undercounts its native shapes.
        public let distinctStepShapes: Int
    }
    public func cacheStatistics() -> CacheStatistics {
        CacheStatistics(maximumContext: capacity, allocatedCapacity: allocatedCapacity,
            materializedTokens: materializedTokens, allocatedBytes: (keys?.length ?? 0) + (values?.length ?? 0), growthCount: growthCount,
            modelReloadCount: modelReloadCount, distinctStepShapes: stepShapes.count)
    }

    private func configureReopening(_ configuration: ReopenConfiguration) { reopenConfiguration = configuration }

    /// Terminal close. New requests fail with `closed`; an active request finishes and
    /// joins its native work first, so cancel its task to end a long generation sooner.
    /// Then the model, function, stream and KV caches are released. Concurrent calls
    /// return once the release completes. Open a new session to load the model again.
    public func unload() async {
        // A request still waiting for GPU admission is withdrawn, so close never waits behind the GPU queue.
        pendingGPU?.withdraw()
        guard await lifecycle.beginClose() else { return }
        function = nil
        imageFunction = nil
        stream = nil
        model = nil
        reopenConfiguration = nil
        keys = nil
        values = nil
        materializedTokens = 0
        promptRope = []
        ropeDelta = 0
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
        function = nil
        imageFunction = nil
        stream = nil
        model = nil
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
        if acceptsImages {
            guard let loadedImage = try loaded.loadFunction(named: "withImage") else {
                throw LanguageGenerationError.invalid("Missing image function after model reload")
            }
            imageFunction = loadedImage
        }
        model = loaded
        function = loadedFunction
        stream = ComputeStream(commandQueue: queue)
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

public enum LanguageGenerationError: Error, Sendable, Equatable {
    case invalid(String), busy, closed, contextLimit(limit: Int)
    case memoryBudget(requiredBytes: UInt64, availableBytes: UInt64)
}
