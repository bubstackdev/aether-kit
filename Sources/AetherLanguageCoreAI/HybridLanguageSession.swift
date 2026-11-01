import CoreAI
import Foundation
import AetherCoreAI
import AetherKit
import Metal

/// The hybrid step contract's pure rules (declared-state validation, the padded plan), shared by
/// ``HybridLanguageSession`` and CPU tests. Nothing here is model-specific: Qwen3.5 (rotary positions and image rows)
/// and Granite 4.0-H (neither) are both checked against it.
@available(macOS 27, iOS 27, *)
enum LanguageHybridContract {
    enum Scalar: Equatable, Sendable { case float16, float32, int32, other }

    /// One function port: its scalar type and shape (`-1` for a dynamic dimension).
    struct Port: Equatable, Sendable {
        let scalar: Scalar
        let shape: [Int]
    }

    /// What `main` takes beyond the required inputs.
    struct Layout: Equatable, Sendable {
        let width: Int
        let vocabularySize: Int
        /// The KV layout [attention layers, 1, kv heads, capacity (-1 when dynamic), head dim].
        let cacheShape: [Int]
        let takesRopePositions: Bool
        /// The hidden size of `image_embeds` [width, hidden], nil without image input.
        let imageHiddenSize: Int?
    }

    static let requiredInputs: Set<String> = ["input_ids", "cache_positions", "keyCache", "valueCache"]
    static let outputs: Set<String> = ["logits", "keyUpdate", "valueUpdate"]

    /// Checks `main` against the registration: exactly the required inputs plus optional `rope_positions` [3, W] and
    /// `image_embeds` [W, hidden]; the three outputs; exactly the declared states, each with its declared scalar type and
    /// shape. Every token-carrying input is exactly `staticWidth` wide; the KV capacity is dynamic, or static at the
    /// registered context (a fully static graph, which Core AI specializes at load rather than on each load's first call).
    static func validate(registration: LanguageHybridRegistration, inputs: [String: Port], outputs: [String: Port],
                         states: [String: Port]) throws -> Layout {
        let width = registration.staticWidth
        let optional: Set<String> = ["rope_positions", "image_embeds"]
        let names = Set(inputs.keys)
        guard names.isSuperset(of: requiredInputs), names.subtracting(requiredInputs).isSubset(of: optional),
              Set(outputs.keys) == Self.outputs
        else { throw LanguageGenerationError.invalid("Unsupported hybrid step contract") }
        guard Set(states.keys) == Set(registration.states.map(\.name)) else {
            throw LanguageGenerationError.invalid("The decoder's states are not the registration's")
        }
        for declared in registration.states {
            let scalar: Scalar = declared.scalarType == .float32 ? .float32 : .float16
            guard let port = states[declared.name], port.scalar == scalar, port.shape == declared.shape else {
                throw LanguageGenerationError.invalid("State \(declared.name) does not match its declaration")
            }
        }
        guard inputs["input_ids"] == Port(scalar: .int32, shape: [1, width]),
              inputs["cache_positions"] == Port(scalar: .int32, shape: [1, width]),
              let key = inputs["keyCache"], let value = inputs["valueCache"], key == value, key.scalar == .float16,
              key.shape.count == 5, key.shape[3] == -1 || key.shape[3] > 0,
              key.shape.enumerated().allSatisfy({ $0.offset == 3 || $0.element > 0 }),
              let logits = outputs["logits"], logits.scalar == .float16, logits.shape.count == 3, logits.shape[0] == 1,
              logits.shape[1] == 1, logits.shape[2] > 0
        else { throw LanguageGenerationError.invalid("Unsupported hybrid token, cache or logits contract") }
        var update = key.shape
        update[3] = width
        guard outputs["keyUpdate"] == Port(scalar: .float16, shape: update), outputs["valueUpdate"] == Port(scalar: .float16, shape: update)
        else { throw LanguageGenerationError.invalid("Unsupported hybrid cache update contract") }
        if let rope = inputs["rope_positions"], rope != Port(scalar: .int32, shape: [3, width]) {
            throw LanguageGenerationError.invalid("Unsupported hybrid rotary position contract")
        }
        var hidden: Int?
        if let image = inputs["image_embeds"] {
            guard inputs["rope_positions"] != nil, image.scalar == .float16, image.shape.count == 2, image.shape[0] == width,
                  image.shape[1] > 0
            else { throw LanguageGenerationError.invalid("Unsupported hybrid image row contract") }
            hidden = image.shape[1]
        }
        return Layout(width: width, vocabularySize: logits.shape[2], cacheShape: key.shape,
                      takesRopePositions: inputs["rope_positions"] != nil, imageHiddenSize: hidden)
    }

    /// Prefill submissions for `count` prompt tokens: `width`-wide chunks in order, the last one shorter (the session pads
    /// it). Never a step back: recurrent state can't rewind.
    static func chunks(count: Int, width: Int) -> [LanguagePrefillChunk] {
        precondition(count > 0 && width > 0)
        return stride(from: 0, to: count, by: width).map { LanguagePrefillChunk(offset: $0, count: min(width, count - $0)) }
    }

    /// A submission's padded token ids and cache positions: `tokens` at `position…`, then token 0 at position -1.
    static func padded(tokens: [Int32], position: Int, width: Int) -> (ids: [Int32], positions: [Int32]) {
        precondition(!tokens.isEmpty && tokens.count <= width)
        let pad = width - tokens.count
        return (tokens + [Int32](repeating: 0, count: pad),
                (position..<(position + tokens.count)).map(Int32.init) + [Int32](repeating: -1, count: pad))
    }
}

/// A hybrid recurrent-state decoder (Qwen3.5, Granite 4.0-H) on Core AI: the ``LanguageHybridRegistration`` contract.
///
/// One `main` at a fixed width W: every prompt chunk and every decode step is a W-wide call, its tail padded (cache
/// position -1), so each request uses one input shape. The KV cache is Kit-owned at the registered context's capacity,
/// passed read-only, with the call's valid update rows copied in after the join. The declared recurrent states are
/// Kit-owned buffers bound as states and zeroed at the start of every request. Image prompts take Qwen3-VL's prompt
/// expansion and M-RoPE positions, with each chunk carrying its image tokens' rows.
///
/// One request at a time; overlap fails with `busy`. With `gpu`, opening and each request are admitted; `unload()`
/// withdraws a request waiting for admission and lets an admitted one finish. There is no memory-growth reload: the
/// session's shapes are fixed, so warm requests add no shape retention.
@available(macOS 27, iOS 27, *)
public actor HybridLanguageSession {
    public typealias Reply = LanguageGenerationSession.Reply
    public typealias Observer = LanguageGenerationSession.Observer

    public struct Statistics: Sendable {
        public let maximumContext: Int
        public let staticWidth: Int
        public let cacheBytes: Int
        public let stateBytes: Int
        public let materializedTokens: Int
        /// Native calls over the session's life (every one W wide).
        public let submissions: Int
    }

    private struct StateBuffer { let name: String; let buffer: any MTLBuffer; let descriptor: NDArrayDescriptor }

    private var model: AIModel?
    private var function: InferenceFunction?
    private var stream: SerialComputeStream?
    private var keys: (any MTLBuffer)?
    private var values: (any MTLBuffer)?
    private let cacheDescriptor: NDArrayDescriptor
    private var states: [StateBuffer]
    private let layout: LanguageHybridContract.Layout
    private let capacity: Int
    private let imageTokenID: Int?
    public nonisolated let vocabularySize: Int
    public nonisolated let stopTokenIDs: Set<Int>
    public nonisolated var maximumContextLength: Int { capacity }
    /// Whether `main` takes image rows (Qwen3.5 does; Granite 4.0-H doesn't).
    public nonisolated let acceptsImages: Bool
    private var materializedTokens = 0
    private var submissions = 0
    private var promptRope: [[Int32]] = []
    private var ropeDelta = 0
    private var zeroRows: NDArray?
    private let lifecycle = LanguageSessionLifecycle()
    private let admission = LanguageRequestAdmission()
    private let gpu: GPUAdmissionOwner?
    private var resourceBudget: LanguageModelBudget?
    private var residency: ResourceReservation?

    public var state: LanguageGenerationSession.State { lifecycle.state }

    /// Opens a registered bundle whose `language-model.json` has a `hybrid` registration (text requests; image
    /// requests go through ``VisionLanguageSession``). No download or fallback; the asset is verified first.
    public static func open(bundle directory: URL, platform: LanguageModelManifest.Platform, gpu: GPUAdmissionOwner? = nil,
                            resources: LanguageResourceAdmission? = nil) async throws -> HybridLanguageSession {
        let manifest = try LanguageModelManifest.load(from: directory)
        guard let hybrid = manifest.hybrid else { throw LanguageGenerationError.invalid("The registration is not a hybrid decoder") }
        let budget = try resources?.budget(modelSHA256: manifest.modelSHA256, contextLength: manifest.maximumContextLength)
        let modelURL = try manifest.verify(in: directory, platform: platform)
        return try await open(at: modelURL, registration: hybrid, maximumContextLength: manifest.maximumContextLength,
                              stopTokenIDs: Set(manifest.stopTokenIDs), imageTokenID: manifest.vision?.imageTokenID, gpu: gpu,
                              budget: budget)
    }

    /// Raw-asset entry point for qualification before registration.
    public static func open(at modelURL: URL, registration: LanguageHybridRegistration, maximumContextLength: Int,
                            stopTokenIDs: Set<Int>, imageTokenID: Int? = nil, gpu: GPUAdmissionOwner? = nil) async throws
        -> HybridLanguageSession {
        try await open(at: modelURL, registration: registration, maximumContextLength: maximumContextLength,
                       stopTokenIDs: stopTokenIDs, imageTokenID: imageTokenID, gpu: gpu, budget: nil)
    }

    static func open(at modelURL: URL, registration: LanguageHybridRegistration, maximumContextLength: Int,
                     stopTokenIDs: Set<Int>, imageTokenID: Int?, gpu: GPUAdmissionOwner?,
                     budget: LanguageModelBudget?) async throws -> HybridLanguageSession {
        try Task.checkCancellation()
        try registration.validate()
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = registration.expectFrequentReshapes ?? false
        let (session, residency) = try await withReservedOpen(gpu, budget: budget) {
            let model = try await AIModel(contentsOf: modelURL, options: options)
            try Task.checkCancellation()
            return try HybridLanguageSession(model: model, registration: registration, capacity: maximumContextLength,
                                             stopTokenIDs: stopTokenIDs, imageTokenID: imageTokenID, gpu: gpu)
        }
        await session.configure(budget: budget, residency: residency)
        return session
    }

    init(model: AIModel, registration: LanguageHybridRegistration, capacity: Int, stopTokenIDs: Set<Int>, imageTokenID: Int?,
         gpu: GPUAdmissionOwner?) throws {
        guard capacity > 0, capacity <= 16384, let d = model.functionDescriptor(for: "main"),
              let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue()
        else { throw LanguageGenerationError.invalid("Unsupported hybrid decoder") }
        func ports(_ names: [String], _ descriptor: (String) -> InferenceValue.Descriptor?) -> [String: LanguageHybridContract.Port] {
            var out: [String: LanguageHybridContract.Port] = [:]
            for name in names {
                guard case .ndArray(let array)? = descriptor(name) else { continue }
                out[name] = LanguageHybridContract.Port(scalar: Self.scalar(array), shape: array.shape)
            }
            return out
        }
        let layout = try LanguageHybridContract.validate(registration: registration,
            inputs: ports(d.inputNames, d.inputDescriptor(of:)), outputs: ports(d.outputNames, d.outputDescriptor(of:)),
            states: ports(d.stateNames, d.stateDescriptor(of:)))
        guard !stopTokenIDs.isEmpty, stopTokenIDs.allSatisfy({ $0 >= 0 && $0 < layout.vocabularySize }),
              layout.imageHiddenSize == nil || imageTokenID != nil,
              layout.cacheShape[3] == -1 || layout.cacheShape[3] == capacity,
              case .ndArray(let key)? = d.inputDescriptor(of: "keyCache")
        else { throw LanguageGenerationError.invalid("Unsupported hybrid decoder registration") }
        var shape = key.shape
        shape[3] = capacity
        let cache = key.resolvingDynamicDimensions(shape)
        guard cache.preferredStrides == LanguageCachePlan.strides(shape),
              let kb = device.makeBuffer(length: cache.minimumByteCount, options: .storageModeShared),
              let vb = device.makeBuffer(length: cache.minimumByteCount, options: .storageModeShared)
        else { throw LanguageGenerationError.invalid("Could not allocate the hybrid KV cache") }
        var states: [StateBuffer] = []
        for declared in registration.states {
            guard case .ndArray(let template)? = d.stateDescriptor(of: declared.name) else {
                throw LanguageGenerationError.invalid("Missing state \(declared.name)")
            }
            let resolved = template.resolvingDynamicDimensions(template.shape)
            guard let buffer = device.makeBuffer(length: resolved.minimumByteCount, options: .storageModeShared) else {
                throw LanguageGenerationError.invalid("Could not allocate state \(declared.name)")
            }
            states.append(StateBuffer(name: declared.name, buffer: buffer, descriptor: resolved))
        }
        guard let function = try model.loadFunction(named: "main") else { throw LanguageGenerationError.invalid("Missing main function") }
        for buffer in [kb, vb] { buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length) }
        self.model = model
        self.function = function
        self.stream = SerialComputeStream(commandQueue: queue)
        self.keys = kb
        self.values = vb
        self.cacheDescriptor = cache
        self.states = states
        self.layout = layout
        self.capacity = capacity
        self.imageTokenID = imageTokenID
        self.vocabularySize = layout.vocabularySize
        self.stopTokenIDs = stopTokenIDs
        self.acceptsImages = layout.imageHiddenSize != nil
        self.gpu = gpu
    }

    private static func scalar(_ array: NDArrayDescriptor) -> LanguageHybridContract.Scalar {
        if array.scalarType == .float16 { return .float16 }
        if array.scalarType == .float32 { return .float32 }
        if array.scalarType == .int32 { return .int32 }
        return .other
    }

    private func configure(budget: LanguageModelBudget?, residency: ResourceReservation?) {
        resourceBudget = budget
        self.residency = residency
    }

    /// For an owning session's close: withdraws a request waiting for admission and refuses later ones.
    package func withdrawPendingAdmission() { admission.withdraw() }

    public func statistics() -> Statistics {
        Statistics(maximumContext: capacity, staticWidth: layout.width, cacheBytes: (keys?.length ?? 0) + (values?.length ?? 0),
                   stateBytes: states.reduce(0) { $0 + $1.buffer.length }, materializedTokens: materializedTokens,
                   submissions: submissions)
    }

    // MARK: requests

    /// ``LanguageBackend``: one text request.
    public func generate(_ request: LanguageGenerationRequest,
                         onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?) async throws
        -> LanguageGenerationResult {
        guard request.maximumTokens > 0 else { throw LanguageGenerationError.invalid("Output limit must be positive") }
        try request.sampling.validate()
        try begin(prompt: request.prompt, reserve: request.maximumTokens)
        defer { lifecycle.finish() }
        let admitted = try await admit()
        defer { admitted.release() }
        let output = try await LanguageDecodeLoop.generate(prompt: request.prompt, maximumTokens: request.maximumTokens,
            stopTokenIDs: stopTokenIDs, vocabularySize: vocabularySize, sampling: request.sampling, processor: request.constraint,
            prefill: { try await self.prefill(request.prompt, plan: nil, images: [], observer: nil) },
            step: { token, position in try await self.step(tokens: [Int32(token)], position: position, observer: nil) },
            onToken: onToken)
        return LanguageGenerationResult(tokens: output.tokens, stopReason: output.stopReason, seed: output.seed,
                                        timeToFirstTokenSeconds: output.timeToFirstTokenSeconds, totalSeconds: output.totalSeconds)
    }

    /// Image + text generation (``VisionLanguageSession``'s decoder): each image placeholder expands to its tokens, the
    /// rotary positions follow Qwen3-VL's M-RoPE rule, and each W-wide chunk carries its image tokens' rows.
    public func generate(prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens, maximumTokens: Int,
                         sampling: LanguageSampling, constraint: (any LanguageLogitsProcessor)? = nil, observer: Observer? = nil,
                         onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)? = nil) async throws
        -> LanguageGenerationResult {
        guard maximumTokens > 0 else { throw LanguageGenerationError.invalid("Output limit must be positive") }
        try sampling.validate()
        let plan = try imagePlan(prompt, imageTokens: imageTokens)
        try begin(prompt: plan.ids, reserve: maximumTokens)
        defer { lifecycle.finish() }
        promptRope = plan.rope
        ropeDelta = plan.ropeDelta
        let admitted = try await admit()
        defer { admitted.release() }
        let output = try await LanguageDecodeLoop.generate(prompt: plan.ids, maximumTokens: maximumTokens,
            stopTokenIDs: stopTokenIDs, vocabularySize: vocabularySize, sampling: sampling, processor: constraint,
            prefill: { try await self.prefill(plan.ids, plan: plan, images: prompt.images, observer: observer) },
            step: { token, position in try await self.step(tokens: [Int32(token)], position: position, observer: observer) },
            onToken: onToken)
        return LanguageGenerationResult(tokens: output.tokens, stopReason: output.stopReason, seed: output.seed,
                                        timeToFirstTokenSeconds: output.timeToFirstTokenSeconds, totalSeconds: output.totalSeconds)
    }

    /// Teacher-forced last-row logits for a text prompt: after the prefill, then after each continuation token.
    public func teacherForcedLogits(prompt: [Int], continuation: [Int]) async throws -> [[Float]] {
        try await teacherForced(ids: prompt, plan: nil, images: [], continuation: continuation)
    }

    /// Teacher-forced last-row logits for an image prompt (qualification against an independent reference).
    public func teacherForcedLogits(prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens,
                                    continuation: [Int]) async throws -> [[Float]] {
        let plan = try imagePlan(prompt, imageTokens: imageTokens)
        return try await teacherForced(ids: plan.ids, plan: plan, images: prompt.images, continuation: continuation)
    }

    private func teacherForced(ids: [Int], plan: LanguagePromptPlan?, images: [LanguageImageFeatures],
                               continuation: [Int]) async throws -> [[Float]] {
        try begin(prompt: ids + continuation, reserve: 0)
        defer { lifecycle.finish() }
        promptRope = plan?.rope ?? []
        ropeDelta = plan?.ropeDelta ?? 0
        let admitted = try await admit()
        defer { admitted.release() }
        var rows = [try await prefill(ids, plan: plan, images: images, observer: nil)]
        for (index, token) in continuation.enumerated() {
            rows.append(try await step(tokens: [Int32(token)], position: ids.count + index, observer: nil))
        }
        return rows
    }

    public func unload() async {
        await admission.close()
        guard await lifecycle.beginClose() else { return }
        function = nil
        stream = nil
        model = nil
        keys = nil
        values = nil
        states = []
        zeroRows = nil
        materializedTokens = 0
        residency?.release()
        residency = nil
        lifecycle.didClose()
    }

    // MARK: steps

    private func admit() async throws -> LanguageAdmittedRequest {
        try await admission.begin(gpu: gpu, reserveScratch: { try self.resourceBudget?.reserveScratch() },
                                  reloadNeeded: { false }, reload: {})
    }

    private func imagePlan(_ prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens) throws -> LanguagePromptPlan {
        guard let hidden = layout.imageHiddenSize, imageTokens.imagePad == imageTokenID else {
            throw LanguageGenerationError.invalid("This asset has no image input")
        }
        guard prompt.images.allSatisfy({ $0.hiddenSize == hidden && $0.deepstackLevels == 0 }) else {
            throw LanguageGenerationError.invalid("Image features do not match the decoder")
        }
        let plan = try LanguagePromptPlan.expand(prompt, tokens: imageTokens)
        try resourceBudget?.check(imageTokens: plan.spans.reduce(0) { $0 + $1.count })
        return plan
    }

    /// Admits the request's prompt and output against the context and zeroes the recurrent states.
    private func begin(prompt: [Int], reserve: Int) throws {
        try Task.checkCancellation()
        try lifecycle.admit()
        guard !prompt.isEmpty, prompt.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            lifecycle.finish()
            throw LanguageGenerationError.invalid("Invalid prompt")
        }
        guard prompt.count <= capacity, reserve <= capacity - prompt.count else {
            lifecycle.finish()
            throw LanguageGenerationError.contextLimit(limit: capacity)
        }
        materializedTokens = 0
        promptRope = []
        ropeDelta = 0
        // Stale KV rows past the request's position are masked by the graph (rows at or after a chunk's start are
        // invisible to it), so only the recurrent states need resetting.
        for state in states { state.buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: state.buffer.length) }
    }

    private func prefill(_ ids: [Int], plan: LanguagePromptPlan?, images: [LanguageImageFeatures],
                         observer: Observer?) async throws -> [Float] {
        var last: [Float] = []
        for chunk in LanguageHybridContract.chunks(count: ids.count, width: layout.width) {
            let rows = plan.flatMap { plan in
                layout.imageHiddenSize.flatMap { plan.alignedImageRows(offset: chunk.offset, count: chunk.count, images: images,
                                                                      hiddenSize: $0, levels: 0)?.embeddings }
            }
            last = try await step(tokens: ids[chunk.offset..<(chunk.offset + chunk.count)].map(Int32.init), position: chunk.offset,
                                  observer: observer, imageRows: rows)
        }
        return last
    }

    /// [3, W] rotary positions: the prompt's planned rows, then `position + ropeDelta` on every axis; padding 0.
    private func ropeRows(at position: Int, count: Int) -> [Int32] {
        let width = layout.width
        var out = [Int32](repeating: 0, count: 3 * width)
        for axis in 0..<3 {
            for i in 0..<count {
                let p = position + i
                out[axis * width + i] = p < promptRope.first?.count ?? 0 ? promptRope[axis][p] : Int32(p + ropeDelta)
            }
        }
        return out
    }

    /// One W-wide call: `tokens` at `position`, the rest padding. `imageRows`: the tokens' image rows [count, hidden]
    /// (nil sends zero rows).
    private func step(tokens: [Int32], position: Int, observer: Observer?, imageRows: [Float16]? = nil) async throws -> [Float] {
        try Task.checkCancellation()
        guard position == materializedTokens else { throw LanguageGenerationError.invalid("Noncontiguous cache position") }
        guard position + tokens.count <= capacity else { throw LanguageGenerationError.contextLimit(limit: capacity) }
        guard let function, let stream, let keys, let values else { throw LanguageGenerationError.closed }
        let width = layout.width
        let (ids, positions) = LanguageHybridContract.padded(tokens: tokens, position: position, width: width)
        var inputs: [String: InferenceFunction.AsyncValue] = [
            "input_ids": .init(NDArray(scalars: ids, shape: [1, width])),
            "cache_positions": .init(NDArray(scalars: positions, shape: [1, width])),
            "keyCache": .init(unsafeBuffer: keys, scalarType: .float16, shape: cacheDescriptor.shape,
                              strides: cacheDescriptor.preferredStrides, interleaveLayout: cacheDescriptor.interleaveLayout),
            "valueCache": .init(unsafeBuffer: values, scalarType: .float16, shape: cacheDescriptor.shape,
                                strides: cacheDescriptor.preferredStrides, interleaveLayout: cacheDescriptor.interleaveLayout),
        ]
        if layout.takesRopePositions {
            inputs["rope_positions"] = .init(NDArray(scalars: ropeRows(at: position, count: tokens.count), shape: [3, width]))
        }
        if let hidden = layout.imageHiddenSize {
            if let imageRows {
                guard imageRows.count == tokens.count * hidden else { throw LanguageGenerationError.invalid("Image rows do not match the submission") }
                inputs["image_embeds"] = .init(NDArray(scalars: imageRows + [Float16](repeating: 0, count: (width - tokens.count) * hidden),
                                                       shape: [width, hidden]))
            } else {
                if zeroRows == nil { zeroRows = NDArray(scalars: [Float16](repeating: 0, count: width * hidden), shape: [width, hidden]) }
                inputs["image_embeds"] = .init(zeroRows!)
            }
        }
        let outputs: [String: InferenceFunction.AsyncValue]
        do {
            outputs = try encode(function, stream: stream, inputs: inputs, index: 0, views: InferenceFunction.AsyncMutableViews())
        } catch {
            await stream.currentWorkCompleted()
            throw error
        }
        observer?(.submitted(position: position, count: tokens.count))
        await stream.currentWorkCompleted()
        observer?(.completed(position: position, count: tokens.count))
        submissions += 1
        try await writeUpdates(outputs, position: position, count: tokens.count)
        materializedTokens = position + tokens.count
        try Task.checkCancellation()
        guard let result = outputs["logits"], let logits = try await result.ndArray, logits.scalarType == .float16,
              logits.shape == [1, 1, vocabularySize]
        else { throw LanguageGenerationError.invalid("Unexpected logits") }
        return logits.view(as: Float16.self).withUnsafePointer { pointer, shape, strides in
            LanguageLogitsRow.lastRow(pointer, rows: shape[1], rowStride: strides[1], count: shape[2], stride: strides[2])
        }
    }

    /// Binds every declared state (each binding lives across the recursive call), then encodes.
    private func encode(_ function: InferenceFunction, stream: SerialComputeStream, inputs: [String: InferenceFunction.AsyncValue],
                        index: Int, views: consuming InferenceFunction.AsyncMutableViews) throws -> [String: InferenceFunction.AsyncValue] {
        if index == states.count { return try stream.encode(function, inputs: inputs, states: views) }
        let state = states[index]
        var value = InferenceFunction.AsyncMutableValue(unsafeBuffer: state.buffer, scalarType: state.descriptor.scalarType,
            shape: state.descriptor.shape, strides: state.descriptor.preferredStrides, interleaveLayout: state.descriptor.interleaveLayout)
        var next = views
        next.insert(&value, for: state.name)
        return try encode(function, stream: stream, inputs: inputs, index: index + 1, views: next)
    }

    /// Copies the call's `count` valid update rows into the caches at `position…` (a padded tail is dropped).
    private func writeUpdates(_ outputs: [String: InferenceFunction.AsyncValue], position: Int, count: Int) async throws {
        guard let keys, let values else { throw LanguageGenerationError.closed }
        let shape = cacheDescriptor.shape
        var updates: [(NDArray, any MTLBuffer)] = []
        for (name, buffer) in [("keyUpdate", keys), ("valueUpdate", values)] {
            guard let value = outputs[name], let update = try await value.ndArray, update.scalarType == .float16,
                  update.shape == [shape[0], shape[1], shape[2], layout.width, shape[4]]
            else { throw LanguageGenerationError.invalid("Unexpected cache update") }
            updates.append((update, buffer))
        }
        for (update, buffer) in updates {
            let cache = buffer.contents().bindMemory(to: Float16.self, capacity: buffer.length / MemoryLayout<Float16>.size)
            update.view(as: Float16.self).withUnsafePointer { pointer, updateShape, strides in
                var valid = (0..<updateShape.count).map { updateShape[$0] }
                valid[3] = count
                LanguageCachePlan.writeUpdateTensor(pointer, shape: valid, strides: (0..<strides.count).map { strides[$0] },
                                                    into: cache, capacity: shape[3], start: position)
            }
        }
    }
}

@available(macOS 27, iOS 27, *)
extension HybridLanguageSession: VisionLanguageDecoding {
    func generate(prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens, maximumTokens: Int, prefillChunkSize: Int?,
                  onToken: (@Sendable (Int) async throws -> Void)?, observer: LanguageGenerationSession.Observer?) async throws
        -> LanguageGenerationSession.Reply {
        var delivery: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?
        if let onToken { delivery = { token in try await onToken(token.id); return .continue } }
        let result = try await generate(prompt: prompt, imageTokens: imageTokens, maximumTokens: maximumTokens, sampling: .greedy,
                                        constraint: nil, observer: observer, onToken: delivery)
        return Reply(tokens: result.tokens, timeToFirstTokenSeconds: result.timeToFirstTokenSeconds, totalSeconds: result.totalSeconds,
                     stopReason: result.stopReason.rawValue)
    }

    /// The prefill width is the registration's; a requested chunk size is ignored (the asset takes only W).
    func generate(prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens, maximumTokens: Int, sampling: LanguageSampling,
                  constraint: (any LanguageLogitsProcessor)?, prefillChunkSize: Int?, observer: LanguageGenerationSession.Observer?,
                  onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?) async throws
        -> LanguageGenerationResult {
        try await generate(prompt: prompt, imageTokens: imageTokens, maximumTokens: maximumTokens, sampling: sampling,
                           constraint: constraint, observer: observer, onToken: onToken)
    }
}
