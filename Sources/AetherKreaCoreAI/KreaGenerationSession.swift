import AetherKit
import AetherAssets
import Synchronization
import AetherCoreAI
import AetherKrea
import Foundation

public enum KreaResidency: Sendable { case balanced, resident }

/// Events are delivered synchronously in order. Return to continue; throw to abort.
/// Latent buffers use Swift value ownership and may be retained for fixture checks.
public enum KreaGenerationEvent: Sendable {
    case conditioningReady(KreaConditioning)
    case denoisersReady(KreaDenoiserSelection)
    case step(index: Int, total: Int, latents: FloatTensor)
    case decoded
}

/// Owned conditioning encoded once for repeated generation with the same bundle.
/// No native model or raw prompt is retained. Keep only the values you need;
/// the session does not cache prepared prompts. Each value holds at most 30 MiB
/// of fp16 conditioning, shared by Swift value copies until mutation.
public struct KreaPreparedPrompt: Sendable {
    public let conditioning: KreaConditioning
    public let bundleFingerprint: String

    /// Tensor payload only; excludes Swift object overhead and native models.
    public var conditioningByteCount: Int { conditioning.tensor.values.count * MemoryLayout<Float16>.stride }
}

/// Experimental complete reference pipeline. All segment boundaries currently copy
/// through owned host tensors; this is not the device-resident optimized path.
public actor KreaGenerationSession {
    public enum State: String, Sendable { case ready, running, closing, closed }
    public private(set) var state: State = .ready
    private let runner: KreaRunner
    private let bundleFingerprint: String
    private var active: Task<Void, Never>?
    private let resources: KreaResourceAdmission?
    private let gpu: GPUAdmissionOwner?
    private var pendingGPU: GPUAdmissionAttempt?

    /// `gpu` opts into a host-shared ``GPUAdmission``: each operation waits for (or, in fail-fast mode, requires) a
    /// grant before any model work and holds it until the operation's native work has joined. With admission, the
    /// operation's resource reservation is taken after the grant, and `unload()` withdraws an operation still waiting.
    public init(assets: KreaAssets, compute: ModelComputePolicy = .preferGPU, residency: KreaResidency = .balanced,
                resources: KreaResourceAdmission? = nil, gpu: GPUAdmissionOwner? = nil) {
        bundleFingerprint = assets.fingerprint
        self.resources = resources
        self.gpu = gpu
        let factory = NativeKreaFactory(compute: compute, installation: nil)
        runner = KreaRunner(assets: assets, factory: Self.factory(factory, resources: resources), residency: residency)
    }

    /// Holds installation protection through every stage and releases it after terminal joined unload.
    /// External specialization/refine models are not imported; undeclared paths refuse.
    public init(installed installation: AssetReadLease, compute: ModelComputePolicy = .preferGPU,
                residency: KreaResidency = .balanced, resources: KreaResourceAdmission? = nil,
                gpu: GPUAdmissionOwner? = nil) throws {
        let assets = try KreaAssets.open(in: installation)
        bundleFingerprint = assets.fingerprint
        self.resources = resources
        self.gpu = gpu
        let factory = NativeKreaFactory(compute: compute, installation: KreaInstallationAccess(installation))
        runner = KreaRunner(assets: assets, factory: Self.factory(factory, resources: resources), residency: residency)
    }

    init(assets: KreaAssets, factory: any KreaModelFactory, residency: KreaResidency = .balanced,
         resources: KreaResourceAdmission? = nil, gpu: GPUAdmissionOwner? = nil) {
        bundleFingerprint = assets.fingerprint
        self.resources = resources
        self.gpu = gpu
        runner = KreaRunner(assets: assets, factory: Self.factory(factory, resources: resources), residency: residency)
    }

    private static func factory(_ base: any KreaModelFactory, resources: KreaResourceAdmission?) -> any KreaModelFactory {
        if let resources { return BudgetedKreaFactory(base: base, resources: resources) }
        return base
    }

    /// Encodes once and releases the encoder before returning. Preparation always
    /// closes retained denoisers before encoder acquisition, even in resident mode.
    /// The decoder remains available. This shares the generation operation slot:
    /// overlap fails with `sessionBusy`; cancellation and unload await active work.
    public func prepare(prompt: KreaPrompt) async throws -> KreaPreparedPrompt {
        try resources?.check(conditioningRows: prompt.conditioningRows)
        return try await perform { [runner, bundleFingerprint] in
            let conditioning = try await runner.encode(prompt, retainingModels: false)
            return KreaPreparedPrompt(conditioning: conditioning, bundleFingerprint: bundleFingerprint)
        }
    }

    /// Reuses owned conditioning without acquiring an encoder. Keep the session
    /// open to reuse denoisers/functions and the decoder for compatible shapes.
    /// Prepared prompts may be shared across sessions using the same bundle.
    public func generate(
        configuration: KreaConfiguration, preparedPrompt: KreaPreparedPrompt, noise: FloatTensor? = nil,
        progress: @escaping @Sendable (KreaGenerationEvent) throws -> Void = { _ in }
    ) async throws -> GeneratedImage {
        guard preparedPrompt.bundleFingerprint == bundleFingerprint else {
            throw AetherError.invalidRequest("prepared prompt belongs to a different Krea bundle")
        }
        return try await generate(configuration: configuration, source: .provided(preparedPrompt.conditioning), noise: noise, progress: progress)
    }

    /// Image-to-image refine: starts from `start` (encoded picture latents + start step) instead of pure noise and runs
    /// only the remaining steps. `noise` defaults to the configuration seed's noise, as for generation.
    public func refine(
        configuration: KreaConfiguration, preparedPrompt: KreaPreparedPrompt, start: KreaRefineStart, noise: FloatTensor? = nil,
        progress: @escaping @Sendable (KreaGenerationEvent) throws -> Void = { _ in }
    ) async throws -> GeneratedImage {
        guard preparedPrompt.bundleFingerprint == bundleFingerprint else {
            throw AetherError.invalidRequest("prepared prompt belongs to a different Krea bundle")
        }
        guard start.latents.shape == [configuration.imageTokenCount, 64], start.startStep < configuration.steps,
              start.startSigma == nil || start.startStep == configuration.steps - 1 else {
            throw AetherError.invalidRequest("refine latents must be [imageTokens,64] and start before the last step ends")
        }
        return try await generate(configuration: configuration, source: .provided(preparedPrompt.conditioning), noise: noise,
                                  start: start, progress: progress)
    }

    /// Encodes `pixels` ([3, 16h, 16w], values in [−1, 1]) with the external VAE encoder into packed latents
    /// [h·w, 64] for `refine`. The encoder is kept for the session's later encodes and released by `unload`.
    public func encodeImage(_ pixels: FloatTensor, encoder: KreaModelAsset) async throws -> FloatTensor {
        guard pixels.shape.count == 3, pixels.shape[0] == 3, pixels.shape[1] % 16 == 0, pixels.shape[2] % 16 == 0,
              pixels.shape[1] > 0, pixels.shape[2] > 0, (pixels.shape[1] / 16) * (pixels.shape[2] / 16) <= 4096 else {
            throw AetherError.invalidRequest("encoder input must be [3, 16h, 16w] within 4096 tokens")
        }
        try resources?.check(imageTokens: (pixels.shape[1] / 16) * (pixels.shape[2] / 16))
        return try await perform { [runner] in try await runner.encodeImage(pixels, encoder: encoder) }
    }

    public func generate(
        configuration: KreaConfiguration, prompt: KreaPrompt, noise: FloatTensor? = nil,
        progress: @escaping @Sendable (KreaGenerationEvent) throws -> Void = { _ in }
    ) async throws -> GeneratedImage {
        try await generate(configuration: configuration, source: .prompt(prompt), noise: noise, progress: progress)
    }

    /// Reference-image generation (`visionTokens`): each reference is encoded once by the vision asset, which is then
    /// released (unless resident), and the text encoder's `encodeWithImages` produces the longer conditioning.
    /// `references` must be the prompt's images in order (matched by content digest and grid). Validation happens
    /// before any model loads.
    public func generate(
        configuration: KreaConfiguration, referencePrompt: KreaReferencePrompt, references: [KreaReferencePixels],
        noise: FloatTensor? = nil, progress: @escaping @Sendable (KreaGenerationEvent) throws -> Void = { _ in }
    ) async throws -> GeneratedImage {
        try runner.checkReferences(referencePrompt, references)
        return try await generate(configuration: configuration, source: .references(referencePrompt, references),
                                  noise: noise, progress: progress)
    }

    /// Encodes a reference prompt once and releases the vision and text encoders before returning.
    public func prepare(referencePrompt: KreaReferencePrompt, references: [KreaReferencePixels]) async throws -> KreaPreparedPrompt {
        try runner.checkReferences(referencePrompt, references)
        try resources?.check(conditioningRows: referencePrompt.conditioningRows)
        return try await perform { [runner, bundleFingerprint] in
            let conditioning = try await runner.encode(referencePrompt, references, retainingModels: false)
            return KreaPreparedPrompt(conditioning: conditioning, bundleFingerprint: bundleFingerprint)
        }
    }

    public func generate(
        configuration: KreaConfiguration, conditioning: KreaConditioning, noise: FloatTensor? = nil,
        progress: @escaping @Sendable (KreaGenerationEvent) throws -> Void = { _ in }
    ) async throws -> GeneratedImage {
        try await generate(configuration: configuration, source: .provided(conditioning), noise: noise, progress: progress)
    }

    private func generate(
        configuration: KreaConfiguration, source: ConditioningSource, noise: FloatTensor?, start: KreaRefineStart? = nil,
        progress: @escaping @Sendable (KreaGenerationEvent) throws -> Void
    ) async throws -> GeneratedImage {
        let rows: Int
        switch source {
        case .provided(let value): rows = value.rows
        case .prompt(let value): rows = value.conditioningRows
        case .references(let value, _): rows = value.conditioningRows
        }
        try resources?.check(imageTokens: configuration.imageTokenCount, conditioningRows: rows)
        return try await perform { [runner] in
            if let noise, noise.shape != [configuration.imageTokenCount, 64] {
                throw AetherError.invalidTensor("initial Krea noise must be [imageTokens,64]")
            }
            return try await runner.generate(configuration: configuration, source: source, noise: noise, start: start, progress: progress)
        }
    }

    private func perform<Value: Sendable>(
        _ operation: @escaping @Sendable () async throws -> Value
    ) async throws -> Value {
        try Task.checkCancellation()
        guard state != .closing, state != .closed else { throw AetherError.sessionClosed }
        guard active == nil else { throw AetherError.sessionBusy }
        let resources = resources
        let reserve: @Sendable () throws -> ResourceReservation? = {
            try resources.map { try $0.coordinator.reserve(bytes: $0.operationBytes) }
        }
        // Without GPU admission, reserve before starting, as before. With it, reserve only once granted, so a request
        // waiting for the GPU never holds capacity the running one needs.
        let early = gpu == nil ? try reserve() : nil
        let attempt = gpu.map(GPUAdmissionAttempt.init)
        let task = Task {
            let grant = try await attempt?.admit(orThrow: { AetherError.sessionClosed })
            defer { grant?.release() }
            let reservation = try attempt == nil ? early : reserve()
            defer { reservation?.release() }
            return try await operation()
        }
        active = Task { _ = await task.result }
        pendingGPU = attempt
        state = .running
        defer {
            active = nil
            pendingGPU = nil
            if state == .running { state = .ready }
        }
        return try await withTaskCancellationHandler {
            let value = try await task.value
            try Task.checkCancellation()
            return value
        } onCancel: { task.cancel() }
    }

    /// Terminal close; waits for generation/preparation before releasing models.
    public func unload() async {
        guard state != .closed else { return }
        state = .closing
        // An operation still waiting for GPU admission is withdrawn instead of waited for.
        pendingGPU?.withdraw()
        if let active { _ = await active.result }
        await runner.unload()
        state = .closed
    }
}

/// A loaded Krea model: the package's shared staged-model protocol, which `ModelSession` and test models adopt.
typealias KreaExecutable = StagedModel

protocol KreaModelFactory: Sendable {
    func open(_ asset: KreaModelAsset, frequentReshapes: Bool) async throws -> any KreaExecutable
    func finish() async
}
extension KreaModelFactory { func finish() async {} }

private final class KreaInstallationAccess: Sendable {
    private let held: Mutex<AssetReadLease?>
    init(_ installation: AssetReadLease) { held = Mutex(installation) }
    var lease: AssetReadLease? { held.withLock { $0 } }
    func release() { held.withLock { $0 = nil } }
}

private struct NativeKreaFactory: KreaModelFactory {
    let compute: ModelComputePolicy
    let installation: KreaInstallationAccess?
    func open(_ asset: KreaModelAsset, frequentReshapes: Bool) async throws -> any KreaExecutable {
        if let installation {
            guard let lease = installation.lease, asset.url.path.hasPrefix(lease.directory.path + "/") else {
                throw AetherError.invalidModel("model is outside the installed Krea bundle")
            }
            let path = String(asset.url.path.dropFirst(lease.directory.path.count + 1))
            return try await ModelSession.open(modelPath: path, in: lease, compute: compute, expectFrequentReshapes: frequentReshapes)
        }
        return try await ModelSession.open(at: asset.url, compute: compute, expectFrequentReshapes: frequentReshapes)
    }
    func finish() async { installation?.release() }
}

private enum ConditioningSource: Sendable {
    case prompt(KreaPrompt), provided(KreaConditioning), references(KreaReferencePrompt, [KreaReferencePixels])
}

private actor KreaRunner {
    let assets: KreaAssets
    let factory: any KreaModelFactory
    let residency: KreaResidency
    var encoder: (any KreaExecutable)?
    var vision: (any KreaExecutable)?
    var decoder: (any KreaExecutable)?
    var imageEncoder: (url: URL, model: any KreaExecutable)?
    var denoisers: [any KreaExecutable] = []
    var selectionIdentity: String?

    init(assets: KreaAssets, factory: any KreaModelFactory, residency: KreaResidency) {
        self.assets = assets
        self.factory = factory
        self.residency = residency
    }

    func encodeImage(_ pixels: FloatTensor, encoder asset: KreaModelAsset) async throws -> FloatTensor {
        try Task.checkCancellation()
        if imageEncoder?.url != asset.url {
            if let old = imageEncoder { await old.model.unload(); imageEncoder = nil }
            imageEncoder = (asset.url, try await factory.open(asset, frequentReshapes: true))
        }
        guard let model = imageEncoder?.model else { throw AetherError.invalidModel("image encoder did not load") }
        let result = try await model.runTensors(function: "encode", inputs: ["image": .float16(try halves(pixels))])
        let h = pixels.shape[1] / 16, w = pixels.shape[2] / 16
        guard result.count == 1, case .float32(let latents) = result["latents"], latents.shape == [h, w, 64],
              latents.values.allSatisfy(\.isFinite) else {
            throw AetherError.invalidTensor("image encoder must return finite float32 [h,w,64] latents")
        }
        return try FloatTensor(shape: [h * w, 64], values: latents.values)
    }

    func generate(
        configuration: KreaConfiguration, source: ConditioningSource, noise: FloatTensor?, start: KreaRefineStart? = nil,
        progress: @Sendable (KreaGenerationEvent) throws -> Void
    ) async throws -> GeneratedImage {
        try Task.checkCancellation()
        let conditioning: KreaConditioning
        switch source {
        case .provided(let provided): conditioning = provided
        case .prompt(let prompt): conditioning = try await encode(prompt, retainingModels: residency == .resident)
        case .references(let prompt, let references):
            conditioning = try await encode(prompt, references, retainingModels: residency == .resident)
        }
        try progress(.conditioningReady(conditioning))
        try Task.checkCancellation()
        let selection = try assets.denoiserSelection(imageTokens: configuration.imageTokenCount, textRows: conditioning.rows)
        try await prepareDenoisers(selection)
        try Task.checkCancellation()
        let signatures = try denoisers.map { try signature($0, named: "segment") }
        try KreaContracts.segments(signatures, tokens: selection.imageTokens, text: selection.textRows)
        try progress(.denoisersReady(selection))
        let rotary = try KreaRotary.image(configuration: configuration)
        let grid = try KreaSchedule(steps: configuration.steps)
        // An off-grid refine start replaces only the last step's sigma; on-grid runs use the grid unchanged.
        let schedule = try start?.startSigma.map { try grid.replacingLastSigma($0) } ?? grid
        var latents = try noise ?? FloatTensor(
            shape: [configuration.imageTokenCount, 64],
            values: KreaNoise.generate(seed: configuration.seed, count: configuration.imageTokenCount * 64)
        )
        if let start {
            latents = try FloatTensor(shape: latents.shape, values: schedule.refineStart(
                x0: start.latents.values, noise: latents.values, step: start.startStep))
        }
        for step in (start?.startStep ?? 0)..<configuration.steps {
            try Task.checkCancellation()
            let roots: [String: TensorValue] = [
                "latents": .float16(try halves(latents)), "text_embeds": .float16(conditioning.tensor),
                "sigma": .float32(try FloatTensor(shape: [1], values: [schedule.timesteps[step]])),
                "rope_image": .float32(rotary),
            ]
            // Carries survive the entire step, including nonadjacent consumers of cond.
            var carries: [String: TensorValue] = [:]
            var velocity: Float16Tensor?
            for (index, model) in denoisers.enumerated() {
                try Task.checkCancellation()
                var inputs: [String: TensorValue] = [:]
                for name in signatures[index].inputs.keys {
                    guard let value = roots[name] ?? carries[name] else {
                        throw AetherError.invalidModel("missing carry '\(name)'")
                    }
                    inputs[name] = value
                }
                let outputs = try await model.runTensors(function: "segment", inputs: inputs)
                guard Set(outputs.keys) == Set(signatures[index].outputs.keys) else {
                    throw AetherError.invalidTensor("denoiser output names differ from its signature")
                }
                for (name, value) in outputs {
                    let expected = try KreaContracts.binding(name, tokens: selection.imageTokens, text: selection.textRows)
                    try KreaContracts.value(value, type: expected.0, shape: expected.1)
                    if name == "velocity", case .float16(let tensor) = value { velocity = tensor }
                    else { carries[name] = value }
                }
            }
            guard let velocity else { throw AetherError.invalidTensor("missing denoiser velocity") }
            latents = try FloatTensor(shape: latents.shape, values: schedule.advance(
                step: step, latents: latents.values, velocity: velocity.values.map(Float.init)
            ))
            try progress(.step(index: step, total: configuration.steps, latents: latents))
        }
        try Task.checkCancellation()
        if decoder == nil { decoder = try await factory.open(assets.decoder, frequentReshapes: false) }
        guard let decoder else { throw AetherError.invalidModel("decoder did not load") }
        try KreaContracts.decoder(signature(decoder, named: "decode"), configuration: configuration)
        let packed = try FloatTensor(shape: [configuration.height / 16, configuration.width / 16, 64], values: latents.values)
        let result = try await decoder.runTensors(function: "decode", inputs: ["latents": .float16(try halves(packed))])
        guard result.count == 1, case .float32(let pixels) = result["image"],
              pixels.shape == [3, configuration.height, configuration.width] else {
            throw AetherError.invalidTensor("decoder must return float32 [3,height,width] pixels")
        }
        try Task.checkCancellation()
        let image = try GeneratedImage(pixelsCHW: pixels)
        try progress(.decoded)
        return image
    }

    func encode(_ prompt: KreaPrompt, retainingModels: Bool) async throws -> KreaConditioning {
        try Task.checkCancellation()
        // A repeated prompt must not acquire an encoder alongside the previous
        // denoiser set in balanced mode. Provided-conditioning calls still reuse
        // denoisers; resident mode deliberately retains both stages.
        if !retainingModels {
            for model in denoisers { await model.unload() }
            denoisers = []
            selectionIdentity = nil
        }
        try Task.checkCancellation()
        if encoder == nil { encoder = try await factory.open(assets.textEncoder, frequentReshapes: true) }
        guard let model = encoder else { throw AetherError.invalidModel("encoder did not load") }
        do {
            let result = try await model.runTensors(function: "encode", inputs: prompt.encoderInputs)
            try Task.checkCancellation()
            guard result.count == 1, case .float16(let tensor) = result["text_embeds"] else {
                throw AetherError.invalidTensor("encoder must return float16 text_embeds")
            }
            let conditioning = try KreaConditioning(encoderOutput: tensor, prompt: prompt)
            if !retainingModels { await model.unload(); encoder = nil }
            return conditioning
        } catch {
            // Failed/cancelled encoders are discarded before a subsequent request.
            await model.unload()
            encoder = nil
            throw error
        }
    }

    nonisolated func checkReferences(_ prompt: KreaReferencePrompt, _ references: [KreaReferencePixels]) throws {
        guard assets.supportsReferences else { throw AetherError.unsupported("this Krea bundle declares no reference images") }
        guard references.count == prompt.images.count,
              zip(prompt.images, references).allSatisfy({ span, image in
                  span.digest == image.digest && span.range.count == image.tokens
                      && (span.t, span.h, span.w) == image.mergedGrid
              }) else {
            throw AetherError.invalidRequest("reference images do not match the prepared reference prompt")
        }
    }

    /// Vision encoder per image, then `encodeWithImages`. Balanced residency releases denoisers before either encoder
    /// loads and releases both encoders afterwards; failures and cancellation always discard the encoders.
    /// `conditioningReady` follows only after the vision encoder is released (unless resident), so existing
    /// observers see the reference phase without a new event case.
    func encode(_ prompt: KreaReferencePrompt, _ references: [KreaReferencePixels],
                retainingModels: Bool) async throws -> KreaConditioning {
        try Task.checkCancellation()
        try checkReferences(prompt, references)
        guard let visionAsset = assets.vision else { throw AetherError.unsupported("this Krea bundle declares no reference images") }
        if !retainingModels {
            for model in denoisers { await model.unload() }
            denoisers = []
            selectionIdentity = nil
        }
        var embeds: [[Float16]] = [], deep: [[Float16]] = []
        do {
            try Task.checkCancellation()
            if vision == nil { vision = try await factory.open(visionAsset, frequentReshapes: false) }
            guard let model = vision else { throw AetherError.invalidModel("vision encoder did not load") }
            for image in references {
                try Task.checkCancellation()
                try KreaContracts.vision(signature(model, named: "encode"), patches: image.patches, tokens: image.tokens)
                var inputs = try KreaPatchTables(grid: image.grid).encoderInputs
                inputs["pixels"] = .float16(try image.encoderPixels)
                let outputs = try await model.runTensors(function: "encode", inputs: inputs)
                try Task.checkCancellation()
                let hidden = KreaReferenceContract.hiddenSize, levels = KreaReferenceContract.deepstackLevels
                guard outputs.count == 2, case .float16(let rows) = outputs["image_embeds"], case .float16(let stack) = outputs["deepstack"],
                      rows.shape == [image.tokens, hidden], stack.shape == [levels, image.tokens, hidden] else {
                    throw AetherError.invalidTensor("vision encoder must return float16 image_embeds [tokens,2560] and deepstack [3,tokens,2560]")
                }
                embeds.append(rows.values); deep.append(stack.values)
            }
            if !retainingModels { await model.unload(); vision = nil }
        } catch {
            await vision?.unload()
            vision = nil
            throw error
        }
        try Task.checkCancellation()
        let hidden = KreaReferenceContract.hiddenSize, levels = KreaReferenceContract.deepstackLevels
        let n = prompt.imageTokens
        let imageEmbeds = try Float16Tensor(shape: [n, hidden], values: Array(embeds.joined()))
        var levelMajor: [Float16] = []
        levelMajor.reserveCapacity(levels * n * hidden)
        for level in 0..<levels {
            for (index, values) in deep.enumerated() {
                let size = references[index].tokens * hidden
                levelMajor += values[(level * size)..<((level + 1) * size)]
            }
        }
        let deepstack = try Float16Tensor(shape: [levels, n, hidden], values: levelMajor)
        if encoder == nil { encoder = try await factory.open(assets.textEncoder, frequentReshapes: true) }
        guard let model = encoder else { throw AetherError.invalidModel("encoder did not load") }
        do {
            let rank3 = try KreaContracts.encodeWithImages(signature(model, named: "encodeWithImages"),
                                                           rows: prompt.tokenIDs.count, images: n)
            var inputs = try prompt.encoderInputs(ropeShape: rank3 ? [3, 1, prompt.tokenIDs.count] : nil)
            inputs["image_embeds"] = .float16(imageEmbeds)
            inputs["deepstack"] = .float16(deepstack)
            let result = try await model.runTensors(function: "encodeWithImages", inputs: inputs)
            try Task.checkCancellation()
            guard result.count == 1, case .float16(let tensor) = result["text_embeds"] else {
                throw AetherError.invalidTensor("encoder must return float16 text_embeds")
            }
            let conditioning = try KreaConditioning(encoderOutput: tensor, referencePrompt: prompt)
            if !retainingModels { await model.unload(); encoder = nil }
            return conditioning
        } catch {
            await model.unload()
            encoder = nil
            throw error
        }
    }

    func prepareDenoisers(_ selection: KreaDenoiserSelection) async throws {
        if selectionIdentity == selection.identity { return }
        for model in denoisers { await model.unload() }
        denoisers = []
        selectionIdentity = nil
        do {
            for asset in selection.assets {
                try Task.checkCancellation()
                denoisers.append(try await factory.open(asset, frequentReshapes: !selection.specialized))
            }
            try KreaContracts.segments(denoisers.map { try signature($0, named: "segment") }, tokens: selection.imageTokens, text: selection.textRows)
            selectionIdentity = selection.identity
        } catch {
            for model in denoisers { await model.unload() }
            denoisers = []
            throw error
        }
    }

    func unload() async {
        if let encoder { await encoder.unload() }
        if let vision { await vision.unload() }
        vision = nil
        if let decoder { await decoder.unload() }
        if let imageEncoder { await imageEncoder.model.unload() }
        imageEncoder = nil
        for model in denoisers { await model.unload() }
        encoder = nil
        decoder = nil
        denoisers = []
        selectionIdentity = nil
        await factory.finish()
    }
}

private func halves(_ tensor: FloatTensor) throws -> Float16Tensor {
    try Float16Tensor(shape: tensor.shape, values: tensor.values.map(Float16.init))
}
