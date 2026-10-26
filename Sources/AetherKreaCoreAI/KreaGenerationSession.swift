import AetherKit
import AetherCoreAI
import AetherKrea

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

    public init(assets: KreaAssets, compute: ModelComputePolicy = .preferGPU, residency: KreaResidency = .balanced) {
        bundleFingerprint = assets.fingerprint
        runner = KreaRunner(assets: assets, factory: NativeKreaFactory(compute: compute), residency: residency)
    }

    init(assets: KreaAssets, factory: any KreaModelFactory, residency: KreaResidency = .balanced) {
        bundleFingerprint = assets.fingerprint
        runner = KreaRunner(assets: assets, factory: factory, residency: residency)
    }

    /// Encodes once and releases the encoder before returning. Preparation always
    /// closes retained denoisers before encoder acquisition, even in resident mode.
    /// The decoder remains available. This shares the generation operation slot:
    /// overlap fails with `sessionBusy`; cancellation and unload await active work.
    public func prepare(prompt: KreaPrompt) async throws -> KreaPreparedPrompt {
        try await perform { [runner, bundleFingerprint] in
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

    public func generate(
        configuration: KreaConfiguration, prompt: KreaPrompt, noise: FloatTensor? = nil,
        progress: @escaping @Sendable (KreaGenerationEvent) throws -> Void = { _ in }
    ) async throws -> GeneratedImage {
        try await generate(configuration: configuration, source: .prompt(prompt), noise: noise, progress: progress)
    }

    public func generate(
        configuration: KreaConfiguration, conditioning: KreaConditioning, noise: FloatTensor? = nil,
        progress: @escaping @Sendable (KreaGenerationEvent) throws -> Void = { _ in }
    ) async throws -> GeneratedImage {
        try await generate(configuration: configuration, source: .provided(conditioning), noise: noise, progress: progress)
    }

    private func generate(
        configuration: KreaConfiguration, source: ConditioningSource, noise: FloatTensor?,
        progress: @escaping @Sendable (KreaGenerationEvent) throws -> Void
    ) async throws -> GeneratedImage {
        try await perform { [runner] in
            if let noise, noise.shape != [configuration.imageTokenCount, 64] {
                throw AetherError.invalidTensor("initial Krea noise must be [imageTokens,64]")
            }
            return try await runner.generate(configuration: configuration, source: source, noise: noise, progress: progress)
        }
    }

    private func perform<Value: Sendable>(
        _ operation: @escaping @Sendable () async throws -> Value
    ) async throws -> Value {
        try Task.checkCancellation()
        guard state != .closing, state != .closed else { throw AetherError.sessionClosed }
        guard active == nil else { throw AetherError.sessionBusy }
        let task = Task { try await operation() }
        active = Task { _ = await task.result }
        state = .running
        defer {
            active = nil
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
        if let active { _ = await active.result }
        await runner.unload()
        state = .closed
    }
}

protocol KreaExecutable: Sendable {
    var functions: [ModelFunctionDescription] { get }
    func runTensors(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue]
    func unload() async
}
extension ModelSession: KreaExecutable {}

protocol KreaModelFactory: Sendable {
    func open(_ asset: KreaModelAsset, frequentReshapes: Bool) async throws -> any KreaExecutable
}

private struct NativeKreaFactory: KreaModelFactory {
    let compute: ModelComputePolicy
    func open(_ asset: KreaModelAsset, frequentReshapes: Bool) async throws -> any KreaExecutable {
        try await ModelSession.open(at: asset.url, compute: compute, expectFrequentReshapes: frequentReshapes)
    }
}

private enum ConditioningSource: Sendable { case prompt(KreaPrompt), provided(KreaConditioning) }

private actor KreaRunner {
    let assets: KreaAssets
    let factory: any KreaModelFactory
    let residency: KreaResidency
    var encoder: (any KreaExecutable)?
    var decoder: (any KreaExecutable)?
    var denoisers: [any KreaExecutable] = []
    var selectionIdentity: String?

    init(assets: KreaAssets, factory: any KreaModelFactory, residency: KreaResidency) {
        self.assets = assets
        self.factory = factory
        self.residency = residency
    }

    func generate(
        configuration: KreaConfiguration, source: ConditioningSource, noise: FloatTensor?,
        progress: @Sendable (KreaGenerationEvent) throws -> Void
    ) async throws -> GeneratedImage {
        try Task.checkCancellation()
        let conditioning: KreaConditioning
        switch source {
        case .provided(let provided): conditioning = provided
        case .prompt(let prompt): conditioning = try await encode(prompt, retainingModels: residency == .resident)
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
        let schedule = try KreaSchedule(steps: configuration.steps)
        var latents = try noise ?? FloatTensor(
            shape: [configuration.imageTokenCount, 64],
            values: KreaNoise.generate(seed: configuration.seed, count: configuration.imageTokenCount * 64)
        )
        for step in 0..<configuration.steps {
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
        if let decoder { await decoder.unload() }
        for model in denoisers { await model.unload() }
        encoder = nil
        decoder = nil
        denoisers = []
        selectionIdentity = nil
    }
}

private func signature(_ model: any KreaExecutable, named name: String) throws -> ModelFunctionDescription {
    guard let value = model.functions.first(where: { $0.name == name }) else {
        throw AetherError.invalidModel("missing function '\(name)'")
    }
    return value
}

private func halves(_ tensor: FloatTensor) throws -> Float16Tensor {
    try Float16Tensor(shape: tensor.shape, values: tensor.values.map(Float16.init))
}
