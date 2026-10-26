import AetherAssets
import AetherCoreAI
import AetherFlux2
import AetherKit
import Foundation
import Synchronization

/// `balanced` releases the text encoder after encoding and releases denoisers before any encoder
/// loads. `resident` keeps the text encoder and denoisers between requests. Both release the
/// reference encoder after each preparation (its specializations are shape-dependent) and recycle
/// denoisers and the decoder when the token count or output size changes, so specializations for
/// different shapes do not accumulate.
public enum Flux2Residency: Sendable { case balanced, resident }

/// Delivered synchronously in order. Return to continue; throw to abort.
public enum Flux2GenerationEvent: Sendable {
    case conditioningReady(Flux2Conditioning)
    case referencesReady(Flux2PreparedReferences)
    /// Total denoiser tokens (generated plus reference) for the loaded segment set.
    case denoisersReady(imageTokens: Int)
    case step(index: Int, total: Int, latents: FloatTensor)
    case decoded
}

/// Owned conditioning encoded once for repeated generation with the same bundle. No native model or
/// raw prompt text is retained. The payload is 512 × 7,680 float16 values (7.5 MiB).
public struct Flux2PreparedPrompt: Sendable {
    public let conditioning: Flux2Conditioning
    public let bundleFingerprint: String
}

/// One encoded reference: its token grid and immutable packed latents `[rows·columns, 128]`.
public struct Flux2EncodedReference: Sendable, Equatable {
    public let rows: Int
    public let columns: Int
    public let latents: FloatTensor
}

/// Reference images encoded once by the VAE encoder, in request order. Reuse across seeds and
/// prompts of the same bundle; reference k is positioned at t = 10·(k + 1).
public struct Flux2PreparedReferences: Sendable {
    public let images: [Flux2EncodedReference]
    public let referenceFingerprint: String
    public var tokens: Int { images.reduce(0) { $0 + $1.rows * $1.columns } }
}

/// FLUX.2 [klein] 4B generation: prompt → Qwen3 text encoder → optional reference encoding → three
/// denoiser segments per step with CPU Euler updates → VAE decode → owned float32 planar RGB.
/// Text-to-image and multi-reference conditioning share one path; references are appended to the
/// generated tokens at every denoiser call and only generated tokens advance and decode.
/// All boundaries copy through owned host tensors; this is not a device-resident optimized path.
///
/// One operation at a time: overlapping prepare/generate fails with `sessionBusy`. Cancelling the
/// caller task stops cooperatively between native calls after any submitted call returns.
/// `unload()` rejects new work, waits for the active operation and closes the session permanently.
public actor Flux2GenerationSession {
    public enum State: String, Sendable { case ready, running, closing, closed }
    public private(set) var state: State = .ready
    private let runner: Flux2Runner
    private let bundleFingerprint: String
    private let referenceFingerprint: String?
    private let limits: Flux2ReferenceLimits?
    private let resources: Flux2ResourceAdmission?
    private let assets: Flux2Assets
    private var active: Task<Void, Never>?

    public init(assets: Flux2Assets, compute: ModelComputePolicy = .preferGPU, residency: Flux2Residency = .balanced,
                resources: Flux2ResourceAdmission? = nil) {
        self.init(assets: assets, factory: NativeFlux2Factory(compute: compute, installation: nil),
                  residency: residency, resources: resources)
    }

    /// Holds installation protection through every stage until terminal joined unload.
    public init(installed installation: AssetReadLease, compute: ModelComputePolicy = .preferGPU,
                residency: Flux2Residency = .balanced, resources: Flux2ResourceAdmission? = nil) throws {
        let assets = try Flux2Assets.open(in: installation)
        self.init(assets: assets, factory: NativeFlux2Factory(compute: compute, installation: InstallationAccess(installation)),
                  residency: residency, resources: resources)
    }

    init(assets: Flux2Assets, factory: any Flux2ModelFactory, residency: Flux2Residency = .balanced,
         resources: Flux2ResourceAdmission? = nil) {
        bundleFingerprint = assets.fingerprint
        referenceFingerprint = assets.referenceFingerprint
        limits = assets.references
        self.resources = resources
        self.assets = assets
        let budgeted: any Flux2ModelFactory = resources.map { BudgetedFlux2Factory(base: factory, resources: $0) } ?? factory
        runner = Flux2Runner(assets: assets, factory: budgeted, residency: residency)
    }

    /// Encodes once and releases the text encoder before returning, even in resident mode.
    /// Retained denoisers are released first; the decoder remains available.
    public func prepare(prompt: Flux2Prompt) async throws -> Flux2PreparedPrompt {
        try requireEstimates(text: true, references: false, generation: false)
        return try await perform { [runner, bundleFingerprint] in
            Flux2PreparedPrompt(conditioning: try await runner.encode(prompt, retaining: false),
                                bundleFingerprint: bundleFingerprint)
        }
    }

    /// Encodes reference images once (one encoder call per image) and releases the encoder.
    /// Bundle bounds and the admission profile (references plus `generatedTokens`, the output's tokens
    /// reserved for the later generation) are checked before any model is released or loaded.
    public func prepare(references: [Flux2ReferenceImage], generatedTokens: Int = 1) async throws -> Flux2PreparedReferences {
        guard !references.isEmpty else { throw AetherError.invalidRequest("no reference images") }
        guard generatedTokens >= 1 else { throw AetherError.invalidRequest("generatedTokens must be at least 1") }
        try Flux2ReferencePolicy.check(references.map { ($0.rows, $0.columns) }, generatedTokens: generatedTokens, limits: limits)
        try resources?.check(imageTokens: references.reduce(generatedTokens) { $0 + $1.tokens })
        try requireEstimates(text: false, references: true, generation: false)
        return try await perform { [runner] in try await runner.encode(references) }
    }

    /// Text-to-image or reference-conditioned generation from a tokenized prompt.
    public func generate(
        configuration: Flux2Configuration, prompt: Flux2Prompt, references: [Flux2ReferenceImage] = [],
        noise: FloatTensor? = nil, progress: @escaping @Sendable (Flux2GenerationEvent) throws -> Void = { _ in }
    ) async throws -> GeneratedImage {
        try Flux2ReferencePolicy.check(references.map { ($0.rows, $0.columns) },
                                       generatedTokens: configuration.imageTokenCount, limits: limits)
        return try await generate(configuration: configuration, text: .prompt(prompt),
                                  references: references.isEmpty ? .none : .images(references), noise: noise, progress: progress)
    }

    /// Reuses owned conditioning (and optionally prepared references) without acquiring encoders.
    public func generate(
        configuration: Flux2Configuration, preparedPrompt: Flux2PreparedPrompt, references: Flux2PreparedReferences? = nil,
        noise: FloatTensor? = nil, progress: @escaping @Sendable (Flux2GenerationEvent) throws -> Void = { _ in }
    ) async throws -> GeneratedImage {
        guard preparedPrompt.bundleFingerprint == bundleFingerprint else {
            throw AetherError.invalidRequest("prepared prompt belongs to a different FLUX.2 bundle")
        }
        return try await generate(configuration: configuration, conditioning: preparedPrompt.conditioning,
                                  references: references, noise: noise, progress: progress)
    }

    /// Fixture path: caller-provided conditioning rows and optional exact initial noise.
    public func generate(
        configuration: Flux2Configuration, conditioning: Flux2Conditioning, references: Flux2PreparedReferences? = nil,
        noise: FloatTensor? = nil, progress: @escaping @Sendable (Flux2GenerationEvent) throws -> Void = { _ in }
    ) async throws -> GeneratedImage {
        if let references {
            guard references.referenceFingerprint == referenceFingerprint else {
                throw AetherError.invalidRequest("prepared references belong to a different FLUX.2 bundle")
            }
            try Flux2ReferencePolicy.check(references.images.map { ($0.rows, $0.columns) },
                                           generatedTokens: configuration.imageTokenCount, limits: limits)
        }
        return try await generate(configuration: configuration, text: .provided(conditioning),
                                  references: references.map { .prepared($0) } ?? .none, noise: noise, progress: progress)
    }

    private func generate(
        configuration: Flux2Configuration, text: TextSource, references: ReferenceSource, noise: FloatTensor?,
        progress: @escaping @Sendable (Flux2GenerationEvent) throws -> Void
    ) async throws -> GeneratedImage {
        if let noise, noise.shape != [configuration.imageTokenCount, Flux2Configuration.latentChannels] {
            throw AetherError.invalidTensor("initial FLUX.2 noise must be [imageTokens,128]")
        }
        try resources?.check(imageTokens: configuration.imageTokenCount + references.tokens)
        let encodesText = if case .prompt = text { true } else { false }
        let encodesReferences = if case .images = references { true } else { false }
        try requireEstimates(text: encodesText, references: encodesReferences, generation: true)
        return try await perform { [runner] in
            try await runner.generate(configuration: configuration, text: text, references: references, noise: noise,
                                      progress: progress)
        }
    }

    /// With admission, every model the request may open needs a qualified estimate; refuse before
    /// any stage is released or loaded rather than part-way through the request.
    private func requireEstimates(text: Bool, references: Bool, generation: Bool) throws {
        guard let resources else { return }
        var models: [DeclaredModelAsset] = []
        if text { models.append(assets.textEncoder) }
        if references, let encoder = assets.referenceEncoder { models.append(encoder) }
        if generation { models += assets.denoisers + [assets.decoder] }
        for model in models { _ = try resources.estimate(for: model) }
    }

    private func perform<Value: Sendable>(_ operation: @escaping @Sendable () async throws -> Value) async throws -> Value {
        try Task.checkCancellation()
        guard state != .closing, state != .closed else { throw AetherError.sessionClosed }
        guard active == nil else { throw AetherError.sessionBusy }
        let reservation = try resources.map { try $0.coordinator.reserve(bytes: $0.operationBytes) }
        let task = Task {
            defer { reservation?.release() }
            return try await operation()
        }
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

    /// Terminal close; waits for the active operation before releasing every model.
    public func unload() async {
        guard state != .closed else { return }
        state = .closing
        if let active { _ = await active.result }
        await runner.unload()
        state = .closed
    }
}

protocol Flux2ModelFactory: Sendable {
    func open(_ asset: DeclaredModelAsset, frequentReshapes: Bool) async throws -> any StagedModel
    func finish() async
}
extension Flux2ModelFactory { func finish() async {} }

private final class InstallationAccess: Sendable {
    private let held: Mutex<AssetReadLease?>
    init(_ installation: AssetReadLease) { held = Mutex(installation) }
    var lease: AssetReadLease? { held.withLock { $0 } }
    func release() { held.withLock { $0 = nil } }
}

private struct NativeFlux2Factory: Flux2ModelFactory {
    let compute: ModelComputePolicy
    let installation: InstallationAccess?
    func open(_ asset: DeclaredModelAsset, frequentReshapes: Bool) async throws -> any StagedModel {
        if let installation {
            guard let lease = installation.lease, asset.url.path.hasPrefix(lease.directory.path + "/") else {
                throw AetherError.invalidModel("model is outside the installed FLUX.2 bundle")
            }
            let path = String(asset.url.path.dropFirst(lease.directory.path.count + 1))
            return try await ModelSession.open(modelPath: path, in: lease, compute: compute, expectFrequentReshapes: frequentReshapes)
        }
        return try await ModelSession.open(at: asset.url, compute: compute, expectFrequentReshapes: frequentReshapes)
    }
    func finish() async { installation?.release() }
}

enum TextSource: Sendable { case prompt(Flux2Prompt), provided(Flux2Conditioning) }
enum ReferenceSource: Sendable {
    case none, images([Flux2ReferenceImage]), prepared(Flux2PreparedReferences)
    var tokens: Int {
        switch self {
        case .none: 0
        case .images(let images): images.reduce(0) { $0 + $1.tokens }
        case .prepared(let value): value.tokens
        }
    }
}

actor Flux2Runner {
    let assets: Flux2Assets
    let factory: any Flux2ModelFactory
    let residency: Flux2Residency
    private var textEncoder: (any StagedModel)?
    private var denoisers: [any StagedModel] = []
    private var denoiserTokens: Int?
    private var decoder: (any StagedModel)?
    private var decoderGrid: [Int]?

    init(assets: Flux2Assets, factory: any Flux2ModelFactory, residency: Flux2Residency) {
        self.assets = assets; self.factory = factory; self.residency = residency
    }

    func generate(
        configuration: Flux2Configuration, text: TextSource, references: ReferenceSource, noise: FloatTensor?,
        progress: @Sendable (Flux2GenerationEvent) throws -> Void
    ) async throws -> GeneratedImage {
        try Task.checkCancellation()
        let conditioning: Flux2Conditioning
        switch text {
        case .provided(let value): conditioning = value
        case .prompt(let prompt): conditioning = try await encode(prompt, retaining: residency == .resident)
        }
        try progress(.conditioningReady(conditioning))
        var prepared: Flux2PreparedReferences?
        switch references {
        case .none: break
        case .prepared(let value): prepared = value
        case .images(let images): prepared = try await encode(images)
        }
        if let prepared { try progress(.referencesReady(prepared)) }
        try Task.checkCancellation()

        let generated = configuration.imageTokenCount
        let referenceTokens = prepared?.tokens ?? 0
        let total = generated + referenceTokens
        try await prepareDenoisers(tokens: total)
        let signatures = try denoisers.map { try signature($0, named: "segment") }
        try Flux2Contracts.segments(signatures, tokens: total)
        try progress(.denoisersReady(imageTokens: total))

        // Immutable per-generation inputs: text rotary, image rotary (generated grid at t = 0, then
        // each reference at t = stride·(k + 1)) and the float16 reference tokens.
        let ropeText = try Flux2Rotary.table(ids: Flux2Rotary.textIDs(length: Flux2Prompt.length))
        var ids = Flux2Rotary.gridIDs(rows: configuration.rows, columns: configuration.columns)
        var referenceHalves: [Float16] = []
        referenceHalves.reserveCapacity(referenceTokens * Flux2Configuration.latentChannels)
        let stride = assets.references?.positionStride ?? 10
        for (index, image) in (prepared?.images ?? []).enumerated() {
            ids += Flux2Rotary.gridIDs(rows: image.rows, columns: image.columns, t: Int32(stride * (index + 1)))
            referenceHalves += image.latents.values.map(Float16.init)
        }
        let ropeImage = try Flux2Rotary.table(ids: ids)
        let schedule = try Flux2Schedule(steps: configuration.steps, imageTokens: generated)
        var latents = try noise ?? configuration.initialNoise()
        let count = generated * Flux2Configuration.latentChannels

        for step in 0..<configuration.steps {
            try Task.checkCancellation()
            let input = try Float16Tensor(shape: [total, Flux2Configuration.latentChannels],
                                          values: latents.values.map(Float16.init) + referenceHalves)
            let roots: [String: TensorValue] = [
                "latents": .float16(input), "text_embeds": .float16(conditioning.tensor),
                "sigma": .float32(try FloatTensor(shape: [1], values: [schedule.timesteps[step]])),
                "rope_text": .float32(ropeText), "rope_image": .float32(ropeImage),
            ]
            var carries: [String: TensorValue] = [:]
            var velocity: Float16Tensor?
            for (index, model) in denoisers.enumerated() {
                try Task.checkCancellation()
                var inputs: [String: TensorValue] = [:]
                for name in signatures[index].inputs.keys {
                    guard let value = roots[name] ?? carries[name] else { throw AetherError.invalidModel("missing carry '\(name)'") }
                    inputs[name] = value
                }
                let outputs = try await model.runTensors(function: "segment", inputs: inputs)
                guard Set(outputs.keys) == Set(signatures[index].outputs.keys) else {
                    throw AetherError.invalidTensor("denoiser output names differ from its signature")
                }
                for (name, value) in outputs {
                    let expected = try Flux2Contracts.binding(name, tokens: total)
                    try Flux2Contracts.value(value, type: expected.0, shape: expected.1)
                    if name == "velocity", case .float16(let tensor) = value { velocity = tensor } else { carries[name] = value }
                }
            }
            guard let velocity else { throw AetherError.invalidTensor("missing denoiser velocity") }
            // Reference tokens' velocities are discarded; only generated tokens advance.
            latents = try FloatTensor(shape: latents.shape, values: schedule.advance(
                step: step, latents: latents.values, velocity: velocity.values[0..<count].map(Float.init)))
            try progress(.step(index: step, total: configuration.steps, latents: latents))
        }
        try Task.checkCancellation()
        let grid = [configuration.rows, configuration.columns]
        if decoderGrid != grid, let old = decoder { await old.unload(); decoder = nil; decoderGrid = nil }
        if decoder == nil {
            decoder = try await factory.open(assets.decoder, frequentReshapes: assets.expectsFrequentReshapes("vaeDecoder"))
            decoderGrid = grid
        }
        guard let decoder else { throw AetherError.invalidModel("decoder did not load") }
        try Flux2Contracts.decoder(signature(decoder, named: "decode"), rows: configuration.rows, columns: configuration.columns)
        let packed = try Float16Tensor(shape: [configuration.rows, configuration.columns, Flux2Configuration.latentChannels],
                                       values: latents.values.map(Float16.init))
        let result = try await decoder.runTensors(function: "decode", inputs: ["latents": .float16(packed)])
        guard result.count == 1, case .float32(let pixels) = result["image"],
              pixels.shape == [3, configuration.height, configuration.width] else {
            throw AetherError.invalidTensor("decoder must return float32 [3,height,width] pixels")
        }
        try Task.checkCancellation()
        let image = try GeneratedImage(pixelsCHW: pixels)
        try progress(.decoded)
        return image
    }

    func encode(_ prompt: Flux2Prompt, retaining: Bool) async throws -> Flux2Conditioning {
        try Task.checkCancellation()
        if !retaining { await releaseDenoisers() }
        try Task.checkCancellation()
        if textEncoder == nil { textEncoder = try await factory.open(assets.textEncoder, frequentReshapes: assets.expectsFrequentReshapes("textEncoder")) }
        guard let model = textEncoder else { throw AetherError.invalidModel("text encoder did not load") }
        do {
            try Flux2Contracts.textEncoder(signature(model, named: "encode"))
            let result = try await model.runTensors(function: "encode", inputs: prompt.encoderInputs)
            try Task.checkCancellation()
            guard result.count == 1, case .float16(let tensor) = result["text_embeds"] else {
                throw AetherError.invalidTensor("text encoder must return float16 text_embeds")
            }
            let conditioning = try Flux2Conditioning(tensor: tensor)
            if !retaining { await model.unload(); textEncoder = nil }
            return conditioning
        } catch {
            // Failed or cancelled encoders are discarded before any later request.
            await model.unload()
            textEncoder = nil
            throw error
        }
    }

    func encode(_ images: [Flux2ReferenceImage]) async throws -> Flux2PreparedReferences {
        guard let asset = assets.referenceEncoder, let fingerprint = assets.referenceFingerprint else {
            throw AetherError.unsupported("this FLUX.2 bundle declares no reference images")
        }
        try Task.checkCancellation()
        if residency == .balanced { await releaseDenoisers() }
        try Task.checkCancellation()
        let model = try await factory.open(asset, frequentReshapes: assets.expectsFrequentReshapes("vaeEncoder"))
        do {
            var encoded: [Flux2EncodedReference] = []
            for image in images {
                try Task.checkCancellation()
                try Flux2Contracts.referenceEncoder(signature(model, named: "encode"), rows: image.rows, columns: image.columns)
                let result = try await model.runTensors(function: "encode", inputs: ["reference_pixels": .float32(image.pixels)])
                try Task.checkCancellation()
                guard result.count == 1, case .float32(let latents) = result["reference_latents"],
                      latents.shape == [image.rows, image.columns, Flux2Configuration.latentChannels] else {
                    throw AetherError.invalidTensor("reference encoder must return float32 [rows,columns,128] latents")
                }
                encoded.append(Flux2EncodedReference(rows: image.rows, columns: image.columns, latents: try FloatTensor(
                    shape: [image.tokens, Flux2Configuration.latentChannels], values: latents.values)))
            }
            await model.unload()
            return Flux2PreparedReferences(images: encoded, referenceFingerprint: fingerprint)
        } catch {
            await model.unload()
            throw error
        }
    }

    private func releaseDenoisers() async {
        for model in denoisers { await model.unload() }
        denoisers = []
        denoiserTokens = nil
    }

    private func prepareDenoisers(tokens: Int) async throws {
        if denoiserTokens == tokens, denoisers.count == Flux2Assets.segmentCount { return }
        await releaseDenoisers()
        do {
            for (index, asset) in assets.denoisers.enumerated() {
                try Task.checkCancellation()
                denoisers.append(try await factory.open(asset, frequentReshapes: assets.expectsFrequentReshapes("denoiser\(index)")))
            }
            try Flux2Contracts.segments(denoisers.map { try signature($0, named: "segment") }, tokens: tokens)
            denoiserTokens = tokens
        } catch {
            await releaseDenoisers()
            throw error
        }
    }

    func unload() async {
        await textEncoder?.unload()
        textEncoder = nil
        await releaseDenoisers()
        await decoder?.unload()
        decoder = nil
        decoderGrid = nil
        await factory.finish()
    }
}
