import AetherAssets
import AetherCoreAI
import AetherKit
import AetherSpeech
import Synchronization

/// One synthesized utterance.
public struct KokoroSpeech: Sendable {
    /// Mono float32 PCM at 24 kHz, unclipped.
    public let audio: GeneratedAudio
    /// Frames per token, including the two boundary tokens; 600 samples per frame.
    public let frames: [Int]
    public let phonemes: KokoroPhonemes
    public let configuration: KokoroSpeechConfiguration
}

/// One segment of a chunked request, delivered as soon as it is decoded.
public struct KokoroSpeechChunk: Sendable {
    public let index: Int
    public let count: Int
    public let speech: KokoroSpeech
}

/// Events are delivered synchronously in order. Return to continue; throw to abort.
public enum KokoroSpeechEvent: Sendable {
    case framesPredicted(segment: Int, frames: [Int])
    case chunk(KokoroSpeechChunk)
}

/// Caller-qualified estimates for opt-in admission. Model estimates are keyed by stream and metadata
/// hash; `operationBytes` must cover host tensors (noise is 21.6 KB per frame) up to `maximumFrames`.
/// These are reservations of estimates, not measured memory or allocator enforcement.
public struct KokoroResourceAdmission: Sendable {
    public struct ModelEstimate: Sendable {
        let identity: String
        let residentBytes: UInt64
        let runScratchBytes: UInt64
        public init(asset: KokoroModelAsset, residentBytes: UInt64, runScratchBytes: UInt64) throws {
            guard residentBytes > 0, runScratchBytes > 0 else { throw ResourceAdmissionError.invalidByteCount }
            identity = asset.declaredSHA256 + ":" + asset.metadataSHA256
            self.residentBytes = residentBytes
            self.runScratchBytes = runScratchBytes
        }
    }

    public let coordinator: ResourceCoordinator
    public let operationBytes: UInt64
    public let maximumFrames: Int
    let models: [String: ModelEstimate]

    public init(coordinator: ResourceCoordinator, operationBytes: UInt64, maximumFrames: Int,
                durations: ModelEstimate, decoder: ModelEstimate) throws {
        guard operationBytes > 0, maximumFrames > 1, durations.identity != decoder.identity else {
            throw AetherError.invalidRequest("invalid Kokoro resource profile")
        }
        self.coordinator = coordinator
        self.operationBytes = operationBytes
        self.maximumFrames = maximumFrames
        models = [durations.identity: durations, decoder.identity: decoder]
    }

    func admission(for asset: KokoroModelAsset) throws -> ModelResourceAdmission {
        guard let estimate = models[asset.declaredSHA256 + ":" + asset.metadataSHA256] else {
            throw AetherError.invalidRequest("no resource estimate for the selected Kokoro model bytes")
        }
        return try ModelResourceAdmission(coordinator: coordinator, residentBytes: estimate.residentBytes,
                                          runScratchBytes: estimate.runScratchBytes)
    }
}

/// Kokoro-82M speech synthesis: validated phonemes in, 24 kHz float32 PCM out.
///
/// The durations graph predicts frames per phoneme; the host rounds and aligns them; the decoder
/// graph renders the waveform with seeded harmonic-source noise. Both models load on first use and
/// stay resident until `unload()`. One request runs at a time: overlap fails with `sessionBusy`.
/// Cancel the caller's task to stop between native calls; submitted calls are awaited first.
/// `unload()` rejects new work, waits for the active request and closes the session permanently.
///
/// Delivery is whole-utterance per segment. `synthesize(segments:)` streams at segment granularity:
/// each segment's audio is delivered through the event callback as soon as it is decoded, so callers
/// can start playback after the first sentence. Sub-utterance streaming would need a windowed decoder.
public actor KokoroSpeechSession {
    public enum State: String, Sendable { case ready, running, closing, closed }
    public private(set) var state: State = .ready
    public nonisolated let assets: KokoroAssets
    private let runner: KokoroRunner
    private let resources: KokoroResourceAdmission?
    private var active: Task<Void, Never>?

    public init(assets: KokoroAssets, compute: ModelComputePolicy = .preferGPU, resources: KokoroResourceAdmission? = nil) {
        self.assets = assets
        self.resources = resources
        runner = KokoroRunner(assets: assets, factory: NativeKokoroFactory(compute: compute, installation: nil, resources: resources))
    }

    /// Holds installation protection until terminal joined unload.
    public init(installed installation: AssetReadLease, compute: ModelComputePolicy = .preferGPU,
                resources: KokoroResourceAdmission? = nil) throws {
        assets = try KokoroAssets.open(in: installation)
        self.resources = resources
        runner = KokoroRunner(assets: assets, factory: NativeKokoroFactory(
            compute: compute, installation: KokoroInstallationAccess(installation), resources: resources))
    }

    init(assets: KokoroAssets, factory: any KokoroModelFactory, resources: KokoroResourceAdmission? = nil) {
        self.assets = assets
        self.resources = resources
        runner = KokoroRunner(assets: assets, factory: factory)
    }

    /// Parses phonemes against this bundle's vocabulary.
    public nonisolated func phonemes(_ string: String) throws -> KokoroPhonemes {
        try KokoroPhonemes(string, vocabulary: assets.vocabulary)
    }

    public func synthesize(
        _ phonemes: KokoroPhonemes, configuration: KokoroSpeechConfiguration,
        progress: @escaping @Sendable (KokoroSpeechEvent) throws -> Void = { _ in }
    ) async throws -> KokoroSpeech {
        let result = try await synthesize(segments: [phonemes], configuration: configuration, progress: progress)
        return result[0]
    }

    /// Synthesizes segments in order, delivering each as a `.chunk` event once decoded. Segment `i`
    /// uses noise seed `configuration.seed &+ i`. Join with `GeneratedAudio.concatenating`.
    public func synthesize(
        segments: [KokoroPhonemes], configuration: KokoroSpeechConfiguration,
        progress: @escaping @Sendable (KokoroSpeechEvent) throws -> Void = { _ in }
    ) async throws -> [KokoroSpeech] {
        guard !segments.isEmpty, segments.count <= 4096 else {
            throw AetherError.invalidRequest("provide 1...4096 Kokoro segments")
        }
        let maximumFrames = min(assets.maximumFrames, resources?.maximumFrames ?? .max)
        return try await perform { [runner, resources] in
            let operation = try resources.map { try $0.coordinator.reserve(bytes: $0.operationBytes) }
            defer { operation?.release() }
            var results: [KokoroSpeech] = []
            for (index, phonemes) in segments.enumerated() {
                try Task.checkCancellation()
                let segment = try KokoroSpeechConfiguration(voice: configuration.voice, speed: configuration.speed,
                                                            seed: configuration.seed &+ UInt64(index))
                let speech = try await runner.synthesize(phonemes, configuration: segment, maximumFrames: maximumFrames) {
                    try progress(.framesPredicted(segment: index, frames: $0))
                }
                try Task.checkCancellation()
                try progress(.chunk(KokoroSpeechChunk(index: index, count: segments.count, speech: speech)))
                results.append(speech)
            }
            return results
        }
    }

    /// Converts text with the app's phonemizer, then synthesizes the returned segments.
    public func synthesize(
        text: String, phonemizer: any SpeechPhonemizer, configuration: KokoroSpeechConfiguration,
        progress: @escaping @Sendable (KokoroSpeechEvent) throws -> Void = { _ in }
    ) async throws -> [KokoroSpeech] {
        let strings = try await phonemizer.phonemize(text, language: configuration.voice.language)
        let segments = try strings.map { try phonemes($0) }
        return try await synthesize(segments: segments, configuration: configuration, progress: progress)
    }

    private func perform<Value: Sendable>(_ operation: @escaping @Sendable () async throws -> Value) async throws -> Value {
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

    /// Terminal close; waits for the active request before releasing models and installation protection.
    public func unload() async {
        guard state != .closed else { return }
        state = .closing
        if let active { _ = await active.result }
        await runner.unload()
        state = .closed
    }
}

protocol KokoroExecutable: Sendable {
    var functions: [ModelFunctionDescription] { get }
    func runTensors(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue]
    func unload() async
}
extension ModelSession: KokoroExecutable {}

protocol KokoroModelFactory: Sendable {
    func open(_ asset: KokoroModelAsset) async throws -> any KokoroExecutable
    func finish() async
}
extension KokoroModelFactory { func finish() async {} }

private final class KokoroInstallationAccess: Sendable {
    private let held: Mutex<AssetReadLease?>
    init(_ installation: AssetReadLease) { held = Mutex(installation) }
    var lease: AssetReadLease? { held.withLock { $0 } }
    func release() { held.withLock { $0 = nil } }
}

private struct NativeKokoroFactory: KokoroModelFactory {
    let compute: ModelComputePolicy
    let installation: KokoroInstallationAccess?
    let resources: KokoroResourceAdmission?

    func open(_ asset: KokoroModelAsset) async throws -> any KokoroExecutable {
        let admission = try resources?.admission(for: asset)
        // Sequence lengths change per utterance; both graphs are dynamic.
        if let installation {
            guard let lease = installation.lease, asset.url.path.hasPrefix(lease.directory.path + "/") else {
                throw AetherError.invalidModel("model is outside the installed Kokoro bundle")
            }
            return try await ModelSession.open(modelPath: asset.relativePath, in: lease, compute: compute,
                                               expectFrequentReshapes: true, admission: admission)
        }
        return try await ModelSession.open(at: asset.url, compute: compute, expectFrequentReshapes: true, admission: admission)
    }

    func finish() async { installation?.release() }
}

private actor KokoroRunner {
    let assets: KokoroAssets
    let factory: any KokoroModelFactory
    var durations: (any KokoroExecutable)?
    var decoder: (any KokoroExecutable)?
    var styles: [KokoroVoice: KokoroVoiceStyles] = [:]

    init(assets: KokoroAssets, factory: any KokoroModelFactory) {
        self.assets = assets
        self.factory = factory
    }

    func synthesize(
        _ phonemes: KokoroPhonemes, configuration: KokoroSpeechConfiguration, maximumFrames: Int,
        framesPredicted: @Sendable ([Int]) throws -> Void
    ) async throws -> KokoroSpeech {
        guard phonemes.tokenIDs.allSatisfy({ $0 == KokoroVocabulary.boundaryToken || assets.vocabulary.tokens.values.contains($0) }) else {
            throw AetherError.invalidRequest("phonemes were parsed with a different vocabulary")
        }
        let style = try voiceStyles(configuration.voice).style(for: phonemes)
        let tokens = phonemes.tokenIDs.count
        let durationsModel = try await model(isDecoder: false, assets.durations, contract: KokoroContracts.durations)
        try Task.checkCancellation()
        let ids = try Int32Tensor(shape: [1, tokens], values: phonemes.tokenIDs)
        let predicted = try await durationsModel.runTensors(function: "durations", inputs: ["input_ids": .int32(ids), "style": .float32(style)])
        guard predicted.count == 3 else { throw AetherError.invalidTensor("unexpected Kokoro durations outputs") }
        let sums = try KokoroContracts.float(predicted, "duration", shape: [1, tokens])
        let prosody = try KokoroContracts.float(predicted, "prosody", shape: [1, tokens, KokoroContracts.prosodyChannels])
        let text = try KokoroContracts.float(predicted, "text", shape: [1, KokoroContracts.textChannels, tokens])
        let frames = try KokoroAlignment.frames(durationSums: sums.values, speed: configuration.speed, maximumFrames: maximumFrames)
        try framesPredicted(frames)
        try Task.checkCancellation()
        let total = frames.reduce(0, +)
        let aligned = try KokoroAlignment.align(prosody: prosody, text: text, frames: frames)
        let noise = try FloatTensor(shape: [1, total, KokoroAssets.samplesPerFrame, KokoroAssets.harmonics],
                                    values: KokoroNoise.generate(seed: configuration.seed,
                                                                 count: total * KokoroAssets.samplesPerFrame * KokoroAssets.harmonics))
        let decoderModel = try await model(isDecoder: true, assets.decoder, contract: KokoroContracts.decoder)
        try Task.checkCancellation()
        let rendered = try await decoderModel.runTensors(function: "decode", inputs: [
            "prosody": .float32(aligned.prosody), "text": .float32(aligned.text),
            "style": .float32(style), "noise": .float32(noise),
        ])
        guard rendered.count == 1 else { throw AetherError.invalidTensor("unexpected Kokoro decoder outputs") }
        let audio = try KokoroContracts.float(rendered, "audio", shape: [1, total * KokoroAssets.samplesPerFrame])
        try Task.checkCancellation()
        return KokoroSpeech(audio: try GeneratedAudio(mono: audio.values, sampleRate: KokoroAssets.sampleRate),
                            frames: frames, phonemes: phonemes, configuration: configuration)
    }

    private func voiceStyles(_ voice: KokoroVoice) throws -> KokoroVoiceStyles {
        if let cached = styles[voice] { return cached }
        let loaded = try assets.styles(for: voice)
        if styles.count >= 8 { styles.removeAll() }  // ~0.5 MB each
        styles[voice] = loaded
        return loaded
    }

    private func model(
        isDecoder: Bool, _ asset: KokoroModelAsset,
        contract: (ModelFunctionDescription) throws -> Void
    ) async throws -> any KokoroExecutable {
        if let loaded = isDecoder ? decoder : durations { return loaded }
        try Task.checkCancellation()
        let opened = try await factory.open(asset)
        do {
            guard let function = opened.functions.first(where: { $0.name == asset.function }) else {
                throw AetherError.invalidModel("missing Kokoro function '\(asset.function)'")
            }
            try contract(function)
            try Task.checkCancellation()
        } catch {
            await opened.unload()
            throw error
        }
        if isDecoder { decoder = opened } else { durations = opened }
        return opened
    }

    func unload() async {
        if let durations { await durations.unload() }
        if let decoder { await decoder.unload() }
        durations = nil
        decoder = nil
        styles = [:]
        await factory.finish()
    }
}
