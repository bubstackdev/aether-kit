public import AetherCore
public import AetherHost
public import AetherRuntime
import Foundation

/// `kokoro@1` (MANIFEST.md §5.9): Kokoro-82M speech from phonemes. Loads a ``KokoroModel``: set its options (a shared
/// coordinator, GPU admission, a ledger, the frame cap), then make the ``KokoroSpeechSession``.
public enum KokoroDriver: ModelDriver {
    public static let id: DriverID = .kokoro

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> KokoroModel {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        try DriverContract.kokoro.validate(manifest)
        return KokoroModel(
            bundle: bundle, manifest: manifest, device: options.device,
            options: .init(cachePolicy: options.cachePolicy, cache: options.cache))
    }
}

/// An opened Kokoro bundle. Nothing is loaded yet; ``options`` start from the `LoadOptions` cache settings.
public struct KokoroModel: Sendable {
    public let bundle: InstalledBundle
    public let manifest: Manifest
    public let device: Device
    public var options: KokoroSpeechSession.Options
    /// The longest utterance in frames; absent, the manifest's cap for the platform.
    public var frameCap: Int?

    public func session() throws -> KokoroSpeechSession {
        try KokoroSpeechSession(bundle: bundle, manifest: manifest, device: device, options: options, frameCap: frameCap)
    }
}

extension Registry {
    /// The drivers AetherSpeech links: `kokoro@1`.
    public static let speech = Registry([KokoroDriver.self])
}

/// One synthesis request's settings.
public struct SpeechRequest: Sendable, Hashable {
    public static let speedRange: ClosedRange<Float> = 0.25...4

    public var voice: KokoroVoice
    /// Divides the predicted durations before rounding, as upstream's `speed` does.
    public var speed: Float
    /// Seeds the vocoder's harmonic-source noise from Kit's ``SeededGaussianNoise`` (upstream draws it from PyTorch's
    /// global generator), so a seed always gives the same audio. Segment `i` of a multi-segment request uses
    /// `seed &+ i`.
    public var seed: UInt64
    /// The request's GPU priority (absent: `.medium`). Keep the job to change it while the request waits.
    public var job: GPUJob?

    public init(voice: KokoroVoice = .afHeart, speed: Float = 1, seed: UInt64 = 0, job: GPUJob? = nil) {
        self.voice = voice
        self.speed = speed
        self.seed = seed
        self.job = job
    }

    func validate() throws(AetherError) {
        guard speed.isFinite, Self.speedRange.contains(speed) else {
            throw .generation(.invalidParameters("the Kokoro speed must be in \(Self.speedRange), not \(speed)"))
        }
    }
}

/// One synthesized utterance.
public struct KokoroSpeech: Sendable, Hashable {
    /// Mono float32 PCM at the bundle's sample rate (24 kHz), unclipped.
    public let audio: GeneratedAudio
    /// Frames per token, the two boundary tokens included (600 samples, 25 ms, a frame).
    public let frames: [Int]
    /// The decoder bucket it ran at.
    public let bucket: Int
    public let phonemes: KokoroPhonemes
    public let request: SpeechRequest
}

/// One loaded Kokoro-82M (DESIGN.md §6.7): validated phonemes → the durations graph (CPU) → host rounding and
/// alignment → the bucketed decoder (GPU) with seeded harmonic-source noise → ``KokoroSpeech``.
///
/// - **Frames:** an utterance is refused before decoding when its frames pass the session's ``frameCap`` (the
///   manifest's `kokoro.frames.cap` for the platform unless the options set one: 1024 frames, 25.6 s, on macOS, 512 on
///   iOS); the decoder's memory grows faster than its bucket. Split longer input into segments.
/// - **Memory and the GPU:** both graphs stay loaded until ``unload()``. Each request is admitted against the measured
///   footprint (``FootprintLedger``, keyed by the cap's bucket) or the files' size, with the shared
///   `ResourceCoordinator` when the options set one, and through the options' GPU admission (by default
///   ``GPUAdmission/shared``), one grant per request.
/// - **One at a time:** requests run in GPU priority order (the request's `job`), in the order made within a priority;
///   ``unload()`` waits for the running one.
/// - **Cancellation** stops between graph calls with `CancellationError`. Any failure releases the graphs; the session
///   stays usable.
public actor KokoroSpeechSession {
    /// AetherHost's ``PipelineOptions``; absent `memory` means ``MemoryBudget/platformDefault`` here.
    public typealias Options = PipelineOptions

    /// The most segments one request takes; each segment's audio is kept in the result.
    public static let maximumSegments = 256

    /// The graphs' tensor names, from the manifest.
    struct Names: Sendable {
        var tokens: String, durationsStyle: String, durations: String, prosody: String, text: String
        var decoderProsody: String, decoderText: String, decoderStyle: String, noise: String, frames: String, audio: String
    }

    public nonisolated let manifest: Manifest
    public nonisolated let kokoro: Manifest.Kokoro
    public nonisolated let vocabulary: KokoroVocabulary
    /// The voices this bundle carries, sorted by identifier.
    public nonisolated let voices: [KokoroVoice]
    /// Utterances longer than this many frames are refused before decoding.
    public nonisolated let frameCap: Int
    public nonisolated let options: Options
    let names: Names
    let device: Device
    private let readTable: @Sendable (Manifest.Table) throws(AetherError) -> [Float]
    private let admission: ResidentAdmission
    private var styles: [KokoroVoice: [Float]] = [:]
    /// One request at a time, in arrival order (after the GPU grant).
    private let gate = SerialGate()

    /// Opens an installed bundle; the graphs load on the first request.
    /// `frameCap` is the longest utterance in frames; absent, the manifest's cap for the platform (at most the
    /// largest bucket).
    public init(
        bundle: InstalledBundle, manifest: Manifest, device: Device, options: Options = Options(), frameCap: Int? = nil
    ) throws(AetherError) {
        let loader = GraphResidency.bundleLoader(
            bundle: bundle, manifest: manifest, device: device, cachePolicy: options.cachePolicy, cache: options.cache)
        try self.init(
            manifest: manifest, device: device, options: options, frameCap: frameCap, bundle: bundle,
            load: Self.validating(loader, manifest: manifest),
            readTable: { table throws(AetherError) in try Self.readTable(table, in: bundle) },
            fileBytes: { role in GraphResidency.fileBytes(bundle: bundle, manifest: manifest, role: role, device: device) })
    }

    /// The session over any graph loader and table reader (tests drive it with fakes).
    init(
        manifest: Manifest, device: Device, options: Options, frameCap requested: Int? = nil, bundle: InstalledBundle? = nil,
        load: @escaping GraphResidency.Loader,
        readTable: @escaping @Sendable (Manifest.Table) throws(AetherError) -> [Float],
        fileBytes: @escaping @Sendable (String) -> Int64
    ) throws(AetherError) {
        try DriverContract.kokoro.validate(manifest)
        guard let kokoro = manifest.kokoro else {
            throw .manifestInvalid(field: "kokoro", reason: "\(KokoroDriver.id) requires the kokoro block")
        }
        let platformCap = kokoro.frames.cap[device.platform.rawValue] ?? kokoro.frames.maxFrames
        let frameCap = requested ?? platformCap
        guard (1...kokoro.frames.maxFrames).contains(frameCap) else {
            throw .generation(.invalidParameters("the frame cap must be in 1...\(kokoro.frames.maxFrames), not \(frameCap)"))
        }
        var voices: [KokoroVoice] = []
        for id in kokoro.voices.keys.sorted() {
            do {
                voices.append(try KokoroVoice(id))
            } catch {
                throw .manifestInvalid(field: "kokoro.voices.\(id)", reason: "not a Kokoro voice identifier")
            }
        }
        let durations = manifest.graphs["durations"]!, decoder = manifest.graphs["decoder"]!
        names = Names(
            tokens: durations.inputs["tokens"]!.name, durationsStyle: durations.inputs["style"]!.name,
            durations: durations.outputs["durations"]!.name, prosody: durations.outputs["prosody"]!.name,
            text: durations.outputs["text"]!.name,
            decoderProsody: decoder.inputs["prosody"]!.name, decoderText: decoder.inputs["text"]!.name,
            decoderStyle: decoder.inputs["style"]!.name, noise: decoder.inputs["noise"]!.name,
            frames: decoder.inputs["frames"]!.name, audio: decoder.outputs["audio"]!.name)
        self.manifest = manifest
        self.kokoro = kokoro
        vocabulary = try KokoroVocabulary(kokoro.phonemes)
        self.voices = voices
        self.frameCap = frameCap
        self.options = options
        self.device = device
        self.readTable = readTable
        // Kokoro's graphs have no residency: both stay loaded, and the session measures them under `perStep`.
        admission = try options.admission(
            manifest: manifest, device: device, defaultMemory: .platformDefault, unphasedPhase: .perStep, bundle: bundle,
            load: load, fileBytes: fileBytes)
    }

    /// Parses phonemes against this bundle's vocabulary.
    public nonisolated func phonemes(_ string: String) throws(AetherError) -> KokoroPhonemes {
        try KokoroPhonemes(string, vocabulary: vocabulary)
    }

    /// Synthesizes one utterance.
    public func synthesize(_ phonemes: KokoroPhonemes, _ request: SpeechRequest = SpeechRequest()) async throws -> KokoroSpeech {
        try await synthesize(segments: [phonemes], request)[0]
    }

    /// Synthesizes `segments` in order; `onSegment` receives each as soon as it is decoded (start playback after the
    /// first sentence). Segment `i` uses noise seed `request.seed &+ i`; join with `GeneratedAudio.concatenating`.
    public func synthesize(
        segments: [KokoroPhonemes], _ request: SpeechRequest = SpeechRequest(),
        onSegment: (@Sendable (Int, KokoroSpeech) -> Void)? = nil
    ) async throws -> [KokoroSpeech] {
        try Task.checkCancellation()
        // Validate the request before waiting for a turn or the GPU.
        guard !segments.isEmpty, segments.count <= Self.maximumSegments else {
            throw AetherError.generation(.invalidParameters("give 1...\(Self.maximumSegments) segments per request"))
        }
        try request.validate()
        guard let voice = kokoro.voices[request.voice.identifier] else {
            throw AetherError.generation(.invalidParameters("this bundle has no voice '\(request.voice)'"))
        }
        for phonemes in segments where (try? KokoroPhonemes(phonemes.string, vocabulary: vocabulary))?.tokenIDs != phonemes.tokenIDs {
            // Not just ids this vocabulary uses: another vocabulary can map the same ids to other symbols.
            throw AetherError.generation(.invalidParameters("'\(phonemes)' was parsed with another vocabulary"))
        }

        // The GPU first, so a request's priority also orders it among this session's own requests.
        let grant = try await options.gpu.admit(job: request.job)
        defer { grant.release() }
        return try await gate.withTurn {
            let residency = try admission.begin(profile: profile)
            do {
                let style = try styles(request.voice, table: voice.table)
                var results: [KokoroSpeech] = []
                for (index, phonemes) in segments.enumerated() {
                    try Task.checkCancellation()
                    var segment = request
                    segment.seed = request.seed &+ UInt64(index)
                    let speech = try await run(residency, phonemes, segment, style: style)
                    onSegment?(index, speech)
                    results.append(speech)
                }
                admission.finish()
                return results
            } catch {
                admission.release()
                throw error
            }
        }
    }

    /// Converts `text` with the app's phonemizer (in the voice's language), then synthesizes the segments it returns.
    public func synthesize(
        text: String, phonemizer: any SpeechPhonemizer, _ request: SpeechRequest = SpeechRequest(),
        onSegment: (@Sendable (Int, KokoroSpeech) -> Void)? = nil
    ) async throws -> [KokoroSpeech] {
        let strings = try await phonemizer.phonemize(text, language: request.voice.language)
        var segments: [KokoroPhonemes] = []
        for string in strings { segments.append(try phonemes(string)) }
        return try await synthesize(segments: segments, request, onSegment: onSegment)
    }

    /// Loads both graphs and runs the decoder once at every bucket up to `maximumFrames` (default and limit: the
    /// frame cap), so later utterances never wait for GPU specialization (each bucket keeps its specialization until
    /// ``unload()``). Returns the buckets warmed.
    @discardableResult
    public func prewarm(maximumFrames: Int? = nil) async throws -> [Int] {
        let limit = min(maximumFrames ?? frameCap, frameCap)
        guard limit >= 1, let top = kokoro.frames.bucket(for: limit) else {
            throw AetherError.generation(.invalidParameters("the prewarm frame limit must be in 1...\(frameCap)"))
        }
        let grant = try await options.gpu.admit()
        defer { grant.release() }
        return try await gate.withTurn {
            let residency = try admission.begin(profile: profile)
            do {
                try await residency.beginPhase(.perStep)
                _ = try await residency.graph("durations")
                let buckets = kokoro.frames.buckets.filter { $0 <= top }
                let style = [Float](repeating: 0, count: kokoro.style.dimension)
                for bucket in buckets {
                    try Task.checkCancellation()
                    _ = try await decode(residency, prosody: nil, text: nil, style: style, frames: bucket, bucket: bucket, seed: 0)
                }
                residency.endPhase(.perStep)
                admission.finish()
                return buckets
            } catch {
                admission.release()
                throw error
            }
        }
    }

    /// Releases both graphs and the cached voice rows (waiting for a running request). The next request loads them
    /// again.
    public func unload() async {
        await gate.withTurnIgnoringCancellation {
            admission.unload()
            styles = [:]
        }
    }

    /// Roles currently loaded (for tests and diagnostics).
    public var residentRoles: [String] { admission.residentRoles }

    // MARK: - One utterance

    private func run(
        _ residency: GraphResidency, _ phonemes: KokoroPhonemes, _ request: SpeechRequest, style table: [Float]
    ) async throws -> KokoroSpeech {
        let dimension = kokoro.style.dimension
        let style = Array(table[(phonemes.styleRow * dimension)..<((phonemes.styleRow + 1) * dimension)])
        let arena = TensorArena()
        let tokens = phonemes.tokenIDs.count

        try await residency.beginPhase(.perStep)
        let durations = try await residency.graph("durations")
        let ids = try durations.makeBuffer(for: names.tokens, role: .input, dimensions: [tokens], in: arena)
        let styleIn = try durations.makeBuffer(for: names.durationsStyle, role: .input, dimensions: [], in: arena)
        let sums = try durations.makeBuffer(for: names.durations, role: .output, dimensions: [tokens], in: arena)
        let prosody = try durations.makeBuffer(for: names.prosody, role: .output, dimensions: [tokens], in: arena)
        let text = try durations.makeBuffer(for: names.text, role: .output, dimensions: [tokens], in: arena)
        guard prosody.shape.count == 3, prosody.shape[1] == tokens, text.shape.count == 3, text.shape[2] == tokens else {
            throw AetherError.manifestMismatch(field: "graphs.durations.outputs", expected: "prosody [1, T, C] and text [1, C, T]",
                                               found: "\(prosody.shape) and \(text.shape)")
        }
        try ids.write(phonemes.tokenIDs)
        try styleIn.write(style)
        var bindings = Bindings(arena: arena)
        try bindings.bindInput(names.tokens, to: ids)
        try bindings.bindInput(names.durationsStyle, to: styleIn)
        try bindings.bindOutput(names.durations, to: sums)
        try bindings.bindOutput(names.prosody, to: prosody)
        try bindings.bindOutput(names.text, to: text)
        try await durations.run(&bindings)
        residency.sample()
        try Task.checkCancellation()

        let frames = try KokoroAlignment.frames(durationSums: try sums.readFloats(), speed: request.speed, maximumFrames: .max)
        let total = frames.reduce(0, +)
        guard total <= frameCap else {
            throw AetherError.generation(.invalidParameters(
                "the utterance needs \(total) frames, above this session's \(frameCap)-frame cap; split the phonemes into "
                + "shorter segments" + (frameCap < kokoro.frames.maxFrames ? " or raise Options.frameCap (more memory)" : "")))
        }
        guard let bucket = kokoro.frames.bucket(for: total) else {
            throw AetherError.generation(.invalidParameters("no decoder bucket holds \(total) frames"))
        }
        let aligned = try KokoroAlignment.align(
            prosody: try prosody.readFloats(), prosodyChannels: prosody.shape[2], text: try text.readFloats(),
            textChannels: text.shape[1], frames: frames, bucket: bucket)
        let samples = try await decode(
            residency, prosody: aligned.prosody, text: aligned.text, style: style, frames: total, bucket: bucket, seed: request.seed)
        residency.endPhase(.perStep)
        return KokoroSpeech(
            audio: try GeneratedAudio(mono: samples, sampleRate: kokoro.sampleRate), frames: frames, bucket: bucket,
            phonemes: phonemes, request: request)
    }

    /// Runs the decoder at `bucket` for `frames` valid frames (nil features: silence, for prewarming) and returns the
    /// valid prefix of the audio. The noise extends the same seeded stream to the bucket.
    private func decode(
        _ residency: GraphResidency, prosody: [Float]?, text: [Float]?, style: [Float], frames: Int, bucket: Int, seed: UInt64
    ) async throws -> [Float] {
        let decoder = try await residency.graph("decoder")
        let arena = TensorArena()
        let perFrame = kokoro.samplesPerFrame
        let prosodyIn = try decoder.makeBuffer(for: names.decoderProsody, role: .input, dimensions: [bucket], in: arena)
        let textIn = try decoder.makeBuffer(for: names.decoderText, role: .input, dimensions: [bucket], in: arena)
        let styleIn = try decoder.makeBuffer(for: names.decoderStyle, role: .input, dimensions: [], in: arena)
        let noise = try decoder.makeBuffer(for: names.noise, role: .input, dimensions: [bucket], in: arena)
        let count = try decoder.makeBuffer(for: names.frames, role: .input, dimensions: [], in: arena)
        let audio = try decoder.makeBuffer(for: names.audio, role: .output, dimensions: [bucket * perFrame], in: arena)
        guard noise.shape == [1, bucket, perFrame, kokoro.harmonics], audio.shape == [1, bucket * perFrame] else {
            throw AetherError.manifestMismatch(
                field: "graphs.decoder", expected: "noise [1, \(bucket), \(perFrame), \(kokoro.harmonics)], audio [1, \(bucket * perFrame)]",
                found: "noise \(noise.shape), audio \(audio.shape)")
        }
        if let prosody { try prosodyIn.write(prosody) } else { try prosodyIn.zero() }
        if let text { try textIn.write(text) } else { try textIn.zero() }
        try styleIn.write(style)
        try noise.write(try SeededGaussianNoise.generate(seed: seed, count: noise.count))
        try count.write([Int32(frames)])
        var bindings = Bindings(arena: arena)
        try bindings.bindInput(names.decoderProsody, to: prosodyIn)
        try bindings.bindInput(names.decoderText, to: textIn)
        try bindings.bindInput(names.decoderStyle, to: styleIn)
        try bindings.bindInput(names.noise, to: noise)
        try bindings.bindInput(names.frames, to: count)
        try bindings.bindOutput(names.audio, to: audio)
        try await decoder.run(&bindings)
        residency.sample()
        try Task.checkCancellation()
        return try audio.readFloats(0..<(frames * perFrame))
    }

    // MARK: - Admission, residency, voices

    /// Footprints are keyed by the bucket the cap allows: each bucket the decoder has run at keeps its memory.
    private var profile: String { "frames<=\(kokoro.frames.bucket(for: frameCap) ?? frameCap)" }

    private func styles(_ voice: KokoroVoice, table name: String) throws(AetherError) -> [Float] {
        if let cached = styles[voice] { return cached }
        guard let table = manifest.tables?[name] else {
            throw .manifestInvalid(field: "kokoro.voices.\(voice).table", reason: "no table '\(name)'")
        }
        let values = try readTable(table)
        guard values.count == kokoro.phonemes.maxPhonemes * kokoro.style.dimension, values.allSatisfy(\.isFinite) else {
            throw .tensor(name: name, reason: .unsupported("a voice table must hold finite float32 [maxPhonemes, dimension]"))
        }
        if styles.count >= 8 { styles.removeAll() }   // 0.5 MB each
        styles[voice] = values
        return values
    }

    /// A float32 little-endian table from the installed bundle (the store verified the file when it installed).
    static func readTable(_ table: Manifest.Table, in bundle: InstalledBundle) throws(AetherError) -> [Float] {
        guard table.dtype == .float32, let url = bundle.url(of: AssetPath(asset: table.asset, path: table.file)) else {
            throw .manifestInvalid(field: "tables", reason: "'\(table.file)' is not a float32 table in an installed asset")
        }
        let data: Data
        do { data = try Data(contentsOf: url, options: .mappedIfSafe) } catch { throw .runtime(underlying: error) }
        guard Int64(data.count) == table.byteCount else {
            throw .tensor(name: table.file, reason: .unsupported("expected \(table.byteCount ?? 0) bytes, found \(data.count)"))
        }
        return data.withUnsafeBytes { raw in
            (0..<(data.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
    }

    /// Wraps `loader` so each graph is checked against its manifest role and the `kokoro@1` contract before use.
    static func validating(_ loader: @escaping GraphResidency.Loader, manifest: Manifest) -> GraphResidency.Loader {
        GraphResidency.validating(loader, manifest: manifest, contract: .kokoro)
    }
}
