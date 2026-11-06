import AetherCore
import AetherRuntime
import CryptoKit
import Foundation
import Synchronization
import Testing
@testable import AetherSpeech

/// `kokoro@1`'s host recipe and session wiring, without the model (DESIGN.md §12): tokenization, the seeded noise and
/// torch.round's rounding against 0.0.x's recorded CPU known answers (`Fixtures/kokoro-cpu-fixture.json`, from
/// `Forge/kokoro/make_cpu_fixtures.py`), the alignment, and the session over fake graphs. The real graphs are the
/// gate's (`KokoroGateTests`).
@Suite("Kokoro")
struct KokoroTests {
    struct CPUFixture: Decodable {
        struct Tokenization: Decodable { var phonemes: String; var ids: [Int32] }
        struct Noise: Decodable { var seed: UInt64; var count: Int; var first: [Float]; var sha256: String }
        struct Durations: Decodable {
            struct Case: Decodable { var speed: Float; var frames: [Int] }
            var sums: [Float]
            var cases: [Case]
        }
        var vocabulary: [String: Int]
        var tokenization: [Tokenization]
        var noise: Noise
        var durations: Durations

        static func load() throws -> CPUFixture {
            let url = try #require(Bundle.module.url(forResource: "kokoro-cpu-fixture", withExtension: "json", subdirectory: "Fixtures"))
            return try JSONDecoder().decode(CPUFixture.self, from: Data(contentsOf: url))
        }
    }

    static let manifestURL = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "AetherCoreTests/Fixtures/kokoro/aether.json")
    static func manifest() throws -> Manifest { try Manifest.decode(Data(contentsOf: manifestURL)) }
    static let mac = Device(platform: .macOS, architecture: "test")
    static let phone = Device(platform: .iOS, architecture: "test")

    // MARK: - Host recipe

    @Test func phonemesTokenizeLikeUpstreamAndRefuseUnknownSymbols() throws {
        let fixture = try CPUFixture.load()
        let vocabulary = KokoroVocabulary(Manifest.Kokoro.Phonemes(vocabulary: fixture.vocabulary))
        #expect(KokoroVocabulary(try Self.manifest().kokoro!.phonemes) == vocabulary, "the bundle's vocabulary is upstream's")
        for item in fixture.tokenization {
            let phonemes = try KokoroPhonemes(item.phonemes, vocabulary: vocabulary)
            #expect(phonemes.tokenIDs == item.ids)
            #expect(phonemes.count == item.ids.count - 2 && phonemes.styleRow == phonemes.count - 1)
        }
        // Precomposed U+00E3 is not a Kokoro symbol (the vocabulary has a + U+0303), though Swift's String == says
        // they are equal: no normalization.
        #expect("\u{E3}" == "a\u{303}")
        #expect(throws: AetherError.self) { try KokoroPhonemes("\u{E3}", vocabulary: vocabulary) }
        #expect(throws: AetherError.self) { try KokoroPhonemes("", vocabulary: vocabulary) }
        #expect(throws: AetherError.self) { try KokoroPhonemes("hello 1", vocabulary: vocabulary) }
        _ = try KokoroPhonemes(String(repeating: "a", count: 510), vocabulary: vocabulary)
        #expect(throws: AetherError.self) { try KokoroPhonemes(String(repeating: "a", count: 511), vocabulary: vocabulary) }
    }

    @Test func noiseIsTheReferenceHarnessStream() throws {
        let fixture = try CPUFixture.load().noise
        let values = try SeededGaussianNoise.generate(seed: fixture.seed, count: fixture.count)
        #expect(Array(values.prefix(fixture.first.count)) == fixture.first)
        let digest = SHA256.hash(data: values.withUnsafeBufferPointer { Data(buffer: $0) })
        #expect(digest.map { String(format: "%02x", $0) }.joined() == fixture.sha256)
    }

    @Test func durationRoundingMatchesTorchRound() throws {
        let fixture = try CPUFixture.load().durations
        for item in fixture.cases {
            #expect(try KokoroAlignment.frames(durationSums: fixture.sums, speed: item.speed, maximumFrames: 10_000) == item.frames)
        }
        #expect(try KokoroAlignment.frames(durationSums: [0.4, 2.5, 3.5, 0], speed: 1, maximumFrames: 100) == [1, 2, 4, 1])
        #expect(throws: AetherError.self) { try KokoroAlignment.frames(durationSums: [3, 3], speed: 1, maximumFrames: 5) }
        #expect(throws: AetherError.self) { try KokoroAlignment.frames(durationSums: [.nan], speed: 1, maximumFrames: 5) }
        #expect(throws: AetherError.self) { try KokoroAlignment.frames(durationSums: [], speed: 1, maximumFrames: 5) }
        // Extreme finite durations throw instead of trapping in the Int conversion or the sum.
        for (sums, speed) in [([Float.greatestFiniteMagnitude], Float(1)), ([3e38], 0.25), ([9.3e18], 1), ([4e18, 4e18, 4e18], 1)] {
            #expect(throws: AetherError.self) { try KokoroAlignment.frames(durationSums: sums, speed: speed, maximumFrames: .max) }
        }
        #expect(try KokoroAlignment.frames(durationSums: [-Float.greatestFiniteMagnitude, 2], speed: 1, maximumFrames: .max) == [1, 2])
    }

    @Test func alignmentRepeatsTokenColumnsAndPadsToTheBucket() throws {
        // prosody [1, T=3, C=2]: 10·t + c; text [1, C=2, T=3]: 100·c + t; frames [1, 3, 2] into a bucket of 8.
        let aligned = try KokoroAlignment.align(
            prosody: [0, 1, 10, 11, 20, 21], prosodyChannels: 2, text: [0, 1, 2, 100, 101, 102], textChannels: 2,
            frames: [1, 3, 2], bucket: 8)
        #expect(aligned.prosody == [0, 10, 10, 10, 20, 20, 0, 0, 1, 11, 11, 11, 21, 21, 0, 0])
        #expect(aligned.text == [0, 1, 1, 1, 2, 2, 0, 0, 100, 101, 101, 101, 102, 102, 0, 0])
        #expect(throws: AetherError.self) {
            try KokoroAlignment.align(prosody: [0, 1], prosodyChannels: 2, text: [0, 1], textChannels: 2, frames: [9], bucket: 8)
        }
        #expect(throws: AetherError.self) {
            try KokoroAlignment.align(prosody: [0, 1], prosodyChannels: 2, text: [0, 1], textChannels: 2, frames: [0], bucket: 8)
        }
    }

    @Test func voicesParseTheirLanguageAndGender() throws {
        let voice = try KokoroVoice("bm_george")
        #expect(voice == .bmGeorge && voice.language == .britishEnglish && voice.gender == .male && voice.language.bcp47 == "en-GB")
        for bad in ["", "af", "xf_heart", "ax_heart", "afheart", "af_Heart", "af_he art"] {
            #expect(throws: AetherError.self) { try KokoroVoice(bad) }
        }
        let encoded = try JSONEncoder().encode(KokoroVoice.afHeart)
        #expect(String(decoding: encoded, as: UTF8.self) == "\"af_heart\"")
        #expect(try JSONDecoder().decode(KokoroVoice.self, from: encoded) == .afHeart)
        // Every voice the bundle carries parses, in the language its manifest entry names.
        let kokoro = try Self.manifest().kokoro!
        for (id, entry) in kokoro.voices { #expect(try KokoroVoice(id).language.bcp47 == entry.language, "\(id)") }
    }

    // MARK: - The session over fake graphs

    final class Log: Sendable {
        private let state = Mutex<[String]>([])
        var entries: [String] { state.withLock { $0 } }
        func append(_ entry: String) { state.withLock { $0.append(entry) } }
    }

    /// What the fake decoder saw, per call.
    struct Seen: Sendable {
        var bucket: Int, frames: Int, noise: Float, style: Float, prosody: [Float], textColumn0: Float
    }

    final class Observed: Sendable {
        private let state = Mutex<[Seen]>([])
        var all: [Seen] { state.withLock { $0 } }
        func append(_ seen: Seen) { state.withLock { $0.append(seen) } }
    }

    /// Durations: `frames` per token (as the duration sum), prosody[t, c] = t, text[c, t] = 100·c + t.
    final class FakeDurations: GraphBackend {
        let signature = GraphSignature(
            function: "durations",
            inputs: [TensorSpec(name: "input_ids", scalarType: .int32, shape: [1, -1]),
                     TensorSpec(name: "style", scalarType: .float32, shape: [1, 256])], states: [],
            outputs: [TensorSpec(name: "duration", scalarType: .float32, shape: [1, -1]),
                      TensorSpec(name: "prosody", scalarType: .float32, shape: [1, -1, 640]),
                      TensorSpec(name: "text", scalarType: .float32, shape: [1, 512, -1])])
        let log: Log
        let perToken: Float
        let gate: Gate?
        init(_ log: Log, perToken: Float, gate: Gate?) {
            self.log = log
            self.perToken = perToken
            self.gate = gate
        }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            log.append("durations")
            await gate?.enter()
            let t = bindings.input("input_ids")!.shape[1]
            try bindings.output("duration")!.write([Float](repeating: perToken, count: t))
            try bindings.output("prosody")!.write((0..<t).flatMap { token in [Float](repeating: Float(token), count: 640) })
            try bindings.output("text")!.write((0..<512).flatMap { c in (0..<t).map { Float(100 * c + $0) } })
        }
    }

    /// Decoder: records its inputs; audio is 0.5 for valid frames and 9 in the padded tail.
    final class FakeDecoder: GraphBackend {
        let signature = GraphSignature(
            function: "decode",
            inputs: [TensorSpec(name: "prosody", scalarType: .float32, shape: [1, 640, -1]),
                     TensorSpec(name: "text", scalarType: .float32, shape: [1, 512, -1]),
                     TensorSpec(name: "style", scalarType: .float32, shape: [1, 256]),
                     TensorSpec(name: "noise", scalarType: .float32, shape: [1, -1, 600, 9]),
                     TensorSpec(name: "frames", scalarType: .int32, shape: [1])], states: [],
            outputs: [TensorSpec(name: "audio", scalarType: .float32, shape: [1, -1])])
        let log: Log
        let observed: Observed
        init(_ log: Log, _ observed: Observed) {
            self.log = log
            self.observed = observed
        }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            let prosody = bindings.input("prosody")!
            let bucket = prosody.shape[2]
            let frames = Int(try bindings.input("frames")!.read(Int32.self)[0])
            log.append("decode \(bucket)")
            observed.append(Seen(
                bucket: bucket, frames: frames, noise: try bindings.input("noise")!.read(Float.self, 0..<1)[0],
                style: try bindings.input("style")!.read(Float.self, 0..<1)[0],
                prosody: try prosody.read(Float.self, 0..<bucket), textColumn0: try bindings.input("text")!.read(Float.self, 0..<1)[0]))
            try bindings.output("audio")!.write((0..<(bucket * 600)).map { $0 < frames * 600 ? Float(0.5) : 9 })
        }
    }

    /// Holds the durations call until released.
    actor Gate {
        private var entered = false
        private var waiting: CheckedContinuation<Void, Never>?
        private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
        private var released = false
        func enter() async {
            entered = true
            enteredWaiters.forEach { $0.resume() }
            enteredWaiters = []
            if !released { await withCheckedContinuation { waiting = $0 } }
        }
        func waitUntilEntered() async { if !entered { await withCheckedContinuation { enteredWaiters.append($0) } } }
        func release() {
            released = true
            waiting?.resume()
            waiting = nil
        }
    }

    /// Voice row r of every table is `1000·(voice index) + r`.
    static func session(
        log: Log, observed: Observed = Observed(), perToken: Float = 2, gate: Gate? = nil, device: Device = mac,
        frameCap: Int? = nil, budget: Int64 = 1 << 40
    ) throws -> KokoroSpeechSession {
        let manifest = try manifest()
        let index = Dictionary(uniqueKeysWithValues: manifest.kokoro!.voices.keys.sorted().enumerated().map { ($1, $0) })
        return try KokoroSpeechSession(
            manifest: manifest, device: device,
            options: .init(memory: .bytes(budget), probe: GPUMemoryProbe { 0 }, frameCap: frameCap),
            load: { role throws(AetherError) in
                log.append("load \(role)")
                return role == "durations" ? FakeDurations(log, perToken: perToken, gate: gate) : FakeDecoder(log, observed)
            },
            readTable: { table throws(AetherError) in
                let voice = String(table.file.dropFirst("voices/".count).dropLast(".f32".count))
                let base = Float(1000 * index[voice]!)
                return (0..<(510 * 256)).map { base + Float($0 / 256) }
            },
            fileBytes: { _ in 10 })
    }

    @Test func synthesisAlignsFramesSeedsNoiseAndPadsToTheBucket() async throws {
        let log = Log(), observed = Observed()
        let session = try Self.session(log: log, observed: observed)
        #expect(session.frameCap == 1024 && session.voices.count == 54)
        let phonemes = try session.phonemes("həlˈO")
        let speech = try await session.synthesize(phonemes, SpeechRequest(voice: .bmGeorge, seed: 5))
        let tokens = phonemes.tokenIDs.count   // 7: two frames each, 14 frames → bucket 64
        #expect(speech.frames == Array(repeating: 2, count: tokens) && speech.bucket == 64)
        #expect(speech.audio.sampleRate == 24_000 && speech.audio.samples == [Float](repeating: 0.5, count: 14 * 600))
        let seen = try #require(observed.all.first)
        #expect(seen.bucket == 64 && seen.frames == 14)
        #expect(seen.noise == (try SeededGaussianNoise.generate(seed: 5, count: 1))[0])
        let george = Float(1000 * session.voices.firstIndex(of: .bmGeorge)!)
        #expect(seen.style == george + Float(phonemes.styleRow))
        // Prosody channel 0 over the bucket: token t repeated twice, then zero padding.
        #expect(seen.prosody == (0..<tokens).flatMap { [Float($0), Float($0)] } + [Float](repeating: 0, count: 64 - 14))
        #expect(log.entries == ["load durations", "durations", "load decoder", "decode 64"])

        // Several segments: seed + i, delivered in order, graphs stay loaded.
        let delivered = Log()
        let parts = try await session.synthesize(
            segments: try ["ðə", "kwˈɪk", "fˈɑks"].map { try session.phonemes($0) }, SpeechRequest(seed: 100)
        ) { index, speech in delivered.append("\(index):\(speech.phonemes)") }
        #expect(delivered.entries == ["0:ðə", "1:kwˈɪk", "2:fˈɑks"])
        #expect(parts.map(\.request.seed) == [100, 101, 102])
        #expect(observed.all.dropFirst().map(\.noise) == (try [100, 101, 102].map { try SeededGaussianNoise.generate(seed: $0, count: 1)[0] }))
        #expect(log.entries.filter { $0.hasPrefix("load") }.count == 2)
        await session.unload()
        #expect(await session.residentRoles.isEmpty)
    }

    @Test func requestsAreRefusedBeforeAnyWork() async throws {
        let log = Log()
        let session = try Self.session(log: log)
        let phonemes = try session.phonemes("ðə")
        for speed: Float in [0, 0.2, 4.5, .nan, .infinity] {
            await #expect(throws: AetherError.self) { try await session.synthesize(phonemes, SpeechRequest(speed: speed)) }
        }
        await #expect(throws: AetherError.self) { try await session.synthesize(segments: []) }
        // A voice this bundle doesn't carry, and phonemes parsed with another vocabulary.
        let other = KokoroVocabulary(Manifest.Kokoro.Phonemes(vocabulary: ["ð": 1, "ə": 2]))
        await #expect(throws: AetherError.self) { try await session.synthesize(try KokoroPhonemes("ðə", vocabulary: other)) }
        #expect(log.entries.isEmpty)
        // Not enough memory for the graphs' files: refused before anything loads.
        let small = try Self.session(log: log, budget: 5)
        await #expect(throws: AetherError.self) { try await small.synthesize(phonemes) }
        #expect(log.entries.isEmpty)
    }

    @Test func theFrameCapRefusesBeforeDecoding() async throws {
        // 7 tokens × 40 frames = 280 frames: past a 256-frame cap, within the platform's.
        let log = Log()
        let capped = try Self.session(log: log, perToken: 40, frameCap: 256)
        let phonemes = try capped.phonemes("həlˈO")
        await #expect(throws: AetherError.self) { try await capped.synthesize(phonemes) }
        #expect(!log.entries.contains { $0.hasPrefix("decode") })
        let wide = try Self.session(log: Log(), perToken: 40)
        #expect(try await wide.synthesize(phonemes).bucket == 512)
        // iOS defaults to the manifest's 512-frame cap.
        let phone = try Self.session(log: Log(), perToken: 80, device: Self.phone)
        #expect(phone.frameCap == 512)
        await #expect(throws: AetherError.self) { try await phone.synthesize(phonemes) }
        #expect(throws: AetherError.self) { try Self.session(log: Log(), frameCap: 5000) }
    }

    @Test func cancellationStopsBetweenCallsAndTheSessionRecovers() async throws {
        let log = Log(), gate = Gate()
        let session = try Self.session(log: log, gate: gate)
        let phonemes = try session.phonemes("ðə")
        let task = Task { try await session.synthesize(phonemes) }
        await gate.waitUntilEntered()
        // A second request waits its turn (no busy error) and runs once the first ends.
        let queued = Task { try await session.synthesize(phonemes) }
        task.cancel()
        await gate.release()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(try await queued.value.frames == [2, 2, 2, 2])
        #expect(log.entries.filter { $0 == "decode 64" }.count == 1, "\(log.entries)")
    }

    @Test func prewarmRunsEveryBucketUpToTheCap() async throws {
        let log = Log()
        let session = try Self.session(log: log)
        #expect(try await session.prewarm(maximumFrames: 300) == [64, 128, 256, 512])
        #expect(log.entries.filter { $0.hasPrefix("decode") } == ["decode 64", "decode 128", "decode 256", "decode 512"])
        #expect(try await session.prewarm() == [64, 128, 256, 512, 1024])
    }

    @Test func textGoesThroughTheAppsPhonemizer() async throws {
        struct Fixed: SpeechPhonemizer {
            func phonemize(_ text: String, language: KokoroVoice.Language) async throws -> [String] {
                language == .britishEnglish ? ["ðə", "kwˈɪk"] : []
            }
        }
        let session = try Self.session(log: Log())
        let parts = try await session.synthesize(text: "the quick", phonemizer: Fixed(), SpeechRequest(voice: .bfEmma))
        #expect(parts.map(\.phonemes.string) == ["ðə", "kwˈɪk"])
    }
}
