import AetherAssets
import AetherKit
import AetherSpeech
import CryptoKit
import Foundation
import Testing

@Test func voiceCatalogMatchesIdentifierParsing() throws {
    #expect(KokoroVoice.v1_0.count == 54)
    #expect(Set(KokoroVoice.v1_0).count == 54)
    for voice in KokoroVoice.v1_0 {
        let parsed = try KokoroVoice(voice.identifier)
        #expect(parsed == voice)
        #expect(parsed.language == voice.language && parsed.gender == voice.gender)
    }
    #expect(KokoroVoice.afHeart.language == .americanEnglish && KokoroVoice.afHeart.gender == .female)
    #expect(KokoroVoice.bmGeorge.language.bcp47 == "en-GB")
    for bad in ["", "af", "xf_heart", "ax_heart", "afheart", "af_Heart", "af_he art"] {
        #expect(throws: AetherError.self) { try KokoroVoice(bad) }
    }
    let encoded = try JSONEncoder().encode(KokoroVoice.zmYunxi)
    #expect(String(decoding: encoded, as: UTF8.self) == "\"zm_yunxi\"")
    #expect(try JSONDecoder().decode(KokoroVoice.self, from: encoded) == .zmYunxi)
}

@Test func phonemesTokenizeLikeUpstreamAndRefuseUnknownSymbols() throws {
    let fixture = try CPUFixture.load()
    let vocabulary = try KokoroVocabulary(fixture.vocabulary)
    #expect(vocabulary.tokens.count == fixture.vocabulary.count)
    for item in fixture.tokenization {
        let phonemes = try KokoroPhonemes(item.phonemes, vocabulary: vocabulary)
        #expect(phonemes.tokenIDs == item.ids)
        #expect(phonemes.count == item.ids.count - 2)
        #expect(phonemes.styleRow == phonemes.count - 1)
    }
    // Precomposed U+00E3 is not a Kokoro symbol; the vocabulary uses a + U+0303. No normalization.
    #expect(throws: AetherError.self) { try KokoroPhonemes("\u{E3}", vocabulary: vocabulary) }
    #expect(throws: AetherError.self) { try KokoroPhonemes("", vocabulary: vocabulary) }
    #expect(throws: AetherError.self) { try KokoroPhonemes("hello 1", vocabulary: vocabulary) }
    _ = try KokoroPhonemes(String(repeating: "a", count: 510), vocabulary: vocabulary)
    #expect(throws: AetherError.self) { try KokoroPhonemes(String(repeating: "a", count: 511), vocabulary: vocabulary) }
    #expect(throws: AetherError.self) { try KokoroVocabulary(["ab": 1]) }
    #expect(throws: AetherError.self) { try KokoroVocabulary(["a": 1, "b": 1]) }
    #expect(throws: AetherError.self) { try KokoroVocabulary(["a": 0]) }
}

@Test func noiseMatchesTheReferenceHarness() throws {
    let fixture = try CPUFixture.load().noise
    let values = try KokoroNoise.generate(seed: fixture.seed, count: fixture.count)
    #expect(Array(values.prefix(fixture.first.count)) == fixture.first)
    let bytes = values.withUnsafeBufferPointer { Data(buffer: $0) }
    #expect(sha256Hex(bytes) == fixture.sha256)
    #expect(try KokoroNoise.generate(seed: 1, count: 3) != KokoroNoise.generate(seed: 2, count: 3))
    #expect(try KokoroNoise.generate(seed: 9, count: 0).isEmpty)
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
}

@Test func alignmentGathersTokenColumnsIntoFrames() throws {
    // prosody [1,T=3,C=2]: value 10*t + c; text [1,C=2,T=3]: value 100*c + t
    let prosody = try FloatTensor(shape: [1, 3, 2], values: [0, 1, 10, 11, 20, 21])
    let text = try FloatTensor(shape: [1, 2, 3], values: [0, 1, 2, 100, 101, 102])
    let aligned = try KokoroAlignment.align(prosody: prosody, text: text, frames: [1, 3, 2])
    #expect(aligned.prosody.shape == [1, 2, 6])
    #expect(aligned.prosody.values == [0, 10, 10, 10, 20, 20, 1, 11, 11, 11, 21, 21])
    #expect(aligned.text.values == [0, 1, 1, 1, 2, 2, 100, 101, 101, 101, 102, 102])
    #expect(throws: AetherError.self) { try KokoroAlignment.align(prosody: prosody, text: text, frames: [1, 1]) }
    #expect(throws: AetherError.self) { try KokoroAlignment.align(prosody: prosody, text: text, frames: [1, 0, 1]) }
}

@Test func configurationValidatesSpeed() throws {
    _ = try KokoroSpeechConfiguration(voice: .afBella, speed: 0.25, seed: 3)
    for bad: Float in [0, 0.2, 4.5, .nan, .infinity, -1] {
        #expect(throws: AetherError.self) { try KokoroSpeechConfiguration(speed: bad) }
    }
}

@Test func generatedAudioWritesPCM16AndFloatWAV() throws {
    let audio = try GeneratedAudio(mono: [0, 0.5, -0.5, 1.5, -2, 0.25 / 32767], sampleRate: 24_000)
    #expect(audio.frameCount == 6 && audio.channelCount == 1 && audio.durationSeconds == 6.0 / 24_000)
    let pcm = try audio.wavData()
    #expect(pcm.count == 44 + 12)
    #expect(String(decoding: pcm[0..<4], as: UTF8.self) == "RIFF" && String(decoding: pcm[8..<16], as: UTF8.self) == "WAVEfmt ")
    func u32(_ d: Data, _ o: Int) -> UInt32 { d[o..<o + 4].enumerated().reduce(0) { $0 | UInt32($1.element) << (8 * $1.offset) } }
    func i16(_ d: Data, _ o: Int) -> Int16 { Int16(bitPattern: UInt16(d[o]) | UInt16(d[o + 1]) << 8) }
    #expect(u32(pcm, 4) == 48 && u32(pcm, 24) == 24_000 && u32(pcm, 28) == 48_000 && u32(pcm, 40) == 12)
    #expect((0..<6).map { i16(pcm, 44 + 2 * $0) } == [0, 16384, -16384, 32767, -32767, 0])
    let float = try audio.wavData(encoding: .float32)
    #expect(float.count == 44 + 24 && float[20] == 3 && float[34] == 32)
    #expect(Float(bitPattern: u32(float, 44 + 12)) == 1.5)

    let stereo = try GeneratedAudio(channels: [[0.1, 0.2], [0.3, 0.4]], sampleRate: 48_000)
    let joined = try GeneratedAudio.concatenating([stereo, stereo])
    #expect(joined.channels == [[0.1, 0.2, 0.1, 0.2], [0.3, 0.4, 0.3, 0.4]])
    #expect(throws: AetherError.self) { try GeneratedAudio.concatenating([stereo, audio]) }
    #expect(throws: AetherError.self) { try GeneratedAudio(mono: [], sampleRate: 24_000) }
    #expect(throws: AetherError.self) { try GeneratedAudio(mono: [.nan], sampleRate: 24_000) }
    #expect(throws: AetherError.self) { try GeneratedAudio(channels: [[1], [1, 2]], sampleRate: 24_000) }

    let url = FileManager.default.temporaryDirectory.appendingPathComponent("aether-\(UUID().uuidString).wav")
    defer { try? FileManager.default.removeItem(at: url) }
    try audio.saveWAV(to: url)
    #expect(try Data(contentsOf: url) == pcm)
}

@Test func assetsOpenVerifyAndLoadVoiceRows() throws {
    let bundle = try FakeBundle.make()
    defer { bundle.remove() }
    let assets = try KokoroAssets.open(at: bundle.root)
    #expect(assets.availableVoices == [.afHeart, .bmGeorge])
    #expect(assets.durations.function == "durations" && assets.decoder.function == "decode")
    #expect(assets.maximumFrames == 4096 && assets.source.license == "Apache-2.0")
    try assets.verify()
    let styles = try assets.styles(for: .bmGeorge)
    let phonemes = try KokoroPhonemes("həlˈO", vocabulary: assets.vocabulary)
    let style = try styles.style(for: phonemes)
    #expect(style.shape == [1, 256] && style.values.allSatisfy { $0 == Float(1000 + phonemes.count - 1) })
    #expect(throws: AetherError.self) { try assets.styles(for: .afBella) }
    #expect(try KokoroAssets.open(at: bundle.root).fingerprint == assets.fingerprint)
}

@Test func assetsRefuseCorruptionAndUnsafeContracts() throws {
    let bundle = try FakeBundle.make()
    defer { bundle.remove() }
    let assets = try KokoroAssets.open(at: bundle.root)
    try Data("tampered".utf8).write(to: bundle.root.appendingPathComponent("decoder.aimodel/main.mlirb"))
    #expect(throws: AetherError.self) { try assets.verify() }
    try Data("{}".utf8).write(to: bundle.root.appendingPathComponent("durations.aimodel/metadata.json"))
    #expect(throws: AetherError.self) { try KokoroAssets.open(at: bundle.root) }

    let voice = try FakeBundle.make()
    defer { voice.remove() }
    let opened = try KokoroAssets.open(at: voice.root)
    var bytes = try Data(contentsOf: voice.root.appendingPathComponent("voices/af_heart.f32"))
    bytes[0] ^= 1
    try bytes.write(to: voice.root.appendingPathComponent("voices/af_heart.f32"))
    #expect(throws: AetherError.self) { try opened.styles(for: .afHeart) }

    let mutations: [(inout [String: Any]) -> Void] = [
        { $0["schema"] = 2 },
        { $0["sampleRate"] = 22_050 },
        { $0["precision"] = "float16" },
        { $0["source"] = ["repository": "other/model", "revision": String(repeating: "a", count: 40),
                          "checkpoint_sha256": String(repeating: "a", count: 64), "license": "Apache-2.0"] },
        { contract in
            var models = contract["models"] as! [String: Any]
            var durations = models["durations"] as! [String: Any]
            durations["path"] = "../durations.aimodel"
            models["durations"] = durations
            contract["models"] = models
        },
        { contract in
            var voices = contract["voices"] as! [[String: Any]]
            voices[1]["file"] = voices[0]["file"]
            voices[1]["sha256"] = voices[0]["sha256"]
            contract["voices"] = voices
        },
        { contract in
            var voices = contract["voices"] as! [[String: Any]]
            voices[0]["language"] = "fr"
            contract["voices"] = voices
        },
    ]
    for mutate in mutations {
        let bad = try FakeBundle.make(mutate: mutate)
        defer { bad.remove() }
        #expect(throws: (any Error).self) { try KokoroAssets.open(at: bad.root) }
    }

    let escaping = try FakeBundle.make()
    defer { escaping.remove() }
    let outside = FileManager.default.temporaryDirectory.appendingPathComponent("kokoro-outside-\(UUID().uuidString).f32")
    try FileManager.default.moveItem(at: escaping.root.appendingPathComponent("voices/af_heart.f32"), to: outside)
    defer { try? FileManager.default.removeItem(at: outside) }
    try FileManager.default.createSymbolicLink(at: escaping.root.appendingPathComponent("voices/af_heart.f32"), withDestinationURL: outside)
    #expect(throws: AetherError.self) { try KokoroAssets.open(at: escaping.root) }
}

@Test func installationRoundTripsThroughTheLocalAssetStore() async throws {
    let bundle = try FakeBundle.make()
    defer { bundle.remove() }
    let storeRoot = FileManager.default.temporaryDirectory.appendingPathComponent("kokoro-store-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: storeRoot) }
    let assets = try KokoroAssets.open(at: bundle.root)
    let manifest = try assets.installationManifest()
    #expect(manifest.files.count == 1 + 6 + 2)
    #expect(manifest.executionContract == "kokoro.speech-bundle/1")
    let store = try LocalAssetStore(root: storeRoot)
    let id = try await store.install(manifest, from: bundle.root)
    let lease = try await store.acquire(id)
    let installed = try KokoroAssets.open(in: lease)
    #expect(installed.fingerprint == assets.fingerprint)
    #expect(installed.root.path.hasPrefix(lease.directory.standardizedFileURL.resolvingSymlinksInPath().path))
    try installed.verify()
}
