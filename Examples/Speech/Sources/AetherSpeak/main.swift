import AetherKit
import AetherSpeech
import AetherSpeechCoreAI
import Foundation

// Usage: aether-speak BUNDLE OUTPUT.wav VOICE 'PHONEMES[|PHONEMES...]' [SPEED] [SEED]
// Phonemes are Kokoro's alphabet (G2P is the app's job); '|' separates segments, which are
// synthesized in order and written as each arrives, then joined into OUTPUT.wav.
let arguments = CommandLine.arguments
guard arguments.count >= 5 else {
    FileHandle.standardError.write(Data("usage: aether-speak BUNDLE OUTPUT.wav VOICE 'PHONEMES[|PHONEMES...]' [SPEED] [SEED]\n".utf8))
    exit(2)
}
let assets = try KokoroAssets.open(at: URL(fileURLWithPath: arguments[1]))
let output = URL(fileURLWithPath: arguments[2])
let configuration = try KokoroSpeechConfiguration(
    voice: KokoroVoice(arguments[3]),
    speed: arguments.count > 5 ? Float(arguments[5]) ?? 1 : 1,
    seed: arguments.count > 6 ? UInt64(arguments[6]) ?? 0 : 0)
let session = KokoroSpeechSession(assets: assets)
do {
    let segments = try arguments[4].split(separator: "|").map { try session.phonemes(String($0)) }
    let started = Date()
    let parts = try await session.synthesize(segments: segments, configuration: configuration) { event in
        if case .chunk(let chunk) = event {
            let elapsed = Date().timeIntervalSince(started)
            print(String(format: "segment %d/%d: %.2f s of audio, ready after %.2f s",
                         chunk.index + 1, chunk.count, chunk.speech.audio.durationSeconds, elapsed))
        }
    }
    let audio = try GeneratedAudio.concatenating(parts.map(\.audio))
    try audio.saveWAV(to: output)
    print(String(format: "wrote %.2f s at %d Hz to %@", audio.durationSeconds, audio.sampleRate, output.path))
    await session.unload()
} catch {
    await session.unload()
    throw error
}
