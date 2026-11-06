import AetherCore
import AetherRuntime
import Foundation
import Testing
@testable import AetherSpeech

/// The Kokoro gate (DESIGN.md §12): the exported `kokoro@1` bundle, run through the session on its manifest placements
/// (durations on the CPU, the bucketed decoder on the GPU), against the unmodified upstream model on the CPU in float32
/// (`Forge/kokoro/reference_kokoro.py`, which seeds the harmonic noise from Kit's stream). Off by default; run with
/// `AETHER_GATES=1 swift test --no-parallel --filter KokoroGateTests`.
///
/// Bars (0.0.x's, declared before its first native run; `Forge/kokoro/check_kokoro.py` holds the Python side):
/// - token ids and integer frames per token equal upstream's exactly;
/// - waveform relative L2 ≤ min(0.05, 4 × the case's own float32-versus-float64 upstream difference);
/// - log-mel mean absolute difference ≤ 0.03 nats;
/// - a repeat synthesis bit-identical.
///
/// The bundle and references default to the Forge outputs beside this repo (`aether/forge-outputs/kokoro-82m/bundle`
/// and `…/reference`), or `AETHER_KOKORO_BUNDLE` and `AETHER_KOKORO_REFERENCE`. `AETHER_GATE_DUMP=<dir>` also writes each
/// waveform as raw float32 (Tools/DeviceGate compares).
@Suite("Kokoro gates", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct KokoroGateTests {
    static let environment = ProcessInfo.processInfo.environment
    static let outputs = forgeOutputs().appending(path: "kokoro-82m")
    static let bundleURL = environment["AETHER_KOKORO_BUNDLE"].map { URL(filePath: $0) } ?? outputs.appending(path: "bundle")
    static let referenceURL = environment["AETHER_KOKORO_REFERENCE"].map { URL(filePath: $0) } ?? outputs.appending(path: "reference")
    static let mac = Device(platform: .macOS, architecture: Device.current.architecture)

    struct Index: Decodable { struct Entry: Decodable { var name: String }; var cases: [Entry] }
    struct Case: Decodable {
        struct Floor: Decodable { var rel_l2: Double }
        var name: String
        var voice: String
        var speed: Float
        var seed: UInt64
        var phonemes: String
        var input_ids: [Int32]
        var frames: [Int]
        var float64_upstream: Floor
    }

    /// A Forge export is laid out as an installed revision root; until it is catalogued its ref is local.
    static func localBundle(_ url: URL) throws -> InstalledBundle {
        let manifest = try Manifest.load(fromBundle: url)
        let ref = ModelRef(
            catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: manifest.variant.platform.rawValue,
            driver: manifest.driver, assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
        return InstalledBundle(ref: ref, url: url, isStale: false)
    }

    @Test func kokoroMatchesUpstream() async throws {
        let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: Self.referenceURL.appending(path: "references.json")))
        let session = try await Registry.speech.load(
            try Self.localBundle(Self.bundleURL), as: KokoroDriver.self,
            options: LoadOptions(device: Self.mac, cachePolicy: .default, cache: .default)).session()
        #expect(session.voices.count == 54 && session.frameCap == 1024)
        for entry in index.cases {
            let directory = Self.referenceURL.appending(path: entry.name)
            let c = try JSONDecoder().decode(Case.self, from: Data(contentsOf: directory.appending(path: "case.json")))
            let expected = try Data(contentsOf: directory.appending(path: "audio.f32")).withUnsafeBytes { raw in
                (0..<(raw.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
            }
            let phonemes = try session.phonemes(c.phonemes)
            #expect(phonemes.tokenIDs == c.input_ids, "\(c.name): token ids")
            let request = SpeechRequest(voice: try KokoroVoice(c.voice), speed: c.speed, seed: c.seed)
            let started = ContinuousClock.now
            let speech = try await session.synthesize(phonemes, request)
            let elapsed = ContinuousClock.now - started
            #expect(speech.frames == c.frames, "\(c.name): frames per token")
            guard speech.audio.samples.count == expected.count else {
                Issue.record("\(c.name): \(speech.audio.samples.count) samples, expected \(expected.count)")
                continue
            }
            let bound = min(0.05, 4 * c.float64_upstream.rel_l2)
            let value = KokoroMetrics.compare(speech.audio.samples, expected)
            print("gate: kokoro \(c.name) (\(speech.frames.reduce(0, +)) frames → bucket \(speech.bucket)): rel L2 "
                  + "\(value.rel_l2) (bound \(bound)), log-mel mean \(value.log_mel_mean_abs), \(elapsed)")
            #expect(value.rel_l2 <= bound, "\(c.name): rel L2")
            #expect(value.log_mel_mean_abs <= 0.03, "\(c.name): log-mel mean")
            if let dump = Self.environment["AETHER_GATE_DUMP"] {   // raw float32, to compare with the device gate's
                try speech.audio.samples.withUnsafeBufferPointer { Data(buffer: $0) }
                    .write(to: URL(filePath: dump).appending(path: "kokoro-\(c.name).f32"))
            }
            let repeated = try await session.synthesize(phonemes, request)
            #expect(repeated.audio == speech.audio, "\(c.name): repeat bit-identical")
        }
        for (key, footprint) in session.options.ledger.snapshot {
            print("gate: kokoro \(key): peak \(footprint.peakBytes >> 20) MiB, held \(footprint.heldBytes >> 20) MiB")
        }
        await session.unload()
        #expect(await session.residentRoles.isEmpty)
    }
}

/// The Forge outputs directory beside the repo: the first `forge-outputs` found walking up from this file, so the gates
/// find it from a worktree under `.worktrees/` as from the main checkout (`AETHER_FORGE_OUTPUTS` overrides).
func forgeOutputs(from file: String = #filePath) -> URL {
    if let path = ProcessInfo.processInfo.environment["AETHER_FORGE_OUTPUTS"] { return URL(filePath: path) }
    var directory = URL(filePath: file).deletingLastPathComponent()
    while directory.path != "/" {
        let candidate = directory.appending(path: "forge-outputs")
        if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        directory = directory.deletingLastPathComponent()
    }
    return URL(filePath: file).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().appending(path: "forge-outputs")
}
