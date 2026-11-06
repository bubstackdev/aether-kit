import AetherCore
import AetherRuntime
import Foundation
import Testing
@testable import AetherVideo

/// The MiniMax H3 gates (DESIGN.md §12), on the real export (`aether/forge-outputs/minimax-h3`, or AETHER_H3_EXPORT;
/// not publicly hosted: see docs/INTEGRATION.md for the license). Off by default; run with
/// `AETHER_GATES=1 swift test --no-parallel --filter MiniMaxH3GateTests`. What each gates:
///
/// - **Host recipe against an independent one:** the driver's first steps against `Forge/h3/reference_steps_h3.py`,
///   which runs the same bundle graphs on the GPU (the path `check_h3.py` holds to the original modules on the CPU at
///   0.02 relative L2 per graph) from Diffusers' own layout, row timesteps and rotary, the unpadded modulation and Kit's
///   noise. Prompt ids exact; the tap, each step's head velocities and advanced latents within 0.02 relative L2 (the
///   0.0.x per-graph limit).
/// - **A whole clip:** the qualified profile, every frame and the soundtrack finite and of the right shape, a repeat
///   bit-identical, and the clip saved for a look (`driver-outputs/`: frames 0, 62, 123 as PNG, the WAV).
/// - **Memory:** each phase's measured Metal peak (the ledger) plus 0.0.x's 256 MiB sampling margin within its 18 GiB
///   bar for this region (960×544, 124 frames).
@Suite("MiniMax H3 gates", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct MiniMaxH3GateTests {
    /// AETHER_H3_EXPORT, else `forge-outputs/minimax-h3` in the nearest directory above this checkout that has one (a
    /// worktree under `.worktrees/` sits one level deeper than the main checkout).
    static let export: URL = {
        if let path = ProcessInfo.processInfo.environment["AETHER_H3_EXPORT"] { return URL(filePath: path) }
        var directory = URL(filePath: #filePath).deletingLastPathComponent()
        while directory.path != "/" {
            let candidate = directory.appending(path: "forge-outputs/minimax-h3")
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            directory = directory.deletingLastPathComponent()
        }
        return URL(filePath: "forge-outputs/minimax-h3")
    }()
    static let mac = Device(platform: .macOS, architecture: Device.current.architecture)
    static let limit = 0.02
    static let barBytes: Int64 = 18 << 30, marginBytes: Int64 = 256 << 20

    struct Spec: Decodable { var prompt: String; var seed: UInt64; var steps: Int; var ids: [Int32]; var profile: String }

    static func pipeline(ledger: FootprintLedger = FootprintLedger()) async throws -> MiniMaxH3Pipeline {
        let url = export.appending(path: "bundle")
        let manifest = try Manifest.load(fromBundle: url)
        let ref = ModelRef(
            catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: manifest.variant.platform.rawValue,
            driver: manifest.driver, assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
        return try await MiniMaxH3Pipeline(bundle: InstalledBundle(ref: ref, url: url, isStale: false), manifest: manifest,
                                           device: mac, options: .init(ledger: ledger))
    }

    static func relativeL2(_ a: [Float], _ b: [Float]) -> Double {
        var d = 0.0, n = 0.0
        for i in a.indices { let x = Double(a[i]), y = Double(b[i]); d += (x - y) * (x - y); n += y * y }
        return (d / max(n, 1e-30)).squareRoot()
    }

    static func floats(_ url: URL) throws -> [Float] {
        try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func bf16(_ bytes: [UInt8]) -> [Float] {
        bytes.withUnsafeBytes { raw in raw.bindMemory(to: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) } }
    }

    @Test func driverMatchesTheGraphReference() async throws {
        let reference = Self.export.appending(path: "driver-reference")
        let spec = try JSONDecoder().decode(Spec.self, from: Data(contentsOf: reference.appending(path: "spec.json")))
        let pipeline = try await Self.pipeline()
        let trace = MiniMaxH3Pipeline.Trace()
        let start = ContinuousClock.now
        _ = try await pipeline.generate(VideoRequest(prompt: spec.prompt, seed: spec.seed, profile: spec.profile),
                                        onProgress: nil, stepLimit: spec.steps, trace: trace)
        print("gate: h3 \(spec.steps) step(s) through the driver in \(ContinuousClock.now - start)")
        #expect(trace.ids == spec.ids, "prompt ids differ")
        let tap = Self.relativeL2(Self.bf16(trace.tap), Self.bf16([UInt8](try Data(contentsOf: reference.appending(path: "tap.raw")))))
        print("gate: h3 tap rel L2 \(tap)")
        #expect(tap <= Self.limit)
        for step in 0..<spec.steps {
            func check(_ name: String, _ actual: [Float]) throws {
                let expected = try Self.floats(reference.appending(path: "step\(step)-\(name).raw"))
                #expect(actual.count == expected.count, "step \(step) \(name): \(actual.count) values, not \(expected.count)")
                let value = Self.relativeL2(actual, expected)
                print("gate: h3 step \(step) \(name) rel L2 \(value)")
                #expect(value <= Self.limit, "step \(step) \(name) rel L2 \(value)")
            }
            try check("velocity-video", trace.velocities[step].video)
            try check("velocity-audio", trace.velocities[step].audio)
            try check("latents-video", trace.latents[step].video)
            try check("latents-audio", trace.latents[step].audio)
        }
    }

    @Test func wholeClipRepeatsAndFitsItsBar() async throws {
        let ledger = FootprintLedger()
        let pipeline = try await Self.pipeline(ledger: ledger)
        let request = VideoRequest(prompt: "A red fox trots through a snowy birch forest at dawn, its breath steaming; soft wind and distant birdsong.",
                                   seed: 20261106)
        let cold = ContinuousClock.now
        let clip = try await pipeline.generate(request)
        let coldTime = ContinuousClock.now - cold
        #expect(clip.frames.count == 124 && clip.width == 960 && clip.height == 544)
        #expect(clip.frames.allSatisfy { $0.count == 960 * 544 * 3 })
        let audio = try #require(clip.audio)
        #expect(audio.channelCount == 2 && audio.frameCount == 800 * 207 && audio.sampleRate == 32000)
        // Not a blank clip: frames vary across the image and over time.
        let mean = { (f: [UInt8]) in Double(f.reduce(0) { $0 + Int($1) }) / Double(f.count) }
        #expect(clip.frames[0] != clip.frames[123] && Set(clip.frames[62]).count > 64)
        print("gate: h3 clip cold \(coldTime); mean RGB frame 0 \(mean(clip.frames[0])), 123 \(mean(clip.frames[123])); "
              + "audio peak \(audio.channels.flatMap { $0 }.map(abs).max() ?? 0)")

        let outputs = Self.export.appending(path: "driver-outputs")
        try FileManager.default.createDirectory(at: outputs, withIntermediateDirectories: true)
        for index in [0, 62, 123] {
            let planar = (0..<3).flatMap { c in stride(from: c, to: clip.frames[index].count, by: 3).map { Float(clip.frames[index][$0]) / 255 * 2 - 1 } }
            try GeneratedImage(width: 960, height: 544, planarRGB: planar).pngData().write(to: outputs.appending(path: "frame-\(index).png"))
        }
        try audio.wavData().write(to: outputs.appending(path: "audio.wav"))

        // Memory: each phase's measured peak within the bar.
        for (key, footprint) in ledger.snapshot.sorted(by: { $0.key.description < $1.key.description }) {
            print("gate: h3 \(key): peak \(footprint.peakBytes >> 20) MiB, held \(footprint.heldBytes >> 20) MiB")
            #expect(footprint.peakBytes + Self.marginBytes <= Self.barBytes, "\(key) peaks at \(footprint.peakBytes >> 20) MiB")
        }

        let warm = ContinuousClock.now
        let again = try await pipeline.generate(request)
        print("gate: h3 clip warm \(ContinuousClock.now - warm)")
        #expect(again.frames == clip.frames, "a repeat differs")
        #expect(again.audio == clip.audio, "a repeat's audio differs")
    }
}
