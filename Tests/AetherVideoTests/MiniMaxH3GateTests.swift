import AetherHost
import AetherCore
import AetherRuntime
import Foundation
import Testing
@_spi(Testing) @testable import AetherVideo

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
/// - **Keyframes (FL2VA):** a clip anchored to a rendered geometric keyframe at its first frame, its last frame, or
///   both: the presentation's labels and vision blocks in place, the anchored frame within ``keyframePSNR`` dB of its
///   keyframe (0.0.x measured 31.9–35.5 dB), every phase within the 18 GiB bar. The frames are saved for a look in
///   AETHER_H3_FL2VA_OUTPUTS (else `driver-outputs-fl2va/` beside the export).
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
    /// The anchored frame's PSNR against its keyframe, at least: below 0.0.x's lowest measured value (31.9 dB, the
    /// 960×544 first + last clip's first frame).
    static let keyframePSNR = 30.0

    struct Spec: Decodable { var prompt: String; var seed: UInt64; var steps: Int; var ids: [Int32]; var profile: String }

    /// The pipeline on the export's bundle. `policy` `.measuring` for what has no measured envelope on this export
    /// (its keyframed clips, before the H2 memory gate recorded them).
    static func pipeline(
        ledger: FootprintLedger = FootprintLedger(), policy: VideoAdmissionPolicy = .measuredOnly
    ) async throws -> MiniMaxH3Pipeline {
        let url = export.appending(path: "bundle")
        let manifest = try Manifest.load(fromBundle: url)
        let ref = ModelRef(
            catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: manifest.variant.platform.rawValue,
            driver: manifest.driver, assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
        return try await MiniMaxH3Pipeline(bundle: InstalledBundle(ref: ref, url: url, isStale: false), manifest: manifest,
                                           device: mac, options: .init(ledger: ledger), admissionPolicy: policy)
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

    /// The seeded footprints (``MiniMaxH3Pipeline/qualifiedFootprints``) are scoped to the export they were measured
    /// on; this is it. A new export fails here until it is measured and the digest updated.
    @Test func theQualifiedFootprintsNameThisExport() throws {
        let manifest = try Manifest.load(fromBundle: Self.export.appending(path: "bundle"))
        print("gate: h3 export digest \(manifest.contentDigest)")
        #expect(manifest.contentDigest == MiniMaxH3Pipeline.qualifiedExportDigest)
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
        #expect(audio.channelCount == 2 && audio.sampleCount == 800 * 207 && audio.sampleRate == 32000)
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

    // MARK: Keyframes

    /// A synthetic, non-graphic keyframe: a sky gradient over a checkered floor, a red disc, a yellow square and a blue
    /// triangle; `shift` moves the shapes (the last keyframe of a first + last clip).
    static func scene(width: Int, height: Int, shift: Int = 0) throws -> VideoKeyframe {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        let horizon = height * 3 / 5
        for y in 0..<height {
            for x in 0..<width {
                var c: (Int, Int, Int)
                if y < horizon {
                    let t = Double(y) / Double(horizon)
                    c = (Int(110 + 90 * t), Int(160 + 60 * t), 235)
                } else {
                    let checker = ((x / 48) + ((y - horizon) / 24)) % 2 == 0
                    c = checker ? (205, 200, 190) : (95, 90, 85)
                }
                let dx = x - (width / 3 + shift), dy = y - (horizon - 40)
                if dx * dx + dy * dy < 70 * 70 { c = (210, 40, 40) }
                if abs(x - (2 * width / 3 - shift)) < 55, abs(y - (horizon - 30)) < 55 { c = (235, 200, 40) }
                let ty = y - (horizon - 170), tx = x - (width / 2)
                if ty >= 0, ty < 120, abs(tx) * 2 < ty { c = (40, 80, 200) }
                let at = (y * width + x) * 3
                rgb[at] = UInt8(c.0); rgb[at + 1] = UInt8(c.1); rgb[at + 2] = UInt8(c.2)
            }
        }
        return try VideoKeyframe(rgb: rgb, width: width, height: height)
    }

    static func psnr(_ a: [UInt8], _ b: [UInt8]) -> Double {
        var squared = 0.0
        for i in a.indices { let d = Double(a[i]) - Double(b[i]); squared += d * d }
        let mse = squared / Double(a.count)
        return mse == 0 ? .infinity : 10 * log10(255 * 255 / mse)
    }

    static func png(_ rgb: [UInt8], width: Int, height: Int) throws -> Data {
        let planar = (0..<3).flatMap { c in stride(from: c, to: rgb.count, by: 3).map { Float(rgb[$0]) / 255 * 2 - 1 } }
        return try GeneratedImage(width: width, height: height, planarRGB: planar).pngData()
    }

    static let fl2vaOutputs: URL = ProcessInfo.processInfo.environment["AETHER_H3_FL2VA_OUTPUTS"].map { URL(filePath: $0) }
        ?? export.appending(path: "driver-outputs-fl2va")

    @Test(arguments: ["first", "last", "first+last"]) func anchoredClipMatchesItsKeyframes(mode: String) async throws {
        let ledger = FootprintLedger()
        let pipeline = try await Self.pipeline(ledger: ledger, policy: .measuring)
        let first = try Self.scene(width: 960, height: 544), last = try Self.scene(width: 960, height: 544, shift: 120)
        let keyframes: VideoKeyframes = switch mode {
        case "first": .first(first)
        case "last": .last(last)
        default: .firstAndLast(first, last)
        }
        let prompt = "A red ball rolls slowly across a checkered floor past a yellow block and a blue cone under a clear sky. "
            + "Soft rolling sounds and a quiet room tone."
        let trace = MiniMaxH3Pipeline.Trace()
        let start = ContinuousClock.now
        let clip = try await pipeline.generate(VideoRequest(prompt: prompt, keyframes: keyframes, seed: 20261108),
                                               onProgress: nil, stepLimit: nil, trace: trace)
        let elapsed = ContinuousClock.now - start
        #expect(clip.frames.count == 124 && clip.width == 960 && clip.height == 544)
        #expect(clip.frames.allSatisfy { $0.count == 960 * 544 * 3 })
        let audio = try #require(clip.audio)
        #expect(audio.channelCount == 2 && audio.sampleCount == 800 * 207)

        // The presentation: per keyframe the "<Picture i>: " label (the FL2VA tokenizer's ids), a 512-row vision
        // block tagged video, then the prompt; the condition rows lead the video latents.
        let count = keyframes.frames.count
        #expect(trace.conditioningIDs.count == count * 518 + trace.ids.count && Array(trace.conditioningIDs.suffix(trace.ids.count)) == trace.ids)
        for index in 0..<count {
            let at = index * 518
            #expect(Array(trace.conditioningIDs[at..<(at + 6)]) == [21604, 3826, 220, 16 + Int32(index), 26818, 220])
            #expect(trace.conditioningIDs[at + 6] == 151652 && trace.conditioningIDs[at + 517] == 151653)
            #expect(trace.tags[(at + 6)..<(at + 518)].allSatisfy { $0 == 0 } && trace.tags[at..<(at + 6)].allSatisfy { $0 == 1 })
        }
        #expect(trace.conditions.count == count * 510 * 96 && trace.conditions.allSatisfy(\.isFinite))

        // Adherence: the anchored frames against their keyframes.
        var report = "gate: h3 fl2va \(mode): \(elapsed); "
        if mode != "last" {
            let value = Self.psnr(clip.frames[0], first.rgb)
            report += "first frame vs keyframe \(String(format: "%.2f", value)) dB; "
            #expect(value >= Self.keyframePSNR, "first frame \(value) dB")
        }
        if mode != "first" {
            let value = Self.psnr(clip.frames[123], last.rgb)
            report += "last frame vs keyframe \(String(format: "%.2f", value)) dB; "
            #expect(value >= Self.keyframePSNR, "last frame \(value) dB")
        }
        report += "first vs last frame \(String(format: "%.2f", Self.psnr(clip.frames[0], clip.frames[123]))) dB"
        print(report)

        // Memory: each phase's measured peak within the bar.
        for (key, footprint) in ledger.snapshot.sorted(by: { $0.key.description < $1.key.description }) {
            print("gate: h3 fl2va \(mode) \(key): peak \(footprint.peakBytes >> 20) MiB, held \(footprint.heldBytes >> 20) MiB")
            #expect(footprint.peakBytes + Self.marginBytes <= Self.barBytes, "\(key) peaks at \(footprint.peakBytes >> 20) MiB")
        }

        let outputs = Self.fl2vaOutputs.appending(path: mode)
        try FileManager.default.createDirectory(at: outputs, withIntermediateDirectories: true)
        for (name, rgb) in [("keyframe-first", first.rgb), ("keyframe-last", last.rgb), ("frame-0", clip.frames[0]),
                            ("frame-62", clip.frames[62]), ("frame-123", clip.frames[123])] {
            try Self.png(rgb, width: 960, height: 544).write(to: outputs.appending(path: "\(name).png"))
        }
        try audio.wavData().write(to: outputs.appending(path: "audio.wav"))
    }
}
