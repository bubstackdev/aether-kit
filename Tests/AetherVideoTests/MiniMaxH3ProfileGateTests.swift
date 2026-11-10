import AetherCore
import AetherHost
import AetherRuntime
import Foundation
import Synchronization
import Testing
@_spi(Testing) @testable import AetherVideo

/// One render of the H2 memory gate (every profile with measured admission), on the GPU. Off unless AETHER_GATES=1 and
/// AETHER_H3_GATE_RENDER names the render; `scripts/h3-memory-gate` runs the whole matrix, one fresh process a render
/// under the GPU lease, and `Forge/h3/record_envelopes_h3.py` turns the records into the manifest's envelopes.
///
/// - AETHER_H3_EXPORT: the export (`bundle/` inside it), as for ``MiniMaxH3GateTests``.
/// - AETHER_H3_GATE_RENDER: `<profile>:<recipe>:<task>`, the task `t2va`, `fl2va1` (a first keyframe) or `fl2va2`
///   (first and last), e.g. `960x544x243:turbo4:fl2va2`.
/// - AETHER_H3_GATE_RECORD: where the render's JSON record goes (schema `aether-h3-memory-render/1`).
///
/// The render runs under ``VideoAdmissionPolicy/measuring`` with a fresh ledger (nothing seeded), so each phase's
/// peak and held bytes are what this render measured (Metal, sampled by the probe). It passes when the clip has the
/// profile's frames and soundtrack and every phase's peak plus 0.0.x's 256 MiB sampling margin is within the profile's
/// bar. The record is written before the expectations, so a failed render is recorded as failed.
@Suite("MiniMax H3 profile memory gate", .serialized,
       .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"
                && ProcessInfo.processInfo.environment["AETHER_H3_GATE_RENDER"] != nil))
struct MiniMaxH3ProfileGateTests {
    static let marginBytes: Int64 = 256 << 20
    static let seed: UInt64 = 20261109
    static let textPrompt = "A red fox trots through a snowy birch forest at dawn, its breath steaming; soft wind and distant birdsong."
    static let keyframePrompt = "A red ball rolls slowly across a checkered floor past a yellow block and a blue cone under a clear sky. "
        + "Soft rolling sounds and a quiet room tone."

    struct Render: Sendable {
        var profile: String
        var recipe: String
        var task: String
        var keyframes: Int { task == "fl2va2" ? 2 : task == "fl2va1" ? 1 : 0 }

        init(parsing text: String) throws {
            let parts = text.split(separator: ":").map(String.init)
            guard parts.count == 3, ["t2va", "fl2va1", "fl2va2"].contains(parts[2]) else {
                throw AetherError.invalidInput("AETHER_H3_GATE_RENDER must be <profile>:<recipe>:<t2va|fl2va1|fl2va2>, not '\(text)'")
            }
            (profile, recipe, task) = (parts[0], parts[1], parts[2])
        }
    }

    final class Events: Sendable {
        let start = ContinuousClock.now
        let log = Mutex<[(String, Double)]>([])
        func note(_ progress: VideoProgress) {
            let seconds = MiniMaxH3ProfileGateTests.seconds(ContinuousClock.now - start)
            let name = switch progress {
            case .conditionerLayer: "layer"
            case .step: "step"
            case .window: "window"
            }
            log.withLock { $0.append((name, seconds)) }
        }
    }

    static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    @Test func oneRender() async throws {
        let environment = ProcessInfo.processInfo.environment
        let render = try Render(parsing: environment["AETHER_H3_GATE_RENDER"] ?? "")
        let output = try #require(environment["AETHER_H3_GATE_RECORD"].map { URL(filePath: $0) }, "AETHER_H3_GATE_RECORD is required")
        let ledger = FootprintLedger()
        let pipeline = try await MiniMaxH3GateTests.pipeline(ledger: ledger, policy: .measuring)
        let manifest = pipeline.manifest, device = MiniMaxH3GateTests.mac
        let profile = try #require(pipeline.h3.profile(render.profile), "the export has no profile \(render.profile)")
        let memory = try #require(profile.memory, "profile \(render.profile) has no memory bar (extend the manifest: Forge/h3/extend_h3.py)")
        let (width, height) = (profile.width, profile.height)

        let first = try MiniMaxH3GateTests.scene(width: width, height: height)
        let last = try MiniMaxH3GateTests.scene(width: width, height: height, shift: width / 8)
        let keyframes: VideoKeyframes? = switch render.keyframes {
        case 1: .first(first)
        case 2: .firstAndLast(first, last)
        default: nil
        }
        let request = VideoRequest(prompt: keyframes == nil ? Self.textPrompt : Self.keyframePrompt, keyframes: keyframes,
                                   seed: Self.seed, profile: render.profile, recipe: render.recipe)
        let key = render.keyframes == 0 ? profile.key : "\(profile.key)/fl2va\(render.keyframes)"
        let started = Date()
        let events = Events()
        var failures: [String] = []
        var clip: GeneratedVideo?
        do {
            clip = try await pipeline.generate(request, onProgress: { events.note($0) })
        } catch {
            failures.append("generation failed: \(error)")
        }
        let total = Self.seconds(ContinuousClock.now - events.start)
        await pipeline.unload()

        // Each phase's measurement for this render's admission key (the largest over specialization counts).
        let model = ResidentAdmission.modelKey(manifest: manifest, device: device)
        var phases: [String: [String: Int64]] = [:]
        for (entry, footprint) in ledger.snapshot where entry.model == model && entry.profile == key {
            let name = entry.phase.rawValue
            phases[name] = ["peakBytes": max(phases[name]?["peakBytes"] ?? 0, footprint.peakBytes),
                            "heldBytes": max(phases[name]?["heldBytes"] ?? 0, footprint.heldBytes)]
        }
        let stagePeak = phases.values.compactMap { $0["peakBytes"] }.max() ?? 0
        for phase in ["perPrompt", "perStep", "perImage"] where phases[phase] == nil { failures.append("\(phase) was not measured") }
        if stagePeak + Self.marginBytes > memory.barBytes {
            failures.append("peak \(stagePeak >> 20) MiB + margin is over the \(memory.barBytes >> 30) GiB bar")
        }
        var quality: [String: Double] = [:]
        if let clip {
            if clip.frames.count != profile.frames || clip.width != width || clip.height != height
                || !clip.frames.allSatisfy({ $0.count == width * height * 3 }) {
                failures.append("the clip is \(clip.frames.count) frames at \(clip.width)x\(clip.height), not the profile's")
            }
            if let audio = clip.audio {
                if audio.sampleCount != 800 * profile.audioLatents || audio.channels.contains(where: { $0.contains { !$0.isFinite } }) {
                    failures.append("the soundtrack has \(audio.sampleCount) samples or non-finite values")
                }
            } else {
                failures.append("no soundtrack")
            }
            if render.keyframes >= 1, let frame = clip.frames.first { quality["firstFramePSNR"] = MiniMaxH3GateTests.psnr(frame, first.rgb) }
            if render.keyframes == 2, let frame = clip.frames.last { quality["lastFramePSNR"] = MiniMaxH3GateTests.psnr(frame, last.rgb) }
        }

        // Progress arrives as each layer, step and window ends: the prompt phase ends with the last conditioner
        // layer, the denoise phase with the last step.
        let log = events.log.withLock { $0 }
        let steps = log.filter { $0.0 == "step" }.map(\.1)
        let promptEnd = log.last { $0.0 == "layer" }?.1 ?? 0, denoiseEnd = steps.last ?? promptEnd
        let stepSeconds = zip(steps, [promptEnd] + steps).map { $0 - $1 }
        let record: [String: Any] = [
            "schema": "aether-h3-memory-render/1",
            "render": ["profile": render.profile, "recipe": render.recipe, "task": render.task],
            "admissionKey": key,
            "export": manifest.contentDigest,
            "contentDigest": manifest.contentDigest,
            "architecture": device.architecture,
            "physicalMemoryBytes": device.physicalMemory,
            "barBytes": memory.barBytes,
            "minimumPhysicalMemoryBytes": memory.minimumPhysicalMemoryBytes,
            "marginBytes": Self.marginBytes,
            "phases": phases,
            "stagePeakBytes": stagePeak,
            "passed": failures.isEmpty,
            "failures": failures,
            "seconds": ["total": total, "prompt": promptEnd, "denoise": denoiseEnd - promptEnd, "decode": total - denoiseEnd],
            "stepSeconds": stepSeconds,
            "frames": clip?.frames.count ?? 0,
            "audioSamples": clip?.audio?.sampleCount ?? 0,
            "quality": quality,
            "seed": Self.seed,
            "startedAt": ISO8601DateFormatter().string(from: started),
        ]
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys]).write(to: output)
        print("gate: h3 \(environment["AETHER_H3_GATE_RENDER"] ?? "") \(key): \(String(format: "%.0f", total)) s, peak "
              + "\(stagePeak >> 20) MiB of \(memory.barBytes >> 20) MiB; " + (failures.isEmpty ? "passed" : failures.joined(separator: "; ")))
        #expect(failures.isEmpty, "\(failures)")
    }
}
