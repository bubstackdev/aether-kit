import AetherKit
import AetherH3
import AetherH3CoreAI
import Foundation
import Metal
import Synchronization

/// A 20 ms memory timeline with named phases, so a peak can be attributed to model load and specialization, a call's
/// scratch, or memory still held after release. Metal is `MTLDevice.currentAllocatedSize`; footprint is the
/// process's physical footprint. The two overlap and cannot be added.
final class H3MemoryTimeline: Sendable {
    struct Sample: Codable, Sendable { let t: Double; let phase: String; let metal: UInt64; let footprint: UInt64; let resident: UInt64 }
    private let state = Mutex((phase: "start", samples: [Sample]()))
    private let device: (any MTLDevice)?
    private let started = ContinuousClock.now
    init(device: (any MTLDevice)?) { self.device = device }
    func mark(_ phase: String) { state.withLock { $0.phase = phase }; sample() }
    func sample() {
        guard let m = try? BenchmarkMemory.read(device: device) else { return }
        let d = ContinuousClock.now - started
        let t = Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
        state.withLock { $0.samples.append(Sample(t: t, phase: $0.phase, metal: m.metalAllocatedBytes ?? 0, footprint: m.physicalFootprintBytes, resident: m.residentBytes)) }
    }
    var samples: [Sample] { state.withLock { $0.samples } }
    struct Phase: Codable { let phase: String; let seconds: Double; let metalStart: UInt64; let metalPeak: UInt64; let metalEnd: UInt64
        let footprintPeak: UInt64; let footprintEnd: UInt64; let samples: Int }
    func phases() -> [Phase] {
        var result: [Phase] = [], current: [Sample] = []
        func flush() {
            guard let first = current.first, let last = current.last else { return }
            result.append(Phase(phase: first.phase, seconds: last.t - first.t, metalStart: first.metal, metalPeak: current.map(\.metal).max()!,
                                metalEnd: last.metal, footprintPeak: current.map(\.footprint).max()!, footprintEnd: last.footprint, samples: current.count))
        }
        for s in samples { if let p = current.last?.phase, p != s.phase { flush(); current = [] }; current.append(s) }
        flush()
        return result
    }
}

/// Memory diagnosis commands (GPU; run them under the GPU lock).
enum H3MemoryDiagnosis {
    static func run(_ arguments: [String]) async throws -> Bool {
        switch arguments.first {
        case "h3-memory-encoder":
            // h3-memory-encoder PROFILE.json WIDTH HEIGHT KEYFRAME.png NEW_DIR: the session's first-frame encoder stage.
            guard arguments.count == 6, let width = Int(arguments[2]), let height = Int(arguments[3]) else {
                throw AetherError.invalidRequest("use h3-memory-encoder PROFILE.json WIDTH HEIGHT KEYFRAME.png NEW_DIR")
            }
            try await encoder(profile: URL(fileURLWithPath: arguments[1]), width: width, height: height, image: URL(fileURLWithPath: arguments[4]),
                              output: URL(fileURLWithPath: arguments[5]))
            return true
        case "h3-memory-render":
            // h3-memory-render PROFILE.json DUMP NEW_DIR: an H3VideoSession render with the timeline split at stage events.
            guard arguments.count == 4 else { throw AetherError.invalidRequest("use h3-memory-render PROFILE.json DUMP NEW_DIR") }
            try await render(profile: URL(fileURLWithPath: arguments[1]), dump: URL(fileURLWithPath: arguments[2]), output: URL(fileURLWithPath: arguments[3]))
            return true
        default: return false
        }
    }

    static func write(_ output: URL, _ timeline: H3MemoryTimeline, extra: [String: String]) throws {
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        struct Report: Encodable { let fields: [String: String]; let phases: [H3MemoryTimeline.Phase]; let samples: [H3MemoryTimeline.Sample] }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(Report(fields: extra, phases: timeline.phases(), samples: timeline.samples)).write(to: output.appendingPathComponent("memory.json"))
    }

    static func idle(_ timeline: H3MemoryTimeline, _ phase: String, _ seconds: Double) async {
        timeline.mark(phase); try? await Task.sleep(for: .milliseconds(Int(seconds * 1000))); timeline.sample()
    }

    /// The session's first-frame encoder stage (whatever the manifest selects: the tiled graphs or the whole-frame
    /// graph), called twice and then released.
    static func encoder(profile url: URL, width: Int, height: Int, image: URL, output: URL) async throws {
        let profile = try H3Profile(url: url)
        let path: String
        switch try profile.keyframeEncoding(width: width, height: height) {
        case .tiled?: path = "tiled"
        case .whole?: path = "whole"
        case nil: throw AetherError.unsupported("profile has no first-frame encoder for \(width)x\(height)")
        }
        let geometry = try H3Geometry(width: width, height: height, requestedFrames: 120, diagnostic: true)
        let keyframe = try H3ProfileCommand.keyframe(image, width: width, height: height)
        let device = MTLCreateSystemDefaultDevice()
        let timeline = H3MemoryTimeline(device: device)
        let sampling = Task.detached { while !Task.isCancelled { timeline.sample(); try? await Task.sleep(for: .milliseconds(20)) } }
        defer { sampling.cancel() }
        await idle(timeline, "baseline", 1)
        let stages = try H3ProfileEngine.stages(profile: profile, tokenize: { _ in [] })
        guard let encoder = stages.keyframes else { throw AetherError.unsupported("profile has no first-frame encoder") }
        var hashes: [String] = []
        for pass in 1...2 {
            timeline.mark(pass == 1 ? "run1 (open, specialize, encode)" : "run2")
            hashes.append(try await encoder.encode(keyframe, geometry: geometry).sha256)
            await idle(timeline, "after-run\(pass)", 1)
        }
        timeline.mark("release"); await encoder.release()
        await idle(timeline, "released+0-1s", 1)
        await idle(timeline, "released+1-6s", 5)
        try write(output, timeline, extra: ["mode": "encoder", "path": path, "canvas": "\(width)x\(height)", "profile_sha256": profile.manifestSHA256,
                                             "condition_sha256_run1": hashes[0], "condition_sha256_run2": hashes[1]])
        print(output.path)
    }

    /// A complete H3VideoSession render with the timeline split at the session's stage and step events.
    static func render(profile url: URL, dump: URL, output: URL) async throws {
        let profile = try H3Profile(url: url)
        struct Manifest: Decodable { struct Entry: Decodable { let file: String; let sha256: String }; let tensors: [String: Entry]; let fields: [String: String] }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: dump.appendingPathComponent("manifest.json")))
        let f = manifest.fields
        guard f["profile_sha256"] == profile.manifestSHA256 else { throw AetherError.invalidModel("dump was made for another profile") }
        let ids = try Data(contentsOf: dump.appendingPathComponent(manifest.tensors["ids"]!.file))
        let tokens = ids.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        let geometry = try H3Geometry(width: Int(f["width"]!)!, height: Int(f["height"]!)!, requestedFrames: Int(f["frames"]!)!, diagnostic: true)
        var first: H3Keyframe? = nil
        if let entry = manifest.tensors["keyframe"] {
            first = try H3Keyframe(width: geometry.width, height: geometry.height, rgb: [UInt8](Data(contentsOf: dump.appendingPathComponent(entry.file))))
        }
        let device = MTLCreateSystemDefaultDevice()
        let timeline = H3MemoryTimeline(device: device)
        let sampling = Task.detached { while !Task.isCancelled { timeline.sample(); try? await Task.sleep(for: .milliseconds(20)) } }
        defer { sampling.cancel() }
        await idle(timeline, "baseline", 1)
        timeline.mark("open")
        let stages = try H3ProfileEngine.stages(profile: profile, recipe: f["recipe"]!, tokenize: { _ in tokens })
        let session = H3VideoSession(stages: stages)
        let request = try H3VideoRequest(prompt: "ids", geometry: geometry, seed: UInt64(f["seed"]!)!, firstFrame: first, audio: true)
        _ = try await session.generate(request) { event in
            switch event {
            case .stageStarted(let stage): timeline.mark("stage.\(stage.rawValue)")
            case .stageReleased(let stage): timeline.mark("released.\(stage.rawValue)")
            case .step(let step, _): timeline.mark("stage.denoising.after-step\(step)")
            default: break
            }
        }
        timeline.mark("unload"); await session.unload()
        await idle(timeline, "unloaded+0-5s", 5)
        try write(output, timeline, extra: ["mode": "render", "dump": dump.path, "profile_sha256": profile.manifestSHA256])
        print(output.path)
    }
}
