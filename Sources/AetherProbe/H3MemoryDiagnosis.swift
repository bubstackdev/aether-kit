import AetherKit
import AetherH3
import AetherH3CoreAI
import Foundation
import Metal
import Synchronization

/// A 20 ms memory timeline with named phases, so a peak can be attributed to model load and specialization, a call's
/// scratch, or memory still held after release. Metal is `MTLDevice.currentAllocatedSize`; footprint is the
/// process's physical footprint. The two overlap and cannot be added.
@available(macOS 27, iOS 27, *)
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
    var last: Sample? { state.withLock { $0.samples.last } }
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
@available(macOS 27, iOS 27, *)
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
            // h3-memory-render PROFILE.json DUMP NEW_DIR [--spike-capture=GiB]: an H3VideoSession render with the timeline split
            // at stage events; see H3SpikeTrigger for the optional capture.
            let options = try RenderOptions(arguments)
            try await render(profile: options.profile, dump: options.dump, output: options.output, spikeCapture: options.spikeCapture)
            return true
        default: return false
        }
    }

    struct RenderOptions: Equatable {
        static let usage = "use h3-memory-render PROFILE.json DUMP NEW_DIR [--spike-capture=GiB]"
        let profile: URL, dump: URL, output: URL
        /// Metal level, in bytes, above which a denoising sample triggers the one-shot capture; nil when off.
        let spikeCapture: UInt64?
        init(_ arguments: [String]) throws {
            let flags = arguments.dropFirst().filter { $0.hasPrefix("--") }, paths = arguments.dropFirst().filter { !$0.hasPrefix("--") }
            guard arguments.first == "h3-memory-render", paths.count == 3, flags.count <= 1 else { throw AetherError.invalidRequest(Self.usage) }
            var spike: UInt64?
            if let flag = flags.first {
                guard flag.hasPrefix("--spike-capture="), let gib = Double(flag.dropFirst("--spike-capture=".count)), gib.isFinite, gib > 0, gib < 1024 else {
                    throw AetherError.invalidRequest(Self.usage)
                }
                spike = UInt64(gib * Double(1 << 30))
            }
            profile = URL(fileURLWithPath: paths[0]); dump = URL(fileURLWithPath: paths[1]); output = URL(fileURLWithPath: paths[2]); spikeCapture = spike
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
    static func render(profile url: URL, dump: URL, output: URL, spikeCapture: UInt64? = nil) async throws {
        let profile = try H3Profile(url: url)
        struct Manifest: Decodable { struct Entry: Decodable { let file: String; let sha256: String }; let tensors: [String: Entry]; let fields: [String: String] }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: dump.appendingPathComponent("manifest.json")))
        let f = manifest.fields
        guard f["profile_sha256"] == profile.manifestSHA256 else { throw AetherError.invalidModel("dump was made for another profile") }
        let ids = try Data(contentsOf: dump.appendingPathComponent(manifest.tensors["ids"]!.file))
        let tokens = ids.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        let geometry = try H3Geometry(width: Int(f["width"]!)!, height: Int(f["height"]!)!, requestedFrames: Int(f["frames"]!)!, diagnostic: true)
        let keyframes = try H3ProfileCommand.dumpKeyframes(dump, geometry: geometry)
        let device = MTLCreateSystemDefaultDevice()
        let timeline = H3MemoryTimeline(device: device)
        // The capture goes beside NEW_DIR (written last) so the render's own output stays unchanged.
        let spikeDirectory = URL(fileURLWithPath: output.path + ".spike")
        let trigger = spikeCapture.map { H3SpikeTrigger(threshold: $0, capture: H3SpikeTrigger.processSnapshot(into: spikeDirectory)) }
        let sampling = Task.detached {
            while !Task.isCancelled { timeline.sample(); if let s = timeline.last { trigger?.observe(s) }; try? await Task.sleep(for: .milliseconds(20)) }
        }
        defer { sampling.cancel() }
        await idle(timeline, "baseline", 1)
        timeline.mark("open")
        let stages = try H3ProfileEngine.stages(profile: profile, recipe: f["recipe"]!, tokenize: { _ in tokens })
        let session = H3VideoSession(stages: stages)
        let request = try H3VideoRequest(prompt: "ids", geometry: geometry, seed: UInt64(f["seed"]!)!, firstFrame: keyframes.first,
                                         lastFrame: keyframes.last, audio: true)
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
        try write(output, timeline, extra: ["mode": "render", "dump": dump.path, "profile_sha256": profile.manifestSHA256,
                                             "execution_identity": H3ProfileEngine.resourceIdentity(profile: profile, recipe: f["recipe"]!),
                                             "device_architecture": H3ProfileEngine.deviceArchitecture,
                                             "width": f["width"]!, "height": f["height"]!, "frames": f["frames"]!,
                                             "task": f["anchors"] == nil ? "t2va" : "fl2va", "anchors": f["anchors"] ?? "",
                                             "spike_capture_bytes": spikeCapture.map(String.init) ?? "off",
                                             "spike_captured": trigger?.fired == true ? spikeDirectory.lastPathComponent : "no"])
        print(output.path)
    }
}

/// Diagnostic: the first time a denoising sample's Metal allocation exceeds `threshold`, `capture` runs once with that
/// sample (`h3-memory-render --spike-capture=GiB`). The default capture snapshots this process with `sample` (thread
/// stacks for 1 s), `vmmap --summary` and `footprint`. `sample` suspends the threads briefly, so timing inside a captured
/// window is not representative; memory is.
@available(macOS 27, iOS 27, *)
final class H3SpikeTrigger: Sendable {
    struct Event: Sendable, Equatable { let phase: String; let t: Double; let metal: UInt64; let footprint: UInt64; let secondsIntoPhase: Double }
    private let threshold: UInt64
    private let capture: @Sendable (Event) -> Void
    private let state = Mutex((phase: "", phaseT: 0.0, fired: false))
    init(threshold: UInt64, capture: @escaping @Sendable (Event) -> Void) { self.threshold = threshold; self.capture = capture }
    var fired: Bool { state.withLock { $0.fired } }

    func observe(_ sample: H3MemoryTimeline.Sample) {
        let event: Event? = state.withLock { s in
            if s.phase != sample.phase { s.phase = sample.phase; s.phaseT = sample.t }
            guard !s.fired, sample.phase.hasPrefix("stage.denoising"), sample.metal > threshold else { return nil }
            s.fired = true
            return Event(phase: sample.phase, t: sample.t, metal: sample.metal, footprint: sample.footprint, secondsIntoPhase: sample.t - s.phaseT)
        }
        if let event { capture(event) }
    }

    static func processSnapshot(into directory: URL) -> @Sendable (Event) -> Void {
        { event in
            try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
            let pid = String(ProcessInfo.processInfo.processIdentifier)
            let info = ["phase": event.phase, "t": String(format: "%.2f", event.t), "secondsIntoPhase": String(format: "%.2f", event.secondsIntoPhase),
                        "metal": String(event.metal), "footprint": String(event.footprint)]
            try? JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys]).write(to: directory.appendingPathComponent("trigger.json"))
            for (tool, arguments, name) in [("/usr/bin/sample", [pid, "1", "1", "-mayDie", "-file", directory.appendingPathComponent("sample.txt").path], "sample.log"),
                                            ("/usr/bin/vmmap", ["--summary", pid], "vmmap.txt"), ("/usr/bin/footprint", ["-p", pid], "footprint.txt")] {
                let path = directory.appendingPathComponent(name).path
                guard FileManager.default.createFile(atPath: path, contents: nil), let handle = FileHandle(forWritingAtPath: path) else { continue }
                let process = Process(); process.executableURL = URL(fileURLWithPath: tool); process.arguments = arguments
                process.standardOutput = handle; process.standardError = handle
                try? process.run()
            }
        }
    }
}
