import AetherKit
import AetherH3
import AetherH3CoreAI
import Foundation
import Metal
import Synchronization

/// `qualify-h3-session-lifecycle PROFILE.json DUMP NEW_DIRECTORY`: per-stage lifecycle of `H3VideoSession` over a schema-3
/// profile, on the native stages with an `H3NativeCallObserver` journal. A T2VA dump runs, in one process:
///
///   baseline     a render whose trajectory is hashed (every denoiser boundary), plus its frames and PCM;
///   cancel-*     cancellation requested while a native call of that stage is in flight (conditioner layer 10, body call
///                60 = step 1 block 10, video tile 1, audio decoder 0); then a recovery render in the same session;
///   unload       `unload()` requested while body call 60 is in flight; the active render completes;
///   reopen       a new session renders again; memory 0, 1 and 5 s after each session's unload.
///
/// An FL2VA dump runs baseline, cancel-keyframe (the first keyframe-encoder call) with recovery, and unload. Every
/// recovery and post-unload render must equal the baseline: all trajectory hashes, the frames and the PCM.
enum H3SessionLifecycleCommand {
    static let usage = "use qualify-h3-session-lifecycle PROFILE.json DUMP NEW_DIRECTORY"

    static func run(_ arguments: [String]) async throws -> Bool {
        guard arguments.first == "qualify-h3-session-lifecycle" else { return false }
        guard arguments.count == 4 else { throw AetherError.invalidRequest(usage) }
        try await execute(profile: URL(fileURLWithPath: arguments[1]), dump: URL(fileURLWithPath: arguments[2]),
                          output: URL(fileURLWithPath: arguments[3]))
        return true
    }

    /// The cancellation scenarios: the graph role and its call index at which cancellation is requested.
    static func cancellations(keyframe: Bool) -> [(name: String, graph: String, call: Int)] {
        keyframe ? [("cancel-keyframe", "keyframe", 0)]
            : [("cancel-conditioner", "conditioner", 10), ("cancel-denoiser", "body", 60), ("cancel-videoTile", "videoTile", 1),
               ("cancel-audio", "audio", 0)]
    }

    static func execute(profile url: URL, dump: URL, output: URL) async throws {
        let profile = try H3Profile(url: url)
        guard try H3RawDirectory.field(dump, "profile_sha256") == profile.manifestSHA256 else { throw AetherError.invalidModel("dump was made for another profile") }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        let geometry = try H3Geometry(width: Int(H3RawDirectory.field(dump, "width"))!, height: Int(H3RawDirectory.field(dump, "height"))!,
                                      requestedFrames: Int(H3RawDirectory.field(dump, "frames"))!, diagnostic: true)
        let recipe = try H3RawDirectory.field(dump, "recipe"), seed = UInt64(try H3RawDirectory.field(dump, "seed"))!
        let ids = try H3RawDirectory.read(dump, "ids")
        let tokens = ids.bytes.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        let first = try H3ProfileCommand.dumpKeyframe(dump, geometry: geometry)
        let request = try H3VideoRequest(prompt: "ids", geometry: geometry, seed: seed, firstFrame: first, audio: true)
        let device = MTLCreateSystemDefaultDevice()
        let report = H3SessionLifecycleReport(url: output.appendingPathComponent("report.json"))
        report.set("profile_sha256", profile.manifestSHA256); report.set("task", first == nil ? "t2va" : "fl2va")
        report.set("canvas", "\(geometry.width)x\(geometry.height)x\(geometry.frames)")

        let journal = H3NativeJournal(), trigger = H3NativeTrigger()
        let recorder = Mutex<H3TrajectoryRecorder?>(nil)
        func makeSession() throws -> H3VideoSession {
            let stages = try H3ProfileEngine.stages(profile: profile, recipe: recipe, tokenize: { _ in tokens }, observer: { event in
                if let current = recorder.withLock({ $0 }) { try await current.record(event) }
            }, nativeObserver: { call in journal.add(.native(call)); trigger.observe(call) })
            return H3VideoSession(stages: stages)
        }
        let baselineDirectory = output.appendingPathComponent("baseline")
        try FileManager.default.createDirectory(at: baselineDirectory, withIntermediateDirectories: false)

        /// One render through `session`; returns the frame and PCM hashes and, when replaying, the trajectory mismatches.
        func render(_ session: H3VideoSession, capture: Bool, cancelAt: (graph: String, call: Int)? = nil,
                    onTarget: (@Sendable (Task<H3RenderDigest, any Error>) -> Void)? = nil) async throws -> H3RenderDigest {
            let made = try capture ? H3TrajectoryRecorder(directory: baselineDirectory.appendingPathComponent("capture"), steps: "0", reference: nil)
                                   : H3TrajectoryRecorder(directory: output, steps: nil, reference: baselineDirectory)
            recorder.withLock { $0 = made }
            defer { recorder.withLock { $0 = nil } }
            let digest = H3DigestTape()
            let task = Task {
                _ = try await session.generate(request) { event in
                    switch event {
                    case .frames(let batch): await digest.add(frames: batch)
                    case .audio(let pcm): await digest.add(pcm: pcm)
                    default: break
                    }
                }
                let summary = try await made.finish()
                return await digest.result(trajectoryHashes: summary.hashes, mismatches: summary.mismatches)
            }
            if let cancelAt, let onTarget { trigger.arm(graph: cancelAt.graph, call: cancelAt.call) { onTarget(task) } }
            defer { trigger.disarm() }
            return try await task.value
        }
        func matches(_ digest: H3RenderDigest, _ baseline: H3RenderDigest) -> Bool {
            digest.mismatches == [] && digest.frames == baseline.frames && digest.pcm == baseline.pcm && digest.trajectoryHashes == baseline.trajectoryHashes
        }

        let session = try makeSession()
        journal.reset()
        let baseline = try await render(session, capture: true)
        report.record("baseline", ["frames": baseline.frames, "pcm": baseline.pcm, "trajectoryHashes": String(baseline.trajectoryHashes)])
        report.check("baselineCompletes", "baseline", baseline.trajectoryHashes > 0)

        for scenario in cancellations(keyframe: first != nil) {
            journal.reset()
            var cancelled = false, otherError: String?
            do {
                _ = try await render(session, capture: false, cancelAt: (scenario.graph, scenario.call)) { task in
                    journal.add(.marker("cancelRequested")); task.cancel()
                }
            } catch is CancellationError { cancelled = true } catch { otherError = String(describing: error) }
            journal.add(.marker("cancellationReturned"))
            let checks = H3NativeJournal.cancellationChecks(journal.entries, target: scenario.graph)
            report.check("throwsCancellation", scenario.name, cancelled, detail: otherError)
            for (name, passed) in checks.sorted(by: { $0.key < $1.key }) { report.check(name, scenario.name, passed) }
            report.check("stateReturnsToReady", scenario.name, await session.state == .ready)
            report.journal(scenario.name, journal.entries)
            journal.reset()
            let recovered = try await render(session, capture: false)
            report.check("recoveryMatchesBaseline", scenario.name, matches(recovered, baseline), detail: recovered.summary)
        }

        // Unload while body call 60 is in flight: the active render completes, then every stage is released.
        journal.reset()
        let unloading = Mutex<Task<Void, Never>?>(nil)
        let active = try await render(session, capture: false, cancelAt: ("body", 60)) { _ in
            journal.add(.marker("unloadRequested"))
            unloading.withLock { $0 = Task { await session.unload(); journal.add(.marker("unloadReturned")) } }
        }
        await unloading.withLock({ $0 })?.value
        report.check("activeRenderCompletesAndMatches", "unload", matches(active, baseline), detail: active.summary)
        let unloadChecks = H3NativeJournal.unloadChecks(journal.entries)
        for (name, passed) in unloadChecks.sorted(by: { $0.key < $1.key }) { report.check(name, "unload", passed) }
        report.check("stateClosed", "unload", await session.state == .closed)
        var rejected = false
        do { _ = try await session.generate(request) } catch AetherError.sessionClosed { rejected = true } catch {}
        report.check("generateAfterUnloadRejected", "unload", rejected)
        report.journal("unload", journal.entries)
        let firstSettled = await settle(device: device)
        report.memory("afterFirstUnload", firstSettled)

        // A new session over new stages renders the same bytes, and settles to the same memory.
        journal.reset()
        let reopened = try makeSession()
        let again = try await render(reopened, capture: false)
        report.check("reopenedMatchesBaseline", "reopen", matches(again, baseline), detail: again.summary)
        await reopened.unload()
        let secondSettled = await settle(device: device)
        report.memory("afterReopenedUnload", secondSettled)
        report.check("settledGrowthWithin512MiB", "reopen", H3NativeJournal.settledGrowthBounded(firstSettled.last!, secondSettled.last!))
        let failures = report.finish()
        print(output.path)
        guard failures.isEmpty else { throw AetherError.invalidModel("H3 session lifecycle failed: \(failures.joined(separator: ", "))") }
    }

    /// Memory 0, 1 and 5 s after an unload.
    static func settle(device: (any MTLDevice)?) async -> [BenchmarkMemory] {
        var samples: [BenchmarkMemory] = []
        for wait in [0, 1, 4] {
            try? await Task.sleep(for: .seconds(wait))
            if let memory = try? BenchmarkMemory.read(device: device) { samples.append(memory) }
        }
        return samples
    }
}

/// The ordered native-call journal of one scenario.
final class H3NativeJournal: Sendable {
    enum Entry: Sendable, Equatable { case native(H3NativeCall), marker(String) }
    private let state = Mutex([Entry]())
    func add(_ entry: Entry) { state.withLock { $0.append(entry) } }
    func reset() { state.withLock { $0 = [] } }
    var entries: [Entry] { state.withLock { $0 } }

    /// After "cancelRequested": the call in flight is of `target` and joins before "cancellationReturned", and no native
    /// call begins.
    static func cancellationChecks(_ entries: [Entry], target: String) -> [String: Bool] {
        guard let request = entries.firstIndex(of: .marker("cancelRequested")),
              let returned = entries.firstIndex(of: .marker("cancellationReturned")) else {
            return ["cancellationRequested": false]
        }
        var open: [String: Int] = [:]
        for entry in entries[..<request] {
            guard case .native(let call) = entry else { continue }
            let key = "\(call.graph)#\(call.call)"
            if call.phase == .begin { open[key] = 0 } else { open[key] = nil }
        }
        let inFlight = open.keys.sorted()
        let joined = inFlight.allSatisfy { key in
            entries[request..<returned].contains { if case .native(let c) = $0 { "\(c.graph)#\(c.call)" == key && c.phase != .begin } else { false } }
        }
        let later = entries[request...].contains { if case .native(let c) = $0 { c.phase == .begin } else { false } }
        return ["inFlightCallIsTheTargetStage": inFlight.count == 1 && inFlight[0].hasPrefix(target + "#"),
                "inFlightCallJoinsBeforeTheError": !inFlight.isEmpty && joined,
                "noNativeCallAfterCancellation": !later]
    }

    /// After "unloadRequested", native calls continue (the active render completes), every one ends, and none begins after
    /// "unloadReturned".
    static func unloadChecks(_ entries: [Entry]) -> [String: Bool] {
        guard let request = entries.firstIndex(of: .marker("unloadRequested")),
              let returned = entries.firstIndex(of: .marker("unloadReturned")) else { return ["unloadRequested": false] }
        var open = Set<String>()
        for entry in entries[..<returned] {
            guard case .native(let call) = entry else { continue }
            if call.phase == .begin { open.insert("\(call.graph)#\(call.call)") } else { open.remove("\(call.graph)#\(call.call)") }
        }
        let continued = entries[request..<returned].contains { if case .native(let c) = $0 { c.phase == .begin } else { false } }
        let later = entries[returned...].contains { if case .native = $0 { true } else { false } }
        return ["activeRenderContinuesAfterUnloadRequest": continued, "everyCallJoinsBeforeUnloadReturns": open.isEmpty,
                "noNativeCallAfterUnload": !later]
    }

    static func settledGrowthBounded(_ first: BenchmarkMemory, _ second: BenchmarkMemory) -> Bool {
        (try? H3LifecycleLimits().checkGrowth(baseline: first, current: second)) != nil
    }
}

/// Runs `action` once, synchronously, when the armed graph's call begins.
final class H3NativeTrigger: Sendable {
    private let state = Mutex<(graph: String, call: Int, action: (@Sendable () -> Void)?)>(("", -1, nil))
    func arm(graph: String, call: Int, _ action: @escaping @Sendable () -> Void) { state.withLock { $0 = (graph, call, action) } }
    func disarm() { state.withLock { $0.action = nil } }
    func observe(_ call: H3NativeCall) {
        let action: (@Sendable () -> Void)? = state.withLock { s in
            guard call.phase == .begin, call.graph == s.graph, call.call == s.call, let action = s.action else { return nil }
            s.action = nil
            return action
        }
        action?()
    }
}

struct H3RenderDigest: Sendable {
    let frames: String, pcm: String, trajectoryHashes: Int, mismatches: [String]?
    var summary: String { "frames \(frames.prefix(12)) pcm \(pcm.prefix(12)) hashes \(trajectoryHashes) mismatches \(mismatches.map { String($0.count) } ?? "-")" }
}

private actor H3DigestTape {
    private var frames = Data(), pcm = Data()
    func add(frames batch: H3DecodedFrames) { for frame in batch.rgb { frames.append(contentsOf: frame) } }
    func add(pcm audio: H3AudioPCM) { for channel in audio.channels { channel.withUnsafeBytes { pcm.append(contentsOf: $0) } } }
    func result(trajectoryHashes: Int, mismatches: [String]?) -> H3RenderDigest {
        H3RenderDigest(frames: h3SHA256(frames), pcm: h3SHA256(pcm), trajectoryHashes: trajectoryHashes, mismatches: mismatches)
    }
}

/// Incrementally written lifecycle report.
final class H3SessionLifecycleReport: Sendable {
    private let url: URL
    private let state = Mutex([String: Any]())
    init(url: URL) { self.url = url; state.withLock { $0 = ["schema": 1, "kind": "h3-session-lifecycle", "checks": [[String: Any]](), "journals": [String: Any]()] } }
    func set(_ key: String, _ value: String) { update { $0[key] = value } }
    func record(_ name: String, _ value: [String: String]) { update { $0[name] = value } }
    func check(_ name: String, _ scenario: String, _ passed: Bool, detail: String? = nil) {
        print("h3 lifecycle \(scenario) \(name): \(passed ? "pass" : "FAIL")\(detail.map { " (\($0))" } ?? "")")
        var entry: [String: Any] = ["name": name, "scenario": scenario, "passed": passed]
        if let detail { entry["detail"] = detail }
        update { $0["checks"] = ($0["checks"] as! [[String: Any]]) + [entry] }
    }
    func journal(_ scenario: String, _ entries: [H3NativeJournal.Entry]) {
        let compact: [String] = entries.map {
            switch $0 { case .native(let c): "\(c.graph)#\(c.call):\(c.phase.rawValue)"; case .marker(let m): "== \(m)" }
        }
        // Keep the journal's ends: the cancellation window is near the end.
        update { var journals = $0["journals"] as! [String: Any]; journals[scenario] = ["entries": compact.count, "tail": Array(compact.suffix(40))]; $0["journals"] = journals }
    }
    func memory(_ name: String, _ samples: [BenchmarkMemory]) {
        update { $0[name] = samples.map { ["metalAllocatedBytes": $0.metalAllocatedBytes ?? 0, "physicalFootprintBytes": $0.physicalFootprintBytes,
                                            "residentBytes": $0.residentBytes] } }
    }
    /// Writes the report and returns the failed checks.
    func finish() -> [String] {
        let failures = state.withLock { s in (s["checks"] as! [[String: Any]]).filter { !($0["passed"] as! Bool) }.map { "\($0["scenario"]!)/\($0["name"]!)" } }
        update { $0["passed"] = failures.isEmpty }
        return failures
    }
    private func update(_ body: (inout [String: Any]) -> Void) {
        state.withLock { s in
            body(&s)
            if let data = try? JSONSerialization.data(withJSONObject: s, options: [.prettyPrinted, .sortedKeys]) { try? data.write(to: url, options: .atomic) }
        }
    }
}
