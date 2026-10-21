@testable import AetherH3
import AetherCoreAI
import AetherKit
import Foundation
import Testing
@testable import AetherH3CoreAI

private actor PhaseGateBackend: H3PhaseBackend {
    var entered = false
    var released = false
    var calls = 0
    var waiter: CheckedContinuation<Void, Never>?
    var startWaiter: CheckedContinuation<Void, Never>?
    let failFirst: Bool
    init(failFirst: Bool = false) { self.failFirst = failFirst }
    func run() async throws -> [String: H3TensorData] {
        calls += 1
        if !released {
            await withCheckedContinuation { waiter = $0; entered = true; startWaiter?.resume(); startWaiter = nil }
        }
        if failFirst && calls == 1 { throw AetherError.invalidTensor("injected native failure") }
        let tensor = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1]), bytes: Data([128,63]))
        return ["video": tensor, "audio": tensor]
    }
    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { startWaiter = $0 }
    }
    func release() { released = true; waiter?.resume(); waiter = nil }
}

@Test(.timeLimit(.minutes(1))) func h3PhaseSessionRejectsOverlapAndRetainsSlotThroughCancellation() async throws {
    let backend = PhaseGateBackend()
    let activeSession = H3PhaseSession(backend: backend)
    let task = Task { try await activeSession.run() }
    await backend.waitUntilEntered(); task.cancel()
    await #expect(throws: AetherError.sessionBusy) { try await activeSession.run() }
    #expect(await activeSession.state == .running)
    await backend.release()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await activeSession.state == .ready)
    let result = try await activeSession.run()
    #expect(try result["video"]!.floatValues() == [1])
    await activeSession.unload()
    await #expect(throws: AetherError.sessionClosed) { try await activeSession.run() }
}

@Test(.timeLimit(.minutes(1))) func h3PhaseUnloadWaitsForSubmittedWorkAndPreservesOwnedOutputs() async throws {
    let backend = PhaseGateBackend()
    let activeSession = H3PhaseSession(backend: backend)
    let task = Task { try await activeSession.run() }
    await backend.waitUntilEntered()
    let closing = Task { await activeSession.unload() }
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await activeSession.state != .closing, ContinuousClock.now < deadline { await Task.yield() }
    #expect(await activeSession.state == .closing)
    await #expect(throws: AetherError.sessionClosed) { try await activeSession.run() }
    await backend.release()
    let output = try await task.value
    await closing.value
    #expect(await activeSession.state == .closed)
    #expect(try output["audio"]!.floatValues() == [1])
}

@Test func h3PhaseNativeFailureReleasesSlotForRetry() async throws {
    let backend = PhaseGateBackend(failFirst: true)
    await backend.release()
    let session = H3PhaseSession(backend: backend)
    await #expect(throws: AetherError.invalidTensor("injected native failure")) { try await session.run() }
    #expect(await session.state == .ready)
    _ = try await session.run()
    #expect(await backend.calls == 2)
    await session.unload()
}

@Test func h3PhaseBindingGateRequiresBothNamedOutputsWithExactPrecision() throws {
    let spec = try H3TensorSpec(scalarType: .float32, shape: [1,2,4])
    let file = try H3TensorFile(name: "hidden", url: URL(fileURLWithPath: "/not-read"), spec: spec, sha256: String(repeating: "0", count: 64))
    let model = try H3ModelAsset(url: URL(fileURLWithPath: "/not-loaded.aimodel"), function: "main", sha256: file.sha256)
    let video = try H3TensorFile(name: "video", url: file.url, spec: spec, sha256: file.sha256)
    let audio = try H3TensorFile(name: "audio", url: file.url, spec: spec, sha256: file.sha256)
    let probe = H3PhaseProbePlan.Case(name: "heads-0", model: model, inputs: [file], references: [video,audio])
    let tensor: [String: Any] = ["kind": "tensor", "scalarType": "float32", "shape": [-1,2,4]]
    let base: [String: Any] = ["name": "main", "inputs": ["hidden": tensor], "outputs": ["video": tensor, "audio": tensor], "states": [:]]
    func validate(_ raw: [String: Any]) throws {
        try H3PhaseBindings.validate(JSONDecoder().decode(ModelFunctionDescription.self, from: JSONSerialization.data(withJSONObject: raw)), probe: probe)
    }
    try validate(base)
    for fault in ["missingAudio", "precision", "dimension", "state", "name"] {
        var raw = base
        switch fault {
        case "missingAudio": raw["outputs"] = ["video": tensor]
        case "precision": var changed = tensor; changed["scalarType"] = "float16"; raw["outputs"] = ["video": tensor,"audio": changed]
        case "dimension": var changed = tensor; changed["shape"] = [1,3,4]; raw["outputs"] = ["video": tensor,"audio": changed]
        case "state": raw["states"] = ["cache": tensor]
        default: raw["inputs"] = ["other": tensor]
        }
        #expect(throws: AetherError.self) { try validate(raw) }
    }
}
