@testable import AetherH3
import AetherCoreAI
import AetherKit
import Foundation
import Testing
@testable import AetherH3CoreAI

private actor GateBackend: H3BlockBackend {
    var entered = false
    var released = false
    var calls = 0
    var waiter: CheckedContinuation<Void, Never>?
    var startWaiter: CheckedContinuation<Void, Never>?
    let failFirst: Bool
    init(failFirst: Bool = false) { self.failFirst = failFirst }
    func run() async throws -> H3TensorData {
        calls += 1
        if !released {
            await withCheckedContinuation { waiter = $0; entered = true; startWaiter?.resume(); startWaiter = nil }
        }
        if failFirst && calls == 1 { throw AetherError.invalidTensor("injected native failure") }
        return try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1]), bytes: Data([128,63]))
    }
    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { startWaiter = $0 }
    }
    func release() { released = true; waiter?.resume(); waiter = nil }
}

@Test(.timeLimit(.minutes(1))) func h3BlockSessionRejectsOverlapAndRetainsSlotThroughCancellation() async throws {
    let backend = GateBackend()
    let activeSession = H3BlockSession(backend: backend)
    let task = Task { try await activeSession.run() }
    await backend.waitUntilEntered(); task.cancel()
    await #expect(throws: AetherError.sessionBusy) { try await activeSession.run() }
    #expect(await activeSession.state == .running)
    await backend.release()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await activeSession.state == .ready)
    let result = try await activeSession.run()
    #expect(try result.floatValues() == [1])
    await activeSession.unload()
    await #expect(throws: AetherError.sessionClosed) { try await activeSession.run() }
}

@Test(.timeLimit(.minutes(1))) func h3UnloadWaitsForSubmittedWorkAndPreservesOwnedOutput() async throws {
    let backend = GateBackend()
    let activeSession = H3BlockSession(backend: backend)
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
    #expect(try output.floatValues() == [1])
}

@Test func h3NativeFailureReleasesSlotForRetry() async throws {
    let backend = GateBackend(failFirst: true)
    await backend.release()
    let session = H3BlockSession(backend: backend)
    await #expect(throws: AetherError.invalidTensor("injected native failure")) { try await session.run() }
    #expect(await session.state == .ready)
    _ = try await session.run()
    #expect(await backend.calls == 2)
    await session.unload()
}

@Test func h3BindingGateRejectsFP16StatesAndWrongDimensionsBeforeDispatch() throws {
    let spec = try H3TensorSpec(scalarType: .bfloat16, shape: [1,2,4])
    let input = try H3TensorFile(name: "hidden", url: URL(fileURLWithPath: "/not-read"), spec: spec, sha256: String(repeating: "0", count: 64))
    let model = try H3ModelAsset(url: URL(fileURLWithPath: "/not-loaded.aimodel"), function: "main", sha256: input.sha256)
    let probe = H3BlockProbePlan.Case(name: "fake", model: model, inputs: [input], output: spec, reference: input, referenceRows: [0])
    let tensor: [String: Any] = ["kind": "tensor", "scalarType": "bfloat16", "shape": [-1,2,4]]
    let base: [String: Any] = ["name": "main", "inputs": ["hidden": tensor], "outputs": ["hidden_out": tensor], "states": [:]]
    func validate(_ raw: [String: Any]) throws {
        try H3BlockBindings.validate(JSONDecoder().decode(ModelFunctionDescription.self,
            from: JSONSerialization.data(withJSONObject: raw)), probe: probe)
    }
    try validate(base)
    for fault in ["dtype", "shape", "state", "name", "output"] {
        var raw = base
        switch fault {
        case "dtype": var value = tensor; value["scalarType"] = "float16"; raw["inputs"] = ["hidden": value]
        case "shape": var value = tensor; value["shape"] = [1,3,4]; raw["inputs"] = ["hidden": value]
        case "state": raw["states"] = ["state": tensor]
        case "name": raw["inputs"] = ["other": tensor]
        default: raw["outputs"] = ["wrong": tensor]
        }
        #expect(throws: AetherError.self) { try validate(raw) }
    }
}
