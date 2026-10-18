import AetherKit
import Foundation
import Testing
@testable import AetherCoreAI

private actor SuspendedBackend: ModelBackend {
    private var started = false
    private var startWaiters: [CheckedContinuation<Void, Never>] = []
    private var completion: CheckedContinuation<Void, Never>?

    func run(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue] {
        await withCheckedContinuation { continuation in
            completion = continuation
            started = true
            for waiter in startWaiters { waiter.resume() }
            startWaiters.removeAll()
        }
        return inputs
    }

    func waitUntilStarted() async {
        if started { return }
        await withCheckedContinuation { startWaiters.append($0) }
    }

    func complete() {
        completion?.resume()
        completion = nil
    }
}

private struct FailingBackend: ModelBackend {
    func run(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue] {
        throw AetherError.invalidModel("controlled failure")
    }
}

private struct EchoBackend: ModelBackend {
    func run(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue] { inputs }
}

@Test func mixedTensorCallsPreserveTypes() async throws {
    let session = ModelSession(backend: EchoBackend())
    let values: [String: TensorValue] = [
        "ids": .int32(try Int32Tensor(shape: [2], values: [0, Int32.max])),
        "half": .float16(try Float16Tensor(shape: [1], values: [0.5])),
        "float": .float32(try FloatTensor(shape: [1], values: [7])),
    ]
    #expect(try await session.runTensors(inputs: values) == values)
    await session.unload()
    #expect(await session.state == .closed)
    await #expect(throws: AetherError.sessionClosed) { try await session.runTensors(inputs: values) }
}

@Test(.timeLimit(.minutes(1))) func rejectsOverlapAndUnloadWaitsForCompletion() async throws {
    let backend = SuspendedBackend()
    let session = ModelSession(backend: backend)
    let input = try FloatTensor(shape: [1], values: [7])
    let run = Task { try await session.run(inputs: ["x": input]) }
    await backend.waitUntilStarted()
    #expect(await session.state == .running)
    await #expect(throws: AetherError.sessionBusy) { try await session.run(inputs: ["x": input]) }
    let unload = Task { await session.unload() }
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await session.state != .closing, ContinuousClock.now < deadline { await Task.yield() }
    #expect(await session.state == .closing)
    await #expect(throws: AetherError.sessionClosed) { try await session.run(inputs: ["x": input]) }
    await backend.complete()
    let output = try await run.value
    await unload.value
    #expect(output["x"] == input)
    #expect(await session.state == .closed)
    await session.unload()
    await #expect(throws: AetherError.sessionClosed) { try await session.run(inputs: ["x": input]) }
}

@Test(.timeLimit(.minutes(1))) func cancellationRetainsSlotUntilBackendCompletes() async throws {
    let backend = SuspendedBackend()
    let session = ModelSession(backend: backend)
    let run = Task { try await session.run(inputs: [:]) }
    await backend.waitUntilStarted()
    run.cancel()
    #expect(await session.state == .running)
    await #expect(throws: AetherError.sessionBusy) { try await session.run(inputs: [:]) }
    await backend.complete()
    await #expect(throws: CancellationError.self) { try await run.value }
    #expect(await session.state == .ready)
    await session.unload()
}

@Test func failedInferenceReleasesSessionSlot() async throws {
    let session = ModelSession(backend: FailingBackend())
    for _ in 0..<2 {
        await #expect(throws: AetherError.invalidModel("controlled failure")) {
            try await session.run(inputs: [:])
        }
        #expect(await session.state == .ready)
    }
    await session.unload()
}

@Test func missingModelFailsBeforeNativeLoad() async {
    await #expect(throws: AetherError.self) {
        try await ModelSession.open(at: URL(fileURLWithPath: "/nonexistent/aether-test.aimodel"))
    }
}
