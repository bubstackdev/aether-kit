import AetherKit
import AetherH3
import Foundation
import Testing
@testable import AetherH3CoreAI

private actor ChainGateBackend: H3ChainBackend {
    var started = false
    var continuation: CheckedContinuation<Void, Never>?
    var start: CheckedContinuation<Void, Never>?
    var calls = 0
    var released = false
    func run() async throws -> [H3TensorData] {
        calls += 1
        if !released {
            await withCheckedContinuation { continuation = $0; started = true; start?.resume(); start = nil }
        }
        return try (1...3).map { try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [1]),
                                               bytes: withUnsafeBytes(of: Float($0)) { Data($0) }) }
    }
    func wait() async {
        if started { return }
        await withCheckedContinuation { start = $0 }
    }
    func release() { released = true; continuation?.resume(); continuation = nil }
}

@Test(.timeLimit(.minutes(1))) func h3ChainCancellationOwnsTheSlotUntilCompletionAndCanRetry() async throws {
    let backend = ChainGateBackend(), session = H3ChainSession(backend: backend)
    let task = Task { try await session.run() }
    await backend.wait(); task.cancel()
    await #expect(throws: AetherError.sessionBusy) { try await session.run() }
    #expect(await session.state == .running)
    await backend.release()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await session.state == .ready)
    let outputs = try await session.run()
    #expect(try outputs.map { try $0.floatValues() } == [[1],[2],[3]])
    await session.unload()
    #expect(try outputs[0].floatValues() == [1])
    await #expect(throws: AetherError.sessionClosed) { try await session.run() }
}

@Test(.timeLimit(.minutes(1))) func h3ChainUnloadWaitsForActiveOutputs() async throws {
    let backend = ChainGateBackend(), session = H3ChainSession(backend: backend)
    let task = Task { try await session.run() }
    await backend.wait()
    let closing = Task { await session.unload() }
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await session.state != .closing, ContinuousClock.now < deadline { await Task.yield() }
    #expect(await session.state == .closing)
    await #expect(throws: AetherError.sessionClosed) { try await session.run() }
    await backend.release()
    let outputs = try await task.value
    await closing.value
    #expect(await session.state == .closed)
    #expect(try outputs.map { try $0.floatValues() } == [[1],[2],[3]])
}

private actor ConsumerGate {
    var entered = false
    var seen: [Int] = []
    var pending: CheckedContinuation<Void, Never>?
    var started: CheckedContinuation<Void, Never>?
    func receive(_ index: Int) async {
        seen.append(index)
        await withCheckedContinuation { pending = $0; entered = true; started?.resume(); started = nil }
    }
    func wait() async {
        if entered { return }
        await withCheckedContinuation { started = $0 }
    }
    func release() { pending?.resume(); pending = nil }
}

@Test(.timeLimit(.minutes(1))) func h3StreamedConsumerBackpressureKeepsOwnershipThroughCancellationAndUnload() async throws {
    let backend = ChainGateBackend(); await backend.release()
    let session = H3ChainSession(backend: backend), consumer = ConsumerGate()
    let run = Task { try await session.run(onOutput: { index, _ in await consumer.receive(index) }) }
    await consumer.wait(); run.cancel()
    await #expect(throws: AetherError.sessionBusy) { try await session.run() }
    let closing = Task { await session.unload() }
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await session.state != .closing, ContinuousClock.now < deadline { await Task.yield() }
    #expect(await session.state == .closing)
    #expect(await consumer.seen == [0])
    await consumer.release()
    await #expect(throws: CancellationError.self) { try await run.value }
    await closing.value
    #expect(await consumer.seen == [0])
    #expect(await session.state == .closed)
}

@Test func h3StreamedConsumerFailureReleasesSlotForFreshRetry() async throws {
    let backend = ChainGateBackend(); await backend.release()
    let session = H3ChainSession(backend: backend)
    await #expect(throws: AetherError.invalidTensor("consumer failure")) {
        try await session.run(onOutput: { index, _ in
            if index == 1 { throw AetherError.invalidTensor("consumer failure") }
        })
    }
    #expect(await session.state == .ready)
    try await session.run(onOutput: { _, value in
        let floats = try value.floatValues(); #expect(floats.count == 1)
    })
    #expect(await backend.calls == 2)
    await session.unload()
}
