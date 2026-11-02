import AetherKit
@testable import AetherLanguage
import Foundation
import Testing

/// Sleeps that end only when the test fires them.
private final class Sleeper: @unchecked Sendable {
    private let lock = NSLock()
    private var waiters: [CheckedContinuation<Void, any Error>] = []
    var pending: Int { lock.withLock { waiters.count } }
    func sleep(_ duration: Duration) async throws {
        try await withCheckedThrowingContinuation { continuation in lock.withLock { waiters.append(continuation) } }
    }
    func fire() {
        let ready = lock.withLock { () -> [CheckedContinuation<Void, any Error>] in defer { waiters = [] }; return waiters }
        for waiter in ready { waiter.resume() }
    }
}

private final class LoadCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    var loads: Int { lock.withLock { count } }
    func increment() { lock.withLock { count += 1 } }
}

private func settle(_ model: AetherLanguageModel, until done: @escaping @Sendable (ModelLifecycleState) -> Bool) async throws {
    try await withThrowingTaskGroup(of: Bool.self) { group in
        group.addTask { for await state in model.lifecycleStates where done(state) { return true }; return false }
        group.addTask { try await Task.sleep(for: .seconds(5)); return false }
        let reached = try await group.next() ?? false
        group.cancelAll()
        #expect(reached, "lifecycle stayed at \(model.lifecycleState)")
    }
}

struct LanguageModelLifecycleTests {
    private let monitor = MemoryPressureMonitor(systemSource: false)
    private let sleeper = Sleeper()
    private let loads = LoadCounter()

    private func model(backend: FakeBackend? = nil,
                       policy: ModelLifecyclePolicy = ModelLifecyclePolicy(idleUnloadAfter: .seconds(60))) -> AetherLanguageModel {
        let lifecycle = ModelLifecycle(policy: policy, monitor: monitor, sleep: { [sleeper] in try await sleeper.sleep($0) })
        return FakeModel.make(backend: backend, lifecycle: lifecycle) { [loads] in loads.increment() }
    }

    @Test func aModelLoadsForARequestAndUnloadsWhenIdle() async throws {
        let backend = FakeBackend { _ in FakeVocabulary.tokens("Hi") }
        let model = model(backend: backend)
        #expect(model.lifecycleState == .unloaded(nil))
        _ = try await AetherLanguageSession(model: model).respond(to: "Hello")
        #expect(model.lifecycleState == .loaded)
        while sleeper.pending == 0 { await Task.yield() }
        sleeper.fire()
        try await settle(model) { $0 == .unloaded(.idle) }
        #expect(await backend.unloads == 1)
        // The next request loads again.
        _ = try await AetherLanguageSession(model: model).respond(to: "Again")
        #expect(loads.loads == 2 && model.lifecycleState == .loaded)
    }

    @Test func anIdleReleaseWaitsForARunningRequest() async throws {
        let backend = FakeBackend(stepDelay: .milliseconds(20)) { _ in FakeVocabulary.tokens("A slow reply") }
        let model = model(backend: backend)
        try await model.prewarm()
        while sleeper.pending == 0 { await Task.yield() }
        let request = Task { try await AetherLanguageSession(model: model).respond(to: "Hello") }
        try await settle(model) { $0 == .running }
        sleeper.fire()  // the timer from before the request is stale
        let reply = try await request.value
        #expect(reply.content == "A slow reply")
        #expect(await backend.unloads == 0 && model.lifecycleState == .loaded)
    }

    @Test func memoryPressureReleasesAnIdleModel() async throws {
        let backend = FakeBackend { _ in FakeVocabulary.tokens("Hi") }
        let model = model(backend: backend)
        try await model.prewarm()
        #expect(model.lifecycleState == .loaded)
        monitor.deliver(.warning)
        try await settle(model) { $0 == .unloaded(.memoryPressure) }
        #expect(await backend.unloads == 1)
        monitor.deliver(.normal)
    }

    @Test func criticalPressureRefusesALoad() async throws {
        let model = model()
        monitor.deliver(.critical)
        await #expect(throws: AetherLanguageError.self) { try await model.prewarm() }
        do {
            _ = try await AetherLanguageSession(model: model).respond(to: "Hello")
            Issue.record("a load ran under critical pressure")
        } catch AetherLanguageError.insufficientMemory {
        }
        #expect(loads.loads == 0 && model.lifecycleState == .unloaded(nil))
        monitor.deliver(.normal)
        _ = try await AetherLanguageSession(model: model).respond(to: "Hello")
        #expect(loads.loads == 1)
    }

    @Test func unloadReportsItWasRequested() async throws {
        let model = model()
        try await model.prewarm()
        await model.unload()
        #expect(model.lifecycleState == .unloaded(.requested))
    }

    @Test func copiesShareOneLifecycle() async throws {
        let model = model()
        let copy = model
        try await copy.prewarm()
        #expect(model.lifecycleState == .loaded && model.lifecyclePolicy == ModelLifecyclePolicy(idleUnloadAfter: .seconds(60)))
    }
}
