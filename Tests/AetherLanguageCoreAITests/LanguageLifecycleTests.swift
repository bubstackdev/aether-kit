@testable import AetherLanguageCoreAI
import Testing

/// Mirrors the session: requests are admitted and finished on the owner's actor,
/// and close releases resources once, after any active request has finished.
private actor LifecycleOwner {
    let lifecycle = LanguageSessionLifecycle()
    private(set) var events: [String] = []
    private var pending: CheckedContinuation<Void, Never>?
    var state: LanguageGenerationSession.State { lifecycle.state }
    var requestSuspended: Bool { pending != nil }

    func request() async throws {
        try lifecycle.admit()
        defer { lifecycle.finish() }
        events.append("start")
        await withCheckedContinuation { pending = $0 }
        events.append("end")
    }
    func completeRequest() { pending?.resume(); pending = nil }
    func close() async {
        guard await lifecycle.beginClose() else { events.append("closeReturned"); return }
        events.append("released")
        lifecycle.didClose()
        events.append("closeReturned")
    }
}

private func waitUntil(_ condition: @Sendable () async -> Bool) async {
    for _ in 0..<1000 where !(await condition()) { await Task.yield() }
}

struct LanguageLifecycleTests {
    @Test(.timeLimit(.minutes(1))) func admissionRejectsOverlapAndClosedSessions() async throws {
        let owner = LifecycleOwner()
        let first = Task { try await owner.request() }
        await waitUntil { await owner.requestSuspended }
        #expect(await owner.state == .running)
        await #expect(throws: LanguageGenerationError.busy) { try await owner.request() }
        await owner.completeRequest()
        try await first.value
        #expect(await owner.state == .ready)
        await owner.close()
        #expect(await owner.state == .closed)
        await #expect(throws: LanguageGenerationError.closed) { try await owner.request() }
        // Closing again is a no-op that does not release twice.
        await owner.close()
        #expect(await owner.events.filter { $0 == "released" }.count == 1)
    }

    @Test(.timeLimit(.minutes(1))) func closeWaitsForActiveRequestBeforeReleasing() async throws {
        let owner = LifecycleOwner()
        let request = Task { try await owner.request() }
        await waitUntil { await owner.requestSuspended }
        let close = Task { await owner.close() }
        await waitUntil { await owner.state == .closing }
        #expect(await owner.state == .closing)
        await #expect(throws: LanguageGenerationError.closed) { try await owner.request() }
        #expect(await !owner.events.contains("released"))
        await owner.completeRequest()
        try await request.value
        await close.value
        #expect(await owner.events == ["start", "end", "released", "closeReturned"])
        #expect(await owner.state == .closed)
    }

    @Test(.timeLimit(.minutes(1))) func concurrentClosesReturnAfterSingleRelease() async throws {
        let owner = LifecycleOwner()
        let request = Task { try await owner.request() }
        await waitUntil { await owner.requestSuspended }
        let closes = (0..<3).map { _ in Task { await owner.close() } }
        await waitUntil { await owner.state == .closing }
        await owner.completeRequest()
        try await request.value
        for close in closes { await close.value }
        let events = await owner.events
        #expect(events.filter { $0 == "released" }.count == 1)
        #expect(events.filter { $0 == "closeReturned" }.count == 3)
        // Every close returns only after resources were released.
        #expect(events.firstIndex(of: "closeReturned")! > events.firstIndex(of: "released")!)
    }
}
