import AetherCoreAI
import AetherKit
import AetherSpeech
import Foundation
import Synchronization
import Testing
@testable import AetherSpeechCoreAI

/// Waits until the session reaches a lifecycle state `done` accepts (at most 5 s).
private func settle(_ speech: KokoroSpeechSession, until done: @escaping @Sendable (ModelLifecycleState) -> Bool) async throws {
    try await withThrowingTaskGroup(of: Bool.self) { group in
        group.addTask { for await state in speech.lifecycleStates where done(state) { return true }; return false }
        group.addTask { try await Task.sleep(for: .seconds(5)); return false }
        let reached = try await group.next() ?? false
        group.cancelAll()
        #expect(reached, "lifecycle stayed at \(speech.lifecycleState)")
    }
}

@Test func kokoroReleasesItsModelsUnderPressureAndStaysUsable() async throws {
    let bundle = try FakeBundle.make()
    defer { bundle.remove() }
    let log = EventLog()
    let monitor = MemoryPressureMonitor(systemSource: false)
    let speech = KokoroSpeechSession(assets: try KokoroAssets.open(at: bundle.root), factory: FakeFactory(log: log),
                                     lifecycle: ModelLifecycle(policy: ModelLifecyclePolicy(idleUnloadAfter: nil), monitor: monitor))
    let phonemes = try speech.phonemes("həlˈO")
    #expect(speech.lifecycleState == .unloaded(nil))
    _ = try await speech.synthesize(phonemes, configuration: try KokoroSpeechConfiguration())
    #expect(speech.lifecycleState == .loaded)

    monitor.deliver(.warning)
    try await settle(speech) { $0 == .unloaded(.memoryPressure) }
    // The state turns unloaded first; the models close right after.
    let deadline = ContinuousClock.now + .seconds(5)
    while log.snapshot.filter({ $0.hasPrefix("close:") }).count < 2, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
    #expect(log.snapshot.filter { $0.hasPrefix("close:") }.sorted() == ["close:decode", "close:durations"])
    #expect(await speech.state == .ready)

    // The next request opens both models again.
    monitor.deliver(.normal)
    _ = try await speech.synthesize(phonemes, configuration: try KokoroSpeechConfiguration())
    #expect(log.snapshot.filter { $0.hasPrefix("open:") }.count == 4)
    #expect(speech.lifecycleState == .loaded)
    await speech.unload()
    #expect(speech.lifecycleState == .unloaded(.requested))
}

@Test func kokoroRefusesToOpenModelsUnderCriticalPressure() async throws {
    let bundle = try FakeBundle.make()
    defer { bundle.remove() }
    let log = EventLog()
    let monitor = MemoryPressureMonitor(systemSource: false)
    let speech = KokoroSpeechSession(assets: try KokoroAssets.open(at: bundle.root), factory: FakeFactory(log: log),
                                     lifecycle: ModelLifecycle(policy: .manual, monitor: monitor))
    monitor.deliver(.critical)
    await #expect(throws: AetherError.memoryPressure) {
        _ = try await speech.synthesize(try speech.phonemes("həlˈO"), configuration: try KokoroSpeechConfiguration())
    }
    #expect(!log.snapshot.contains { $0.hasPrefix("open:") })
    #expect(await speech.state == .ready && speech.lifecycleState == .unloaded(nil))
}
