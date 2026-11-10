import Synchronization
import Testing
@testable import AetherRuntime
import AetherTestSupport

/// The reload budget (docs/VLM.md, V6) on the fake decoder: a level probe the test moves by hand stands in for Core
/// AI's retention, and the reloader hands back fresh fakes.
@Suite("Reload budget (fake)")
struct ReloadBudgetTests {
    final class Probe: Sendable {
        let level = Mutex<Int64>(1 << 30)
        let loads = Mutex(0)
        let failNext = Mutex(false)
        func raise(by n: Int64) { level.withLock { $0 += n } }
    }

    static func engine(budget: Int64?, probe: Probe, config: FakeLM.Config = FakeLM.Config()) async throws -> DecoderEngine {
        var options = DecoderEngineTests.options()
        options.reloadBudgetBytes = budget
        let step = FakeLM(config)
        return try await DecoderEngine(
            step: step, graph: step.manifestGraph(), device: DecoderEngineTests.mac, options: options,
            reloader: { () async throws(AetherError) -> (step: any GraphBackend, prefill: (any GraphBackend)?) in
                probe.loads.withLock { $0 += 1 }
                if probe.failNext.withLock({ let f = $0; $0 = false; return f }) {
                    throw .runtime(underlying: CancellationError())
                }
                return (FakeLM(config), nil)
            },
            levelProbe: { probe.level.withLock { $0 } })
    }

    /// The first call sets the reference level; a level past it by more than the budget reloads, once, and the state
    /// survives: the position stays, and decoding continues exactly as on an engine that never reloaded.
    @Test func retentionPastTheBudgetReloadsAndKeepsTheState() async throws {
        let probe = Probe()
        let engine = try await Self.engine(budget: 100, probe: probe)
        #expect(engine.reloadBudgetBytes == 100)
        let prompt = DecoderEngineTests.prompt(40)
        _ = try await engine.prefill(Array(prompt.prefix(30)))
        #expect(try await engine.reloadIfRetainedPastBudget() == false)
        probe.raise(by: 100)
        #expect(try await engine.reloadIfRetainedPastBudget() == false)  // at the budget, not past it
        probe.raise(by: 1)
        #expect(try await engine.reloadIfRetainedPastBudget() == true)
        let reloads = await engine.reloads
        #expect(reloads.count == 1 && reloads.lastRetainedBytes == 101 && probe.loads.withLock { $0 } == 1)
        #expect(await engine.position == 30)
        let continued = try await LanguageEngineInvariants.continueGreedy(engine, after: Array(prompt.dropFirst(30)), count: 10)
        #expect(continued.tokens == DecoderEngineTests.expectedGreedy(prompt, count: 10))
        // Still past the reference after the reload (nothing here frees the fake level): it reloads again, so the
        // reference is the first call's, not the level after a reload.
        #expect(try await engine.reloadIfRetainedPastBudget() == true)
        probe.raise(by: -101)
        #expect(try await engine.reloadIfRetainedPastBudget() == false)
        #expect(await engine.reloads.count == 2)
    }

    /// A failed load leaves the graphs unloaded: feeding refuses nothing, it loads them first.
    @Test func aFailedReloadIsRetriedByTheNextFeed() async throws {
        let probe = Probe()
        let engine = try await Self.engine(budget: 10, probe: probe)
        let prompt = DecoderEngineTests.prompt(24)
        _ = try await engine.prefill(Array(prompt.prefix(12)))
        _ = try await engine.reloadIfRetainedPastBudget()
        probe.raise(by: 11)
        probe.failNext.withLock { $0 = true }
        await #expect(throws: AetherError.self) { try await engine.reloadIfRetainedPastBudget() }
        #expect(await engine.reloads.count == 0)
        let continued = try await LanguageEngineInvariants.continueGreedy(engine, after: Array(prompt.dropFirst(12)), count: 6)
        #expect(continued.tokens == DecoderEngineTests.expectedGreedy(prompt, count: 6))
        #expect(await engine.reloads.count == 1 && probe.loads.withLock { $0 } == 2)
    }

    /// Who reloads: only an engine that can (a bundle load, or here a reloader), only without recurrent state, and only
    /// a Core AI graph by default; 0 opts out, a negative budget is refused.
    @Test func theBudgetResolves() async throws {
        let probe = Probe()
        #expect(try await DecoderEngineTests.engine().0.reloadBudgetBytes == nil)
        #expect(try await Self.engine(budget: nil, probe: probe).reloadBudgetBytes == nil)  // a fake is not Core AI
        #expect(try await Self.engine(budget: 0, probe: probe).reloadBudgetBytes == nil)
        #expect(try await Self.engine(budget: 5, probe: probe, config: FakeLM.Config(staticStep: true, recurrent: true))
            .reloadBudgetBytes == nil)
        await #expect(throws: AetherError.self) { _ = try await Self.engine(budget: -1, probe: probe) }
        let off = try await Self.engine(budget: 0, probe: probe)
        probe.raise(by: 1 << 40)
        #expect(try await off.reloadIfRetainedPastBudget() == false)
        #expect(DecoderEngine.defaultReloadBudgetBytes == 2 << 30)
    }
}
