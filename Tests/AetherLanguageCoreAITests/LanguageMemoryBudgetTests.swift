@testable import AetherLanguageCoreAI
import Testing

struct LanguageMemoryBudgetTests {
    private static let mib = 1 << 20

    /// A fake process-wide allocation source: this model's retention per request plus whatever another
    /// session allocates or frees inside the same window, measured like the session does (end minus
    /// start). Returns the request count at which the budget first asks for a reload, if any.
    private static func simulate(requests: Int, limit: Int = 1024 * mib, tokens: Int = 457, retention: Int,
                                 foreign: [Int: Int] = [:]) -> (reloadAt: Int?, budget: LanguageMemoryBudget) {
        var budget = LanguageMemoryBudget(limitBytes: limit)
        var allocated = 8 * 1024 * mib
        for index in 0..<requests {
            let start = allocated
            allocated += retention + (foreign[index] ?? 0)
            budget.record(requestGrowthBytes: allocated - start, submittedTokens: tokens, warming: false)
            if budget.exhausted { return (index, budget) }
        }
        return (nil, budget)
    }

    /// The measured leak pattern (the writer: 137 MiB per 457-token request) reloads after the eighth request.
    @Test func retentionReloadsOnceItPassesTheLimit() {
        let run = Self.simulate(requests: 60, retention: 137 * Self.mib)
        #expect(run.reloadAt == 7)
        #expect(run.budget.foreignRequests == 0)
    }

    /// Another session loading (+3 GiB) and later unloading (-3 GiB) inside this model's windows, once its
    /// rate is known, neither triggers a reload nor hides the model's own retention.
    @Test func concurrentForeignGrowthAndFreesAreSetAside() {
        let noLeak = Self.simulate(requests: 60, retention: 2 * Self.mib, foreign: [5: 3072 * Self.mib, 9: -3072 * Self.mib])
        #expect(noLeak.reloadAt == nil)
        #expect(noLeak.budget.foreignRequests == 2 && noLeak.budget.foreignBytes == 6144 * Self.mib)
        #expect(noLeak.budget.retainedBytes == 58 * 2 * Self.mib)
        // The same events around the real leak: the reload comes after the same number of counted requests.
        let leak = Self.simulate(requests: 60, retention: 137 * Self.mib, foreign: [3: 3072 * Self.mib, 5: -3072 * Self.mib])
        #expect(leak.reloadAt == 9)
        #expect(leak.budget.foreignRequests == 2)
    }

    /// A net decrease is never this model's (it frees nothing of its own inside a window) and never offsets
    /// its retention.
    @Test func netDecreaseIsForeign() {
        var budget = LanguageMemoryBudget(limitBytes: 1000)
        #expect(budget.record(requestGrowthBytes: 400, submittedTokens: 10, warming: false) == .counted)
        #expect(budget.record(requestGrowthBytes: -900, submittedTokens: 10, warming: false) == .foreign)
        #expect(budget.retainedBytes == 400 && budget.foreignBytes == 900)
    }

    /// Before the per-token rate is known, large growth counts: one unnecessary reload is cheaper than
    /// unbounded retention. A longer prompt's proportionally larger retention is not an outlier.
    @Test func largeGrowthCountsUntilTheRateIsKnownAndScalesWithTokens() {
        var budget = LanguageMemoryBudget(limitBytes: 1024 * Self.mib)
        #expect(budget.record(requestGrowthBytes: 700 * Self.mib, submittedTokens: 457, warming: false) == .counted)
        budget.reset()
        for _ in 0..<3 { budget.record(requestGrowthBytes: 137 * Self.mib, submittedTokens: 457, warming: false) }
        // A 4,096-token prompt retaining at the same rate (about 1.2 GiB) is the model's own.
        #expect(budget.record(requestGrowthBytes: 1228 * Self.mib, submittedTokens: 4096, warming: false) == .counted)
        #expect(budget.exhausted)
        // The same growth on a short request is far above the rate: foreign.
        #expect(budget.record(requestGrowthBytes: 1228 * Self.mib, submittedTokens: 457, warming: false) == .foreign)
    }

    /// A request that warms a function or shape is not counted, however large: its growth is the one-time
    /// weight copy or specialization.
    @Test func warmingRequestsAreNotCounted() {
        var budget = LanguageMemoryBudget(limitBytes: 1000)
        #expect(budget.record(requestGrowthBytes: 9_000_000_000, submittedTokens: 457, warming: true) == .warming)
        #expect(budget.retainedBytes == 0 && budget.foreignRequests == 0 && !budget.exhausted)
    }

    /// nil never reloads.
    @Test func noLimitNeverExhausts() {
        var budget = LanguageMemoryBudget(limitBytes: nil)
        for _ in 0..<100 { budget.record(requestGrowthBytes: 1 << 30, submittedTokens: 457, warming: false) }
        #expect(!budget.exhausted)
    }
}
