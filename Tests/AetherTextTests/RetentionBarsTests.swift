import Testing
@testable import AetherText

/// G5's memory bars (``RetentionBars``), hermetic, on the readings of the 2026-11-11 G5 run (M5 Max, the 4B Qwen3-VL
/// export, 20 requests, warm-up 8, MiB) and on variants of it.
@Suite struct RetentionBarsTests {
    static let mib: Int64 = 1 << 20
    static let budget: Int64 = 2048 << 20

    /// (level, peak, owned, reloads) after each request, in MiB: one reload, after request 16.
    static let run: [(Int64, Int64, Int64, Int)] = [
        (10436, 10740, 251, 0), (10620, 10740, 251, 0), (10620, 10740, 251, 0), (10620, 10740, 251, 0),
        (10620, 10740, 251, 0), (11634, 11634, 395, 0), (11630, 11634, 395, 0), (11645, 11645, 395, 0),
        (11904, 11904, 395, 0), (11904, 11904, 395, 0), (11904, 11904, 395, 0), (11904, 11904, 395, 0),
        (11904, 11904, 395, 0), (12514, 12514, 395, 0), (12511, 12515, 395, 0), (12526, 12526, 395, 0),
        (10301, 12784, 395, 1), (10541, 12784, 395, 1), (10541, 12784, 395, 1), (10541, 12784, 395, 1),
    ]

    static func samples(_ rows: [(Int64, Int64, Int64, Int)]) -> [RetentionBars.Sample] {
        rows.map { RetentionBars.Sample(currentBytes: $0.0 * mib, peakBytes: $0.1 * mib, ownedBytes: $0.2 * mib, reloads: $0.3) }
    }

    @Test func theMeasuredRunPassesAndRecordsBothGrowths() {
        let r = RetentionBars.evaluate(Self.samples(Self.run), warmUp: 8, reloadBudgetBytes: Self.budget)
        #expect(r.failures.isEmpty, "\(r.failures)")
        #expect(r.growthAfterWarmUpBytes == 881 * Self.mib)  // the old bar's figure (printed 880, truncated)
        #expect(r.peakGrowthAfterWarmUpBytes == 1139 * Self.mib)  // over 1 GiB: why it can't be the bar
        #expect(r.floorBytes == (10436 - 251) * Self.mib)
        #expect(r.allowanceBytes == (11634 - 10620) * Self.mib)  // the three-frame turn's first run
        #expect(r.highestRetainedAfterRequestBytes == (12526 - 395 - 10185) * Self.mib)
    }

    @Test func aMissedReloadFails() {
        // The same run with the reload after request 16 missing: the level keeps the request's growth.
        var rows = Self.run
        for i in 16..<rows.count { rows[i] = (12784, 12784, 395, 0) }
        let r = RetentionBars.evaluate(Self.samples(rows), warmUp: 8, reloadBudgetBytes: Self.budget)
        #expect(r.failures.count == 4)
        #expect(r.failures.allSatisfy { $0.contains("a reload was missed") })
    }

    @Test func aReloadThatDoesNotReleaseFails() {
        var rows = Self.run
        for i in 16..<rows.count { rows[i] = (12700, 12784, 395, 1) }
        let r = RetentionBars.evaluate(Self.samples(rows), warmUp: 8, reloadBudgetBytes: Self.budget)
        #expect(!r.failures.isEmpty)
    }

    @Test func aPeakPastOneRequestOverTheBudgetFails() {
        var rows = Self.run
        rows[16] = (10301, 14000, 395, 1)
        let r = RetentionBars.evaluate(Self.samples(rows), warmUp: 8, reloadBudgetBytes: Self.budget)
        #expect(r.failures.count == 1)
        #expect(r.failures.first?.hasPrefix("peak level 14000 MiB") == true)
    }

    @Test func withoutABudgetTheBarIsGrowthAtThePeak() {
        let r = RetentionBars.evaluate(Self.samples(Self.run), warmUp: 8, reloadBudgetBytes: nil)
        #expect(r.failures.count == 1)  // 1,139 MiB at the peak
        let flat = Self.samples(Array(repeating: (10436, 10740, 251, 0), count: 20))
        #expect(RetentionBars.evaluate(flat, warmUp: 8, reloadBudgetBytes: 0).failures.isEmpty)
    }

    @Test func tooFewRequestsFail() {
        let r = RetentionBars.evaluate(Self.samples(Array(Self.run.prefix(8))), warmUp: 8, reloadBudgetBytes: Self.budget)
        #expect(!r.failures.isEmpty)
    }
}
