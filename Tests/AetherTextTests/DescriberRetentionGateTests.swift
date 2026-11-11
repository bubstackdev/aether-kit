import AetherCore
import AetherHost
import AetherRuntime
import Foundation
import Metal
import Testing
@testable import AetherText

/// The describer's retention and speed gate (docs/VLM.md §6, G5), opt-in, on the same export, pictures and 0.0.7
/// reference as ``DescriberGateTests``:
///
///     AETHER_GATES=1 AETHER_VLM_RETENTION=1 AETHER_VLM_BUNDLE=<export> AETHER_VLM_PICTURES=<pictures> \
///         AETHER_VLM_REFERENCE=<reference> [AETHER_VLM_RETENTION_REPORT=<new .json>] \
///         scripts/gpu-lease -- swift test -c release --no-parallel --filter DescriberRetentionGateTests
///
/// One engine (capacity 4,096) answers 20 requests, cycling through the gate's cases, which mix grids, image counts and
/// text-only. The warm-up is the first pass over every case (each grid and image count seen once). Recorded per
/// request: the engine's ``DecoderEngine/Memory`` after it (and the device's Metal allocation), its reload count,
/// prefill seconds and decode ms per token beside the 0.0.7 reference's. The memory bars are
/// ``RetentionBars`` (the reload budget's bound when the engine has one, growth after warm-up under 1 GiB when it
/// doesn't); the measured model bytes, charged at a second load as `Options.modelCharge` `.declared`, must still leave
/// the capacity of 4,096. Then the second load's charge and capacity are reported, and the run prints the export's
/// ``DecoderEngine/qualifiedModelBytesTable`` entry (`GATE qualified model bytes entry:`, also `qualifiedEntry` in the
/// report) ready to add to `Sources/AetherRuntime/Decoder/QualifiedModelBytes.swift`. When the kit already has an entry
/// for the export on this chip, it must cover the measurement.
@Suite("DescriberRetentionGateTests", .serialized,
       .enabled(if: DescriberGate.enabled && ProcessInfo.processInfo.environment["AETHER_VLM_RETENTION"] == "1"))
struct DescriberRetentionGateTests {
    static let requests = 20

    struct Row: Codable {
        let request: Int
        let name: String
        let pictures: Int
        let promptTokens: Int
        let generated: Int
        let prefillSeconds: Double
        let decodeMillisecondsPerToken: Double
        let referenceTimeToFirstTokenSeconds: Double
        let currentBytes: Int64
        let peakBytes: Int64
        let ownedBytes: Int64
        let measuredModelBytes: Int64
        let metalAllocatedBytes: Int64
        /// The engine's reloads since load, after this request (``DecoderEngine/reloads``).
        let reloads: Int
    }

    struct Report: Codable {
        var loadMemory: [String: Int64] = [:]
        var rows: [Row] = []
        var warmUpRequests = 0
        /// Recorded, not a bar when the engine reloads: the highest level after a request, past the warm-up's last.
        var growthAfterWarmUpBytes: Int64 = 0
        /// Recorded: the highest level read at all (``DecoderEngine/Memory/peakBytes``, inside requests too), past
        /// the warm-up's last level.
        var peakGrowthAfterWarmUpBytes: Int64 = 0
        var reloadBudgetBytes: Int64?
        var bars: RetentionBars.Result?
        var redeclaredChargeBytes: Int64 = 0
        var redeclaredCapacity = 0
        var qualifiedModelBytes: Int64?
        var qualifiedEntry = ""
    }

    @Test func metalGrowthAfterWarmUpStaysUnderABar() async throws {
        let (cases, reference) = try DescriberGateTests.load()
        let pictures = DescriberGateTests.url("AETHER_VLM_PICTURES")
        let bundleURL = DescriberGateTests.url("AETHER_VLM_BUNDLE")
        let bundle = try DescriberGateTests.localBundle(bundleURL)
        let manifest = try Manifest.load(fromBundle: bundleURL)
        let metal = MTLCreateSystemDefaultDevice()
        var report = Report()
        var measured: Int64 = 0
        do {
            let engine = try await DecoderEngine(bundle: bundle, manifest: manifest, device: .current,
                                                 options: DecoderEngine.Options(capacity: 4096))
            report.reloadBudgetBytes = engine.reloadBudgetBytes
            let loaded = await engine.memory
            report.loadMemory = [
                "baseline": loaded.baselineBytes, "current": loaded.currentBytes, "owned": loaded.ownedBytes,
                "charged": loaded.chargedModelBytes, "measuredModel": loaded.measuredModelBytes,
            ]
            print("GATE retention load: charged \(loaded.chargedModelBytes >> 20) MiB, resident \(loaded.liveResidentBytes >> 20) MiB, "
                  + "measured model \(loaded.measuredModelBytes >> 20) MiB, owned \(loaded.ownedBytes >> 20) MiB, "
                  + "reload budget \(engine.reloadBudgetBytes.map { "\($0 >> 20) MiB" } ?? "none")")
            let generator = try await ChatSession.generator(engine: engine, bundle: bundle, manifest: manifest)
            report.warmUpRequests = cases.cases.count
            for i in 0..<Self.requests {
                let c = cases.cases[i % cases.cases.count]
                let record = reference.cases.first { $0.name == c.name }
                let images = c.pictures.map { ChatImage(ImageInput(contentsOf: pictures.appending(path: $0))) }
                let messages: [ChatMessage] = (c.instructions.isEmpty ? [] : [.system(c.instructions)])
                    + [.user(c.question, images: images)]
                var result: GenerationSummary?
                for try await event in generator.generate(messages, parameters: GenerationParameters(maxTokens: c.maximumTokens)) {
                    if case .finished(let s) = event { result = s }
                }
                let r = try #require(result)
                let m = await engine.memory
                let reloads = await engine.reloads.count
                let row = Row(
                    request: i, name: c.name, pictures: c.pictures.count, promptTokens: r.promptTokens, generated: r.tokens.count,
                    prefillSeconds: r.prefillSeconds, decodeMillisecondsPerToken: r.decodeSeconds * 1000 / Double(max(1, r.tokens.count - 1)),
                    referenceTimeToFirstTokenSeconds: record?.timeToFirstTokenSeconds ?? 0, currentBytes: m.currentBytes,
                    peakBytes: m.peakBytes, ownedBytes: m.ownedBytes, measuredModelBytes: m.measuredModelBytes,
                    metalAllocatedBytes: Int64(metal?.currentAllocatedSize ?? 0), reloads: reloads)
                report.rows.append(row)
                print("GATE retention \(i) \(c.name): \(c.pictures.count) picture(s), prefill \(String(format: "%.3f", row.prefillSeconds)) s "
                      + "(0.0.7 first token \(String(format: "%.3f", row.referenceTimeToFirstTokenSeconds)) s), decode "
                      + "\(String(format: "%.1f", row.decodeMillisecondsPerToken)) ms/token, level \(m.currentBytes >> 20) MiB, "
                      + "peak \(m.peakBytes >> 20) MiB, Metal \(row.metalAllocatedBytes >> 20) MiB, "
                      + "measured model \(m.measuredModelBytes >> 20) MiB, reloads \(reloads)")
            }
            measured = await engine.memory.measuredModelBytes
        }
        let samples = report.rows.map {
            RetentionBars.Sample(currentBytes: $0.currentBytes, peakBytes: $0.peakBytes, ownedBytes: $0.ownedBytes, reloads: $0.reloads)
        }
        let bars = RetentionBars.evaluate(samples, warmUp: report.warmUpRequests, reloadBudgetBytes: report.reloadBudgetBytes)
        report.bars = bars
        report.growthAfterWarmUpBytes = bars.growthAfterWarmUpBytes
        report.peakGrowthAfterWarmUpBytes = bars.peakGrowthAfterWarmUpBytes
        print("GATE retention: growth after warm-up \(bars.growthAfterWarmUpBytes >> 20) MiB after requests, "
              + "\(bars.peakGrowthAfterWarmUpBytes >> 20) MiB at the peak (recorded), \(samples.last?.reloads ?? 0) reload(s), "
              + "measured model \(measured >> 20) MiB")
        for line in bars.lines { print("GATE retention bar: \(line)") }
        for failure in bars.failures { Issue.record(Comment(rawValue: failure)) }
        report.qualifiedEntry = DecoderEngine.qualifiedEntry(
            measured: measured, manifest: manifest, device: .current,
            note: "G5 retention gate, \(Self.requests) mixed requests at capacity 4,096")
        print("GATE qualified model bytes entry: \(report.qualifiedEntry)")
        report.qualifiedModelBytes = DecoderEngine.qualifiedModelBytes(for: manifest, device: .current)
        if let qualified = report.qualifiedModelBytes {
            print("GATE retention: the kit's qualified figure \(qualified >> 20) MiB, measured \(measured >> 20) MiB")
            #expect(qualified >= measured, "the qualified entry no longer covers the measurement; replace it with the one above")
        }
        // The first engine is released: a second load charges the measurement.
        var options = DecoderEngine.Options(capacity: 4096)
        options.modelCharge = .declared(bytes: measured)
        let again = try await DecoderEngine(bundle: bundle, manifest: manifest, device: .current, options: options)
        report.redeclaredChargeBytes = again.sizing.modelBytes
        report.redeclaredCapacity = again.maxCapacity
        print("GATE retention redeclared: charged \(again.sizing.modelBytes >> 20) MiB, capacity \(again.maxCapacity)")
        #expect(again.sizing.modelBytes >= measured && again.maxCapacity >= 4096)
        if let path = ProcessInfo.processInfo.environment["AETHER_VLM_RETENTION_REPORT"] {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: URL(fileURLWithPath: path))
        }
    }
}

/// G5's memory bars over the engine's ``DecoderEngine/Memory`` after each request (docs/VLM.md §6, G5 and V6).
///
/// With a reload budget, the engine's guarantee is checked, not a growth figure. The floor is the level after the
/// first request less the engine's buffers (``DecoderEngine/reloadIfRetainedPastBudget()`` takes the same reading at the
/// same point). The allowance is the most one warm-up request raised the level, its peak inside the request included.
/// The bars:
/// 1. **After every request**, the level less the engine's buffers is at most the floor plus the budget. The
///    ``Generator`` reloads at the end of a request that passed it, so this fails when a reload is missed or
///    doesn't release.
/// 2. **At the peak** (``DecoderEngine/Memory/peakBytes``, read inside requests too), the level is at most the floor
///    plus the budget plus the allowance plus the engine's largest buffers: one request on top of the most a reload
///    lets stand.
///
/// The growth after warm-up is recorded either way. It can't be the bar with a budget: the reload counts from the first
/// request, so what the growth past the warm-up's last level can reach is the budget less whatever the warm-up
/// happened to retain (880 MiB measured, 2026-11-11, against 1,209 MiB retained in the warm-up), and the after-request
/// reading misses the peak a reload releases (1,139 MiB). Without a budget (the engine never reloads), the bar
/// stays the growth after warm-up, at the peak, under 1 GiB.
enum RetentionBars {
    static let growthBar: Int64 = 1 << 30
    /// Leeway on bar 1 for the readings' timing: the engine reads its level before the request's stream ends, the gate
    /// after.
    static let readingSlackBytes: Int64 = 16 << 20

    struct Sample: Sendable, Hashable {
        var currentBytes: Int64
        var peakBytes: Int64
        var ownedBytes: Int64
        var reloads: Int
    }

    struct Result: Codable, Sendable, Hashable {
        var growthAfterWarmUpBytes: Int64 = 0
        var peakGrowthAfterWarmUpBytes: Int64 = 0
        var floorBytes: Int64 = 0
        var allowanceBytes: Int64 = 0
        var afterRequestBarBytes: Int64?
        var peakBarBytes: Int64?
        var highestRetainedAfterRequestBytes: Int64 = 0
        var lines: [String] = []
        var failures: [String] = []
    }

    static func evaluate(_ samples: [Sample], warmUp: Int, reloadBudgetBytes: Int64?) -> Result {
        var result = Result()
        guard warmUp >= 1, samples.count > warmUp else {
            result.failures.append("\(samples.count) requests for a warm-up of \(warmUp): nothing after warm-up to check")
            return result
        }
        let warm = samples[warmUp - 1].currentBytes
        let after = samples.dropFirst(warmUp)
        result.growthAfterWarmUpBytes = (after.map(\.currentBytes).max() ?? warm) - warm
        result.peakGrowthAfterWarmUpBytes = (after.map(\.peakBytes).max() ?? warm) - warm

        guard let budget = reloadBudgetBytes, budget > 0 else {
            result.lines.append("no reload budget: growth after warm-up at the peak "
                                + "\(result.peakGrowthAfterWarmUpBytes >> 20) MiB, bar \(growthBar >> 20) MiB")
            if result.peakGrowthAfterWarmUpBytes >= growthBar {
                result.failures.append("growth after warm-up \(result.peakGrowthAfterWarmUpBytes >> 20) MiB at the peak, "
                                       + "bar \(growthBar >> 20) MiB, and the engine has no reload budget")
            }
            return result
        }

        let floor = samples[0].currentBytes - samples[0].ownedBytes
        result.floorBytes = floor
        // The most one warm-up request raised the level: its end level, or its peak when that was a new high.
        var allowance: Int64 = 0
        for i in 1..<warmUp {
            let before = samples[i - 1]
            let rise = samples[i].peakBytes > before.peakBytes ? samples[i].peakBytes : samples[i].currentBytes
            allowance = max(allowance, rise - before.currentBytes)
        }
        result.allowanceBytes = allowance
        let afterRequestBar = floor + budget + readingSlackBytes
        result.afterRequestBarBytes = afterRequestBar
        let ownedMax = samples.map(\.ownedBytes).max() ?? 0
        let peakBar = floor + budget + allowance + ownedMax
        result.peakBarBytes = peakBar

        for (i, s) in samples.enumerated() {
            let retained = s.currentBytes - s.ownedBytes
            result.highestRetainedAfterRequestBytes = max(result.highestRetainedAfterRequestBytes, retained - floor)
            if retained > afterRequestBar {
                result.failures.append("after request \(i) the model holds \((retained - floor) >> 20) MiB past the floor, "
                                       + "over the \(budget >> 20) MiB reload budget (\(s.reloads) reload(s)): a reload was "
                                       + "missed or didn't release")
            }
        }
        let peak = samples.map(\.peakBytes).max() ?? 0
        if peak > peakBar {
            result.failures.append("peak level \(peak >> 20) MiB, over floor \(floor >> 20) + budget \(budget >> 20) + "
                                   + "one request \(allowance >> 20) + buffers \(ownedMax >> 20) = \(peakBar >> 20) MiB")
        }
        result.lines.append("after each request at most \(result.highestRetainedAfterRequestBytes >> 20) MiB past the floor "
                            + "\(floor >> 20) MiB, bar \(budget >> 20) MiB (the reload budget)")
        result.lines.append("peak \(peak >> 20) MiB, bar \(peakBar >> 20) MiB (floor + budget + one warm-up request "
                            + "\(allowance >> 20) MiB + buffers \(ownedMax >> 20) MiB)")
        return result
    }
}
