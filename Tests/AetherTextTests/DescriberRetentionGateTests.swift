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
/// request: the engine's ``DecoderEngine/Memory`` after it (and the device's Metal allocation), prefill seconds and
/// decode ms per token beside the 0.0.7 reference's. Bars: the level after warm-up grows by less than 1 GiB over the
/// remaining requests; the measured model bytes, charged at a second load as `Options.modelBytes`, still leave the
/// capacity of 4,096. Then the second load's charge and capacity are reported.
@Suite("DescriberRetentionGateTests", .serialized,
       .enabled(if: DescriberGate.enabled && ProcessInfo.processInfo.environment["AETHER_VLM_RETENTION"] == "1"))
struct DescriberRetentionGateTests {
    static let requests = 20
    static let growthBar: Int64 = 1 << 30

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
    }

    struct Report: Codable {
        var loadMemory: [String: Int64] = [:]
        var rows: [Row] = []
        var warmUpRequests = 0
        var growthAfterWarmUpBytes: Int64 = 0
        var redeclaredChargeBytes: Int64 = 0
        var redeclaredCapacity = 0
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
            let loaded = await engine.memory
            report.loadMemory = [
                "baseline": loaded.baselineBytes, "current": loaded.currentBytes, "owned": loaded.ownedBytes,
                "charged": loaded.chargedModelBytes, "measuredModel": loaded.measuredModelBytes,
            ]
            print("GATE retention load: charged \(loaded.chargedModelBytes >> 20) MiB, resident \(loaded.liveResidentBytes >> 20) MiB, "
                  + "measured model \(loaded.measuredModelBytes >> 20) MiB, owned \(loaded.ownedBytes >> 20) MiB")
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
                let row = Row(
                    request: i, name: c.name, pictures: c.pictures.count, promptTokens: r.promptTokens, generated: r.tokens.count,
                    prefillSeconds: r.prefillSeconds, decodeMillisecondsPerToken: r.decodeSeconds * 1000 / Double(max(1, r.tokens.count - 1)),
                    referenceTimeToFirstTokenSeconds: record?.timeToFirstTokenSeconds ?? 0, currentBytes: m.currentBytes,
                    peakBytes: m.peakBytes, ownedBytes: m.ownedBytes, measuredModelBytes: m.measuredModelBytes,
                    metalAllocatedBytes: Int64(metal?.currentAllocatedSize ?? 0))
                report.rows.append(row)
                print("GATE retention \(i) \(c.name): \(c.pictures.count) picture(s), prefill \(String(format: "%.3f", row.prefillSeconds)) s "
                      + "(0.0.7 first token \(String(format: "%.3f", row.referenceTimeToFirstTokenSeconds)) s), decode "
                      + "\(String(format: "%.1f", row.decodeMillisecondsPerToken)) ms/token, level \(m.currentBytes >> 20) MiB, "
                      + "Metal \(row.metalAllocatedBytes >> 20) MiB, measured model \(m.measuredModelBytes >> 20) MiB")
            }
            let warm = report.rows[report.warmUpRequests - 1].currentBytes
            let later = report.rows.dropFirst(report.warmUpRequests).map(\.currentBytes).max() ?? warm
            report.growthAfterWarmUpBytes = later - warm
            measured = await engine.memory.measuredModelBytes
        }
        print("GATE retention: growth after warm-up \(report.growthAfterWarmUpBytes >> 20) MiB (bar \(Self.growthBar >> 20) MiB), "
              + "measured model \(measured >> 20) MiB")
        #expect(report.growthAfterWarmUpBytes < Self.growthBar)
        // The first engine is released: a second load charges the measurement.
        var options = DecoderEngine.Options(capacity: 4096)
        options.modelBytes = measured
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
