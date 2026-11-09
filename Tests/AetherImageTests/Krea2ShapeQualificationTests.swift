import AetherCore
import AetherHost
import AetherRuntime
import CryptoKit
import Foundation
import Metal
import Testing
@testable import AetherImage

/// Native qualification of Krea 2's per-row-count denoiser specializations (``ResidentAdmission/Settings``'s
/// `maximumShapes`), conditioning reuse and the compiled-cache lookup, on the realism export. Off by default: run under
/// the shared GPU lease with
///
///     AETHER_KREA2_SHAPES=1 AETHER_KREA2_SHAPES_REPORT=<new .json> \
///       scripts/gpu-lease -- swift test --no-parallel --filter Krea2ShapeQualificationTests
///
/// optional AETHER_KREA2_SHAPES_BUNDLE (default: the Forge realism output), _SIZE (1024), _STEPS (8).
///
/// One pipeline renders neutral-object prompts of five row counts in the order A B C D A B E A at one seed and size:
/// new row counts, revisits, and a fifth past `maximumShapes`; admission keeps a row count's specialization beside the
/// others only while it fits with every graph resident, and otherwise reloads for it alone. Each
/// render's metrics (wait, phases, every step, loaded roles, Metal peak) and the Metal allocation and resident bytes
/// after it go to the report. Then the compile-cache lookup is timed with the graphs released, and A is encoded once
/// and rendered from the reused conditioning, which must match A's pixels. No image is saved; the filter allows
/// everything (a measurement, not a deployment).
///
/// Recorded 2026-11-08 (realism, 1024², 8 steps, M5 Max 64 GB): a second row count kept about 27 GB more (Metal 47.3 →
/// 74.5 GB), so on 64 GB each new row count reloads. With one specialization, a change of row count costs the reload
/// (15–25 s) and a first step of 35–43 s against 2.5–3 s for later steps: 74–98 s a render. Revisits are
/// bit-identical, reused conditioning matches to the pixel, and the compile-cache lookup took 5 ms and no Metal. The
/// denoisers loaded with the reshape hint give a non-finite velocity on the first render, so that's not an option.
@Suite("Krea 2 shape qualification", .serialized,
       .enabled(if: ProcessInfo.processInfo.environment["AETHER_KREA2_SHAPES"] == "1"
                && ProcessInfo.processInfo.environment["AETHER_KREA2_SHAPES_REPORT"] != nil))
struct Krea2ShapeQualificationTests {
    static let environment = ProcessInfo.processInfo.environment

    static let texts: [(name: String, text: String)] = [
        ("A", "A red ceramic mug."),
        ("B", "A red ceramic mug on a white tabletop beside a folded linen napkin."),
        ("C", "A red ceramic mug on a white tabletop beside a folded linen napkin, soft morning light from a window."),
        ("D", "A red ceramic mug on a white tabletop beside a folded linen napkin, soft morning light from a window, a small potted fern in the background."),
        ("E", "A red ceramic mug on a white tabletop beside a folded linen napkin, soft morning light from a window, a small potted fern in the background, shallow depth of field, muted colors."),
    ]
    static let order = ["A", "B", "C", "D", "A", "B", "E", "A"]

    struct Render: Codable {
        let index: Int
        let prompt: String
        let rows: Int
        let firstUseOfRows: Bool
        let metrics: ImageGenerationMetrics
        let metalAllocatedAfter: Int
        let residentBytesAfter: Int64
        let pixelsSHA256: String
    }

    struct CacheLookup: Codable {
        let seconds: Double
        let metalGrowthBytes: Int
        let entries: [String: Bool]
        let complete: Bool
    }

    struct Report: Codable {
        let bundle: String
        let side: Int
        let steps: Int
        let seed: UInt64
        let architecture: String
        let renders: [Render]
        let cacheLookup: CacheLookup
        let encodeSeconds: Double
        let reuse: ImageGenerationMetrics
        let reuseMatchesFirstA: Bool
        let reuseMaximumPixelDifference: Float
        let revisitsIdentical: Bool
    }

    final class AllowAll: ContentFilter {
        func review(prompt: String) async throws -> ContentVerdict { .allow }
        func review(image: GeneratedImage, prompt: String) async throws -> ContentVerdict { .allow }
    }

    static func digest(_ image: GeneratedImage) -> String {
        image.planarRGB.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
    }

    @Test(.timeLimit(.minutes(90)))
    func perRowCountSpecializationsReuseAndCacheLookup() async throws {
        let report = URL(filePath: try #require(Self.environment["AETHER_KREA2_SHAPES_REPORT"]))
        #expect(!FileManager.default.fileExists(atPath: report.path), "the report must be a new file")
        let side = Int(Self.environment["AETHER_KREA2_SHAPES_SIZE"] ?? "") ?? 1024
        let steps = Int(Self.environment["AETHER_KREA2_SHAPES_STEPS"] ?? "") ?? 8
        let seed: UInt64 = 18
        let bundleURL = Self.environment["AETHER_KREA2_SHAPES_BUNDLE"].map { URL(filePath: $0) }
            ?? LatentDiffusionGateTests.outputs.appending(path: "krea-2-turbo-realism")
        let bundle = try LatentDiffusionGateTests.localBundle(bundleURL)
        let device = LatentDiffusionGateTests.mac
        let metal = try #require(MTLCreateSystemDefaultDevice())
        let model = try await Registry.image.load(
            bundle, as: Krea2Driver.self, options: LoadOptions(device: device, cachePolicy: .default, cache: .default))
        let pipeline = try await model.pipeline(filter: AllowAll())

        let prompts = Dictionary(uniqueKeysWithValues: Self.texts)
        var rowsSeen = Set<Int>()
        var renders: [Render] = []
        var images: [String: GeneratedImage] = [:]
        var revisitsIdentical = true
        for (index, name) in Self.order.enumerated() {
            let text = prompts[name]!
            let rows = try pipeline.prepare(prompt: text).conditioningRows
            let generation = try await pipeline.generate(
                ImageRequest(prompt: text, seed: seed, steps: steps, width: side, height: side), observing: nil)
            let first = rowsSeen.insert(rows).inserted
            if let earlier = images[name] { revisitsIdentical = revisitsIdentical && earlier == generation.image }
            images[name] = images[name] ?? generation.image
            let render = Render(
                index: index, prompt: name, rows: rows, firstUseOfRows: first, metrics: generation.metrics,
                metalAllocatedAfter: metal.currentAllocatedSize, residentBytesAfter: await pipeline.residentBytes,
                pixelsSHA256: Self.digest(generation.image))
            renders.append(render)
            print("shapes: \(index) \(name) rows \(rows) first \(first) total \(generation.metrics.totalSeconds)s "
                  + "step0 \(generation.metrics.stepSeconds.first ?? 0)s loaded \(generation.metrics.loadedRoles) "
                  + "metal \(metal.currentAllocatedSize >> 20) MiB")
        }

        // The compile-cache lookup, with nothing loaded.
        await pipeline.unload()
        let before = metal.currentAllocatedSize
        let started = ContinuousClock.now
        let status = try CompiledCache.status(of: bundle, manifest: model.manifest, device: device)
        let lookupSeconds = ImageGenerationMeter.seconds(ContinuousClock.now - started)
        let lookup = CacheLookup(
            seconds: lookupSeconds, metalGrowthBytes: metal.currentAllocatedSize - before,
            entries: Dictionary(uniqueKeysWithValues: status.entries.map { ($0.role, $0.compiled) }), complete: status.isComplete)
        print("shapes: cache lookup \(lookupSeconds)s complete \(status.isComplete) metal +\(lookup.metalGrowthBytes >> 20) MiB")

        // Conditioning reuse against A's first render.
        let encodeStarted = ContinuousClock.now
        let conditioning = try await pipeline.encode(prompt: prompts["A"]!)
        let encodeSeconds = ImageGenerationMeter.seconds(ContinuousClock.now - encodeStarted)
        let reused = try await pipeline.generate(
            ImageRequest(prompt: prompts["A"]!, seed: seed, steps: steps, width: side, height: side), reusing: conditioning)
        let firstA = try #require(images["A"])
        let difference = zip(firstA.planarRGB, reused.image.planarRGB).map { abs($0 - $1) }.max() ?? 0
        await pipeline.unload()

        let result = Report(
            bundle: bundleURL.lastPathComponent, side: side, steps: steps, seed: seed, architecture: device.architecture,
            renders: renders, cacheLookup: lookup, encodeSeconds: encodeSeconds, reuse: reused.metrics,
            reuseMatchesFirstA: reused.image == firstA, reuseMaximumPixelDifference: difference,
            revisitsIdentical: revisitsIdentical)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(result).write(to: report, options: .withoutOverwriting)

        #expect(Set(renders.map(\.rows)).count == Set(Self.order).count, "every prompt must have its own row count")
        #expect(revisitsIdentical, "a revisited prompt must render the same image")
        #expect(result.reuseMatchesFirstA, "reused conditioning must render A's pixels (max diff \(difference))")
        #expect(!reused.metrics.loadedRoles.contains("textEncoder"))
    }
}
