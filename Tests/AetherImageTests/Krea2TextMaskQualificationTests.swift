import AetherCore
import AetherHost
import AetherRuntime
import CryptoKit
import Foundation
import Metal
import Testing
@testable import AetherImage

/// Native qualification of a masked Krea 2 export (denoisers at a fixed text length with `textMask`), off by default.
/// Run one mode per fresh process, under the shared GPU lease:
///
///     AETHER_KREA2_MASK=<mode> AETHER_KREA2_MASK_REPORT=<new .json> \
///       scripts/gpu-lease -- swift test --no-parallel --filter Krea2TextMaskQualificationTests
///
/// optional AETHER_KREA2_MASK_BUNDLE (default: the Forge `krea-2-turbo-realism-masked` output), _EXACT (the
/// exact-length export compared against, default `krea-2-turbo-realism`), _SIZE (1024), _STEPS (8). Modes:
/// - `timing`: render A, then A again, then a very short and a very long prompt, each with its metrics (wall time, first
///   step, loaded roles, Metal peak) and the Metal allocation after it; then one render from four-reference
///   conditioning (the `references` function) with the Metal before and after, which shows whether the second function
///   shares the first's weights.
/// - `revisit`: render A once (a second fresh process at a prompt the first already ran: whether the first-step cost
///   is paid per process).
/// - `compare`: the three prompts at one seed and size on the masked export, then on the exact-length export, each
///   pair's PSNR and SSIM, reported only. The masked export's gate is against the Diffusers oracle (`check_krea2.py`,
///   SSIM ≥ 0.94): on a 258-row prompt at 1024², 0.9777 masked against 0.9821 exact length, identical per step.
/// - `sharing`: one masked segment's file loaded with `segment` alone, then with both functions on one specialization:
///   the Metal each load and each function's first run takes.
/// The filter allows everything (a measurement, not a deployment); no image is saved.
@Suite("Krea 2 text mask qualification", .serialized,
       .enabled(if: ProcessInfo.processInfo.environment["AETHER_KREA2_MASK"] != nil
                && ProcessInfo.processInfo.environment["AETHER_KREA2_MASK_REPORT"] != nil))
struct Krea2TextMaskQualificationTests {
    static let environment = ProcessInfo.processInfo.environment
    static let seed: UInt64 = 18

    static let prompts: [(name: String, text: String)] = [
        ("A", "A red ceramic mug on a white tabletop beside a folded linen napkin, soft morning light from a window."),
        ("short", "A pear."),
        ("long", """
            A wide establishing photograph of a quiet harbor town at the blue hour, seen from a low stone sea wall. In \
            the foreground, a row of weathered wooden fishing boats rests on the wet shingle, their hulls painted in \
            faded teal, ochre and brick red, coiled ropes and lobster pots stacked beside them. A narrow cobbled \
            street climbs the hill behind the harbor between tall slate-roofed houses with white window frames, warm \
            lamplight glowing in a few upper windows and a bakery's shopfront spilling yellow light onto the stones. \
            Strings of small bulbs hang between iron lamp posts along the quay. A lighthouse with a red lantern room \
            stands on the far headland, its beam just visible against a sky that fades from deep indigo overhead to a \
            thin band of pale peach at the horizon. Gulls sit on the mooring posts. The water is calm and glassy, \
            reflecting the lights in long trembling streaks. Light mist drifts across the hills, softening the \
            distant pine trees. Photographed on a full-frame camera with a 35 mm lens at f/8, long exposure, natural \
            color, fine detail in the textures of wood, stone and rope, no people, no text, no watermark.
            """),
    ]

    struct Render: Codable {
        let label: String
        let prompt: String
        let rows: Int
        let metrics: ImageGenerationMetrics
        let firstStepSeconds: Double
        let laterStepMeanSeconds: Double
        let metalAllocatedAfter: Int
        let residentBytesAfter: Int64
        /// `liveResidentBytes` after the render: the probe's level above the one before the graphs loaded.
        var liveResidentBytesAfter: Int64 = 0
        let pixelsSHA256: String
        /// The footprint ledger after the render: `key: peak / held` in MiB.
        var ledger: [String] = []
    }

    struct Pair: Codable {
        let prompt: String
        let rows: Int
        let psnrDB: Double
        let ssim: Double
        let maxAbs: Float
        let masked: ImageGenerationMetrics
        let exact: ImageGenerationMetrics
    }

    struct Sharing: Codable {
        let role: String
        let fileBytes: Int64
        let baseline: Int
        let stepOnlyAfterLoad: Int
        let stepOnlyAfterRun: Int
        let releasedTo: Int
        let bothAfterLoad: Int
        let bothAfterStepRun: Int
        let bothAfterReferencesRun: Int
        let tokens: Int
    }

    struct Report: Codable {
        let mode: String
        let bundle: String
        let side: Int
        let steps: Int
        let seed: UInt64
        let architecture: String
        let pid: Int32
        let metalBeforeLoad: Int
        var renders: [Render] = []
        var referenceRender: Render?
        var referenceMetalBefore: Int?
        var pairs: [Pair] = []
        var sharing: Sharing?
        var repeatIdentical: Bool?
    }

    final class AllowAll: ContentFilter {
        func review(prompt: String) async throws -> ContentVerdict { .allow }
        func review(image: GeneratedImage, prompt: String) async throws -> ContentVerdict { .allow }
    }

    static func digest(_ image: GeneratedImage) -> String {
        image.planarRGB.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
    }

    static func pipeline(_ url: URL) async throws -> Krea2Pipeline {
        let bundle = try LatentDiffusionGateTests.localBundle(url)
        let model = try await Registry.image.load(
            bundle, as: Krea2Driver.self,
            options: LoadOptions(device: LatentDiffusionGateTests.mac, cachePolicy: .default, cache: .default))
        return try await model.pipeline(filter: AllowAll())
    }

    static func render(
        _ pipeline: Krea2Pipeline, label: String, prompt: String, side: Int, steps: Int, metal: any MTLDevice
    ) async throws -> (Render, GeneratedImage) {
        let rows = try pipeline.prepare(prompt: prompt).conditioningRows
        let generation = try await pipeline.generate(
            ImageRequest(prompt: prompt, seed: seed, steps: steps, width: side, height: side), observing: nil)
        return (try await summarize(generation, label: label, rows: rows, pipeline: pipeline, metal: metal), generation.image)
    }

    static func summarize(
        _ generation: ImageGeneration, label: String, rows: Int, pipeline: Krea2Pipeline, metal: any MTLDevice
    ) async throws -> Render {
        let steps = generation.metrics.stepSeconds
        let later = steps.dropFirst()
        var render = Render(
            label: label, prompt: String(label.split(separator: " ").first ?? ""), rows: rows, metrics: generation.metrics,
            firstStepSeconds: steps.first ?? 0, laterStepMeanSeconds: later.isEmpty ? 0 : later.reduce(0, +) / Double(later.count),
            metalAllocatedAfter: metal.currentAllocatedSize, residentBytesAfter: await pipeline.plannedResidentBytes,
            pixelsSHA256: digest(generation.image))
        render.liveResidentBytesAfter = await pipeline.liveResidentBytes
        render.ledger = pipeline.options.ledger.snapshot.sorted { $0.key.description < $1.key.description }.map {
            "\($0.key): peak \($0.value.peakBytes >> 20) held \($0.value.heldBytes >> 20)"
        }
        print("mask: \(label) rows \(rows) total \(generation.metrics.totalSeconds)s step0 \(render.firstStepSeconds)s "
              + "later \(render.laterStepMeanSeconds)s prep \(generation.metrics.denoiserPreparationSeconds)s "
              + "text \(generation.metrics.conditioningSeconds)s loaded \(generation.metrics.loadedRoles) "
              + "peak \(generation.metrics.peakAllocatedBytes >> 20) MiB metal \(metal.currentAllocatedSize >> 20) MiB "
              + "resident \(render.residentBytesAfter >> 20) MiB live \(render.liveResidentBytesAfter >> 20) MiB")
        return render
    }

    /// Four synthetic 384 × 384 references (576 image tokens): conditioning past the text-only bucket.
    static func references() throws -> [Krea2ReferenceImage] {
        try (0..<4).map { k in
            let side = 384
            var rgb = [UInt8](repeating: 0, count: side * side * 3)
            for y in 0..<side {
                for x in 0..<side {
                    let i = (y * side + x) * 3
                    rgb[i] = UInt8((x * (k + 1)) % 256)
                    rgb[i + 1] = UInt8((y * (4 - k)) % 256)
                    rgb[i + 2] = UInt8(((x + y) * 3 + 60 * k) % 256)
                }
            }
            return try Krea2ReferenceImage(rgb: rgb, width: side, height: side)
        }
    }

    @Test(.timeLimit(.minutes(120)))
    func maskedExport() async throws {
        let mode = try #require(Self.environment["AETHER_KREA2_MASK"])
        let reportURL = URL(filePath: try #require(Self.environment["AETHER_KREA2_MASK_REPORT"]))
        #expect(!FileManager.default.fileExists(atPath: reportURL.path), "the report must be a new file")
        let side = Int(Self.environment["AETHER_KREA2_MASK_SIZE"] ?? "") ?? 1024
        let steps = Int(Self.environment["AETHER_KREA2_MASK_STEPS"] ?? "") ?? 8
        let bundleURL = Self.environment["AETHER_KREA2_MASK_BUNDLE"].map { URL(filePath: $0) }
            ?? LatentDiffusionGateTests.outputs.appending(path: "krea-2-turbo-realism-masked")
        let exactURL = Self.environment["AETHER_KREA2_MASK_EXACT"].map { URL(filePath: $0) }
            ?? LatentDiffusionGateTests.outputs.appending(path: "krea-2-turbo-realism")
        let metal = try #require(MTLCreateSystemDefaultDevice())
        let texts = Dictionary(uniqueKeysWithValues: Self.prompts)
        var report = Report(
            mode: mode, bundle: bundleURL.lastPathComponent, side: side, steps: steps, seed: Self.seed,
            architecture: LatentDiffusionGateTests.mac.architecture, pid: ProcessInfo.processInfo.processIdentifier,
            metalBeforeLoad: metal.currentAllocatedSize)

        switch mode {
        case "timing", "revisit":
            let pipeline = try await Self.pipeline(bundleURL)
            #expect(pipeline.wiring.textMask != nil, "the bundle's denoisers must take the text mask")
            let order = mode == "timing" ? [("A", "A"), ("A", "A repeat"), ("short", "short"), ("long", "long")] : [("A", "A")]
            var images: [String: GeneratedImage] = [:]
            for (name, label) in order {
                let (render, image) = try await Self.render(pipeline, label: label, prompt: texts[name]!, side: side, steps: steps, metal: metal)
                report.renders.append(render)
                if let earlier = images[name] { report.repeatIdentical = earlier == image }
                images[name] = images[name] ?? image
            }
            if mode == "timing" {
                report.referenceMetalBefore = metal.currentAllocatedSize
                let text = "The objects from the four pictures arranged on a wooden table."
                let references = try Self.references()
                let conditioning = try await pipeline.encode(prompt: text, references: references)
                let generation = try await pipeline.generate(
                    ImageRequest(prompt: text, seed: Self.seed, steps: steps, width: side, height: side), reusing: conditioning)
                report.referenceRender = try await Self.summarize(
                    generation, label: "references", rows: conditioning.rows, pipeline: pipeline, metal: metal)
                #expect(conditioning.rows > 512, "the reference render must run the references bucket")
            }
            await pipeline.unload()
        case "compare":
            var masked: [String: ImageGeneration] = [:]
            let pipeline = try await Self.pipeline(bundleURL)
            for (name, text) in Self.prompts {
                masked[name] = try await pipeline.generate(
                    ImageRequest(prompt: text, seed: Self.seed, steps: steps, width: side, height: side), observing: nil)
            }
            await pipeline.unload()
            let exact = try await Self.pipeline(exactURL)
            #expect(exact.wiring.textMask == nil, "the compared export must be the exact-length one")
            for (name, text) in Self.prompts {
                let other = try await exact.generate(
                    ImageRequest(prompt: text, seed: Self.seed, steps: steps, width: side, height: side), observing: nil)
                let a = masked[name]!.image.planarRGB, b = other.image.planarRGB
                let pair = Pair(
                    prompt: name, rows: try exact.prepare(prompt: text).conditioningRows,
                    psnrDB: Flux2KleinGateTests.psnr(a, b), ssim: Flux2KleinGateTests.ssim(a, b, width: side, height: side),
                    maxAbs: zip(a, b).map { abs($0 - $1) }.max() ?? 0, masked: masked[name]!.metrics, exact: other.metrics)
                print("mask: compare \(name) rows \(pair.rows): PSNR \(pair.psnrDB) dB, SSIM \(pair.ssim), max |Δ| \(pair.maxAbs)")
                report.pairs.append(pair)
            }
            await exact.unload()
        case "sharing":
            report.sharing = try await Self.sharing(bundleURL, metal: metal)
        default:
            Issue.record("unknown mode \(mode)")
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: reportURL, options: .withoutOverwriting)
        if mode == "timing" { #expect(report.repeatIdentical == true, "a repeated prompt must render the same image") }
    }

    /// denoiser1's file: `segment` alone (load, one run at 1024² tokens), released; then both functions on one
    /// specialization (load, a `segment` run, a `segmentReferences` run). Metal after each.
    static func sharing(_ url: URL, metal: any MTLDevice) async throws -> Sharing {
        let bundle = try LatentDiffusionGateTests.localBundle(url)
        let manifest = try Manifest.load(fromBundle: bundle.url)
        let role = "denoiser1"
        let spec = try #require(manifest.graphs[role])
        let options = try GraphOptions(manifest: spec, at: "graphs.\(role)", cachePolicy: .default, cache: .default)
        let path = try #require(spec.file.resolve(architecture: LatentDiffusionGateTests.mac.architecture))
        let fileURL = try #require(bundle.url(of: path))
        let tokens = 4096
        func run(_ graph: any GraphBackend) async throws {
            let arena = TensorArena()
            var bindings = Bindings(arena: arena)
            for input in graph.signature.inputs {
                let dims = input.dynamicAxes.map { _ in tokens }
                let buffer = try graph.makeBuffer(for: input.name, role: .input, dimensions: dims, in: arena)
                try buffer.zero()
                if input.name == "text_mask" { try buffer.write([Float](repeating: 1, count: buffer.count)) }
                try bindings.bindInput(input.name, to: buffer)
            }
            for output in graph.signature.outputs {
                let dims = output.dynamicAxes.map { _ in tokens }
                try bindings.bindOutput(output.name, to: try graph.makeBuffer(for: output.name, role: .output, dimensions: dims, in: arena))
            }
            try await graph.run(&bindings)
        }
        let baseline = metal.currentAllocatedSize
        var single: Graph? = try await Graph(url: fileURL, function: "segment", options: options)
        let stepOnlyAfterLoad = metal.currentAllocatedSize
        try await run(single!)
        let stepOnlyAfterRun = metal.currentAllocatedSize
        single = nil
        try await Task.sleep(for: .seconds(5))
        let releasedTo = metal.currentAllocatedSize
        let both = try await Graph.load(url: fileURL, functions: ["segment", "segmentReferences"], options: options)
        let bothAfterLoad = metal.currentAllocatedSize
        try await run(both["segment"]!)
        let bothAfterStepRun = metal.currentAllocatedSize
        try await run(both["segmentReferences"]!)
        let bothAfterReferencesRun = metal.currentAllocatedSize
        let result = Sharing(
            role: role, fileBytes: GraphResidency.fileBytes(bundle: bundle, manifest: manifest, role: role, device: LatentDiffusionGateTests.mac),
            baseline: baseline, stepOnlyAfterLoad: stepOnlyAfterLoad, stepOnlyAfterRun: stepOnlyAfterRun, releasedTo: releasedTo,
            bothAfterLoad: bothAfterLoad, bothAfterStepRun: bothAfterStepRun, bothAfterReferencesRun: bothAfterReferencesRun,
            tokens: tokens)
        print("mask: sharing \(result)")
        return result
    }
}
