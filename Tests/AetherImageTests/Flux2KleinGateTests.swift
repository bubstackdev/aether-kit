import AetherCore
import AetherRuntime
import Foundation
import Testing
@testable import AetherImage

/// The FLUX.2 [klein] gate (DESIGN.md §12): the exported `flux2-klein@1` bundle, run through the driver on the GPU from
/// the case's seed, against the CPU float32 Diffusers reference (`Forge/flux2/reference_flux2.py`, which draws its
/// initial latents from Kit's seeded stream). Off by default; run with
/// `AETHER_GATES=1 swift test --no-parallel --filter Flux2KleinGateTests`.
///
/// Bars (0.0.x's, declared before its native run):
/// - the padded prompt ids and mask exact;
/// - the text rows: cosine ≥ 0.999 per real row (padding rows are ill-conditioned in every precision);
/// - the image from the reference's own text rows: SSIM ≥ 0.94 (int8 weights);
/// - the whole pipeline (tokenizer and native text encoder too): SSIM recorded, not gated. All 512 rows condition the
///   denoiser, padding included, and Diffusers' own bf16 pipeline lands at 0.865–0.999 against float32;
/// - a repeat generation bit-identical.
///
/// The bundle and reference default to the Forge outputs beside this repo (`aether/forge-outputs/flux2-klein-4b/bundle`
/// and `…/reference`), or `AETHER_FLUX2_BUNDLE` and `AETHER_FLUX2_REFERENCE`.
@Suite("FLUX.2 [klein] gates", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct Flux2KleinGateTests {
    static let environment = ProcessInfo.processInfo.environment
    static let root = LatentDiffusionGateTests.outputs.appending(path: "flux2-klein-4b")
    static let bundleURL = environment["AETHER_FLUX2_BUNDLE"].map { URL(filePath: $0) } ?? root.appending(path: "bundle")
    static let referenceURL = environment["AETHER_FLUX2_REFERENCE"].map { URL(filePath: $0) } ?? root.appending(path: "reference")

    struct Reference: Decodable {
        var prompt: String
        var seed: UInt64
        var width: Int
        var height: Int
        var steps: Int
        var realTokens: Int
    }

    struct Tokens: Decodable {
        var padded: [Int32]
        var mask: [Int32]
    }

    @Test func flux2KleinMatchesDiffusers() async throws {
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: Self.referenceURL.appending(path: "reference.json")))
        let tokens = try JSONDecoder().decode(Tokens.self, from: Data(contentsOf: Self.referenceURL.appending(path: "tokens.json")))
        let bundle = try LatentDiffusionGateTests.localBundle(Self.bundleURL)
        let pipeline = try await Registry.image.load(
            bundle, as: Flux2KleinDriver.self,
            options: LoadOptions(device: LatentDiffusionGateTests.mac, cachePolicy: .default, cache: .default))
        let expectedImage = try LatentDiffusionGateTests.readNPY(Self.referenceURL.appending(path: "image.npy"))
        let referenceRows = try LatentDiffusionGateTests.readNPY(Self.referenceURL.appending(path: "text_embeds.npy"))
        let width = pipeline.flux2.conditioning.width

        // The prompt.
        let prompt = try Flux2KleinPipeline.Prompt(rendered: try pipeline.encodePrompt(reference.prompt), rule: pipeline.flux2.prompt)
        #expect(prompt.ids == tokens.padded && prompt.mask == tokens.mask)

        // The text rows.
        let rows = try await pipeline.conditioningRows(for: reference.prompt)
        #expect(rows.count == referenceRows.count)
        var worst = 1.0
        for row in 0..<reference.realTokens {
            let range = (row * width)..<((row + 1) * width)
            worst = min(worst, Self.cosine(rows[range], referenceRows[range]))
        }
        print("gate: flux2-klein text real-row cosine min \(worst) over \(reference.realTokens) rows")
        #expect(worst >= 0.999)

        // The image from the reference's rows, then the whole pipeline.
        let request = Flux2KleinPipeline.Request(
            prompt: reference.prompt, seed: reference.seed, steps: reference.steps, width: reference.width, height: reference.height)
        var started = ContinuousClock.now
        let fromRows = try await pipeline.generate(request, conditioning: referenceRows)
        let imagePath = Self.ssim(fromRows.planarRGB, expectedImage, width: reference.width, height: reference.height)
        print("gate: flux2-klein image path (reference rows): SSIM \(imagePath), PSNR \(Self.psnr(fromRows.planarRGB, expectedImage)) dB, "
              + "\(ContinuousClock.now - started)")
        #expect(imagePath >= 0.94)
        try fromRows.pngData().write(to: Self.referenceURL.appending(path: "native-reference-rows.png"))

        started = ContinuousClock.now
        let whole = try await pipeline.generate(request)
        let wholeSSIM = Self.ssim(whole.planarRGB, expectedImage, width: reference.width, height: reference.height)
        print("gate: flux2-klein whole pipeline: SSIM \(wholeSSIM) (recorded), PSNR \(Self.psnr(whole.planarRGB, expectedImage)) dB, "
              + "\(ContinuousClock.now - started)")
        try whole.pngData().write(to: Self.referenceURL.appending(path: "native.png"))
        started = ContinuousClock.now
        let repeated = try await pipeline.generate(request)
        print("gate: flux2-klein warm generation \(ContinuousClock.now - started)")
        #expect(repeated == whole, "a repeat generation differs")

        for (key, footprint) in pipeline.options.ledger.snapshot.sorted(by: { $0.key.description < $1.key.description }) {
            print("gate: flux2-klein \(key): peak \(footprint.peakBytes >> 20) MiB, held \(footprint.heldBytes >> 20) MiB")
        }
        await pipeline.unload()
    }

    static func cosine(_ a: ArraySlice<Float>, _ b: ArraySlice<Float>) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for (x, y) in zip(a, b) { dot += Double(x) * Double(y); na += Double(x) * Double(x); nb += Double(y) * Double(y) }
        return dot / max((na * nb).squareRoot(), .leastNormalMagnitude)
    }

    /// PSNR on [0, 1] of two planar `[3, H, W]` images on [-1, 1].
    static func psnr(_ a: [Float], _ b: [Float]) -> Double {
        LatentDiffusionGateTests.compare(a, b).psnr
    }

    /// SSIM on BT.601 luma of two planar `[3, H, W]` images on [-1, 1], mapped to [0, 1] and clamped: an 11×11
    /// Gaussian window (σ 1.5), K = 0.01/0.03, valid region only (0.0.x's `diffusion_fixtures.ssim`).
    static func ssim(_ a: [Float], _ b: [Float], width: Int, height: Int) -> Double {
        let plane = width * height
        guard a.count == plane * 3, b.count == a.count, width >= 11, height >= 11 else { return -1 }
        func luma(_ v: [Float]) -> [Double] {
            func unit(_ x: Float) -> Double { Double(min(max((x + 1) / 2, 0), 1)) }
            return (0..<plane).map { p in 0.299 * unit(v[p]) + 0.587 * unit(v[plane + p]) + 0.114 * unit(v[2 * plane + p]) }
        }
        let x = luma(a), y = luma(b)
        var g = (0..<11).map { exp(-pow(Double($0 - 5), 2) / (2 * 1.5 * 1.5)) }
        let total = g.reduce(0, +)
        g = g.map { $0 / total }
        func filter(_ v: [Double]) -> [Double] {
            let ow = width - 10, oh = height - 10
            var horizontal = [Double](repeating: 0, count: ow * height)
            for r in 0..<height { for c in 0..<ow { var s = 0.0; for k in 0..<11 { s += g[k] * v[r * width + c + k] }; horizontal[r * ow + c] = s } }
            var out = [Double](repeating: 0, count: ow * oh)
            for r in 0..<oh { for c in 0..<ow { var s = 0.0; for k in 0..<11 { s += g[k] * horizontal[(r + k) * ow + c] }; out[r * ow + c] = s } }
            return out
        }
        let mx = filter(x), my = filter(y)
        let sxx = filter(x.map { $0 * $0 }), syy = filter(y.map { $0 * $0 }), sxy = filter(zip(x, y).map { $0 * $1 })
        let c1 = 0.0001, c2 = 0.0009
        var sum = 0.0
        for i in mx.indices {
            let vx = sxx[i] - mx[i] * mx[i], vy = syy[i] - my[i] * my[i], cv = sxy[i] - mx[i] * my[i]
            sum += ((2 * mx[i] * my[i] + c1) * (2 * cv + c2)) / ((mx[i] * mx[i] + my[i] * my[i] + c1) * (vx + vy + c2))
        }
        return sum / Double(mx.count)
    }
}
