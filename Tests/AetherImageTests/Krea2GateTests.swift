import AetherCore
import AetherRuntime
import Foundation
import Synchronization
import Testing
@testable import AetherImage

/// The Krea 2 gates (DESIGN.md §12): each exported `krea2@1` bundle (vanilla, and realism with its merged adapter), run
/// through the driver on the GPU from the case's seed, against the CPU float32 Diffusers reference
/// (`Forge/krea2/reference_krea2.py`, which draws its initial latents from Kit's seeded stream and, for realism,
/// applies the exporter's exact merge). Off by default; run with
/// `AETHER_GATES=1 swift test --no-parallel --filter Krea2GateTests`.
///
/// Bars (the exporter's `check_krea2.py`, whose closed loop reached SSIM 0.998 vanilla and 0.993 realism):
/// - the rendered prompt ids exact;
/// - the text conditioning: cosine ≥ 0.999 per row;
/// - the whole pipeline from the seed (tokenizer, native text encoder, 8 × 4 segments, host Euler, VAE): SSIM ≥ 0.94;
/// - a repeat generation bit-identical.
///
/// The bundles and references default to the Forge outputs beside this repo (`aether/forge-outputs/krea-2-turbo[-realism]`
/// and `…/references/…`). The gate's filter lets everything through: it is a parity test, not a deployment.
@Suite("Krea 2 gates", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct Krea2GateTests {
    static let outputs = LatentDiffusionGateTests.outputs

    struct Reference: Decodable {
        var prompt: String
        var seed: UInt64
        var width: Int
        var height: Int
        var steps: Int
        var textRows: Int
    }

    struct Tokens: Decodable { var real: [Int32] }

    /// Records every review and allows it (parity only).
    final class GateFilter: ContentFilter {
        let reviews = Mutex(0)
        func review(prompt: String) async throws -> ContentVerdict { reviews.withLock { $0 += 1 }; return .allow }
        func review(image: GeneratedImage, prompt: String) async throws -> ContentVerdict { reviews.withLock { $0 += 1 }; return .allow }
    }

    @Test(arguments: ["krea-2-turbo", "krea-2-turbo-realism"])
    func krea2MatchesDiffusers(bundleName: String) async throws {
        let bundleURL = Self.outputs.appending(path: bundleName)
        let referenceURL = Self.outputs.appending(path: "references/\(bundleName)")
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: referenceURL.appending(path: "reference.json")))
        let tokens = try JSONDecoder().decode(Tokens.self, from: Data(contentsOf: referenceURL.appending(path: "tokens.json")))
        let bundle = try LatentDiffusionGateTests.localBundle(bundleURL)
        let model = try await Registry.image.load(
            bundle, as: Krea2Driver.self,
            options: LoadOptions(device: LatentDiffusionGateTests.mac, cachePolicy: .default, cache: .default))
        let filter = GateFilter()
        let pipeline = try await model.pipeline(filter: filter)
        let expectedImage = try LatentDiffusionGateTests.readNPY(referenceURL.appending(path: "image.npy"))
        let referenceRows = try LatentDiffusionGateTests.readNPY(referenceURL.appending(path: "text_embeds.npy"))
        let columns = pipeline.krea2.conditioning.columns

        // The prompt.
        let rendered = try pipeline.encodePrompt(reference.prompt)
        #expect(rendered == tokens.real, "\(bundleName): rendered ids differ")

        // The text conditioning.
        let rows = try await pipeline.conditioningRows(for: reference.prompt)
        #expect(rows.count == referenceRows.count && rows.count == reference.textRows * columns)
        var worst = 1.0
        for row in 0..<reference.textRows {
            let range = (row * columns)..<((row + 1) * columns)
            worst = min(worst, Flux2KleinGateTests.cosine(rows[range], referenceRows[range]))
        }
        print("gate: \(bundleName) text row cosine min \(worst) over \(reference.textRows) rows")
        #expect(worst >= 0.999)

        // The whole pipeline from the seed.
        let request = ImageRequest(
            prompt: reference.prompt, seed: reference.seed, steps: reference.steps, width: reference.width, height: reference.height)
        var started = ContinuousClock.now
        let image = try await pipeline.generate(request)
        let ssim = Flux2KleinGateTests.ssim(image.planarRGB, expectedImage, width: reference.width, height: reference.height)
        print("gate: \(bundleName) whole pipeline: SSIM \(ssim), PSNR \(Flux2KleinGateTests.psnr(image.planarRGB, expectedImage)) dB, "
              + "\(ContinuousClock.now - started)")
        #expect(ssim >= 0.94)
        try image.pngData().write(to: referenceURL.appending(path: "native.png"))
        started = ContinuousClock.now
        let repeated = try await pipeline.generate(request)
        print("gate: \(bundleName) warm generation \(ContinuousClock.now - started)")
        #expect(repeated == image, "\(bundleName): a repeat generation differs")
        #expect(filter.reviews.withLock { $0 } == 4)

        for (key, footprint) in pipeline.options.ledger.snapshot.sorted(by: { $0.key.description < $1.key.description }) {
            print("gate: \(bundleName) \(key): peak \(footprint.peakBytes >> 20) MiB, held \(footprint.heldBytes >> 20) MiB")
        }
        await pipeline.unload()
    }
}
