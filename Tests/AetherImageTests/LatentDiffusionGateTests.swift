import AetherCore
import AetherRuntime
import Foundation
import Testing
@testable import AetherImage

/// The SD-Turbo gate (DESIGN.md §12): the exported `latent-diffusion@1` bundle, run through the driver on the GPU from
/// each case's seed alone, against the CPU float32 Diffusers reference (`Forge/sdturbo/reference_sdturbo.py`, which
/// draws its initial latents from Kit's seeded stream). Off by default; run with
/// `AETHER_GATES=1 swift test --no-parallel --filter LatentDiffusionGateTests`.
///
/// Bars (0.0.x's, declared before its native run): tokenizer ids exact on every edge case; the whole image RMSE ≤ 0.03
/// on the decoder's [-1, 1] scale; a repeat generation bit-identical. The bundle and reference default to the Forge
/// outputs beside this repo (`aether/forge-outputs/sd-turbo`, `…/references/sd-turbo`), or `AETHER_SDTURBO_BUNDLE` and
/// `AETHER_SDTURBO_REFERENCE`. `AETHER_GATE_DUMP=<dir>` also writes each image as raw float32 (Tools/DeviceGate compares).
@Suite("Latent diffusion gates", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct LatentDiffusionGateTests {
    static let environment = ProcessInfo.processInfo.environment
    static let outputs = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().appending(path: "forge-outputs")
    static let bundleURL = environment["AETHER_SDTURBO_BUNDLE"].map { URL(filePath: $0) } ?? outputs.appending(path: "sd-turbo")
    static let referenceURL = environment["AETHER_SDTURBO_REFERENCE"].map { URL(filePath: $0) }
        ?? outputs.appending(path: "references/sd-turbo")
    static let mac = Device(platform: .macOS, architecture: Device.current.architecture)

    struct Reference: Decodable {
        struct Case: Decodable { var id: String; var prompt: String; var seed: UInt64; var steps: Int; var ids: [Int32] }
        struct TokenizerCase: Decodable { var prompt: String; var ids: [Int32] }
        var size: Int
        var cases: [Case]
        var tokenizerCases: [TokenizerCase]
    }

    /// A Forge export is laid out as an installed revision root; it isn't catalogued yet, so its ref is local.
    static func localBundle(_ url: URL) throws -> InstalledBundle {
        let manifest = try Manifest.load(fromBundle: url)
        let ref = ModelRef(
            catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: manifest.variant.platform.rawValue,
            driver: manifest.driver, assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
        return InstalledBundle(ref: ref, url: url, isStale: false)
    }

    @Test func latentDiffusionMatchesDiffusers() async throws {
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: Self.referenceURL.appending(path: "reference.json")))
        let bundle = try Self.localBundle(Self.bundleURL)
        let tokenizer = try CLIPTokenizer(directory: Self.bundleURL.appending(path: "tokenizer"))
        for c in reference.tokenizerCases {
            #expect(try tokenizer.encode(c.prompt) == c.ids, "tokenizer: \(c.prompt.debugDescription)")
        }
        let pipeline = try await Registry.image.load(
            bundle, as: LatentDiffusionDriver.self, options: LoadOptions(device: Self.mac, cachePolicy: .default, cache: .default))
        for c in reference.cases {
            #expect(try tokenizer.encode(c.prompt) == c.ids)
            let started = ContinuousClock.now
            let image = try await pipeline.generate(.init(prompt: c.prompt, seed: c.seed, steps: c.steps))
            let seconds = (ContinuousClock.now - started).components.seconds
            let expected = try Self.readNPY(Self.referenceURL.appending(path: "\(c.id)/image.npy"))
            #expect(expected.count == image.planarRGB.count)
            let (rmse, psnr) = Self.compare(image.planarRGB, expected)
            print("gate: sd-turbo \(c.id) (\(c.steps) steps): RMSE \(rmse), PSNR \(psnr) dB, \(seconds) s")
            #expect(rmse <= 0.03, "\(c.id): RMSE \(rmse)")
            try image.pngData().write(to: Self.referenceURL.appending(path: "\(c.id)/native.png"))
            if let dump = Self.environment["AETHER_GATE_DUMP"] {   // raw float32, to compare with the device gate's
                try image.planarRGB.withUnsafeBufferPointer { Data(buffer: $0) }
                    .write(to: URL(filePath: dump).appending(path: "sd-turbo-\(c.id).f32"))
            }
            let repeated = try await pipeline.generate(.init(prompt: c.prompt, seed: c.seed, steps: c.steps))
            #expect(repeated == image, "\(c.id): a repeat generation differs")
        }
        for (key, footprint) in pipeline.options.ledger.snapshot.sorted(by: { $0.key.description < $1.key.description }) {
            print("gate: sd-turbo \(key): peak \(footprint.peakBytes >> 20) MiB, held \(footprint.heldBytes >> 20) MiB")
        }
        await pipeline.unload()
    }

    /// RMSE on [-1, 1] and PSNR on [0, 1].
    static func compare(_ a: [Float], _ b: [Float]) -> (rmse: Double, psnr: Double) {
        var sum = 0.0
        for (x, y) in zip(a, b) { sum += Double(x - y) * Double(x - y) }
        let mse = sum / Double(max(1, a.count))
        return (mse.squareRoot(), mse == 0 ? .infinity : 10 * log10(1 / (mse / 4)))
    }

    /// A little-endian float32 `.npy` (format 1.0), flattened.
    static func readNPY(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        try #require(data.count > 10 && data.prefix(6) == Data([0x93] + Array("NUMPY".utf8)))
        let headerLength = Int(data[8]) | Int(data[9]) << 8
        let header = String(decoding: data[10..<(10 + headerLength)], as: UTF8.self)
        try #require(header.contains("'descr': '<f4'") && header.contains("'fortran_order': False"))
        let body = data.dropFirst(10 + headerLength)
        return body.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}
