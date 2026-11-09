import AetherKit
import AetherKrea
import AetherKreaCoreAI
import AetherKreaTokenizers
import CryptoKit
import Foundation
import Metal
import Testing

/// Explicit native measurement across prompt row counts. One resident session renders neutral-object prompts of
/// different lengths in turn; each render's wall time and Metal growth go to a JSON report, then a fresh session renders
/// the fourth case again and the two must be pixel-identical. On this code every new row count reopens the denoisers.
/// Recorded 2026-11-08 (1024², 8 steps, M5 Max 64 GB) with an unshipped variant that kept them open across row counts:
/// pixel-identical, 0.39–0.64 GB of Metal retained per new shape, and 67–75 s per render, no faster than reopening.
/// No image is saved. Ordinary `swift test` never loads these models. The test waits on the shared GPU lease; a host
/// app that keeps its turn while its model is loaded (stepping aside only for jobs queued through the GPU-lock wrapper)
/// must be closed first.
///
/// AETHER_NATIVE_KREA_RESHAPE=1 AETHER_KREA_BUNDLE=<bundle> AETHER_KREA_RESHAPE_REPORT=<new .json>
/// optional AETHER_KREA_RESHAPE_SIZE (default 1024), AETHER_KREA_RESHAPE_STEPS (default 8).
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_KREA_RESHAPE"] == "1"
               && nativeEnvironment["AETHER_KREA_BUNDLE"] != nil && nativeEnvironment["AETHER_KREA_RESHAPE_REPORT"] != nil),
      .timeLimit(.minutes(60)))
func promptRowCountsCostAndRenderWhatAFreshSessionDoes() async throws {
    let report = URL(fileURLWithPath: try #require(nativeEnvironment["AETHER_KREA_RESHAPE_REPORT"]))
    #expect(!FileManager.default.fileExists(atPath: report.path), "the report must be a new file")
    let side = Int(nativeEnvironment["AETHER_KREA_RESHAPE_SIZE"] ?? "") ?? 1024
    let steps = Int(nativeEnvironment["AETHER_KREA_RESHAPE_STEPS"] ?? "") ?? 8
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let assets = try KreaAssets.open(at: URL(fileURLWithPath: try #require(nativeEnvironment["AETHER_KREA_BUNDLE"])))
    try assets.verify()
    let tokenizer = try await KreaTokenizer.open(assets: assets)
    let device = try #require(MTLCreateSystemDefaultDevice())

    // Neutral objects, increasing in length so every render after the first is a new row count.
    let texts = [
        "A red ceramic mug.",
        "A red ceramic mug on a white tabletop.",
        "A red ceramic mug on a white tabletop beside a folded linen napkin.",
        "A red ceramic mug on a white tabletop beside a folded linen napkin, soft morning light from a window.",
        "A red ceramic mug on a white tabletop beside a folded linen napkin, soft morning light from a window, a small potted fern in the background.",
        "A red ceramic mug on a white tabletop beside a folded linen napkin, soft morning light from a window, a small potted fern in the background, shallow depth of field, muted colors.",
        "A blue glass bottle on a wooden shelf.",
        "A red ceramic mug.",
    ]
    let prompts = try texts.map { try tokenizer.prepare($0) }
    let configuration = try KreaConfiguration(width: side, height: side, steps: steps, seed: 18)

    struct Render: Codable { let index: Int; let rows: Int; let seconds: Double; let metalGrowthBytes: Int; let pixelsSHA256: String }
    func digest(_ image: GeneratedImage) -> String {
        image.pixelsCHW.values.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
    }

    var renders: [Render] = []
    var pixels: [Int: [Float]] = [:]
    let warm = KreaGenerationSession(assets: assets, compute: .preferGPU, residency: .resident, lifecycle: .manual)
    do {
        for (index, prompt) in prompts.enumerated() {
            let before = device.currentAllocatedSize, started = ProcessInfo.processInfo.systemUptime
            let image = try await warm.generate(configuration: configuration, prompt: prompt)
            renders.append(Render(index: index, rows: prompt.conditioningRows, seconds: ProcessInfo.processInfo.systemUptime - started,
                                  metalGrowthBytes: device.currentAllocatedSize - before, pixelsSHA256: digest(image)))
            if index == 3 { pixels[index] = image.pixelsCHW.values }
        }
        await warm.unload()
    } catch {
        await warm.unload()
        throw error
    }

    // A fresh session renders the fourth case alone.
    let fresh = KreaGenerationSession(assets: assets, compute: .preferGPU, residency: .resident, lifecycle: .manual)
    let direct: GeneratedImage
    do {
        direct = try await fresh.generate(configuration: configuration, prompt: prompts[3])
        await fresh.unload()
    } catch {
        await fresh.unload()
        throw error
    }
    let kept = try #require(pixels[3])
    let maximumDifference = zip(kept, direct.pixelsCHW.values).map { abs($0 - $1) }.max() ?? 0

    struct Report: Codable {
        let side: Int, steps: Int, renders: [Render]
        let freshPixelsSHA256: String, keptMatchesFresh: Bool, maximumPixelDifference: Float
    }
    let result = Report(side: side, steps: steps, renders: renders, freshPixelsSHA256: digest(direct),
                        keptMatchesFresh: renders[3].pixelsSHA256 == digest(direct), maximumPixelDifference: maximumDifference)
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(result).write(to: report, options: .withoutOverwriting)

    #expect(Set(renders.map(\.rows)).count >= 6, "the prompts must cover several row counts")
    #expect(result.keptMatchesFresh, "the session must render exactly what a fresh one does (max diff \(maximumDifference))")
    #expect(renders.allSatisfy { $0.pixelsSHA256 != renders[0].pixelsSHA256 || $0.rows == renders[0].rows })
}
