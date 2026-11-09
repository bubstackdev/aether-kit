import AetherKit
import AetherKrea
import AetherKreaCoreAI
import AetherKreaTokenizers
import Foundation
import Testing

/// Explicit native measurement: can padding the conditioning to a fixed row count leave the image unchanged? (A fixed
/// count would avoid Core AI's re-specialization on every prompt-length change; docs/ROADMAP-0.2.0.md K4.) One prompt
/// renders at its own row count, then padded to a fixed count two ways: zero conditioning rows appended, and pad
/// tokens (`<|endoftext|>`) appended after the rendered template before encoding. Same seed, same session. The report
/// gives each padded image's difference from the unpadded one. No image is saved; nothing is asserted (the numbers
/// are the result).
///
/// AETHER_NATIVE_KREA_ROW_PADDING=1 AETHER_KREA_BUNDLE=<bundle> AETHER_KREA_RESHAPE_REPORT=<new .json>
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_KREA_ROW_PADDING"] == "1"
               && nativeEnvironment["AETHER_KREA_BUNDLE"] != nil && nativeEnvironment["AETHER_KREA_RESHAPE_REPORT"] != nil),
      .timeLimit(.minutes(30)))
func paddedConditioningRowsAgainstTheUnpaddedImage() async throws {
    let report = URL(fileURLWithPath: try #require(nativeEnvironment["AETHER_KREA_RESHAPE_REPORT"]))
    #expect(!FileManager.default.fileExists(atPath: report.path), "the report must be a new file")
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let assets = try KreaAssets.open(at: URL(fileURLWithPath: try #require(nativeEnvironment["AETHER_KREA_BUNDLE"])))
    try assets.verify()
    let tokenizer = try await KreaTokenizer.open(assets: assets)
    let prompt = try tokenizer.prepare("A red ceramic mug on a white tabletop.")
    let target = 64, padToken: Int32 = 151_643
    #expect(prompt.conditioningRows < target)
    let configuration = try KreaConfiguration(width: 1024, height: 1024, steps: 8, seed: 18)

    struct Comparison: Codable {
        let variant: String; let rows: Int; let seconds: Double
        let identical: Bool; let maximumDifference: Float; let meanDifference: Float; let psnr: Double
    }
    var results: [Comparison] = []
    let session = KreaGenerationSession(assets: assets, compute: .preferGPU, residency: .resident, lifecycle: .manual)
    do {
        let prepared = try await session.prepare(prompt: prompt)
        var started = ProcessInfo.processInfo.systemUptime
        let base = try await session.generate(configuration: configuration, conditioning: prepared.conditioning).pixelsCHW.values
        let baseSeconds = ProcessInfo.processInfo.systemUptime - started
        let range = Double((base.max() ?? 1) - (base.min() ?? 0))

        func compare(_ variant: String, _ rows: Int, _ seconds: Double, _ pixels: [Float]) -> Comparison {
            let differences = zip(base, pixels).map { abs($0 - $1) }
            let mse = differences.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(differences.count)
            return Comparison(variant: variant, rows: rows, seconds: seconds, identical: pixels == base,
                              maximumDifference: differences.max() ?? 0,
                              meanDifference: Float(differences.reduce(0.0) { $0 + Double($1) } / Double(differences.count)),
                              psnr: mse == 0 ? .infinity : 10 * log10(range * range / mse))
        }
        results.append(compare("unpadded", prompt.conditioningRows, baseSeconds, base))

        // Zero rows after the prompt's own rows.
        let original = prepared.conditioning.tensor
        let zeros = [Float16](repeating: 0, count: (target - prompt.conditioningRows) * KreaConditioning.channels)
        let zeroPadded = try KreaConditioning(tensor: Float16Tensor(shape: [target, KreaConditioning.channels],
                                                                    values: original.values + zeros))
        started = ProcessInfo.processInfo.systemUptime
        let zeroPixels = try await session.generate(configuration: configuration, conditioning: zeroPadded).pixelsCHW.values
        results.append(compare("zero rows", target, ProcessInfo.processInfo.systemUptime - started, zeroPixels))

        // Pad tokens after the rendered template, encoded with the prompt (causal: the prompt's own rows are unchanged).
        let tokenPadded = try KreaPrompt(tokenIDs: prompt.tokenIDs
            + [Int32](repeating: padToken, count: target - prompt.conditioningRows))
        let tokenPrepared = try await session.prepare(prompt: tokenPadded)
        let ownRows = prompt.conditioningRows * KreaConditioning.channels
        let causalRowsMatch = Array(tokenPrepared.conditioning.tensor.values.prefix(ownRows)) == Array(original.values)
        started = ProcessInfo.processInfo.systemUptime
        let tokenPixels = try await session.generate(configuration: configuration, conditioning: tokenPrepared.conditioning).pixelsCHW.values
        var tokenResult = compare("pad tokens", target, ProcessInfo.processInfo.systemUptime - started, tokenPixels)
        tokenResult = Comparison(variant: causalRowsMatch ? "pad tokens (own rows unchanged)" : "pad tokens (own rows changed)",
                                 rows: tokenResult.rows, seconds: tokenResult.seconds, identical: tokenResult.identical,
                                 maximumDifference: tokenResult.maximumDifference, meanDifference: tokenResult.meanDifference,
                                 psnr: tokenResult.psnr)
        results.append(tokenResult)
        await session.unload()
    } catch {
        await session.unload()
        throw error
    }
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    encoder.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
    try encoder.encode(results).write(to: report, options: .withoutOverwriting)
}
