import AetherKit
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import AetherVision
import Foundation
import Testing

/// Opt-in timing of Qwen3-VL decoders, to compare a one-function bundle with a 0.0.2 two-function one: time to first
/// token and decode seconds per token for text-only and image requests on the decoder alone (the image is encoded once
/// up front, so encoding is not timed). Bundles are opened one at a time, in the listed order, for `rounds` rounds
/// (ABAB...), so host drift falls on both. Each opening is warmed with two requests of each kind, then runs `requests`
/// requests of each kind, alternating, without a memory budget (no reload inside the timed requests). Medians over
/// every round. Every reply of a kind is recorded once per distinct token sequence (`replies`), so bundles of the same
/// checkpoint can be compared token for token.
///
/// AETHER_NATIVE_VL_TIMING=1
/// AETHER_VL_TIMING_BUNDLES=<name=bundle,...>
/// AETHER_VL_TIMING_OUTPUT=<report .json>
/// AETHER_VL_TIMING_ROUNDS=<default 2>
/// AETHER_VL_TIMING_REQUESTS=<per kind and round, default 8>
private struct TimingSample: Codable { let kind: String, promptTokens: Int, tokens: Int, timeToFirstTokenSeconds: Double, decodeSecondsPerToken: Double }

private struct TimingBundleReport: Codable {
    var name = ""
    var bundle = ""
    var imagesThroughMain = false
    var samples: [TimingSample] = []
    var replies: [String: [[Int]]] = [:]
    var median: [String: [String: Double]] = [:]
}

private struct TimingReport: Codable {
    var device = LanguageGenerationSession.deviceArchitectureName
    var hostLoad = "Shared host: GPU jobs of cooperating agents are serialized by a lock, other host work is not"
    var rounds = 0
    var requests = 0
    var bundles: [TimingBundleReport] = []
    var error: String?
}

private func timingGradient(width: Int, height: Int) throws -> VisionRGBImage {
    var rgb = [UInt8](repeating: 0, count: width * height * 3)
    for y in 0..<height {
        for x in 0..<width {
            let i = 3 * (y * width + x)
            rgb[i] = UInt8(x * 255 / max(1, width - 1))
            rgb[i + 1] = UInt8(y * 255 / max(1, height - 1))
            rgb[i + 2] = UInt8((x + y) * 127 / max(1, width + height - 2))
        }
    }
    return try VisionRGBImage(rgb: rgb, width: width, height: height)
}

private func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    return sorted.isEmpty ? 0 : sorted.count % 2 == 1 ? sorted[sorted.count / 2] : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_VL_TIMING"] == "1"), .timeLimit(.minutes(60)))
func visionLanguageDecoderTiming() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_VL_TIMING_OUTPUT"]))
    var report = TimingReport()
    report.rounds = Int(nativeEnvironment["AETHER_VL_TIMING_ROUNDS"] ?? "") ?? 2
    report.requests = Int(nativeEnvironment["AETHER_VL_TIMING_REQUESTS"] ?? "") ?? 8
    defer {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    do {
        let entries = try #require(nativeEnvironment["AETHER_VL_TIMING_BUNDLES"]).split(separator: ",").map { $0.split(separator: "=", maxSplits: 1) }
        report.bundles = entries.map { var b = TimingBundleReport(); b.name = String($0[0]); b.bundle = URL(filePath: String($0[1])).lastPathComponent; return b }
        for _ in 0..<report.rounds {
            for (index, entry) in entries.enumerated() {
                let bundle = URL(filePath: String(entry[1]))
                let manifest = try LanguageModelManifest.load(from: bundle)
                let textModel = try manifest.verify(in: bundle, platform: .macos)
                let registration = try #require(manifest.vision)
                let visionModel = try #require(try manifest.verifyVision(in: bundle))
                let grid = try #require(registration.imageGrids?.first)
                let tokenizer = try await LanguageTokenizer.open(bundle: bundle)
                let tokens = LanguageImageTokens(visionStart: registration.visionStartTokenID, imagePad: registration.imageTokenID,
                                                 visionEnd: registration.visionEndTokenID)
                var configuration = Qwen3VLImagePreprocessor.Configuration()
                configuration.imageGrids = registration.imageGrids?.map { .init(height: $0.height, width: $0.width) }
                let encoder = try await LanguageVisionEncoder.open(at: visionModel,
                    configuration: .init(mergeSize: registration.mergeSize, positionTableSide: registration.positionTableSide,
                                         imageGrids: configuration.imageGrids ?? []))
                let image = try timingGradient(width: grid.width * registration.patchSize, height: grid.height * registration.patchSize)
                let processed = try Qwen3VLImagePreprocessor.process(rgb: image.rgb, width: image.width, height: image.height,
                                                                     configuration: configuration)
                let feature = try await encoder.encode(pixels: processed.pixels.map { Float16($0) }, grid: processed.grid)
                await encoder.unload()
                let session = try await LanguageGenerationSession.open(at: textModel, maximumContextLength: manifest.maximumContextLength,
                                                                       stopTokenIDs: Set(manifest.stopTokenIDs), memoryGrowthBudgetBytes: nil,
                                                                       exportBounds: manifest.exportBounds)
                report.bundles[index].imagesThroughMain = session.imagesThroughMain
                let text = try tokenizer.render([.user(String(repeating: "The lighthouse keeper paints the railing a calm teal. ", count: 40)
                                                        + "Now write a long story about the lighthouse.")]).tokenIDs
                let rendered = try tokenizer.render([.user("Describe this picture in as much detail as you can.", images: 1)]).tokenIDs
                let imagePrompt = LanguageVisionPrompt(renderedTokenIDs: rendered, images: [feature])
                func run(_ kind: String) async throws -> TimingSample {
                    let reply = kind == "text"
                        ? try await session.generate(prompt: text, maximumTokens: 64)
                        : try await session.generate(prompt: imagePrompt, imageTokens: tokens, maximumTokens: 64)
                    if !(report.bundles[index].replies[kind] ?? []).contains(reply.tokens) {
                        report.bundles[index].replies[kind, default: []].append(reply.tokens)
                    }
                    let decode = reply.tokens.count > 1 ? (reply.totalSeconds - reply.timeToFirstTokenSeconds) / Double(reply.tokens.count - 1) : 0
                    return TimingSample(kind: kind, promptTokens: kind == "text" ? text.count : rendered.count - 1 + feature.tokenCount,
                                        tokens: reply.tokens.count, timeToFirstTokenSeconds: reply.timeToFirstTokenSeconds,
                                        decodeSecondsPerToken: decode)
                }
                for kind in ["text", "image", "text", "image"] { _ = try await run(kind) }
                for _ in 0..<report.requests {
                    for kind in ["text", "image"] { report.bundles[index].samples.append(try await run(kind)) }
                }
                await session.unload()
            }
        }
        for index in report.bundles.indices {
            for kind in ["text", "image"] {
                let samples = report.bundles[index].samples.filter { $0.kind == kind }
                report.bundles[index].median[kind] = ["timeToFirstTokenSeconds": median(samples.map(\.timeToFirstTokenSeconds)),
                                                      "decodeSecondsPerToken": median(samples.map(\.decodeSecondsPerToken))]
            }
        }
    } catch {
        report.error = String(describing: error)
        throw error
    }
}
