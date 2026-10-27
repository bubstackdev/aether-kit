import AetherKit
import AetherLanguageCoreAI
import AetherVision
import Foundation
import Testing

/// Opt-in native check of the public `VisionLanguageSession` on a registered Qwen3-VL bundle: it
/// must reproduce, token for token, the component-level Kit run that `qwen3VLImageTextQualification`
/// gated (Kit decoding and preprocessing), and it records agreement with upstream.
///
/// AETHER_NATIVE_QWEN3VL_SESSION=1
/// AETHER_QWEN3VL_BUNDLE=<bundle with language-model.json and a vision registration>
/// AETHER_QWEN3VL_REFERENCE=<photo reference dir with reference.json>
/// AETHER_QWEN3VL_COMPONENT_REPORT=<qwen3VLImageTextQualification report for the same photos, Kit preprocessing>
/// AETHER_QWEN3VL_OUTPUT=<report .json>
private struct PhotoReference: Decodable {
    struct File: Decodable { let file: String }
    struct Case: Decodable { let name: String; let promptTokens: [Int]; let greedyTokens: [Int]; let image: File }
    let cases: [Case]
    let imageGrids: [[Int]]
}

private struct ComponentReport: Decodable {
    struct Case: Decodable { let name: String; let greedyTokens: [Int]; let preprocessing: String }
    let passed: Bool
    let cases: [Case]
}

private struct SessionCase: Codable {
    let name: String
    let tokens: [Int]
    let matchesComponentRun: Bool
    let upstreamAgreementTokens: Int
    let upstreamTokens: Int
    let timeToFirstTokenSeconds: Double
    let totalSeconds: Double
}

private struct SessionReport: Codable {
    var passed = false
    var stage = "preparation"
    var error: String?
    var imageGrids: [[Int]] = []
    var gate = "VisionLanguageSession.open(bundle:) plus generate(renderedTokenIDs:images:) on the photo files reproduces the gated component-level Kit run token for token; unload closes the session"
    var cases: [SessionCase] = []
    var closedAfterUnload = false
    /// The registration lists every tokenizer file (`tokenizerFiles`) and `verifyTokenizer` accepts the bundle.
    var tokenizerFilesVerified = false
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_QWEN3VL_SESSION"] == "1"), .timeLimit(.minutes(20)))
func visionLanguageSessionReproducesQualifiedRun() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    var report = SessionReport()
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_OUTPUT"]))
    defer {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    do {
        let bundle = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_BUNDLE"]))
        let directory = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_REFERENCE"]))
        let reference = try JSONDecoder().decode(PhotoReference.self, from: Data(contentsOf: directory.appendingPathComponent("reference.json")))
        let component = try JSONDecoder().decode(ComponentReport.self, from: Data(contentsOf:
            URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_COMPONENT_REPORT"]))))
        guard component.passed, component.cases.allSatisfy({ $0.preprocessing == "swift" }) else {
            throw NativeFailure(description: "the component report must be a passing Kit-preprocessing run")
        }
        report.stage = "tokenizer"
        let manifest = try LanguageModelManifest.load(from: bundle)
        _ = try manifest.verifyTokenizer(in: bundle)
        report.tokenizerFilesVerified = manifest.tokenizerFiles?.isEmpty == false
        report.stage = "open"
        let session = try await VisionLanguageSession.open(bundle: bundle, platform: .macos)
        report.imageGrids = (await session.imagePreprocessing.imageGrids ?? []).map { [$0.height, $0.width] }
        guard report.imageGrids == reference.imageGrids else {
            throw NativeFailure(description: "the bundle's fixed image grids differ from the reference's")
        }
        for c in reference.cases {
            report.stage = "case \(c.name)"
            let image = try VisionRGBImage(contentsOf: directory.appendingPathComponent(c.image.file))
            var rendered: [Int] = []
            for id in c.promptTokens where !(id == 151655 && rendered.last == 151655) { rendered.append(id) }
            let reply = try await session.generate(renderedTokenIDs: rendered, images: [image], maximumTokens: c.greedyTokens.count)
            var upstream = c.greedyTokens
            while let last = upstream.last, [151645, 151643].contains(last) { upstream.removeLast() }
            let expected = try #require(component.cases.first { $0.name == c.name }).greedyTokens
            report.cases.append(SessionCase(name: c.name, tokens: reply.tokens, matchesComponentRun: reply.tokens == expected,
                upstreamAgreementTokens: zip(reply.tokens, upstream).prefix { $0 == $1 }.count, upstreamTokens: upstream.count,
                timeToFirstTokenSeconds: reply.timeToFirstTokenSeconds, totalSeconds: reply.totalSeconds))
        }
        report.stage = "unload"
        await session.unload()
        report.closedAfterUnload = await session.state == .closed
        await #expect(throws: LanguageGenerationError.closed) {
            _ = try await session.generate(renderedTokenIDs: [1], images: [], maximumTokens: 1)
        }
        report.passed = report.cases.count == reference.cases.count && report.cases.allSatisfy(\.matchesComponentRun)
            && report.closedAfterUnload && report.tokenizerFilesVerified
        report.stage = "complete"
    } catch {
        report.error = String(describing: error)
        throw error
    }
    #expect(report.passed)
}
