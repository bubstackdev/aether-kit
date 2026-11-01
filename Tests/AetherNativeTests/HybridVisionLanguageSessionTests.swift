import AetherKit
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import AetherVision
import Darwin
import Foundation
import Metal
import Testing

/// Opt-in native check of a registered Qwen3.5 bundle (hybrid decoder, `hybrid` registration) through the public API a
/// host drives: `VisionLanguageSession.open(bundle:)` and `generate(renderedTokenIDs:images:)` from the raw image file
/// (Kit's decoding and preprocessing), and `HybridLanguageSession.open(bundle:)` for text, each against the upstream
/// fp32 references (`reference_qwen35_image.py`, `reference_qwen35.py`). `AetherLanguageTokenizers` must render every
/// reference prompt (thinking off) to the same tokens. Records time to first token, decode rate, process footprint,
/// Metal's allocated size and wired mapped-file memory before opening, after each request and after unload.
///
/// AETHER_NATIVE_QWEN35_SESSION=1
/// AETHER_QWEN35_BUNDLE=<bundle with language-model.json, a vision and a hybrid registration>
/// AETHER_QWEN35_IMAGE_REFERENCE=<reference_qwen35_image.py output>
/// AETHER_QWEN35_TEXT_REFERENCE=<reference_qwen35.py output> (optional)
/// AETHER_QWEN35_IMAGE_FILE=<the image to decode instead of the reference's copy> (optional: the raw JPEG of a
///     same-decode reference, which upstream saw as Kit's lossless decode)
/// AETHER_QWEN35_REPEATS=<image requests per case, default 3>
/// AETHER_QWEN35_OUTPUT=<report .json>
private struct ImageReference: Decodable {
    struct File: Decodable { let file: String }
    struct Prompt: Decodable { let system: String; let user: String }
    struct Case: Decodable { let name: String; let promptTokens: [Int]; let greedyTokens: [Int]; let image: File; let grid: [Int] }
    let prompt: Prompt
    let cases: [Case]
}

private struct TextReference: Decodable {
    struct Case: Decodable { let name: String; let prompt: String; let promptTokens: [Int]; let greedyTokens: [Int] }
    let cases: [Case]
}

private struct Run: Codable {
    let name: String
    let tokens: Int
    let stopReason: String
    /// Leading tokens equal to upstream's greedy run, and whether the whole reply is (a stop token where upstream
    /// emits it counts as agreement: upstream's reference run ignored end of sequence).
    let upstreamAgreementTokens: Int
    let identicalToUpstream: Bool
    let timeToFirstTokenSeconds: Double
    let totalSeconds: Double
    let decodeTokensPerSecond: Double
    let text: String
}

private struct Report: Codable {
    var passed = false
    var stage = "preparation"
    var error: String?
    var tokenizerRendersReferencePrompts = true
    var imageRuns: [Run] = []
    var textRuns: [Run] = []
    var memory: [Sample] = []
    var closedAfterUnload = false
    var imageOpenSeconds = 0.0
    var textOpenSeconds = 0.0
    var imageFile = ""
}

private struct Sample: Codable {
    let phase: String, footprintMiB: Int, metalMiB: Int, wiredMappedMiB: Int
    static let device = MTLCreateSystemDefaultDevice()

    init(_ phase: String) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        self.phase = phase
        footprintMiB = result == KERN_SUCCESS ? Int(info.phys_footprint >> 20) : 0
        metalMiB = (Self.device?.currentAllocatedSize ?? 0) >> 20
        wiredMappedMiB = Self.wiredMappedMiB()
    }

    /// Wired "mapped file" memory (the model weights Core AI maps) from `footprint -w -j`; 0 when unavailable.
    static func wiredMappedMiB() -> Int {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("qwen35-footprint-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        let tool = Process()
        tool.executableURL = URL(filePath: "/usr/bin/footprint")
        tool.arguments = ["-p", String(getpid()), "-f", "bytes", "-w", "-j", file.path]
        tool.standardOutput = FileHandle.nullDevice
        tool.standardError = FileHandle.nullDevice
        guard (try? tool.run()) != nil else { return 0 }
        tool.waitUntilExit()
        guard let data = try? Data(contentsOf: file), let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let process = (json["processes"] as? [[String: Any]])?.first,
              let mapped = (process["categories"] as? [String: [String: Any]])?["mapped file"] else { return 0 }
        return ((mapped["wired"] as? Int) ?? 0) >> 20
    }
}

private func run(_ name: String, result: LanguageGenerationResult, upstream: [Int], stops: Set<Int>, text: String) -> Run {
    let agreement = zip(result.tokens, upstream).prefix { $0 == $1 }.count
    let identical = agreement == result.tokens.count
        && (result.stopReason != .endOfSequence || (result.tokens.count < upstream.count && stops.contains(upstream[result.tokens.count])))
    let decode = result.tokens.count > 1 ? Double(result.tokens.count - 1) / max(result.totalSeconds - result.timeToFirstTokenSeconds, 1e-9) : 0
    return Run(name: name, tokens: result.tokens.count, stopReason: result.stopReason.rawValue, upstreamAgreementTokens: agreement,
               identicalToUpstream: identical, timeToFirstTokenSeconds: result.timeToFirstTokenSeconds,
               totalSeconds: result.totalSeconds, decodeTokensPerSecond: decode, text: text)
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_QWEN35_SESSION"] == "1"), .timeLimit(.minutes(30)))
func hybridVisionLanguageSessionMatchesUpstream() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    var report = Report()
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN35_OUTPUT"]))
    defer {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    do {
        let bundle = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN35_BUNDLE"]))
        let directory = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN35_IMAGE_REFERENCE"]))
        let reference = try JSONDecoder().decode(ImageReference.self, from: Data(contentsOf: directory.appendingPathComponent("reference.json")))
        let repeats = Int(nativeEnvironment["AETHER_QWEN35_REPEATS"] ?? "3") ?? 3
        let manifest = try LanguageModelManifest.load(from: bundle)
        let vision = try #require(manifest.vision)
        let stops = Set(manifest.stopTokenIDs)
        let tokenizer = try await LanguageTokenizer.open(bundle: bundle)

        report.stage = "image"
        report.memory.append(Sample("before open"))
        let clock = ContinuousClock()
        var started = clock.now
        let session = try await VisionLanguageSession.open(bundle: bundle, platform: .macos)
        report.imageOpenSeconds = Double((clock.now - started).components.attoseconds) / 1e18 + Double((clock.now - started).components.seconds)
        report.memory.append(Sample("after open"))
        for c in reference.cases {
            var rendered: [Int] = []
            for id in c.promptTokens where !(id == vision.imageTokenID && rendered.last == vision.imageTokenID) { rendered.append(id) }
            let mine = try tokenizer.render([.system(reference.prompt.system), .user(reference.prompt.user, images: 1)], thinking: false)
            report.tokenizerRendersReferencePrompts = report.tokenizerRendersReferencePrompts && mine.tokenIDs == rendered
            let file = nativeEnvironment["AETHER_QWEN35_IMAGE_FILE"].map { URL(filePath: $0) } ?? directory.appendingPathComponent(c.image.file)
            report.imageFile = file.lastPathComponent
            let image = try VisionRGBImage(contentsOf: file)
            for index in 0..<repeats {
                let result = try await session.generate(renderedTokenIDs: mine.tokenIDs, images: [image],
                                                        maximumTokens: c.greedyTokens.count, sampling: .greedy)
                report.memory.append(Sample("after \(c.name) \(index + 1)"))
                report.imageRuns.append(run("\(c.name) \(index + 1)", result: result, upstream: c.greedyTokens, stops: stops,
                                            text: tokenizer.decode(result.tokens)))
            }
        }
        await session.unload()
        report.memory.append(Sample("after image unload"))
        report.closedAfterUnload = await session.state == .closed

        if let text = nativeEnvironment["AETHER_QWEN35_TEXT_REFERENCE"] {
            report.stage = "text"
            let textReference = try JSONDecoder().decode(TextReference.self, from: Data(contentsOf: URL(filePath: text).appendingPathComponent("reference.json")))
            started = clock.now
            let language = try await HybridLanguageSession.open(bundle: bundle, platform: .macos)
            report.textOpenSeconds = Double((clock.now - started).components.attoseconds) / 1e18 + Double((clock.now - started).components.seconds)
            report.memory.append(Sample("after text open"))
            for c in textReference.cases {
                let mine = try tokenizer.render([.user(c.prompt)], thinking: false)
                report.tokenizerRendersReferencePrompts = report.tokenizerRendersReferencePrompts && mine.tokenIDs == c.promptTokens
                let result = try await language.generate(LanguageGenerationRequest(prompt: mine.tokenIDs, maximumTokens: c.greedyTokens.count),
                                                         onToken: nil)
                report.textRuns.append(run(c.name, result: result, upstream: c.greedyTokens, stops: stops,
                                           text: tokenizer.decode(result.tokens)))
            }
            await language.unload()
            report.memory.append(Sample("after text unload"))
        }
        report.passed = report.closedAfterUnload && report.tokenizerRendersReferencePrompts
            && (report.imageRuns + report.textRuns).allSatisfy(\.identicalToUpstream)
        report.stage = "complete"
    } catch {
        report.error = String(describing: error)
        throw error
    }
    #expect(report.passed)
}
