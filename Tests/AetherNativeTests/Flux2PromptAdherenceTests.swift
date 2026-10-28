import AetherFlux2
import AetherFlux2CoreAI
import AetherFlux2Tokenizers
import AetherKit
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import AetherVision
import Foundation
import Testing

/// Opt-in FLUX.2 [klein] prompt adherence (`Benchmarks/flux2-prompt-adherence.json`), in two GPU holds:
///
/// 1. `AETHER_NATIVE_FLUX2_ADHERENCE=generate`: for each text-to-image fixture prompt, Kit's image with the fixture's
///    own size, steps and initial noise (paired with the float32 Diffusers reference image), and one image per extra
///    seed with Kit's seeded noise. PNGs and `images.json` go to `AETHER_FLUX2_ADHERENCE_OUTPUT`.
/// 2. `AETHER_NATIVE_FLUX2_ADHERENCE=judge`: a local Qwen3-VL bundle (`AETHER_FLUX2_ADHERENCE_JUDGE`) answers each
///    prompt's yes/no questions about every Kit image and the reference image. Writes `report.json` and a
///    `contact-sheet.html` for a subjective review.
///
/// Kit passes if its score on the paired images is no lower than the reference's by more than the plan's margin, and
/// the judge answers the control questions (expected no) correctly at least `minimumControlAccuracy` of the time.
/// Both phases need `AETHER_FLUX2_FIXTURES`; generation also needs `AETHER_FLUX2_BUNDLE`. Run each alone under the
/// shared GPU lock.
private struct AdherencePlan: Codable {
    struct Prompt: Codable { let `case`: String; let questions: [String]; let control: String }
    let schema: Int
    let textToImageFixture: String
    let marginPercentagePoints: Double
    let minimumControlAccuracy: Double
    let extraSeeds: [UInt64]
    let questionSuffix: String
    let maximumAnswerTokens: Int
    let prompts: [Prompt]

    static func load() throws -> (Self, sha256: String) {
        let url = nativeEnvironment["AETHER_FLUX2_ADHERENCE_PLAN"].map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("Benchmarks/flux2-prompt-adherence.json")
        let data = try Data(contentsOf: url)
        return (try JSONDecoder().decode(Self.self, from: data), nativeByteHash(data))
    }
}

/// One image the judge reads: Kit's (paired or extra seed) or the reference's.
private struct AdherenceImage: Codable {
    let `case`: String
    let source: String   // "kit-paired", "kit-seed", "reference"
    let seed: UInt64
    let file: String
    let sha256: String
}

private struct AdherenceAnswer: Codable {
    let `case`: String
    let source: String
    let seed: UInt64
    let question: String
    let control: Bool
    let reply: String
    /// true: yes, false: no, nil: neither (counted as not satisfied).
    let yes: Bool?
}

private func adherenceEnabled(_ phase: String) -> Bool {
    nativeEnvironment["AETHER_NATIVE_FLUX2_ADHERENCE"] == phase && nativeEnvironment["AETHER_FLUX2_FIXTURES"] != nil
        && nativeEnvironment["AETHER_FLUX2_ADHERENCE_OUTPUT"] != nil
}

@Test(.enabled(if: adherenceEnabled("generate") && nativeEnvironment["AETHER_FLUX2_BUNDLE"] != nil), .timeLimit(.minutes(30)))
func flux2PromptAdherenceGenerate() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let (plan, planHash) = try AdherencePlan.load()
    let output = try nativePath("AETHER_FLUX2_ADHERENCE_OUTPUT")
    #expect(!FileManager.default.fileExists(atPath: output.path), "output directory must be new")
    let images = output.appendingPathComponent("images", isDirectory: true)
    try FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
    let fixture = try Flux2Fixture(try nativePath("AETHER_FLUX2_FIXTURES"))
    try #require(fixture.id == plan.textToImageFixture, "the plan names fixture \(plan.textToImageFixture)")
    let assets = try Flux2Assets.open(at: try nativePath("AETHER_FLUX2_BUNDLE"))
    let tokenizer = try await Flux2Tokenizer.open(assets: assets)
    let session = Flux2GenerationSession(assets: assets)
    var records: [AdherenceImage] = []
    var timings: [String: Double] = [:]
    let started = Date()
    for prompt in plan.prompts {
        let c = try #require(fixture.cases.first { $0.id == prompt.case })
        let prepared = try tokenizer.prepare(c.prompt)
        let paired = try FloatTensor(shape: [c.tokens, 128], values: c.noise)
        for seed in [UInt64(c.seed)] + plan.extraSeeds {
            let configuration = try Flux2Configuration(width: c.width, height: c.height, steps: c.steps, seed: seed)
            let begin = Date()
            let image = try await session.generate(configuration: configuration, prompt: prepared,
                                                   noise: seed == UInt64(c.seed) ? paired : nil)
            let source = seed == UInt64(c.seed) ? "kit-paired" : "kit-seed"
            timings["\(c.id)-\(seed)"] = Date().timeIntervalSince(begin)
            let data = try image.pngData()
            let name = "\(c.id)-\(source)-\(seed).png"
            try data.write(to: images.appendingPathComponent(name))
            records.append(AdherenceImage(case: c.id, source: source, seed: seed, file: "images/" + name, sha256: nativeByteHash(data)))
        }
    }
    await session.unload()
    struct Manifest: Encodable {
        let plan: String; let fixture: String; let bundleFingerprint: String
        let images: [AdherenceImage]; let generationSeconds: [String: Double]; let elapsedSeconds: Double
    }
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(Manifest(plan: planHash, fixture: fixture.id, bundleFingerprint: assets.fingerprint, images: records,
                                generationSeconds: timings, elapsedSeconds: Date().timeIntervalSince(started)))
        .write(to: output.appendingPathComponent("images.json"))
    #expect(records.count == plan.prompts.count * (1 + plan.extraSeeds.count))
}

@Test(.enabled(if: adherenceEnabled("judge") && nativeEnvironment["AETHER_FLUX2_ADHERENCE_JUDGE"] != nil), .timeLimit(.minutes(30)))
func flux2PromptAdherenceJudge() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let (plan, planHash) = try AdherencePlan.load()
    let output = try nativePath("AETHER_FLUX2_ADHERENCE_OUTPUT")
    struct Generated: Decodable { let plan: String; let fixture: String; let bundleFingerprint: String; let images: [AdherenceImage] }
    let generated = try JSONDecoder().decode(Generated.self, from: Data(contentsOf: output.appendingPathComponent("images.json")))
    try #require(generated.plan == planHash, "images.json was generated from another plan")
    let fixtureRoot = try nativePath("AETHER_FLUX2_FIXTURES")
    let fixture = try Flux2Fixture(fixtureRoot)
    try #require(fixture.id == generated.fixture)
    // The reference images: the fixture's own PNGs (copied beside Kit's, hashed against the fixture document).
    struct FixtureDocument: Decodable { let sha256: [String: String] }
    let document = try JSONDecoder().decode(FixtureDocument.self, from: Data(contentsOf: fixtureRoot.appendingPathComponent("fixture.json")))
    var images = generated.images
    for prompt in plan.prompts {
        let c = try #require(fixture.cases.first { $0.id == prompt.case })
        let relative = "cases/\(c.id)/image.png"
        let data = try Data(contentsOf: fixtureRoot.appendingPathComponent(relative))
        try #require(nativeByteHash(data) == document.sha256[relative], "\(relative) differs from the fixture document")
        let name = "images/\(c.id)-reference-\(c.seed).png"
        try data.write(to: output.appendingPathComponent(name))
        images.append(AdherenceImage(case: c.id, source: "reference", seed: UInt64(c.seed), file: name, sha256: nativeByteHash(data)))
    }
    for image in images {
        try #require(nativeByteHash(try Data(contentsOf: output.appendingPathComponent(image.file))) == image.sha256, "\(image.file) changed")
    }

    let judge = try nativePath("AETHER_FLUX2_ADHERENCE_JUDGE")
    let tokenizer = try await LanguageTokenizer.open(bundle: judge)
    let session = try await VisionLanguageSession.open(bundle: judge, platform: .macos)
    var answers: [AdherenceAnswer] = []
    let started = Date()
    for image in images {
        let prompt = try #require(plan.prompts.first { $0.case == image.case })
        let pixels = try VisionRGBImage(contentsOf: output.appendingPathComponent(image.file))
        for (question, control) in prompt.questions.map({ ($0, false) }) + [(prompt.control, true)] {
            let rendered = try tokenizer.render([.user(question + plan.questionSuffix, images: 1)])
            let reply = try await session.generate(renderedTokenIDs: rendered.tokenIDs, images: [pixels],
                                                   maximumTokens: plan.maximumAnswerTokens)
            let text = tokenizer.decode(reply.tokens).trimmingCharacters(in: .whitespacesAndNewlines)
            let word = text.lowercased().prefix { $0.isLetter }
            answers.append(AdherenceAnswer(case: image.case, source: image.source, seed: image.seed, question: question,
                                           control: control, reply: text, yes: word == "yes" ? true : word == "no" ? false : nil))
        }
    }
    await session.unload()

    func score(_ source: String) -> Double {
        let positives = answers.filter { $0.source == source && !$0.control }
        return positives.isEmpty ? 0 : Double(positives.filter { $0.yes == true }.count) / Double(positives.count)
    }
    let controls = answers.filter(\.control)
    let controlAccuracy = Double(controls.filter { $0.yes == false }.count) / Double(max(1, controls.count))
    let unclear = answers.filter { $0.yes == nil }.count
    let paired = score("kit-paired"), reference = score("reference"), seeds = score("kit-seed")
    var perPrompt: [String: [String: Double]] = [:]
    for prompt in plan.prompts {
        for source in ["kit-paired", "kit-seed", "reference"] {
            let positives = answers.filter { $0.case == prompt.case && $0.source == source && !$0.control }
            perPrompt[prompt.case, default: [:]][source] = Double(positives.filter { $0.yes == true }.count) / Double(max(1, positives.count))
        }
    }
    let passed = paired >= reference - plan.marginPercentagePoints / 100 && controlAccuracy >= plan.minimumControlAccuracy
    struct Report: Encodable {
        let kind: String; let plan: String; let fixture: String; let bundleFingerprint: String; let judge: String
        let kitPairedScore: Double; let referenceScore: Double; let kitExtraSeedScore: Double
        let marginPercentagePoints: Double; let controlAccuracy: Double; let unclearAnswers: Int; let passed: Bool
        let perPrompt: [String: [String: Double]]; let answers: [AdherenceAnswer]; let images: [AdherenceImage]; let judgeSeconds: Double
    }
    let manifest = try LanguageModelManifest.load(from: judge)
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(Report(kind: "flux2-prompt-adherence", plan: planHash, fixture: fixture.id,
                              bundleFingerprint: generated.bundleFingerprint,
                              judge: "\(manifest.sourceRepository)@\(manifest.sourceRevision) model \(manifest.modelSHA256.prefix(16))",
                              kitPairedScore: paired, referenceScore: reference, kitExtraSeedScore: seeds,
                              marginPercentagePoints: plan.marginPercentagePoints, controlAccuracy: controlAccuracy,
                              unclearAnswers: unclear, passed: passed, perPrompt: perPrompt, answers: answers, images: images,
                              judgeSeconds: Date().timeIntervalSince(started)))
        .write(to: output.appendingPathComponent("report.json"))
    try contactSheet(plan: plan, images: images, answers: answers, scores: (paired, reference, seeds))
        .write(to: output.appendingPathComponent("contact-sheet.html"), atomically: true, encoding: .utf8)
    print(String(format: "flux2 adherence: Kit paired %.3f, reference %.3f, Kit extra seeds %.3f, control accuracy %.3f, unclear %d",
                 paired, reference, seeds, controlAccuracy, unclear))
    #expect(controlAccuracy >= plan.minimumControlAccuracy, "the judge answered yes to control questions")
    #expect(paired >= reference - plan.marginPercentagePoints / 100, "Kit's adherence is below the reference's beyond the margin")
}

/// A static page: one row per prompt (reference, Kit paired, Kit extra seeds), each image captioned with its answers.
private func contactSheet(plan: AdherencePlan, images: [AdherenceImage], answers: [AdherenceAnswer],
                          scores: (Double, Double, Double)) -> String {
    func escape(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
    }
    var html = """
    <!doctype html><html><head><meta charset="utf-8"><title>FLUX.2 prompt adherence</title>
    <style>body{font:13px -apple-system,sans-serif;margin:16px}figure{display:inline-block;margin:4px;vertical-align:top;width:260px}
    img{width:256px;border:1px solid #ccc}figcaption{font-size:11px}.no{color:#b00}.ctl{color:#666}</style></head><body>
    <h1>FLUX.2 [klein] prompt adherence</h1>
    """
    html += String(format: "<p>Positive-question score: Kit paired %.1f%%, reference %.1f%%, Kit extra seeds %.1f%%.</p>\n",
                   scores.0 * 100, scores.1 * 100, scores.2 * 100)
    let order = ["reference": 0, "kit-paired": 1, "kit-seed": 2]
    for prompt in plan.prompts {
        html += "<h2>\(escape(prompt.case))</h2>\n"
        for image in images.filter({ $0.case == prompt.case }).sorted(by: { (order[$0.source] ?? 3, $0.seed) < (order[$1.source] ?? 3, $1.seed) }) {
            let lines = answers.filter { $0.case == image.case && $0.source == image.source && $0.seed == image.seed }.map { a -> String in
                let ok = a.control ? a.yes == false : a.yes == true
                return "<div class=\"\(a.control ? "ctl" : ok ? "" : "no")\">\(ok ? "✓" : "✗") \(escape(a.question)) — \(escape(a.reply))</div>"
            }
            html += "<figure><img src=\"\(image.file)\"><figcaption><b>\(image.source) · seed \(image.seed)</b>\(lines.joined())</figcaption></figure>\n"
        }
    }
    return html + "</body></html>\n"
}
