import AetherCore
import AetherHost
import AetherRuntime
import CryptoKit
import Foundation
import Metal
import Testing
@testable import AetherText

/// The describer gate (docs/VLM.md §6, G4), opt-in: a `vlm-static-inject@1` export through the Generator against the
/// earlier 0.0.7 describer (`VisionLanguageSession` on a 0.0.x export of the same weights), greedy, on the fixed
/// synthetic pictures of `Forge/qwen3vl/synthetic_pictures.py`.
///
///     AETHER_GATES=1 AETHER_VLM_BUNDLE=<export> AETHER_VLM_PICTURES=<pictures> AETHER_VLM_REFERENCE=<reference> \
///         scripts/gpu-lease -- swift test --filter DescriberGateTests
///
/// The reference is `Tools/DescriberReference007`'s output. Bars:
/// - the pictures are the reference's (SHA-256), and each picture's patch rows equal 0.0.7's, which fed fp16, at fp16;
/// - every case renders the reference's prompt ids exactly;
/// - every generated token equals the reference's; at a divergence the reference's own top-1 minus top-2 probability
///   must be below 0.1 and our token in its top five (INTEGRATION.md's near-tie rule), and the case ends there.
@Suite("DescriberGateTests", .serialized, .enabled(if: DescriberGate.enabled))
struct DescriberGateTests {
    static let env = ProcessInfo.processInfo.environment

    struct Cases: Decodable {
        struct Case: Decodable {
            let name: String
            let instructions: String
            let question: String
            let pictures: [String]
            let maximumTokens: Int
        }
        let files: [String: String]
        let cases: [Case]
    }

    struct Reference: Decodable {
        struct Picture: Decodable {
            let file: String
            let grid: [Int]
            let pixelsFile: String
        }
        struct Step: Decodable {
            let top: [Int]
            let probabilities: [Double]
        }
        struct Record: Decodable {
            let name: String
            let renderedTokenIDs: [Int]
            let tokens: [Int]
            let stopReason: String
            let text: String
            let timeToFirstTokenSeconds: Double
            let totalSeconds: Double
            let steps: [Step]
            let teacherArgmaxAgrees: Bool
        }
        let pictures: [String: String]
        let preprocessed: [Picture]
        let cases: [Record]
    }

    static let stops: Set<Int> = [151_643, 151_645]
    static let nearTieMargin = 0.1

    static func url(_ key: String) -> URL { URL(fileURLWithPath: env[key]!) }

    static func localBundle(_ url: URL) throws -> InstalledBundle {
        let manifest = try Manifest.load(fromBundle: url)
        let ref = ModelRef(
            catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: manifest.variant.platform.rawValue,
            driver: manifest.driver, assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
        return InstalledBundle(ref: ref, url: url, isStale: false)
    }

    static func sha256(_ url: URL) throws -> String {
        SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
    }

    static func load() throws -> (Cases, Reference) {
        let cases = try JSONDecoder().decode(Cases.self, from: Data(contentsOf: url("AETHER_VLM_PICTURES").appending(path: "cases.json")))
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url("AETHER_VLM_REFERENCE").appending(path: "reference.json")))
        return (cases, reference)
    }

    @Test func picturesAndPreprocessingAreTheReferences() throws {
        let (cases, reference) = try Self.load()
        let pictures = Self.url("AETHER_VLM_PICTURES")
        for (file, sha) in cases.files.sorted(by: { $0.key < $1.key }) {
            #expect(try Self.sha256(pictures.appending(path: file)) == sha, "\(file)")
            #expect(reference.pictures[file] == sha, "\(file): the reference was recorded on another picture")
        }
        let manifest = try Manifest.load(fromBundle: Self.url("AETHER_VLM_BUNDLE"))
        let spec = ImageInputSpec(
            padID: 151_655, openID: 151_652, closeID: 151_653, idBase: 151_936, slots: manifest.image?.slots ?? 0,
            maxImages: manifest.image?.maxImages ?? 0, patchRows: 2048, merge: 2,
            preprocessor: try ImagePreprocessor(manifest: manifest))
        for picture in reference.preprocessed {
            let ours = try spec.prepare(ImageInput(contentsOf: pictures.appending(path: picture.file)))
            let theirs = try Data(contentsOf: Self.url("AETHER_VLM_REFERENCE").appending(path: picture.pixelsFile)).withUnsafeBytes {
                Array($0.bindMemory(to: Float16.self))
            }
            #expect([1, ours.gridHeight, ours.gridWidth] == picture.grid, "\(picture.file) grid")
            #expect(ours.values.count == theirs.count, "\(picture.file) rows")
            let differing = zip(ours.values, theirs).filter { Float16($0) != $1 }.count
            let worst = zip(ours.values, theirs).map { abs($0 - Float($1)) }.max() ?? 0
            print("GATE describer preprocess \(picture.file): grid \(picture.grid), \(ours.values.count) values, "
                  + "\(differing) differ at fp16, largest f32-vs-fp16 difference \(worst)")
            #expect(differing == 0, "\(picture.file)")
        }
    }

    @Test func greedyRepliesAreThe007Describers() async throws {
        let (cases, reference) = try Self.load()
        let pictures = Self.url("AETHER_VLM_PICTURES")
        let bundleURL = Self.url("AETHER_VLM_BUNDLE")
        let bundle = try Self.localBundle(bundleURL)
        let manifest = try Manifest.load(fromBundle: bundleURL)
        let metal = MTLCreateSystemDefaultDevice()
        let clock = ContinuousClock(), t0 = clock.now
        let engine = try await DecoderEngine(bundle: bundle, manifest: manifest, device: .current,
                                             options: DecoderEngine.Options(capacity: 4096))
        let loadSeconds = (clock.now - t0) / .seconds(1)
        print("GATE describer load: \(String(format: "%.1f", loadSeconds)) s, decode \(await engine.decodeModeDescription), "
              + "model \(await engine.loadedModelBytes), Metal \((metal?.currentAllocatedSize ?? 0) >> 20) MiB")
        let generator = try await ChatSession.generator(engine: engine, bundle: bundle, manifest: manifest)
        var exact = 0, nearTies = 0
        var summary: [[String: String]] = []
        for c in cases.cases {
            let record = try #require(reference.cases.first { $0.name == c.name }, "\(c.name) in the reference")
            let images = c.pictures.map { ChatImage(ImageInput(contentsOf: pictures.appending(path: $0))) }
            let messages: [ChatMessage] = (c.instructions.isEmpty ? [] : [.system(c.instructions)]) + [.user(c.question, images: images)]
            let rendered = try generator.renderer.render(messages, options: RenderOptions())
            #expect(rendered.tokens.map(Int.init) == record.renderedTokenIDs, "\(c.name): rendered ids")
            var result: GenerationSummary?
            var text = ""
            let start = clock.now
            for try await event in generator.generate(messages, parameters: GenerationParameters(maxTokens: c.maximumTokens)) {
                if case .text(let delta) = event { text += delta }
                if case .finished(let s) = event { result = s }
            }
            let seconds = (clock.now - start) / .seconds(1)
            let ours = try #require(result).tokens.map(Int.init)
            let theirs = record.tokens
            let strip = { (t: [Int]) in t.last.map { Self.stops.contains($0) } == true ? Array(t.dropLast()) : t }
            let a = strip(ours), b = strip(theirs)
            let divergence = zip(a, b).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                ?? (a.count == b.count ? nil : min(a.count, b.count))
            var verdict = "exact"
            if let i = divergence {
                if i < record.steps.count, i < a.count {
                    let step = record.steps[i]
                    let margin = step.probabilities[0] - step.probabilities[1]
                    let inTop = step.top.prefix(5).contains(a[i])
                    verdict = String(format: "near-tie flip at %d (reference margin %.4f, ours %d in its top five: %@)", i, margin, a[i],
                                     inTop ? "yes" : "no")
                    #expect(margin < Self.nearTieMargin && inTop, "\(c.name): divergence at \(i) is not a near tie: \(verdict)")
                    if margin < Self.nearTieMargin && inTop { nearTies += 1 }
                } else {
                    verdict = "length differs at \(i): ours \(a.count), reference \(b.count)"
                    Issue.record("\(c.name): \(verdict)")
                }
            } else {
                exact += 1
            }
            #expect(record.teacherArgmaxAgrees, "\(c.name): the reference's teacher forcing disagrees with its own decode")
            let r = try #require(result)
            print("GATE describer \(c.name): \(c.pictures.count) picture(s), prompt \(r.promptTokens) tokens, \(ours.count) generated "
                  + "(reference \(theirs.count)), \(verdict); prefill \(String(format: "%.3f", r.prefillSeconds)) s, decode "
                  + "\(String(format: "%.1f", r.decodeSeconds * 1000 / Double(max(1, ours.count - 1)))) ms/token "
                  + "(reference total \(String(format: "%.2f", record.totalSeconds)) s, ours \(String(format: "%.2f", seconds)) s), "
                  + "Metal \((metal?.currentAllocatedSize ?? 0) >> 20) MiB")
            print("GATE describer \(c.name) text: \(text.prefix(200))")
            summary.append(["case": c.name, "verdict": verdict, "ours": "\(ours.count)", "reference": "\(theirs.count)"])
        }
        print("GATE describer: \(exact) of \(cases.cases.count) cases token-exact, \(nearTies) near-tie flips")
        if let log = Self.env["AETHER_VLM_GATE_LOG"] {
            try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted]).write(to: URL(fileURLWithPath: log))
        }
    }
}

enum DescriberGate {
    static var enabled: Bool {
        let env = ProcessInfo.processInfo.environment
        return env["AETHER_GATES"] == "1" && env["AETHER_VLM_BUNDLE"] != nil && env["AETHER_VLM_PICTURES"] != nil
            && env["AETHER_VLM_REFERENCE"] != nil
    }
}
