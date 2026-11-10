import AetherCore
import AetherHost
import AetherRuntime
import Foundation
import Metal
import Testing
@testable import AetherText

/// The Qwen3.5 / Qwen3.6 export gate, opt-in: a `Forge/qwen3_5` bundle (a hybrid decoder with `main` and `prefill`
/// functions over recurrent states, and a vision tower) through the engine and the Generator, against a reference
/// recorded from the same checkpoint (`Forge/qwen3_5`'s reference tools: transformers in float32 for a size that fits,
/// mlx-vlm at 8 bits for 27B), from the same pictures and prompt.
///
///     AETHER_GATES=1 AETHER_QWEN35_BUNDLE=<export> AETHER_QWEN35_REFERENCE=<reference.json> \
///         scripts/gpu-lease -- swift test --filter Qwen35GateTests
///
/// Bars: the chat template renders the reference's prompt ids; the pictures prepare to the reference's grids and the
/// prompt expands to its length; teacher-forced on the reference's greedy tokens, the reference's token is in our top
/// five at every step and our argmax agrees on at least `AETHER_QWEN35_AGREE` of the steps (default 0.75: a 4-bit
/// export against an 8-bit reference, where MLX's own 4-bit agrees on 0.875); our greedy reply's first token is the
/// reference's. Timings, memory, KL and the greedy text are printed (`GATE qwen35 …`).
enum Qwen35Gate {
    static let env = ProcessInfo.processInfo.environment
    static let enabled = env["AETHER_GATES"] == "1" && env["AETHER_QWEN35_BUNDLE"] != nil && env["AETHER_QWEN35_REFERENCE"] != nil
    static let retention = enabled && env["AETHER_QWEN35_RETENTION"] == "1"  // the slower diagnostics
}

@Suite("Qwen35GateTests", .serialized, .enabled(if: Qwen35Gate.enabled))
struct Qwen35GateTests {
    static let env = Qwen35Gate.env

    struct Reference: Decodable {
        struct Step: Decodable {
            let token: Int
            let top: [Int]
            let logprobs: [Double]
        }
        let reference: String
        let instructions: String
        let question: String
        let pictures: [String]
        let grids: [[Int]]
        let promptTokens: Int
        let renderedTokenIDs: [Int]
        let tokens: [Int]
        let text: String
        let steps: [Step]
    }

    static func localBundle(_ url: URL) throws -> InstalledBundle {
        let manifest = try Manifest.load(fromBundle: url)
        let ref = ModelRef(
            catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: manifest.variant.platform.rawValue,
            driver: manifest.driver, assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
        return InstalledBundle(ref: ref, url: url, isStale: false)
    }

    /// log-softmax of one logits row.
    static func logSoftmax(_ row: ArraySlice<Float>) -> [Double] {
        let top = Double(row.max() ?? 0)
        let sum = row.reduce(0.0) { $0 + exp(Double($1) - top) }
        return row.map { Double($0) - top - log(sum) }
    }

    @Test func teacherForcedAndGreedyAgainstTheReference() async throws {
        let reference = try JSONDecoder().decode(
            Reference.self, from: Data(contentsOf: URL(fileURLWithPath: Self.env["AETHER_QWEN35_REFERENCE"]!)))
        let bundleURL = URL(fileURLWithPath: Self.env["AETHER_QWEN35_BUNDLE"]!)
        let bundle = try Self.localBundle(bundleURL)
        let manifest = try Manifest.load(fromBundle: bundleURL)
        let metal = MTLCreateSystemDefaultDevice()
        let clock = ContinuousClock(), t0 = clock.now
        let engine = try await DecoderEngine(bundle: bundle, manifest: manifest, device: .current,
                                             options: DecoderEngine.Options(capacity: 2048))
        print("GATE qwen35 load: \(String(format: "%.1f", (clock.now - t0) / .seconds(1))) s, decode "
              + "\(await engine.decodeModeDescription), model \(await engine.loadedModelBytes), Metal "
              + "\((metal?.currentAllocatedSize ?? 0) >> 20) MiB")
        let generator = try await ChatSession.generator(engine: engine, bundle: bundle, manifest: manifest)
        let images = reference.pictures.map { ChatImage(ImageInput(contentsOf: URL(fileURLWithPath: $0))) }
        let messages: [ChatMessage] = [.system(reference.instructions), .user(reference.question, images: images)]
        let rendered = try generator.renderer.render(messages, options: RenderOptions(thinking: false))
        #expect(rendered.tokens.map(Int.init) == reference.renderedTokenIDs, "rendered ids")

        // Teacher forcing at the engine: the plan's expanded prompt, then the reference's tokens one at a time.
        let spec = try #require(engine.imageInput)
        let prepared = try images.map { try $0.prepared(for: spec) }
        #expect(prepared.map { [1, $0.gridH, $0.gridW] } == reference.grids, "grids")
        let plan = try ImagePromptPlan(rendered: rendered.tokens, images: prepared, spec: spec)
        #expect(plan.tokens.count == reference.promptTokens, "expanded prompt length")
        try await engine.reset()
        let t1 = clock.now
        try await engine.bindImages(plan, prepared: prepared)
        let bindSeconds = (clock.now - t1) / .seconds(1)
        let t2 = clock.now
        var logits = try #require(try await engine.prefill(plan.tokens, logits: .last))
        let promptSeconds = (clock.now - t2) / .seconds(1)
        var agree = 0, inTop5 = 0, kls: [Double] = [], disagreements: [Int] = []
        var stepSeconds: [Double] = []
        for (k, step) in reference.steps.enumerated() {
            let mine = Self.logSoftmax(logits.last)
            let order = mine.indices.sorted { mine[$0] > mine[$1] }
            if order[0] == step.token { agree += 1 } else { disagreements.append(k) }
            if order.prefix(5).contains(step.token) { inTop5 += 1 }
            #expect(order.prefix(5).contains(step.token), "step \(k): the reference's token \(step.token) is not in our top five")
            // KL(reference || ours) over the reference's top 20 (renormalized on both sides).
            let p = step.logprobs.map { exp($0) }, ps = p.reduce(0, +)
            let q = step.top.map { exp(mine[$0]) }, qs = q.reduce(0, +)
            kls.append(zip(p, q).reduce(0.0) { $0 + ($1.0 / ps) * log(($1.0 / ps) / max($1.1 / qs, 1e-30)) })
            if k + 1 < reference.steps.count {
                let t = clock.now
                logits = try #require(try await engine.prefill([Int32(step.token)], logits: .last))
                stepSeconds.append((clock.now - t) / .seconds(1))
            }
        }
        let n = reference.steps.count
        let bar = Double(Self.env["AETHER_QWEN35_AGREE"] ?? "") ?? 0.75
        print("GATE qwen35 teacher-forced against \(reference.reference): argmax \(agree)/\(n) (disagree at \(disagreements)), "
              + "reference token in our top 5 \(inTop5)/\(n), KL mean \(String(format: "%.4f", kls.reduce(0, +) / Double(n))) "
              + "max \(String(format: "%.4f", kls.max() ?? 0))")
        let median = stepSeconds.sorted()[stepSeconds.count / 2]
        print("GATE qwen35 timing: bind \(String(format: "%.2f", bindSeconds)) s (\(prepared.count) pictures), prompt "
              + "\(plan.tokens.count) tokens \(String(format: "%.2f", promptSeconds)) s, step median "
              + "\(String(format: "%.1f", median * 1000)) ms; Metal \((metal?.currentAllocatedSize ?? 0) >> 20) MiB")
        #expect(Double(agree) >= bar * Double(n), "argmax agreement \(agree)/\(n) below \(bar)")

        // Greedy through the Generator (the path Atelier takes), from an empty state.
        try await engine.reset()
        var text = "", summary: GenerationSummary?
        let t3 = clock.now
        for try await event in generator.generate(messages, parameters: GenerationParameters(maxTokens: n, thinking: false)) {
            if case .text(let delta) = event { text += delta }
            if case .finished(let s) = event { summary = s }
        }
        let total = (clock.now - t3) / .seconds(1)
        let tokens = try #require(summary).tokens.map(Int.init)
        let same = zip(tokens, reference.tokens).prefix { $0 == $1 }.count
        print("GATE qwen35 greedy: \(tokens.count) tokens in \(String(format: "%.1f", total)) s, the reference's first \(same)")
        print("GATE qwen35 greedy text: \(text)")
        print("GATE qwen35 reference text: \(reference.text)")
        #expect(tokens.first == reference.tokens.first, "the first greedy token")
    }

    /// Metal across mixed requests (opt-in, AETHER_QWEN35_RETENTION=1): the two-picture describer prompt, the same
    /// again, another picture alone, text only, and round again. Prints the device's Metal after each (Core AI retains
    /// memory on reshaping calls: docs/VLM.md, G5).
    @Test(.enabled(if: Qwen35Gate.retention)) func metalAcrossRequests() async throws {
        let reference = try JSONDecoder().decode(
            Reference.self, from: Data(contentsOf: URL(fileURLWithPath: Self.env["AETHER_QWEN35_REFERENCE"]!)))
        let bundleURL = URL(fileURLWithPath: Self.env["AETHER_QWEN35_BUNDLE"]!)
        let bundle = try Self.localBundle(bundleURL)
        let manifest = try Manifest.load(fromBundle: bundleURL)
        let engine = try await DecoderEngine(bundle: bundle, manifest: manifest, device: .current,
                                             options: DecoderEngine.Options(capacity: 2048))
        let generator = try await ChatSession.generator(engine: engine, bundle: bundle, manifest: manifest)
        func mib() -> Int64 { DecoderEngine.metalAllocatedBytes() >> 20 }
        print("GATE qwen35 retention: after load \(mib()) MiB, reload budget \(engine.reloadBudgetBytes.map(String.init) ?? "none")")
        let both = reference.pictures.map { ChatImage(ImageInput(contentsOf: URL(fileURLWithPath: $0))) }
        let one = [ChatImage(ImageInput(contentsOf: URL(fileURLWithPath: reference.pictures[0])))]
        let requests: [(String, [ChatMessage])] = [
            ("two pictures", [.system(reference.instructions), .user(reference.question, images: both)]),
            ("two pictures again", [.system(reference.instructions), .user(reference.question, images: both)]),
            ("one picture", [.system(reference.instructions), .user(reference.question, images: one)]),
            ("text only", [.user("Describe a red bodysuit with gold trim in one sentence.")]),
            ("two pictures", [.system(reference.instructions), .user(reference.question, images: both)]),
            ("one picture", [.system(reference.instructions), .user(reference.question, images: one)]),
            ("two pictures", [.system(reference.instructions), .user(reference.question, images: both)]),
        ]
        let clock = ContinuousClock()
        var levels: [Int64] = []
        for (i, (name, messages)) in requests.enumerated() {
            let t = clock.now
            var summary: GenerationSummary?
            for try await event in generator.generate(messages, parameters: GenerationParameters(maxTokens: 24, thinking: false)) {
                if case .finished(let s) = event { summary = s }
            }
            levels.append(mib())
            print("GATE qwen35 retention: request \(i + 1) (\(name), \(summary?.promptTokens ?? 0) prompt tokens, reused "
                  + "\(summary?.reusedTokens ?? 0)): \(String(format: "%.1f", (clock.now - t) / .seconds(1))) s, Metal "
                  + "\(levels.last!) MiB, reloads \(await engine.reloads.count)")
        }
        print("GATE qwen35 retention: growth after the first request \(levels.last! - levels[0]) MiB")
    }

    /// Logits sanity by call size (opt-in, AETHER_QWEN35_RETENTION=1): no canary, the reference's prompt text fed in
    /// calls of 1, 3, 32 and a shorter one, then single steps; NaN count and largest logit after each.
    @Test(.enabled(if: Qwen35Gate.retention)) func logitsByCallSize() async throws {
        let reference = try JSONDecoder().decode(
            Reference.self, from: Data(contentsOf: URL(fileURLWithPath: Self.env["AETHER_QWEN35_REFERENCE"]!)))
        let bundleURL = URL(fileURLWithPath: Self.env["AETHER_QWEN35_BUNDLE"]!)
        let engine = try await DecoderEngine(bundle: try Self.localBundle(bundleURL), manifest: try Manifest.load(fromBundle: bundleURL),
                                             device: .current, options: DecoderEngine.Options(capacity: 2048, canary: false))
        let text = reference.renderedTokenIDs.prefix(60).map(Int32.init).filter { $0 < 248_000 }
        var at = 0
        for n in [1, 3, 32, 12, 1, 1, 1] {
            let logits = try #require(try await engine.prefill(Array(text[at..<(at + n)]), logits: .last))
            at += n
            let row = logits.last
            let nans = row.filter { $0.isNaN }.count, finite = row.filter { $0.isFinite }
            print("GATE qwen35 logits: call of \(n) (position \(at)): NaN \(nans), max \(finite.max() ?? .nan), "
                  + "argmax \(row.indices.max { row[$0] < row[$1] }.map { $0 - row.startIndex } ?? -1)")
        }
    }
}
