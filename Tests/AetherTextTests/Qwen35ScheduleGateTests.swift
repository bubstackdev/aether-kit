import AetherCore
import AetherHost
import AetherRuntime
import Foundation
import Testing
@testable import AetherText

/// The Qwen3.5 / Qwen3.6 prefill-schedule gate, opt-in: one chat prompt (a system prompt, tools and a user turn, from
/// `AETHER_QWEN35_CHAT`) fed in different call schedules from an empty state must reach the same next-token logits.
/// A hybrid decoder can't be rewound, so a wrong call anywhere in the prompt changes everything after it.
///
///     AETHER_GATES=1 AETHER_QWEN35_BUNDLE=<export> AETHER_QWEN35_CHAT=<chat.json> \
///         scripts/gpu-lease -- swift test --filter Qwen35ScheduleGateTests
///
/// `chat.json`: `{"system": "...", "user": "...", "tools": [{"name", "description", "parameters"}]}`.
/// Prints `GATE qwen35-schedule …` lines: each schedule's argmax, top five and cosine to one-token calls, and the
/// greedy text after it; `GATE qwen35-chained …`: one whole-prompt prefill call (its chunks drained one by one since
/// this fix; chained without a sync, 2 of 8 came out different) repeated against per-chunk calls, then the pipelined
/// decode against host steps and the Generator's first token. `AETHER_QWEN35_ROUNDS` (default 8) repeats.
enum Qwen35ScheduleGate {
    static let env = ProcessInfo.processInfo.environment
    static let enabled = env["AETHER_GATES"] == "1" && env["AETHER_QWEN35_BUNDLE"] != nil && env["AETHER_QWEN35_CHAT"] != nil
}

@Suite("Qwen35ScheduleGateTests", .serialized, .enabled(if: Qwen35ScheduleGate.enabled))
struct Qwen35ScheduleGateTests {
    struct Chat: Decodable {
        struct Tool: Decodable {
            let name: String
            let description: String
            let parameters: JSONValue
        }
        let system: String
        let user: String
        let tools: [Tool]
        let greedyTokens: Int?
    }

    static func cosine(_ a: ArraySlice<Float>, _ b: ArraySlice<Float>) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for (x, y) in zip(a, b) { dot += Double(x) * Double(y); na += Double(x) * Double(x); nb += Double(y) * Double(y) }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
    }

    static func top(_ row: ArraySlice<Float>, _ k: Int) -> [Int] {
        Array(row.indices.sorted { row[$0] > row[$1] }.prefix(k).map { $0 - row.startIndex })
    }

    /// Every call size against one-token calls from the same state: the prompt's first 64 tokens fed, then its next
    /// S tokens in one call or in S calls of one.
    @Test func everyCallSizeMatchesOneTokenCalls() async throws {
        let env = Qwen35ScheduleGate.env
        let chat = try JSONDecoder().decode(Chat.self, from: Data(contentsOf: URL(fileURLWithPath: env["AETHER_QWEN35_CHAT"]!)))
        let url = URL(fileURLWithPath: env["AETHER_QWEN35_BUNDLE"]!)
        let bundle = try Qwen35GateTests.localBundle(url)
        let manifest = try Manifest.load(fromBundle: url)
        let engine = try await DecoderEngine(bundle: bundle, manifest: manifest, device: .current, options: .init(canary: false))
        let generator = try await ChatSession.generator(engine: engine, bundle: bundle, manifest: manifest)
        let tools = chat.tools.map { ToolDefinition(name: $0.name, description: $0.description, parameters: $0.parameters) }
        let prompt = try generator.renderer.render([.system(chat.system), .user(chat.user)],
                                                   options: RenderOptions(thinking: false, tools: tools)).tokens
        let start = Int(env["AETHER_QWEN35_SWEEP_AT"] ?? "") ?? 64
        try await engine.reset()
        _ = try await engine.prefill(Array(prompt[..<start]), logits: .none)
        let base = try await engine.snapshot()
        var bad: [Int] = []
        for s in 1...32 {
            let chunk = Array(prompt[start..<(start + s)])
            try await engine.restore(base)
            let one = try #require(try await engine.prefill(chunk, logits: .last))
            try await engine.restore(base)
            var steps: Logits?
            for t in chunk { steps = try await engine.step(t) }
            let cos = Self.cosine(one.last, steps!.last)
            let same = try Logits.argmax(one.last) == Logits.argmax(steps!.last)
            if cos < 0.999 || !same { bad.append(s) }
            print("GATE qwen35-schedule call of \(s) at \(start): cos \(String(format: "%.6f", cos)), argmax \(same ? "same" : "differs")")
        }
        print("GATE qwen35-schedule call sizes that differ: \(bad)")
        #expect(bad.isEmpty)
    }

    /// One prefill call of the whole prompt (its chunks chained on the GPU with no host sync between them), from an
    /// empty state, again and again: the logits must match bit for bit, and match calls fed one at a time.
    @Test func chainedPrefillIsDeterministic() async throws {
        let env = Qwen35ScheduleGate.env
        let chat = try JSONDecoder().decode(Chat.self, from: Data(contentsOf: URL(fileURLWithPath: env["AETHER_QWEN35_CHAT"]!)))
        let url = URL(fileURLWithPath: env["AETHER_QWEN35_BUNDLE"]!)
        let bundle = try Qwen35GateTests.localBundle(url)
        let manifest = try Manifest.load(fromBundle: url)
        let engine = try await DecoderEngine(bundle: bundle, manifest: manifest, device: .current)
        let generator = try await ChatSession.generator(engine: engine, bundle: bundle, manifest: manifest)
        let tools = chat.tools.map { ToolDefinition(name: $0.name, description: $0.description, parameters: $0.parameters) }
        let prompt = try generator.renderer.render([.system(chat.system), .user(chat.user)],
                                                   options: RenderOptions(thinking: false, tools: tools)).tokens
        // The reference: one engine call per 32-token chunk (each call synchronizes before it returns).
        try await engine.reset()
        var reference: Logits?
        for at in stride(from: 0, to: prompt.count, by: 32) {
            reference = try await engine.prefill(Array(prompt[at..<min(at + 32, prompt.count)]), logits: .last)
        }
        let ref = reference!.last
        var worst: Float = 0, flips = 0
        let rounds = Int(env["AETHER_QWEN35_ROUNDS"] ?? "") ?? 8
        for round in 1...rounds {
            try await engine.reset()
            let chained = try #require(try await engine.prefill(prompt, logits: .last)).last
            let diff = zip(chained, ref).map { abs($0 - $1) }.max() ?? 0
            worst = max(worst, diff)
            let t3 = Self.top(chained, 3)
            if t3[0] != Self.top(ref, 1)[0] { flips += 1 }
            print("GATE qwen35-chained round \(round): max |diff| \(diff), top3 \(t3) \(t3.map { String(format: "%.2f", chained[chained.startIndex + $0]) })")
        }
        print("GATE qwen35-chained worst |diff| \(worst), argmax flips \(flips)/\(rounds)")
        // Decoding after it: the engine's pipelined greedy decode against one host step at a time, and the Generator's
        // reply (its history-then-generation-prompt prefill) against the reference's first token.
        let steps = 48
        try await engine.reset()
        var row = try #require(try await engine.prefill(prompt, logits: .last)).last
        var host: [Int32] = []
        for _ in 0..<steps {
            host.append(try Logits.argmax(row))
            row = try await engine.step(host.last!).last
        }
        var decodeMismatches = 0, generatorMismatches = 0
        for round in 1...max(1, rounds / 2) {
            try await engine.reset()
            let first = try Logits.argmax(try #require(try await engine.prefill(prompt, logits: .last)).last)
            try await engine.beginGreedy(first: first, limit: steps - 1)
            var piped = [first]
            while let t = try await engine.nextGreedy() { piped.append(t) }
            try await engine.endGreedy()
            if piped != host { decodeMismatches += 1 }
            try await engine.reset()
            var summary: GenerationSummary?
            for try await event in generator.generate([.system(chat.system), .user(chat.user)],
                                                       parameters: GenerationParameters(maxTokens: 4, thinking: false, tools: tools)) {
                if case .finished(let s) = event { summary = s }
            }
            if summary?.tokens.first != host.first { generatorMismatches += 1 }
            print("GATE qwen35-chained decode round \(round): pipelined \(piped == host ? "equals" : "differs from") host steps; "
                  + "Generator's first token \(summary?.tokens.first.map(String.init) ?? "none") (reference \(host[0]))")
        }
        print("GATE qwen35-chained host reply: \(generator.decoder.decode(host).debugDescription)")
        #expect(worst < 0.5 && flips == 0)
        #expect(decodeMismatches == 0 && generatorMismatches == 0)
    }

    @Test func scheduleInvariance() async throws {
        let env = Qwen35ScheduleGate.env
        let chat = try JSONDecoder().decode(Chat.self, from: Data(contentsOf: URL(fileURLWithPath: env["AETHER_QWEN35_CHAT"]!)))
        let url = URL(fileURLWithPath: env["AETHER_QWEN35_BUNDLE"]!)
        let bundle = try Qwen35GateTests.localBundle(url)
        let manifest = try Manifest.load(fromBundle: url)
        let engine = try await DecoderEngine(bundle: bundle, manifest: manifest, device: .current)
        let generator = try await ChatSession.generator(engine: engine, bundle: bundle, manifest: manifest)
        let tools = chat.tools.map { ToolDefinition(name: $0.name, description: $0.description, parameters: $0.parameters) }
        let messages: [ChatMessage] = [.system(chat.system), .user(chat.user)]
        let options = RenderOptions(thinking: false, tools: tools)
        let prompt = try generator.renderer.render(messages, options: options).tokens
        let history = try generator.renderer.render(
            messages, options: RenderOptions(thinking: false, addGenerationPrompt: false, tools: tools)).tokens
        let boundary = prompt.starts(with: history) ? history.count : prompt.count
        let greedy = chat.greedyTokens ?? 24
        print("GATE qwen35-schedule prompt \(prompt.count) tokens, history \(boundary), capacity \(await engine.capacity)")

        /// Feeds `prompt` from an empty state in calls of `sizes` (the rest in calls of the last size).
        func feed(_ sizes: [Int]) async throws -> Logits {
            try await engine.reset()
            var at = 0, i = 0, logits: Logits?
            while at < prompt.count {
                let n = min(sizes[min(i, sizes.count - 1)], prompt.count - at)
                logits = try await engine.prefill(Array(prompt[at..<(at + n)]), logits: .last)
                at += n
                i += 1
            }
            return logits!
        }
        func decode(_ first: Logits) async throws -> String {
            var row = first.last, out: [Int32] = []
            for _ in 0..<greedy {
                let t = try Logits.argmax(row)
                out.append(t)
                if generator.stopTokens.contains(t) { break }
                row = try await engine.step(t).last
            }
            return generator.decoder.decode(out)
        }

        let reference = try await feed([1])
        let referenceText = try await decode(reference)
        var schedules: [(String, [Int])] = [
            ("calls of 32", [32]), ("calls of 16", [16]), ("calls of 8", [8]), ("calls of 3", [3]),
            ("the Generator's split", Array(repeating: 32, count: boundary / 32) + [boundary % 32].filter { $0 > 0 } + [prompt.count - boundary]),
        ]
        for width in [31, 30, 29, 4, 2] { schedules.append(("calls of \(width)", [width])) }
        print("GATE qwen35-schedule calls of 1: top5 \(Self.top(reference.last, 5)), text \(referenceText.debugDescription)")
        var worst = 1.0
        for (name, sizes) in schedules {
            let logits = try await feed(sizes)
            let cos = Self.cosine(logits.last, reference.last)
            worst = min(worst, cos)
            let text = try await decode(logits)
            print("GATE qwen35-schedule \(name): cos \(String(format: "%.6f", cos)), top5 \(Self.top(logits.last, 5)), text \(text.debugDescription)")
        }
        // The Generator itself (prefix reuse, its boundary snapshot, the pipelined greedy decode), from an empty state.
        try await engine.reset()
        var text = ""
        for try await event in generator.generate(messages, parameters: GenerationParameters(maxTokens: greedy, thinking: false, tools: tools)) {
            switch event {
            case .text(let delta): text += delta
            case .toolCall(let body): text += "<tool_call>\(body)</tool_call>"
            default: break
            }
        }
        print("GATE qwen35-schedule Generator: \(text.debugDescription)")
        // After the state grew past its first allocation (a long request), the same prompt again.
        try await engine.reset()
        let long = Array(repeating: prompt, count: max(1, 2600 / prompt.count + 1)).flatMap { $0 }.prefix(2600)
        _ = try await engine.prefill(Array(long), logits: .none)
        let grownCapacity = await engine.capacity
        let grown = try await feed([32])
        print("GATE qwen35-schedule after growth to \(grownCapacity): cos \(String(format: "%.6f", Self.cosine(grown.last, reference.last))), "
              + "top5 \(Self.top(grown.last, 5)), text \(try await decode(grown).debugDescription)")
        // After a request with a picture (AETHER_QWEN35_PICTURE), the same text prompt from an empty state again.
        if let picture = env["AETHER_QWEN35_PICTURE"] {
            let image = ChatImage(ImageInput(contentsOf: URL(fileURLWithPath: picture)))
            for try await _ in generator.generate([.user("Describe the footwear.", images: [image])],
                                                   parameters: GenerationParameters(maxTokens: 8, thinking: false)) {}
            let afterPicture = try await feed([32])
            print("GATE qwen35-schedule after a picture: cos \(String(format: "%.6f", Self.cosine(afterPicture.last, reference.last))), "
                  + "top5 \(Self.top(afterPicture.last, 5)), text \(try await decode(afterPicture).debugDescription)")
        }
        #expect(worst > 0.99, "a schedule's logits differ from one-token calls (worst cosine \(worst))")
    }
}
