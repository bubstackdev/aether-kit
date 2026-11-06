import Foundation
@_spi(Testing) import AetherHost
@testable import AetherText
import AetherTestSupport

/// A tiny vocabulary for the fake chat model: control tokens, a–z and space.
enum Vocab {
    static let pad: Int32 = 0, bos: Int32 = 1, user: Int32 = 2, assistant: Int32 = 3, endOfTurn: Int32 = 4, stop: Int32 = 5
    static let thinkOpen: Int32 = 6, thinkClose: Int32 = 7, toolOpen: Int32 = 8, toolClose: Int32 = 9, space: Int32 = 36
    static let specials = ["<pad>", "<s>", "<user>", "<asst>", "<eot>", "<stop>", "<think>", "</think>", "<tool_call>", "</tool_call>"]

    static func token(_ c: Character) -> Int32 {
        if let a = c.asciiValue, a >= 97, a <= 122 { return Int32(a - 97) + 10 }
        return space
    }

    static func string(_ id: Int32) -> String {
        if id < 10 { return specials[Int(id)] }
        if id < 36 { return String(UnicodeScalar(UInt8(id - 10 + 97))) }
        return " "
    }

    static func tokens(_ text: String) -> [Int32] { text.map(token) }
}

struct FakeDecoding: TokenDecoding {
    func decode(_ ids: [Int32]) -> String { ids.map(Vocab.string).joined() }
}

/// Renders `<s>` then `<role> content <eot>` per message, then `<asst>` (and `<think>` when thinking is on).
struct FakeRenderer: PromptRenderer {
    /// Tokens only the generation prompt carries (like Qwen's empty think block with thinking off).
    var generationSuffix: [Int32] = []

    func render(_ messages: [ChatMessage], options: RenderOptions) throws -> RenderedPrompt {
        var ids: [Int32] = [Vocab.bos]
        for m in messages {
            ids.append(m.role == .assistant ? Vocab.assistant : Vocab.user)
            ids += Vocab.tokens(m.content)
            ids.append(Vocab.endOfTurn)
        }
        if options.addGenerationPrompt { ids += [Vocab.assistant] + generationSuffix }
        if options.thinking == true { ids.append(Vocab.thinkOpen) }
        return RenderedPrompt(tokens: ids, startsInThinking: options.thinking == true)
    }
}

enum FakeChat {
    static let device = Device(platform: .macOS, architecture: "fake9", physicalMemory: 64 << 30)

    /// A predictor that, after the last `<asst>`, replies with `reply(prefix)` then `<stop>`.
    static func scripted(_ reply: @escaping @Sendable ([Int32]) -> [Int32]) -> @Sendable ([Int32]) -> Int32 {
        { context in
            guard let start = context.lastIndex(where: { $0 == Vocab.assistant || $0 == Vocab.thinkOpen }) else { return Vocab.stop }
            let script = reply(Array(context[..<(context.lastIndex(of: Vocab.assistant) ?? start)]))
            let n = context.count - 1 - start
            return n < script.count ? script[n] : Vocab.stop
        }
    }

    /// Replies with letters derived from the user's last message: a deterministic function of the conversation.
    static let echo: @Sendable ([Int32]) -> [Int32] = { prefix in
        let user = prefix.lastIndex(of: Vocab.user).map { Array(prefix[($0 + 1)...]).filter { $0 >= 10 && $0 < 36 } } ?? []
        let base = Int(prefix.count)
        return (0..<(3 + base % 4)).map { i in Int32(10 + (base + i * 7 + Int(user.first ?? 0)) % 26) }
    }

    static func generator(
        recurrent: Bool = false, reply: @escaping @Sendable ([Int32]) -> [Int32] = echo, profile: OutputProfile = .plain,
        log: CallLog? = nil, renderer: FakeRenderer = FakeRenderer(), gpu: GPUAdmissionOwner = .unadmitted,
        rowOverride: (@Sendable ([Int32]) -> [Float]?)? = nil
    ) async throws -> Generator {
        var config = FakeLM.Config(staticStep: recurrent, recurrent: recurrent, log: log, predict: scripted(reply))
        config.rowOverride = rowOverride
        let fake = FakeLM(config)
        let graph = fake.manifestGraph()
        let engine = try await DecoderEngine(
            step: fake, graph: graph, device: device, options: DecoderEngine.Options(memoryBudget: .bytes(1 << 30)))
        return Generator(engine: engine, decoder: FakeDecoding(), renderer: renderer, stopTokens: [Vocab.stop], profile: profile,
                         gpu: gpu)
    }

    static func collect(_ stream: AsyncThrowingStream<GenerationEvent, any Error>) async throws -> (text: String, thinking: String, tools: [String], summary: GenerationSummary?) {
        var text = "", thinking = "", tools: [String] = []
        var summary: GenerationSummary?
        for try await event in stream {
            switch event {
            case .text(let s): text += s
            case .thinking(let s): thinking += s
            case .toolCall(let s): tools.append(s)
            case .finished(let s): summary = s
            }
        }
        return (text, thinking, tools, summary)
    }
}
