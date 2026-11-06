import Foundation
import Testing
@testable import AetherText

/// ChatSession gates on real bundles (AETHER_GATES=1): two turns with prefix reuse, on an attention model and a
/// hybrid. The second reply must equal a fresh session's reply to the same history (no reuse), token for token.
@Suite("TextGateTests", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct TextGateTests {
    static func session(_ id: String) async throws -> ChatSession {
        try await ChatSession.load(
            bundle: try Lab.bundle(id), manifest: try Lab.manifest(id), device: Lab.mac,
            engine: DecoderEngine.Options(capacity: 4096),
            configuration: .init(parameters: GenerationParameters(maxTokens: 96, thinking: false)))
    }

    @Test(arguments: ["qwen3-0.6b", "qwen3.5-0.8b"])
    func twoTurnChatReusesThePrefix(_ id: String) async throws {
        let chat = try await Self.session(id)
        let first = try await collect(await chat.respond(to: "Name three rivers in Japan. Answer briefly."))
        let second = try await collect(await chat.respond(to: "Which of them is the longest?"))
        let one = try #require(first.summary), two = try #require(second.summary)

        let fresh = try await Self.session(id)
        let replay = try await collect(fresh.generator.generate(
            Array(await chat.history.prefix(3)), parameters: GenerationParameters(maxTokens: 96, thinking: false)))
        print("GATE chat \(id): turn 1 \(one.promptTokens) prompt tokens, \(one.tokens.count) generated (\(one.decodeMode)); "
              + "turn 2 \(two.promptTokens) prompt tokens, \(two.reusedTokens) reused, \(two.tokens.count) generated; "
              + "replay without reuse \(replay.summary?.tokens == two.tokens ? "identical" : "DIFFERS")\n"
              + "  1: \(first.text.prefix(160).debugDescription)\n  2: \(second.text.prefix(160).debugDescription)")
        // Turn 2 shares turn 1's history; the generation prompt (`<think>\n\n</think>\n\n` with thinking off) is
        // not part of it, so reuse ends there on both model classes (by rewind, or by the history-end snapshot).
        #expect(two.reusedTokens > 0)
        #expect(replay.summary?.reusedTokens == 0)
        #expect(replay.summary?.tokens == two.tokens)
        #expect(!first.text.isEmpty && !second.text.isEmpty)
    }

    func collect(_ stream: AsyncThrowingStream<GenerationEvent, any Error>) async throws -> (text: String, summary: GenerationSummary?) {
        var text = ""
        var summary: GenerationSummary?
        for try await event in stream {
            if case .text(let s) = event { text += s }
            if case .finished(let s) = event { summary = s }
        }
        return (text, summary)
    }
}
