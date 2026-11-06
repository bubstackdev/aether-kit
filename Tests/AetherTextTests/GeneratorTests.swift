import Foundation
import Synchronization
import Testing
@testable import AetherText
import AetherTestSupport

@Suite("Generator and ChatSession (fake)")
struct GeneratorTests {
    static func chat(recurrent: Bool) async throws -> ChatSession {
        ChatSession(generator: try await FakeChat.generator(recurrent: recurrent),
                    configuration: .init(parameters: GenerationParameters(maxTokens: 32)))
    }

    /// Two turns with prefix reuse produce exactly what a fresh model fed the whole history produces.
    @Test(arguments: [false, true])
    func secondTurnReusesThePrefix(recurrent: Bool) async throws {
        let session = try await Self.chat(recurrent: recurrent)
        let first = try await FakeChat.collect(await session.respond(to: "hello there"))
        let second = try await FakeChat.collect(await session.respond(to: "and now"))
        let one = try #require(first.summary), two = try #require(second.summary)
        #expect(one.reusedTokens == 0)
        #expect(one.stopReason == .stopToken(Vocab.stop))
        // Everything the engine held (turn 1's prompt and reply) is reused; only the new turn is fed.
        #expect(two.reusedTokens == one.promptTokens + one.tokens.count - 1)
        #expect(await session.history.count == 4)

        let fresh = try await Self.chat(recurrent: recurrent)
        let history = await session.history
        let replay = try await FakeChat.collect(fresh.generator.generate(Array(history.prefix(3))))
        #expect(replay.text == second.text && replay.summary?.reusedTokens == 0)
        #expect(!second.text.isEmpty)
    }

    /// When the generation prompt is not part of the next turn's history, a hybrid reuses the snapshot taken at the
    /// end of the rendered history (an attention model rewinds to the same point).
    @Test(arguments: [false, true])
    func reuseStopsAtTheHistoryBoundary(recurrent: Bool) async throws {
        let renderer = FakeRenderer(generationSuffix: [Vocab.thinkOpen, Vocab.thinkClose])
        let g = try await FakeChat.generator(recurrent: recurrent, renderer: renderer)
        let chat = ChatSession(generator: g, configuration: .init(parameters: GenerationParameters(maxTokens: 32)))
        let first = try await FakeChat.collect(await chat.respond(to: "hello"))
        let second = try await FakeChat.collect(await chat.respond(to: "more"))
        let history = try renderer.render([.user("hello")], options: RenderOptions(addGenerationPrompt: false)).tokens
        #expect(first.summary?.promptTokens == history.count + 3)
        // Attention: the common prefix runs one token past the history (the next <asst>). Hybrid: the history-end
        // snapshot (prompt 1's own end diverges at its generation suffix).
        #expect(second.summary?.reusedTokens == history.count + (recurrent ? 0 : 1))
        let fresh = try await FakeChat.generator(recurrent: recurrent, renderer: renderer)
        let replay = try await FakeChat.collect(fresh.generate(Array(await chat.history.prefix(3))))
        #expect(replay.text == second.text)
    }

    /// Regenerating the same prompt on a hybrid restores the end-of-prompt snapshot (with its logits).
    @Test func hybridRegenerateRestoresTheSnapshot() async throws {
        let generator = try await FakeChat.generator(recurrent: true)
        let prompt = try FakeRenderer().render([.user("abc")], options: RenderOptions())
        let a = try await FakeChat.collect(generator.generate(prompt: prompt))
        let b = try await FakeChat.collect(generator.generate(prompt: prompt))
        #expect(a.text == b.text)
        #expect(b.summary?.reusedTokens == prompt.tokens.count)
    }

    @Test func attentionModelRewindsToTheCommonPrefix() async throws {
        let generator = try await FakeChat.generator()
        let a = try FakeRenderer().render([.user("abcdef")], options: RenderOptions())
        let b = try FakeRenderer().render([.user("abcxyz")], options: RenderOptions())
        _ = try await FakeChat.collect(generator.generate(prompt: a))
        let r = try await FakeChat.collect(generator.generate(prompt: b))
        #expect(r.summary?.reusedTokens == 5)  // <s> <user> a b c
        let fresh = try await FakeChat.collect(try await FakeChat.generator().generate(prompt: b))
        #expect(r.text == fresh.text)
    }

    @Test func stopsOnTokensSequencesAndLength() async throws {
        let long: @Sendable ([Int32]) -> [Int32] = { _ in Vocab.tokens("abcdefghij") }
        let g = try await FakeChat.generator(reply: long)
        let prompt = try FakeRenderer().render([.user("q")], options: RenderOptions())
        let full = try await FakeChat.collect(g.generate(prompt: prompt))
        #expect(full.text == "abcdefghij" && full.summary?.stopReason == .stopToken(Vocab.stop))
        let seq = try await FakeChat.collect(g.generate(prompt: prompt, parameters: GenerationParameters(stopSequences: ["def"])))
        #expect(seq.text == "abc" && seq.summary?.stopReason == .stopSequence("def"))
        let short = try await FakeChat.collect(g.generate(prompt: prompt, parameters: GenerationParameters(maxTokens: 4)))
        #expect(short.text == "abcd" && short.summary?.stopReason == .maxTokens)
    }

    @Test func thinkingAndToolCallTagsAreParsed() async throws {
        let script: @Sendable ([Int32]) -> [Int32] = { _ in
            Vocab.tokens("why") + [Vocab.thinkClose] + Vocab.tokens("ok ") + [Vocab.toolOpen] + Vocab.tokens("call x") + [Vocab.toolClose]
                + Vocab.tokens(" done")
        }
        let profile = OutputProfile(tags: [
            .init(open: "<think>", close: "</think>", kind: .thinking), .init(open: "<tool_call>", close: "</tool_call>", kind: .toolCall),
        ])
        let g = try await FakeChat.generator(reply: script, profile: profile)
        let prompt = try FakeRenderer().render([.user("q")], options: RenderOptions(thinking: true))
        #expect(prompt.startsInThinking)
        let r = try await FakeChat.collect(g.generate(prompt: prompt))
        #expect(r.thinking == "why")
        #expect(r.tools == ["call x"])
        #expect(r.text == "ok  done")
    }

    @Test func samplingIsSeededAndProcessorsApply() async throws {
        struct Ban: LogitsProcessor {
            let token: Int32
            var all: TokenSet {
                var set = TokenSet(range: 0..<64, vocabularySize: 64)
                set.remove(Int(token))
                return set
            }
            func begin(prompt: [Int32]) -> TokenConstraint { .allowed(all) }
            func accept(_ token: Int32) -> TokenConstraint { .allowed(all) }
        }
        let g = try await FakeChat.generator()
        let prompt = try FakeRenderer().render([.user("hello")], options: RenderOptions())
        let params = GenerationParameters(maxTokens: 20, sampling: SamplingParameters(temperature: 3, seed: 11), logitsProcessors: [Ban(token: Vocab.stop)])
        let a = try await FakeChat.collect(g.generate(prompt: prompt, parameters: params))
        let b = try await FakeChat.collect(g.generate(prompt: prompt, parameters: params))
        #expect(a.summary?.tokens == b.summary?.tokens)
        #expect(a.summary?.tokens.count == 20 && a.summary?.tokens.contains(Vocab.stop) == false)
        #expect(a.summary?.decodeMode.contains("CPU sampler") == true)
        var other = params
        other.sampling.seed = 12
        let c = try await FakeChat.collect(g.generate(prompt: prompt, parameters: other))
        #expect(c.summary?.tokens != a.summary?.tokens)
    }

    @Test func cancellationLeavesTheEngineUsable() async throws {
        let long: @Sendable ([Int32]) -> [Int32] = { _ in Array(repeating: 12, count: 200) }
        let g = try await FakeChat.generator(reply: long)
        let prompt = try FakeRenderer().render([.user("q")], options: RenderOptions())
        for try await event in g.generate(prompt: prompt, parameters: GenerationParameters(maxTokens: 300)) {
            if case .text = event { break }
        }
        let r = try await FakeChat.collect(g.generate(prompt: prompt, parameters: GenerationParameters(maxTokens: 8)))
        #expect(r.text == String(repeating: "c", count: 8))
        #expect(await g.mirroredTokens.count == prompt.tokens.count + 7)
        #expect(await g.engine.position == prompt.tokens.count + 7)
    }
}

@Suite("Text registry")
struct TextRegistryTests {
    @Test func linksTheDecoderDriver() throws {
        #expect(Registry.text.supports(.decoderLM))
        #expect(try Registry.text.driver(for: .decoderLM) == DecoderLMDriver.self)
        let ios = try ModelRef.resolve("qwen3-0.6b", for: Device(platform: .iOS, architecture: "h18p"))
        #expect(throws: AetherError.self) { try Registry.text.require(ios) }  // static-chunked-lm@1 is not in M2
    }
}

@Suite("Output parsing")
struct OutputParsingTests {
    static let profile = OutputProfile(tags: [
        .init(open: "<think>", close: "</think>", kind: .thinking), .init(open: "<tool_call>", close: "</tool_call>", kind: .toolCall),
    ])

    @Test func tagsSplitAcrossChunksAreHeldBack() {
        var p = TagParser(profile: Self.profile)
        var out: [TagParser.Segment] = []
        for chunk in ["Hi <th", "ink>ra", "in</thi", "nk> sun <tool_", "call>{\"a\":1}</tool_call", ">!", " <"] {
            out += p.feed(chunk)
        }
        out += p.finish()
        let text = out.compactMap { if case .text(let s) = $0 { s } else { nil } }.joined()
        let thinking = out.compactMap { if case .thinking(let s) = $0 { s } else { nil } }.joined()
        #expect(text == "Hi  sun ! <")
        #expect(thinking == "rain")
        #expect(out.contains(.toolCall("{\"a\":1}")))
        #expect(!out.contains(.text("<th")))
    }

    @Test func manifestDrivesTheProfile() throws {
        let manifest = try Manifest.decode(Data(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appending(path: "AetherCoreTests/Fixtures/zoo/qwen3-0.6b/aether.json")))
        let profile = OutputProfile(manifest: manifest.tokenizer)
        #expect(profile.tags.map(\.open) == ["<think>", "<tool_call>"])
        #expect(OutputProfile(manifest: nil).tags.isEmpty)
    }

    @Test func stopSequencesHoldBackPartials() {
        var s = StopMatcher(["END"])
        #expect(s.feed("abcE").0 == "abc")
        #expect(s.feed("N").0 == "")
        let (text, hit) = s.feed("Dxyz")
        #expect(text == "" && hit == "END")
        var t = StopMatcher(["END"])
        _ = t.feed("aE")
        #expect(t.feed("x").0 == "Ex")
    }

    /// A byte-level decoder: multi-byte characters span tokens, and a partial sequence decodes to U+FFFD.
    struct Bytes: TokenDecoding {
        final class Counter: Sendable {
            let tally = Mutex(0)
            var decoded: Int { tally.withLock { $0 } }
        }
        let counter = Counter()
        func decode(_ ids: [Int32]) -> String {
            counter.tally.withLock { $0 += ids.count }
            return String(decoding: ids.map { UInt8($0) }, as: UTF8.self)
        }
    }

    @Test func detokenizerIsIncrementalAndUTF8Safe() {
        let text = "naïve café — ok ✓ " + String(repeating: "x", count: 400)
        let ids = Array(text.utf8).map(Int32.init)
        let decoder = Bytes()
        var d = IncrementalDetokenizer()
        var out = ""
        for id in ids {
            let delta = d.append(id, decoder)
            #expect(!delta.contains("\u{FFFD}"))
            out += delta
        }
        out += d.finish(decoder)
        #expect(out == text)
        // Linear, not quadratic: each step decodes a window of a few tokens.
        #expect(decoder.counter.decoded < 6 * ids.count)
    }
}

