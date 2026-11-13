import AetherCore
@_spi(Testing) import AetherHost
import AetherRuntime
import Foundation
import Testing
@testable import AetherText
import AetherTestSupport

/// The Generator's prefix reuse across chat turns, side requests with pictures and conversations (fakes). Every reuse
/// is checked against a fresh engine fed the same tokens from an empty state: the next row's logits must be equal bit
/// for bit, and the reply equal to a fresh generator's.
@Suite("Generator prefix reuse (fake)")
struct PrefixReuseTests {
    /// Renders like Qwen3.6's template on the fake vocabulary: `<s>`, then `<user>` content `<eot>` for system and user
    /// turns (tools listed in a system turn of their own), `<asst>` content `<eot>` for assistant turns, and the
    /// generation prompt `<asst>`, with the empty block `<think></think>` when thinking is off. A past assistant turn
    /// shows that block only after the last user turn, or with `preserveThinking` (`preserve_thinking`).
    struct QwenLikeRenderer: PromptRenderer {
        func render(_ messages: [ChatMessage], options: RenderOptions) throws -> RenderedPrompt {
            var ids: [Int32] = [Vocab.bos]
            if !options.tools.isEmpty { ids += [Vocab.user] + Vocab.tokens("tools " + options.tools.map(\.name).joined(separator: " ")) + [Vocab.endOfTurn] }
            let lastUser = messages.lastIndex { $0.role == .user } ?? -1
            for (i, m) in messages.enumerated() {
                if m.role == .assistant {
                    ids.append(Vocab.assistant)
                    if options.preserveThinking || i > lastUser { ids += [Vocab.thinkOpen, Vocab.thinkClose] }
                } else {
                    ids.append(Vocab.user)
                }
                ids += Vocab.tokens(m.content)
                ids.append(Vocab.endOfTurn)
            }
            if options.addGenerationPrompt {
                ids.append(Vocab.assistant)
                if options.thinking == false { ids += [Vocab.thinkOpen, Vocab.thinkClose] }
            }
            return RenderedPrompt(tokens: ids)
        }
    }

    static func reply(_ generator: Generator, _ messages: [ChatMessage], thinking: Bool? = false, tokens: Int = 8)
        async throws -> (text: String, summary: GenerationSummary)
    {
        let r = try await FakeChat.collect(generator.generate(messages, parameters: GenerationParameters(maxTokens: tokens, thinking: thinking)))
        return (r.text, try #require(r.summary))
    }

    /// The next row after `probe` from `generator`'s engine as it stands, against a fresh engine (`fresh`) fed the
    /// tokens the generator says it holds, from an empty state, then `probe`: equal bit for bit.
    static func expectStateIsExact(_ generator: Generator, fresh: Generator, probe: Int32, sourceLocation: SourceLocation = #_sourceLocation) async throws {
        let held = await generator.mirroredTokens
        #expect(await generator.engine.position == held.count, sourceLocation: sourceLocation)
        let reused = try #require(try await generator.engine.prefill([probe], logits: .last))
        try await fresh.engine.reset()
        let reference = try #require(try await fresh.engine.prefill(held + [probe], logits: .last))
        #expect(Array(reused.last) == Array(reference.last), "the reused state's logits differ from a fresh prefill's", sourceLocation: sourceLocation)
    }

    // MARK: thinking off: past turns render as they were fed

    /// With thinking off a past turn renders with the empty block its generation prompt fed, so the next turn extends
    /// the live state: everything the engine holds is reused, and the state is exact.
    @Test(arguments: [true, false])
    func theNextTurnExtendsTheLiveState(recurrent: Bool) async throws {
        let renderer = QwenLikeRenderer()
        let generator = try await FakeChat.generator(recurrent: recurrent, renderer: FakeRenderer())
        let g = Generator(engine: generator.engine, decoder: FakeDecoding(), renderer: renderer, stopTokens: [Vocab.stop], profile: .plain, gpu: .unadmitted)
        var history: [ChatMessage] = [.system("you help"), .user("hello there")]
        let one = try await Self.reply(g, history)
        history += [.assistant(one.text), .user("and now")]
        let two = try await Self.reply(g, history)
        // Everything turn 1 left in the engine (its prompt and reply but the stop token).
        #expect(two.summary.reusedTokens == one.summary.promptTokens + one.summary.tokens.count - 1)
        // The template's default (the block dropped from turns before the last user turn) would not have extended it.
        let dropped = try renderer.render(history, options: RenderOptions(thinking: false)).tokens
        #expect(!dropped.starts(with: try renderer.render(Array(history.prefix(2)), options: RenderOptions(thinking: false)).tokens + one.summary.tokens.dropLast()))
        let fresh = try await FakeChat.generator(recurrent: recurrent)
        let replay = Generator(engine: fresh.engine, decoder: FakeDecoding(), renderer: renderer, stopTokens: [Vocab.stop], profile: .plain, gpu: .unadmitted)
        #expect(try await Self.reply(replay, history).text == two.text)
        try await Self.expectStateIsExact(g, fresh: replay, probe: Vocab.token("x"))
    }

    /// Thinking on or the template's default: the prompt is rendered as before (no preserved blocks), and the reply
    /// still equals a fresh generator's.
    @Test func withThinkingOnTheTemplatesDefaultStays() async throws {
        let renderer = QwenLikeRenderer()
        let base = try await FakeChat.generator(recurrent: true)
        let g = Generator(engine: base.engine, decoder: FakeDecoding(), renderer: renderer, stopTokens: [Vocab.stop], profile: .plain, gpu: .unadmitted)
        var history: [ChatMessage] = [.system("you help"), .user("hello there")]
        let one = try await Self.reply(g, history, thinking: nil)
        history += [.assistant(one.text), .user("and now")]
        let two = try await Self.reply(g, history, thinking: nil)
        #expect(two.summary.promptTokens == (try renderer.render(history, options: RenderOptions())).tokens.count)
        let fresh = try await FakeChat.generator(recurrent: true)
        let replay = Generator(engine: fresh.engine, decoder: FakeDecoding(), renderer: renderer, stopTokens: [Vocab.stop], profile: .plain, gpu: .unadmitted)
        #expect(try await Self.reply(replay, history, thinking: nil).text == two.text)
        try await Self.expectStateIsExact(g, fresh: replay, probe: Vocab.token("x"))
    }

    // MARK: the system turn's snapshot

    /// A new conversation with the same system turn starts from the snapshot at its end, on a hybrid; one with another
    /// system turn starts over and takes its place. At most `maxSnapshots` are held.
    @Test func aNewConversationStartsFromTheSystemTurn() async throws {
        let generator = try await FakeChat.generator(recurrent: true, renderer: FakeRenderer())
        let fresh = try await FakeChat.generator(recurrent: true, renderer: FakeRenderer())
        let system = ChatMessage.system("you help with looks")
        // <s> <user> system <eot> <user>: through the opening of the first user turn.
        let systemTurn = 1 + 1 + Vocab.tokens("you help with looks").count + 1 + 1
        let a = try await Self.reply(generator, [system, .user("hello there")])
        #expect(a.summary.reusedTokens == 0)
        #expect(await generator.heldSnapshots.systemTokens == systemTurn)
        #expect(await generator.heldSnapshots.count <= Generator.maxSnapshots)
        let b = try await Self.reply(generator, [system, .user("something else")])
        #expect(b.summary.reusedTokens == systemTurn)
        #expect(try await Self.reply(fresh, [system, .user("something else")]).text == b.text)
        try await Self.expectStateIsExact(generator, fresh: fresh, probe: Vocab.token("q"))
        // The next turn of conversation b continues it as before.
        let c = try await Self.reply(generator, [system, .user("something else"), .assistant(b.text), .user("more")])
        #expect(c.summary.reusedTokens > systemTurn)
        #expect(await generator.heldSnapshots.count <= Generator.maxSnapshots)
        // Another system turn replaces it.
        let other = try await Self.reply(generator, [.system("you write"), .user("hello there")])
        #expect(other.summary.reusedTokens == 0)
        #expect(await generator.heldSnapshots.systemTokens == 1 + 1 + Vocab.tokens("you write").count + 1 + 1)
        #expect(await generator.heldSnapshots.count <= Generator.maxSnapshots)
        let back = try await Self.reply(generator, [system, .user("hello there")])
        #expect(back.summary.reusedTokens == 0)
        #expect(back.text == a.text)
        // Without a system turn there is none.
        let bare = try await FakeChat.generator(recurrent: true)
        _ = try await Self.reply(bare, [.user("hello there")])
        #expect(await bare.heldSnapshots.systemTokens == nil)
    }

    // MARK: pictures

    static func image(_ seed: Int) -> ChatImage { ChatImage(ImageInput(FakeVLM.picture(width: 8, height: 8, seed: seed))) }

    static func vlmReply(_ generator: Generator, _ messages: [ChatMessage], tokens: Int = 6) async throws -> GenerationSummary {
        var summary: GenerationSummary?
        for try await event in generator.generate(messages, parameters: GenerationParameters(maxTokens: tokens)) {
            if case .finished(let s) = event { summary = s }
        }
        return try #require(summary)
    }

    /// LabChat's look_at_picture: a one-picture side request between two text turns of a chat, on a hybrid. The text
    /// snapshots outlive it, and the next text turn restores one with the picture unbound; a new conversation starts
    /// from the system turn's. Each state equals a fresh engine's fed the same tokens.
    @Test func aTextTurnAfterAPictureRestoresTheTextSnapshots() async throws {
        let generator = try await FakeVLM.generator(recurrent: true)
        let fresh = try await FakeVLM.generator(recurrent: true)
        let system = ChatMessage.system("You help the owner.")
        var chat: [ChatMessage] = [system, .user("Change the hair.")]
        let one = try await Self.vlmReply(generator, chat)
        let look = try await Self.vlmReply(generator, [.system("Describe only the hair."), .user("Describe the hair.", images: [Self.image(1)])])
        #expect(look.reusedTokens == 0)
        #expect(await (generator.engine as! DecoderEngine).boundImages.count == 1)
        chat += [.assistant(ByteChat.text(one.tokens.dropLast().map { $0 })), .user("Thanks.")]
        let two = try await Self.vlmReply(generator, chat)
        // Turn 1's prompt (it ends with the generation prompt, which this turn's history repeats) is reused.
        #expect(two.reusedTokens >= one.promptTokens)
        #expect(await (generator.engine as! DecoderEngine).boundImages.isEmpty)
        #expect(try await Self.vlmReply(fresh, chat).tokens == two.tokens)
        try await Self.expectStateIsExact(generator, fresh: fresh, probe: ByteVocab.tokens("x")[0])
        // A picture again, then a new conversation with the same system turn.
        _ = try await Self.vlmReply(generator, [.system("Describe only the hair."), .user("Describe the hair.", images: [Self.image(2)])])
        let other: [ChatMessage] = [system, .user("Change the boots.")]
        let three = try await Self.vlmReply(generator, other)
        let systemTurn = try VLMByteRenderer().render([system], options: RenderOptions(addGenerationPrompt: false)).tokens.count
            + 1 + ByteVocab.tokens("user\n").count
        #expect(three.reusedTokens == systemTurn)
        #expect(try await Self.vlmReply(fresh, other).tokens == three.tokens)
        try await Self.expectStateIsExact(generator, fresh: fresh, probe: ByteVocab.tokens("x")[0])
        // And a picture in the chat itself.
        let withPicture = other + [.assistant(ByteChat.text(three.tokens.dropLast().map { $0 })), .user("This one?", images: [Self.image(3)])]
        let four = try await Self.vlmReply(generator, withPicture)
        #expect(try await Self.vlmReply(fresh, withPicture).tokens == four.tokens)
        #expect(await generator.heldSnapshots.count <= Generator.maxSnapshots)
    }

    /// Two generators' engines hold the same tokens and the same state: the next row after `probe` is bit-identical.
    static func expectSameState(_ a: Generator, _ b: Generator, probe: Int32, sourceLocation: SourceLocation = #_sourceLocation) async throws {
        let held = await a.mirroredTokens
        #expect(held == (await b.mirroredTokens), sourceLocation: sourceLocation)
        let x = try #require(try await a.engine.prefill([probe], logits: .last))
        let y = try #require(try await b.engine.prefill([probe], logits: .last))
        #expect(Array(x.last) == Array(y.last), "the reused state's logits differ from a fresh one's", sourceLocation: sourceLocation)
    }

    /// Studio's outfit describer: several questions about one picture, the picture and the shared instructions first,
    /// each question in a turn of its own after them. The second question restores the snapshot at the end of the
    /// picture's turn: no tower call, no prefill of the shared part. Another picture starts over (or from a text
    /// prompt's system turn); each state equals a fresh generator's bit for bit.
    @Test func aRepeatedPictureSkipsTheTowerAndItsPrefill() async throws {
        let log = CallLog()
        let generator = try await FakeVLM.generator(towerLog: log, recurrent: true)
        let system = ChatMessage.system("You describe outfits.")
        func ask(_ picture: Int, _ question: String) -> [ChatMessage] {
            [system, .user("The outfit in this picture, shared rules.", images: [Self.image(picture)]), .user(question)]
        }
        func fresh(_ messages: [ChatMessage]) async throws -> (Generator, GenerationSummary) {
            let g = try await FakeVLM.generator(recurrent: true)
            return (g, try await Self.vlmReply(g, messages))
        }
        let probe = ByteVocab.tokens("x")[0]
        let one = try await Self.vlmReply(generator, ask(1, "The footwear?"))
        #expect(one.reusedTokens == 0 && log.calls.count == 1)
        let imageTurn = try #require(await generator.heldSnapshots.imageTokens)
        // The picture's turn: the system turn, `<|im_start|>user\n`, the image's open, rows and close, the text.
        let rows = one.promptTokens - (try VLMByteRenderer().render(ask(1, "The footwear?"), options: RenderOptions())).tokens.count + 1
        #expect(rows > 1)

        for question in ["The belt?", "The gloves?"] {
            let next = try await Self.vlmReply(generator, ask(1, question))
            #expect(log.calls.count == 1, "the picture isn't encoded again")
            #expect(next.reusedTokens == imageTurn)
            let (reference, summary) = try await fresh(ask(1, question))
            #expect(summary.tokens == next.tokens)
            try await Self.expectSameState(generator, reference, probe: probe)
        }
        #expect(await generator.heldSnapshots.count <= Generator.maxSnapshots)

        // Another picture of the same size: it is encoded, and nothing is reused (the image turn's snapshot is the
        // other picture's; no text prompt left a system turn).
        let other = try await Self.vlmReply(generator, ask(2, "The belt?"))
        #expect(log.calls.count == 2)
        #expect(other.reusedTokens == 0)
        let (reference, summary) = try await fresh(ask(2, "The belt?"))
        #expect(summary.tokens == other.tokens)
        try await Self.expectSameState(generator, reference, probe: probe)
        // Its own questions now reuse its turn.
        let again = try await Self.vlmReply(generator, ask(2, "The gloves?"))
        #expect(log.calls.count == 2 && again.reusedTokens == imageTurn)
        // A text prompt after it (it leaves the system turn's snapshot), then the first picture once more: no longer
        // held, so encoded again, from that system turn.
        let text: [ChatMessage] = [system, .user("No picture now.")]
        let t = try await Self.vlmReply(generator, text)
        #expect(t.reusedTokens == 0)
        #expect(try await fresh(text).1.tokens == t.tokens)
        let systemTurn = try #require(await generator.heldSnapshots.systemTokens)
        #expect(systemTurn < imageTurn)
        let back = try await Self.vlmReply(generator, ask(1, "The belt?"))
        #expect(log.calls.count == 3 && back.reusedTokens == systemTurn)
        let (ref2, s2) = try await fresh(ask(1, "The belt?"))
        #expect(s2.tokens == back.tokens)
        try await Self.expectSameState(generator, ref2, probe: probe)
        #expect(await generator.heldSnapshots.count <= Generator.maxSnapshots)
    }

    /// The question in the picture's own turn (no later turn): the snapshot is at the end of the system turn, and every
    /// later call starts there; still exact.
    @Test func aQuestionInThePicturesTurnReusesOnlyTheSystemTurn() async throws {
        let log = CallLog()
        let generator = try await FakeVLM.generator(towerLog: log, recurrent: true)
        let system = ChatMessage.system("You describe outfits.")
        _ = try await Self.vlmReply(generator, [system, .user("The footwear?", images: [Self.image(1)])])
        let two = try await Self.vlmReply(generator, [system, .user("The belt?", images: [Self.image(1)])])
        let systemTurn = 1 + ByteVocab.tokens("system\nYou describe outfits.").count + 2 + 1 + ByteVocab.tokens("user\n").count
        #expect(await generator.heldSnapshots.imageTokens == systemTurn)
        #expect(two.reusedTokens == systemTurn)
        #expect(log.calls.count == 1, "the picture the statics hold is not encoded again")
        let fresh = try await FakeVLM.generator(recurrent: true)
        #expect(try await Self.vlmReply(fresh, [system, .user("The belt?", images: [Self.image(1)])]).tokens == two.tokens)
        try await Self.expectSameState(generator, fresh, probe: ByteVocab.tokens("x")[0])
    }

    // MARK: the template variable

    /// `preserveThinking` reaches the Jinja template as `preserve_thinking`; without it the variable is undefined.
    @Test func preserveThinkingReachesTheTemplate() async throws {
        let tiny = try #require(Bundle.module.url(forResource: "Fixtures/tiny-tokenizer", withExtension: nil))
        let adapter = try await TokenizerAdapter(
            directory: tiny, block: Manifest.Tokenizer(source: .dir(AssetPath(asset: "main", path: "tokenizer")), stop: ["<eos>"]))
        let template = "{%- for m in messages %}{{ m.content }}{%- if preserve_thinking is defined and preserve_thinking is true %} kept{%- endif %} {% endfor %}"
        let renderer = JinjaChatTemplate(tokenizer: adapter, template: template, thinking: nil)
        let messages: [ChatMessage] = [.user("a"), .assistant("b")]
        let plain = try renderer.render(messages, options: RenderOptions(addGenerationPrompt: false))
        let kept = try renderer.render(messages, options: RenderOptions(thinking: nil, addGenerationPrompt: false, preserveThinking: true))
        #expect(kept.tokens.count > plain.tokens.count)
        #expect(Array(kept.tokens.prefix(1)) == Array(plain.tokens.prefix(1)))
    }
}
