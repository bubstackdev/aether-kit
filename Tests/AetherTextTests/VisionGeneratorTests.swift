import AetherCore
import AetherRuntime
import Foundation
import Testing
@testable import AetherText
import AetherTestSupport

/// The Generator with images (docs/VLM.md §3.4, G2) on a fake VLM: image-aware prefix reuse, the encode skipped for
/// images already bound, the rendered-prompt entry point, and refusals.
@Suite("Generator with images (fake)")
struct VisionGeneratorTests {
    static func image(_ seed: Int, _ width: Int = 8, _ height: Int = 8) -> ChatImage {
        ChatImage(ImageInput(FakeVLM.picture(width: width, height: height, seed: seed)))
    }

    static func reply(_ generator: Generator, _ messages: [ChatMessage], tokens: Int = 6) async throws -> GenerationSummary {
        var summary: GenerationSummary?
        for try await event in generator.generate(messages, parameters: GenerationParameters(maxTokens: tokens)) {
            if case .finished(let s) = event { summary = s }
        }
        return try #require(summary)
    }

    @Test func aFollowUpTurnReusesTheImagesWithoutEncoding() async throws {
        let log = CallLog()
        let generator = try await FakeVLM.generator(towerLog: log)
        let a = Self.image(1), b = Self.image(2, 16, 8)
        let first: [ChatMessage] = [.system("Describe."), .user("What changes?", images: [a, b])]
        let one = try await Self.reply(generator, first)
        #expect(log.calls.count == 2)
        let history = first + [.assistant(ByteChat.text(one.tokens.dropLast().map { $0 })), .user("And then?")]
        let two = try await Self.reply(generator, history)
        #expect(log.calls.count == 2)                                                // no new encode
        let bound = await (generator.engine as! DecoderEngine).boundImages
        #expect(two.reusedTokens >= bound.last!.end)
        // The same conversation on a fresh generator gives the same reply.
        let fresh = try await FakeVLM.generator()
        #expect(try await Self.reply(fresh, history).tokens == two.tokens)
    }

    @Test func anotherPictureOfTheSameSizeIsNotReused() async throws {
        let log = CallLog()
        let generator = try await FakeVLM.generator(towerLog: log)
        let one = try await Self.reply(generator, [.user("Look.", images: [Self.image(1)])])
        let other = [ChatMessage.user("Look.", images: [Self.image(7)])]
        let two = try await Self.reply(generator, other)
        #expect(log.calls.count == 2)
        let start = await (generator.engine as! DecoderEngine).boundImages[0].start
        #expect(two.reusedTokens <= start)
        #expect(two.promptTokens == one.promptTokens)
        let fresh = try await FakeVLM.generator()
        #expect(try await Self.reply(fresh, other).tokens == two.tokens)
        #expect(two.tokens != one.tokens)
    }

    @Test func textAfterImagesEqualsAFreshTextConversation() async throws {
        let generator = try await FakeVLM.generator()
        _ = try await Self.reply(generator, [.user("Look.", images: [Self.image(1), Self.image(2)])])
        let text: [ChatMessage] = [.user("Only words now, please.")]
        let after = try await Self.reply(generator, text)
        let fresh = try await FakeVLM.generator()
        #expect(try await Self.reply(fresh, text).tokens == after.tokens)
        #expect(await (generator.engine as! DecoderEngine).boundImages.isEmpty)
    }

    @Test func aRenderedPromptWithImagesEqualsTheMessages() async throws {
        let messages: [ChatMessage] = [.user("Two frames.", images: [Self.image(3), Self.image(4, 8, 16)])]
        let viaMessages = try await Self.reply(try await FakeVLM.generator(), messages)
        let generator = try await FakeVLM.generator()
        let rendered = try VLMByteRenderer().render(messages, options: RenderOptions())
        var summary: GenerationSummary?
        for try await event in generator.generate(prompt: rendered, images: messages[0].images, parameters: GenerationParameters(maxTokens: 6)) {
            if case .finished(let s) = event { summary = s }
        }
        #expect(summary?.tokens == viaMessages.tokens)
        // The prompt counts the expanded image tokens: 4 for the 8 × 8 picture, 2 for the 8 × 16 one (smart-resized
        // into the fake tower's 64 pixels: 4 × 8, a 2 × 4 patch grid).
        #expect(summary?.promptTokens == viaMessages.promptTokens)
        #expect(summary?.promptTokens == rendered.tokens.count - 2 + 6)
    }

    @Test func aFailedEncodeLeavesTheGeneratorUsable() async throws {
        let gate = TowerGate()
        let generator = try await FakeVLM.generator(towerGate: gate)
        let a = Self.image(1), b = Self.image(2, 16, 8)
        let first: [ChatMessage] = [.user("Look.", images: [a])]
        let one = try await Self.reply(generator, first)
        // The follow-up adds a picture whose encode fails: the turn fails and the state holding the first is dropped.
        let history = first + [.assistant(ByteChat.text(one.tokens.dropLast().map { $0 }))]
        gate.failing = true
        await #expect(throws: AetherError.self) { _ = try await Self.reply(generator, history + [.user("And this?", images: [b])]) }
        gate.failing = false
        // Text-only, then the same turn again: each equals a fresh generator's reply.
        let text: [ChatMessage] = history + [.user("Only words.")]
        #expect(try await Self.reply(generator, text).tokens == (try await Self.reply(try await FakeVLM.generator(), text)).tokens)
        let again = history + [.user("And this?", images: [b])]
        #expect(try await Self.reply(generator, again).tokens == (try await Self.reply(try await FakeVLM.generator(), again)).tokens)
    }

    @Test func refusals() async throws {
        let generator = try await FakeVLM.generator(images: 2)
        await #expect(throws: AetherError.self) {
            _ = try await Self.reply(generator, [.user("Three.", images: [Self.image(1), Self.image(2), Self.image(3)])])
        }
        // Still usable after a refusal.
        _ = try await Self.reply(generator, [.user("Two.", images: [Self.image(1), Self.image(2)])])
        let text = try await ByteChat.generator { _ in ByteVocab.tokens("ok") }
        await #expect(throws: AetherError.self) {
            for try await _ in text.generate([.user("Look.", images: [Self.image(1)])]) {}
        }
        #expect(text.imageInput == nil)
        #expect(Registry.text.supports(.vlmStaticInject))
    }
}
