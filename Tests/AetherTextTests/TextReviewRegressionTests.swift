import Foundation
import Synchronization
import Testing
@testable import AetherText
import AetherTestSupport

/// Regression tests for the M2 review findings in AetherText. Each failed before its fix.
@Suite("AetherText review regressions")
struct TextReviewRegressionTests {
    static var tiny: URL { Bundle.module.url(forResource: "Fixtures/tiny-tokenizer", withExtension: nil)! }

    static func block(stop: [Manifest.StopToken]? = ["<eos>"], addBos: Manifest.AddBos? = nil) -> Manifest.Tokenizer {
        Manifest.Tokenizer(source: .dir(AssetPath(asset: "main", path: "tokenizer")), addBos: addBos, stop: stop)
    }

    /// #8: a generating tokenizer without stops fails the load (MANIFEST §2.4: `stop` is required for generation).
    @Test func missingStopsFailTheLoad() async throws {
        let ok = try await TokenizerAdapter(directory: Self.tiny, block: Self.block())
        #expect(ok.stopTokenIDs == [0])
        for stop in [nil, [] as [Manifest.StopToken]] {
            await #expect(throws: AetherError.self) { _ = try await TokenizerAdapter(directory: Self.tiny, block: Self.block(stop: stop)) }
        }
    }

    /// #9: `addBos` other than the template's (or none) is refused until implemented, not ignored.
    @Test func addBosIsHonouredOrRefused() async throws {
        for ok in [nil, Manifest.AddBos.template, .never] {
            _ = try await TokenizerAdapter(directory: Self.tiny, block: Self.block(addBos: ok))
        }
        for refused in [Manifest.AddBos.always, .renderer] {
            await #expect(throws: AetherError.self) { _ = try await TokenizerAdapter(directory: Self.tiny, block: Self.block(addBos: refused)) }
        }
    }

    /// #11: when the decoded window re-segments, the detokenizer must not split a UTF-8 scalar (U+FFFD).
    @Test func resegmentationKeepsScalarsWhole() {
        /// "a" alone, but "éz" once token 2 follows: the window's text is not a prefix of the new one.
        struct Resegmenting: TokenDecoding {
            func decode(_ ids: [Int32]) -> String {
                ids == [1] ? "a" : ids == [1, 2] ? "éz" : ids == [2] ? "z" : ids.map { String($0) }.joined()
            }
        }
        var d = IncrementalDetokenizer()
        var out = d.append(1, Resegmenting())
        out += d.append(2, Resegmenting())
        out += d.finish(Resegmenting())
        #expect(!out.contains("\u{FFFD}"), "\(out.debugDescription)")
        #expect(out.hasSuffix("éz"))
    }

    static func chat(reply: @escaping @Sendable ([Int32]) -> [Int32] = FakeChat.echo) async throws -> ChatSession {
        ChatSession(generator: try await FakeChat.generator(reply: reply), configuration: .init(parameters: GenerationParameters(maxTokens: 300)))
    }

    /// #1: a cancelled reply is not appended to the history.
    @Test func cancelledReplyIsNotAppended() async throws {
        let long: @Sendable ([Int32]) -> [Int32] = { _ in Array(repeating: 12, count: 200) }
        let session = try await Self.chat(reply: long)
        for try await event in await session.respond(to: "hello") {
            if case .text = event { break }
        }
        // The next turn waits for the cancelled one to unwind (turns are serialized), then completes.
        _ = try await FakeChat.collect(await session.respond(to: "next"))
        let history = await session.history
        #expect(history.map(\.content).first == "next", "\(history)")
        #expect(history.count == 2)
    }

    /// #2: a turn sent before the previous one finished is rendered after it, with its reply in the prompt.
    @Test func concurrentTurnsAreSerialized() async throws {
        let session = try await Self.chat()
        let a = await session.respond(to: "first")
        let b = await session.respond(to: "second")
        async let ra = FakeChat.collect(a)
        async let rb = FakeChat.collect(b)
        let (first, second) = try await (ra, rb)
        let history = await session.history
        #expect(history.map(\.content) == ["first", first.text, "second", second.text])
        let expected = try FakeRenderer().render(Array(history.prefix(3)), options: RenderOptions()).tokens.count
        #expect(second.summary?.promptTokens == expected, "turn 2's prompt holds turn 1 and its reply")
    }
}
