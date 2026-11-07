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
        // The reply can't finish before the consumer below has stopped listening: halfway through its 200 tokens (well
        // after its first text) the model is held until the consumer has walked away, so the cancellation is already
        // set when it resumes. A fixed pace (1 ms a token) raced a consumer starved on a loaded machine, which could
        // let every token through first and keep the reply. The hold is bounded so a regression fails, not hangs.
        let calls = Mutex(0), stopped = Mutex(false)
        let long: @Sendable ([Int32]) -> [Int32] = { _ in
            if calls.withLock({ $0 += 1; return $0 }) == 100 {
                let deadline = ContinuousClock.now.advanced(by: .seconds(10))
                while !stopped.withLock({ $0 }), ContinuousClock.now < deadline { usleep(1_000) }
            }
            return Array(repeating: 12, count: 200)
        }
        let session = try await Self.chat(reply: long)
        for try await event in await session.respond(to: "hello") {
            if case .text = event { break }
        }
        stopped.withLock { $0 = true }
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

/// The second pre-release review's AetherText items (3, 6, 7, 10).
@Suite("AetherText second review regressions")
struct TextSecondReviewTests {
    /// Item 3: a value cut at `maxTokens` that still decodes (`100` cut to `1`) broke the schema's bounds and was
    /// accepted and kept in the history.
    @Test func aStructuredReplyMustCompleteTheSchema() async throws {
        let schema = try OutputSchema(json: Data(#"{"type":"integer","minimum":100,"maximum":100}"#.utf8))
        let generator = try await ByteChat.generator { _ in ByteVocab.tokens("100") }
        let session = ChatSession(generator: generator, configuration: .init(parameters: .init(maxTokens: 1)))
        await #expect(throws: AetherError.self) {
            _ = try await session.respond(to: "q", generating: Int.self, schema: schema, includeSchemaInPrompt: false)
        }
        #expect(await session.history.isEmpty)
        let enough = ChatSession(generator: generator, configuration: .init(parameters: .init(maxTokens: 8)))
        #expect(try await enough.respond(to: "q", generating: Int.self, schema: schema, includeSchemaInPrompt: false) == 100)
    }

    /// Item 6: integer bounds went through Double, so 2^53 + 1 became 2^53 and the exact value was refused.
    @Test func integerBoundsAreExact() throws {
        let schema = try OutputSchema(json: Data(#"{"type":"integer","minimum":9007199254740993,"maximum":9007199254740993}"#.utf8))
        let grammar = try JSONGrammar(schema: schema)
        let matcher = JSONMatcher(grammar: grammar)
        #expect(matcher.feeding(Array("9007199254740993".utf8), grammar)?.canEnd(grammar) == true)
        #expect(matcher.feeding(Array("9007199254740992".utf8), grammar)?.canEnd(grammar) != true)
    }

    /// Item 7: a constraint beside `$ref` or `anyOf` was silently dropped (an `anyOf` integer with a sibling minimum
    /// 10 accepted 1); it is refused as unsupported now.
    @Test func constraintsBesideACombinatorAreRefused() {
        for text in [
            ##"{"$defs":{"Value":{"type":"integer"}},"$ref":"#/$defs/Value","minimum":10}"##,
            #"{"anyOf":[{"type":"integer"}],"minimum":10}"#,
        ] {
            #expect(throws: AetherError.self) { _ = try JSONGrammar(schema: OutputSchema(json: Data(text.utf8))) }
        }
        // A description beside them stays fine.
        #expect(throws: Never.self) { _ = try JSONGrammar(schema: OutputSchema(json: Data(#"{"anyOf":[{"type":"integer"}],"description":"n"}"#.utf8))) }
    }

    /// Item 10: a request cancelled before its turn still rendered and prefilled its prompt.
    @Test func aCancelledRequestPrefillsNothing() async throws {
        let log = CallLog()
        let generator = try await FakeChat.generator(log: log)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try? await FakeChat.collect(generator.generate([.user("cancelled prompt")]))
        }
        await task.value
        _ = try await FakeChat.collect(generator.generate([.user("next")]))
        let cancelledPrompt = try FakeRenderer().render([.user("cancelled prompt")], options: .init())
        #expect(!log.calls.contains(cancelledPrompt.tokens.count))
    }
}

/// The owner's second code review: the chat items.
@Suite("Owner review 2: chat")
struct ChatOwnerReviewTests {
    /// Finding 5: a reply in progress when the history is cleared streams to its end but does not put its turn back,
    /// and the next turn starts from the empty history.
    @Test(.timeLimit(.minutes(1))) func clearingTheHistoryDuringAReplyKeepsItCleared() async throws {
        let calls = Mutex(0), cleared = Mutex(false)
        let held: @Sendable ([Int32]) -> [Int32] = { _ in
            // Halfway through the first reply's 200 tokens (well after its first text), hold the model until the test
            // has cleared the history. Bounded, so a regression fails instead of hanging.
            if calls.withLock({ $0 += 1; return $0 }) == 100 {
                let deadline = ContinuousClock.now.advanced(by: .seconds(10))
                while !cleared.withLock({ $0 }), ContinuousClock.now < deadline { usleep(1_000) }
            }
            return Array(repeating: 12, count: 200)
        }
        let session = try await TextReviewRegressionTests.chat(reply: held)
        let first = await session.respond(to: "hello")
        var iterator = first.makeAsyncIterator()
        while let event = try await iterator.next() {
            if case .text = event { break }
        }
        await session.clearHistory()
        cleared.withLock { $0 = true }
        var finished = false
        while let event = try await iterator.next() {
            if case .finished = event { finished = true }
        }
        #expect(finished, "the reply in progress still streams to its end")
        let afterClear = await session.history
        #expect(afterClear.isEmpty, "\(afterClear)")

        let next = try await FakeChat.collect(await session.respond(to: "next"))
        let history = await session.history
        #expect(history.map(\.content) == ["next", next.text])
        let expected = try FakeRenderer().render([.user("next")], options: RenderOptions()).tokens.count
        #expect(next.summary?.promptTokens == expected, "the next turn renders from the cleared history")
    }

    /// Finding 2: plain greedy took the prompt's argmax with no check, so an all -inf row became token 0 (`<pad>`)
    /// instead of an error. A later request still runs: the engine is left consistent.
    @Test func plainGreedyRefusesAnInvalidFirstRow() async throws {
        let poisoned = Mutex(true)
        let generator = try await FakeChat.generator(rowOverride: { context in
            poisoned.withLock { $0 } && context.last == Vocab.assistant ? [Float](repeating: -.infinity, count: 64) : nil
        })
        do {
            let reply = try await FakeChat.collect(generator.generate([.user("hello")]))
            Issue.record("an all -inf row produced \(reply.summary?.tokens ?? [])")
        } catch AetherError.generation(.invalidLogits) {
        }
        poisoned.withLock { $0 = false }
        let reply = try await FakeChat.collect(generator.generate([.user("hello")]))
        #expect(reply.summary?.stopReason == .stopToken(Vocab.stop))
    }

    /// Finding 1: the Generator holds at most `maxSnapshots` engine snapshots at once (it held four: the last prompt's
    /// two beside the two it was taking), and the engine's default reserve is that many.
    @Test func theGeneratorHoldsAtMostMaxSnapshots() async throws {
        #expect(DecoderEngine.Options().reservedSnapshots == Generator.maxSnapshots)
        let inner = try await FakeChat.generator(recurrent: true, renderer: FakeRenderer(generationSuffix: [Vocab.thinkOpen, Vocab.thinkClose]))
        let counting = await SnapshotCountingEngine(inner.engine)
        let generator = Generator(engine: counting, decoder: FakeDecoding(), renderer: inner.renderer, stopTokens: [Vocab.stop],
                                  profile: .plain, gpu: inner.gpu)
        let session = ChatSession(generator: generator, configuration: .init(parameters: GenerationParameters(maxTokens: 8)))
        var reused: [Int] = []
        for turn in ["hello", "more", "again", "and more"] {
            reused.append(try await FakeChat.collect(await session.respond(to: turn)).summary?.reusedTokens ?? 0)
        }
        #expect(reused.dropFirst().allSatisfy { $0 > 0 }, "every later turn reuses a snapshot: \(reused)")
        #expect(counting.live.peak <= Generator.maxSnapshots, "peak \(counting.live.peak)")
        #expect(counting.live.current <= 2)
    }
}

/// Counts the snapshots alive (taken and not yet released) through an engine, and the most at once.
final class SnapshotTally: Sendable {
    private let state = Mutex((current: 0, peak: 0))
    var current: Int { state.withLock { $0.current } }
    var peak: Int { state.withLock { $0.peak } }
    func taken() { state.withLock { $0.current += 1; $0.peak = max($0.peak, $0.current) } }
    func released() { state.withLock { $0.current -= 1 } }
}

/// Wraps an engine's snapshot payloads in a token whose lifetime the tally sees. Forwards everything else, and
/// mirrors the base's position after each call (the protocol's state properties are synchronous).
actor SnapshotCountingEngine: LanguageEngine {
    final class Tracked: Sendable {
        let inner: EngineSnapshot
        let tally: SnapshotTally
        init(_ inner: EngineSnapshot, _ tally: SnapshotTally) {
            self.inner = inner
            self.tally = tally
            tally.taken()
        }
        deinit { tally.released() }
    }

    let base: any LanguageEngine
    nonisolated let live = SnapshotTally()
    private(set) var position = 0
    private(set) var stateRanAhead = false
    let decodeModeDescription: String

    init(_ base: any LanguageEngine) async {
        self.base = base
        decodeModeDescription = await base.decodeModeDescription
        position = await base.position
    }

    nonisolated var vocab: Int { base.vocab }
    nonisolated var maxCapacity: Int { base.maxCapacity }
    nonisolated var hasRecurrentState: Bool { base.hasRecurrentState }

    private func sync() async {
        position = await base.position
        stateRanAhead = await base.stateRanAhead
    }

    func prefill(_ tokens: [Int32], logits: LogitsSelection) async throws(AetherError) -> Logits? {
        let out = try await base.prefill(tokens, logits: logits)
        await sync()
        return out
    }
    func step(_ token: Int32) async throws(AetherError) -> Logits {
        let out = try await base.step(token)
        await sync()
        return out
    }
    func rewind(to position: Int) async throws(AetherError) {
        try await base.rewind(to: position)
        await sync()
    }
    func reset() async throws(AetherError) {
        try await base.reset()
        await sync()
    }
    func snapshot() async throws(AetherError) -> EngineSnapshot {
        let inner = try await base.snapshot()
        return EngineSnapshot(position: inner.position, byteCount: inner.byteCount, payload: Tracked(inner, live))
    }
    func restore(_ snapshot: EngineSnapshot) async throws(AetherError) {
        guard let tracked = snapshot.payload as? Tracked else { throw .unsupportedOperation("another engine's snapshot") }
        try await base.restore(tracked.inner)
        await sync()
    }
    func beginGreedy(first: Int32, limit: Int) async throws(AetherError) { try await base.beginGreedy(first: first, limit: limit) }
    func nextGreedy() async throws(AetherError) -> Int32? { try await base.nextGreedy() }
    @discardableResult
    func endGreedy() async throws(AetherError) -> (fed: Int, overshoot: Int) {
        let out = try await base.endGreedy()
        await sync()
        return out
    }
}

/// Second code review, batch C item 5: the generator and the chat session queue through `SerialGate`, so a request
/// cancelled while it waits for its turn leaves at once (before: it waited for the running one, then ended).
@Suite("Text turns")
struct TextTurnQueueTests {
    final class Flag: Sendable {
        private let value = Mutex(false)
        var isSet: Bool { value.withLock { $0 } }
        func set() { value.withLock { $0 = true } }
    }

    /// A reply model that holds its calls until `released` (bounded, so a regression fails instead of hanging).
    static func held(_ released: Flag, _ entered: Flag) -> @Sendable ([Int32]) -> [Int32] {
        { _ in
            entered.set()
            let deadline = ContinuousClock.now.advanced(by: .seconds(10))
            while !released.isSet, ContinuousClock.now < deadline { usleep(1_000) }
            return [12, 12]
        }
    }

    @Test(.timeLimit(.minutes(1))) func aCancelledGeneratorRequestLeavesAtOnce() async throws {
        let released = Flag(), entered = Flag()
        let generator = try await FakeChat.generator(reply: Self.held(released, entered))
        let running = Task { try await FakeChat.collect(generator.generate([ChatMessage.user("first")])) }
        while !entered.isSet { await Task.yield() }
        let queued = Task { try await FakeChat.collect(generator.generate([ChatMessage.user("second")])) }
        try await Task.sleep(for: .milliseconds(50))
        let cancelledAt = ContinuousClock.now
        queued.cancel()
        // A cancelled consumer's stream just ends; what matters is that it ends now, not after the 10 s hold.
        _ = try? await queued.value
        #expect(ContinuousClock.now - cancelledAt < .seconds(5), "the queued request left while the first still ran")
        released.set()
        _ = try await running.value
    }

    @Test(.timeLimit(.minutes(1))) func aCancelledChatTurnLeavesAtOnceAndIsDropped() async throws {
        let released = Flag(), entered = Flag()
        let session = try await TextReviewRegressionTests.chat(reply: Self.held(released, entered))
        let running = Task { try await FakeChat.collect(await session.respond(to: "first")) }
        while !entered.isSet { await Task.yield() }
        let queued = Task { try await FakeChat.collect(await session.respond(to: "second")) }
        try await Task.sleep(for: .milliseconds(50))
        let cancelledAt = ContinuousClock.now
        queued.cancel()
        // A cancelled consumer's stream just ends; what matters is that it ends now, not after the 10 s hold.
        _ = try? await queued.value
        #expect(ContinuousClock.now - cancelledAt < .seconds(5), "the queued turn left while the first still ran")
        released.set()
        _ = try await running.value
        let history = await session.history
        #expect(history.map(\.content).first == "first" && history.count == 2)
    }
}
