@testable import AetherLanguage
import AetherLanguageRuntime
import AetherLanguageTokenizers
import Foundation
import os
import Testing

struct AetherLanguageSessionTests {
    @Test func respondRendersTheTranscriptAndKeepsTheTurn() async throws {
        let backend = FakeBackend { _ in FakeVocabulary.tokens("Teal, with a silver clasp.") }
        let session = AetherLanguageSession(model: FakeModel.make(backend: backend), instructions: "Describe looks.")
        let reply = try await session.respond(to: "Her coat?")
        #expect(reply.content == "Teal, with a silver clasp." && reply.stopReason == .endOfTurn && reply.reasoning == nil && reply.seed == nil)
        #expect(await session.transcript.entries == [.instructions("Describe looks."), .prompt("Her coat?"), .response("Teal, with a silver clasp.")])
        let expectedPrompt = try FakeCodec().render([.system("Describe looks."), .user("Her coat?")], thinking: false)
        #expect(await backend.prompts == [expectedPrompt])
        #expect(reply.usage == .init(promptTokens: expectedPrompt.count, outputTokens: 26))
        #expect(await backend.requests.first?.sampling == .greedy, "greedy unless the request opts in")
    }

    @Test func laterTurnsRenderEarlierOnesButNotTheirReasoning() async throws {
        let turns = LockedCounter()
        let backend = FakeBackend { _ in
            let index = turns.next()
            return [FakeVocabulary.thinkStart] + FakeVocabulary.tokens("\nthink \(index)\n") + [FakeVocabulary.thinkEnd]
                + FakeVocabulary.tokens("\n\nanswer \(index)")
        }
        let session = AetherLanguageSession(model: FakeModel.make(backend: backend), instructions: "Be brief.")
        for prompt in ["one", "two", "three"] {
            let reply = try await session.respond(to: prompt, options: .init(reasoning: .on))
            #expect(reply.reasoning?.hasPrefix("think") == true && reply.content.hasPrefix("answer"))
        }
        let prompts = await backend.prompts
        let last = try FakeCodec().render([.system("Be brief."), .user("one"), .assistant("answer 0"), .user("two"),
                                           .assistant("answer 1"), .user("three")], thinking: true)
        #expect(prompts.last == last, "earlier answers render; earlier reasoning doesn't")
        #expect(await session.transcript.entries.contains(.reasoning("think 1")))
    }

    @Test func reasoningIsOptInAndOnlyForModelsThatHaveIt() async throws {
        let backend = FakeBackend { _ in FakeVocabulary.tokens("ok") }
        let session = AetherLanguageSession(model: FakeModel.make(backend: backend))
        _ = try await session.respond(to: "hi")
        #expect(await backend.prompts.last == (try FakeCodec().render([.user("hi")], thinking: false)), "off by default")
        let plain = AetherLanguageSession(model: FakeModel.make(supportsReasoning: false))
        await #expect(throws: AetherLanguageError.self) { try await plain.respond(to: "hi", options: .init(reasoning: .on)) }
        #expect(await plain.transcript.entries.isEmpty)
    }

    @Test func optionsMapOntoTheRuntimesSampling() async throws {
        let backend = FakeBackend { _ in FakeVocabulary.tokens("x") }
        let session = AetherLanguageSession(model: FakeModel.make(backend: backend))
        let cases: [(AetherGenerationOptions, LanguageSampling)] = [
            (.init(), .greedy),
            (.init(samplingMode: .greedy, temperature: 0.5), .greedy),
            (.init(samplingMode: .random(top: 40, seed: 7)), .random(temperature: 1, topK: 40, seed: 7)),
            (.init(samplingMode: .random(top: 40, seed: 7), temperature: 0.7), .random(temperature: 0.7, topK: 40, seed: 7)),
            (.init(samplingMode: .random(probabilityThreshold: 0.9, seed: 3)), .random(temperature: 1, topP: 0.9, seed: 3)),
            (.init(repetitionPenalty: 1.1), LanguageSampling(repetitionPenalty: 1.1)),
        ]
        for (options, _) in cases { _ = try await session.respond(to: "q", options: options) }
        #expect(await backend.requests.map(\.sampling) == cases.map(\.1))
        await #expect(throws: AetherLanguageError.self) {
            try await session.respond(to: "q", options: .init(samplingMode: .random(probabilityThreshold: 1.5)))
        }
    }

    @Test func outputLimitDefaultsToTheRemainingContextCapped() async throws {
        let backend = FakeBackend(context: 10_000) { _ in FakeVocabulary.tokens("x") }
        let session = AetherLanguageSession(model: FakeModel.make(context: 10_000, backend: backend))
        _ = try await session.respond(to: "q")
        _ = try await session.respond(to: "q", options: .init(maximumResponseTokens: 12))
        #expect(await backend.requests.map(\.maximumTokens) == [4096, 12])
        let small = FakeBackend(context: 64) { _ in FakeVocabulary.tokens("x") }
        let tight = AetherLanguageSession(model: FakeModel.make(context: 64, backend: small))
        _ = try await tight.respond(to: "q")
        let prompt = try FakeCodec().render([.user("q")], thinking: false)
        #expect(await small.requests.last?.maximumTokens == 64 - prompt.count)
        do {
            _ = try await tight.respond(to: String(repeating: "long ", count: 20))
            Issue.record("expected contextWindowExceeded")
        } catch AetherLanguageError.contextWindowExceeded(let promptTokens, _, let limit) {
            #expect(promptTokens > 64 && limit == 64)
        }
        #expect(await tight.transcript.entries.count == 2, "the failed turn wasn't kept")
    }

    @Test func stopSequencesEndTheReplyEarly() async throws {
        let backend = FakeBackend { _ in FakeVocabulary.tokens("Name: Aria\nAge: 30\nCity: Vale") }
        let session = AetherLanguageSession(model: FakeModel.make(backend: backend))
        let reply = try await session.respond(to: "q", options: .init(stopSequences: ["\nAge"]))
        #expect(reply.content == "Name: Aria" && reply.stopReason == .stopSequence)
        #expect(await backend.stepsRun < FakeVocabulary.tokens("Name: Aria\nAge: 30\nCity: Vale").count, "generation stopped early")
        let limited = try await session.respond(to: "q", options: .init(maximumResponseTokens: 4))
        #expect(limited.content == "Name" && limited.stopReason == .outputLimit)
    }

    @Test func seededSamplingReplays() async throws {
        let backend = FakeBackend { _ in FakeVocabulary.tokens("abcdefghijklmnop") }
        let session = AetherLanguageSession(model: FakeModel.make(backend: backend))
        let a = try await session.respond(to: "q", options: .init(samplingMode: .random(top: 2, seed: 11)))
        let b = try await session.respond(to: "q", options: .init(samplingMode: .random(top: 2, seed: 11)))
        let unseeded = try await session.respond(to: "q", options: .init(samplingMode: .random(top: 2)))
        // Unwrapped on its own: passed straight to `seed:` (itself optional), the check would be promoted away.
        let seed: UInt64 = try #require(unseeded.seed, "an unseeded request reports the seed it drew")
        let replay = try await session.respond(to: "q", options: .init(samplingMode: .random(top: 2, seed: seed)))
        #expect(a.content == b.content && a.seed == 11 && replay.content == unseeded.content)
    }

    @Test(.timeLimit(.minutes(1))) func overlapIsBusyAndCancellationLeavesTheTranscript() async throws {
        let backend = FakeBackend(stepDelay: .milliseconds(20)) { _ in FakeVocabulary.tokens(String(repeating: "slow ", count: 40)) }
        let session = AetherLanguageSession(model: FakeModel.make(backend: backend))
        let running = Task { try await session.respond(to: "long") }
        while await !session.isResponding { await Task.yield() }
        await #expect(throws: AetherLanguageError.self) { try await session.respond(to: "again") }
        running.cancel()
        await #expect(throws: CancellationError.self) { try await running.value }
        let entries = await session.transcript.entries, responding = await session.isResponding
        #expect(entries.isEmpty && !responding)
        let steps = await backend.stepsRun
        #expect(steps < 200, "cancelled before the end")
        let quick = AetherLanguageSession(model: FakeModel.make())
        #expect(try await quick.respond(to: "after").content == "Hello")
    }

    @Test func streamSnapshotsGrowAndTheirDeltasAddUp() async throws {
        let reply = "Café 🌙 — e\u{301} 👨\u{200D}👩\u{200D}👧 done."
        let backend = FakeBackend { _ in FakeVocabulary.tokens(reply) }
        let session = AetherLanguageSession(model: FakeModel.make(backend: backend))
        var deltas = ""
        var previous = ""
        var snapshots = 0
        for try await snapshot in session.streamResponse(to: "q") {
            #expect(snapshot.content.hasPrefix(previous) && snapshot.content == previous + snapshot.delta)
            #expect(!snapshot.delta.contains("\u{FFFD}"))
            previous = snapshot.content
            deltas += snapshot.delta
            snapshots += 1
        }
        #expect(deltas == reply && snapshots > 1)
        #expect(await session.transcript.entries == [.prompt("q"), .response(reply)])
        let collected = try await session.streamResponse(to: "again").collect()
        #expect(collected.content == reply)
    }

    @Test(.timeLimit(.minutes(1))) func abandoningAStreamCancelsTheRequest() async throws {
        let backend = FakeBackend(stepDelay: .milliseconds(10)) { _ in FakeVocabulary.tokens(String(repeating: "word ", count: 100)) }
        let session = AetherLanguageSession(model: FakeModel.make(backend: backend))
        for try await snapshot in session.streamResponse(to: "q") where !snapshot.content.isEmpty { break }
        while await session.isResponding { try await Task.sleep(for: .milliseconds(5)) }
        let steps = await backend.stepsRun
        try await Task.sleep(for: .milliseconds(100))
        #expect(await backend.stepsRun == steps && steps < 10, "no steps after the stream was abandoned")
        #expect(await session.transcript.entries.isEmpty)
        let stream = AetherLanguageSession(model: FakeModel.make()).streamResponse(to: "twice")
        _ = try await stream.collect()
        await #expect(throws: AetherLanguageError.self) { for try await _ in stream {} }
    }

    @Test func twoSessionsOnOneModelQueueRatherThanFail() async throws {
        let backend = FakeBackend(stepDelay: .milliseconds(2)) { _ in FakeVocabulary.tokens("shared model") }
        let model = FakeModel.make(backend: backend)
        let first = AetherLanguageSession(model: model), second = AetherLanguageSession(model: model)
        async let a = first.respond(to: "a")
        async let b = second.respond(to: "b")
        let (ra, rb) = try await (a, b)
        #expect(ra.content == "shared model" && rb.content == "shared model")
    }

    @Test func transcriptsRoundTripThroughCodable() throws {
        let transcript = AetherTranscript(entries: [.instructions("i"), .prompt("p"), .reasoning("r"), .response("a")])
        let decoded = try JSONDecoder().decode(AetherTranscript.self, from: JSONEncoder().encode(transcript))
        #expect(decoded == transcript)
        #expect(transcript.messages == [.system("i"), .user("p"), .assistant("a")])
    }
}

final class LockedCounter: Sendable {
    private let value = OSAllocatedUnfairLock(initialState: 0)
    func next() -> Int { value.withLock { value in defer { value += 1 }; return value } }
}
