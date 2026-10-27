import AetherKit
import AetherLanguage
import AetherLanguageCoreAI
import AetherLanguageFoundationModels
import AetherLanguageTokenizers
import Foundation
import FoundationModels
import Testing

/// Opt-in native qualification of the `AetherLanguage` facade on a registered bundle, against the runtime and tokenizer
/// alone on the same rendered prompts:
/// - a greedy `respond` is the runtime's greedy reply, decoded by the tokenizer (content, token counts);
/// - a stream's deltas add up to that reply, with no replacement characters;
/// - a stop sequence ends the reply before it; a seeded request repeats; a cancelled stream leaves the session usable;
/// - a three-turn conversation; reasoning on (for models that declare it) separates thinking from the answer;
/// - per-token time through the facade against the runtime alone;
/// - FoundationModels' `LanguageModelSession(model:)` on the same model gives the facade's replies (respond, stream,
///   and with a reasoning level, for models that declare reasoning).
/// Run under the shared GPU lease (`with-gpu-lock.sh`), release build:
///
/// AETHER_NATIVE_LANGUAGE_FACADE=1
/// AETHER_LANGUAGE_FACADE_BUNDLE=<registered bundle with language-model.json>
/// AETHER_LANGUAGE_FACADE_OUTPUT=<report .json>
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_LANGUAGE_FACADE"] == "1"), .timeLimit(.minutes(30)))
func languageFacadeQualification() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let bundle = URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_FACADE_BUNDLE"]))
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_FACADE_OUTPUT"]))
    var report = FacadeReport()
    let manifest = try LanguageModelManifest.load(from: bundle)
    report.model = manifest.sourceRepository
    let model = AetherLanguageModel(bundle: bundle)
    #expect(model.availability == .available)
    var started = ContinuousClock.now
    try await model.prewarm()
    report.prewarmSeconds = seconds(since: started)

    let instructions = "You are a concise assistant."
    let prompts = [
        "Name three colours, each with one emoji, and give the French word for coffee.",
        "Write two sentences about a lighthouse keeper's cat.",
    ]
    var greedy: [String: AetherLanguageSession.Response<String>] = [:]
    for prompt in prompts {
        let session = AetherLanguageSession(model: model, instructions: instructions)
        started = ContinuousClock.now
        let reply = try await session.respond(to: prompt, options: .init(maximumResponseTokens: 160))
        let wall = seconds(since: started)
        greedy[prompt] = reply
        #expect(!reply.content.isEmpty && !reply.content.contains("\u{FFFD}") && !reply.content.contains("<|"))
        // The same request as a stream.
        var deltas = "", snapshots = 0, clean = true
        for try await snapshot in AetherLanguageSession(model: model, instructions: instructions)
            .streamResponse(to: prompt, options: .init(maximumResponseTokens: 160)) {
            deltas += snapshot.delta
            snapshots += 1
            clean = clean && !snapshot.delta.contains("\u{FFFD}")
        }
        #expect(deltas == reply.content && clean, "stream deltas add up to the reply")
        report.greedy.append(.init(prompt: prompt, content: reply.content, outputTokens: reply.usage.outputTokens,
                                   promptTokens: reply.usage.promptTokens, stopReason: "\(reply.stopReason)",
                                   facadeMillisecondsPerToken: wall * 1000 / Double(max(reply.usage.outputTokens, 1)),
                                   streamSnapshots: snapshots, streamDeltasEqualContent: deltas == reply.content && clean))
    }

    // Stop sequence: a word from the middle of the first reply.
    let first = try #require(greedy[prompts[1]]?.content)
    let words = first.split(separator: " ")
    if words.count > 6 {
        let stop = " " + words[4]
        let stopped = try await AetherLanguageSession(model: model, instructions: instructions)
            .respond(to: prompts[1], options: .init(maximumResponseTokens: 160, stopSequences: [String(stop)]))
        let match = try #require(first.range(of: stop))
        let expected = String(first[..<match.lowerBound])
        report.stopSequence = .init(stop: String(stop), content: stopped.content, expected: expected, reason: "\(stopped.stopReason)")
        #expect(stopped.content == expected && stopped.stopReason == .stopSequence)
    }

    // Seeded sampling repeats; another seed differs.
    let story = "Tell a short story about a cartographer who maps clouds."
    func sampled(_ seed: UInt64) async throws -> String {
        try await AetherLanguageSession(model: model).respond(to: story, options: .init(samplingMode: .random(top: 40, seed: seed),
                                                                                       temperature: 0.8, maximumResponseTokens: 64)).content
    }
    let seeded = try await sampled(1234), again = try await sampled(1234), other = try await sampled(1235)
    report.seededRepeats = seeded == again
    report.otherSeedDiffers = seeded != other
    #expect(seeded == again && seeded != other)

    // Cancel a stream after a few snapshots; the session then answers normally.
    let cancelSession = AetherLanguageSession(model: model, instructions: instructions)
    var seen = 0
    for try await _ in cancelSession.streamResponse(to: story, options: .init(maximumResponseTokens: 400)) {
        seen += 1
        if seen == 5 { break }
    }
    while await cancelSession.isResponding { try await Task.sleep(for: .milliseconds(10)) }
    let afterCancel = try await cancelSession.respond(to: prompts[0], options: .init(maximumResponseTokens: 160))
    let cancelledThenAnswered = afterCancel.content == greedy[prompts[0]]?.content
    let keptTurns = await cancelSession.transcript.entries.count
    report.cancelledThenAnswered = cancelledThenAnswered
    #expect(keptTurns == 3 && cancelledThenAnswered, "the cancelled turn wasn't kept; the next one answered as before")

    // Three turns.
    let chat = AetherLanguageSession(model: model, instructions: "You help name fictional characters. Keep replies to one line.")
    for prompt in ["Suggest a name for a cartographer.", "Now a surname for her.", "What was the first name you suggested?"] {
        let reply = try await chat.respond(to: prompt, options: .init(maximumResponseTokens: 64))
        report.conversation.append(reply.content)
        #expect(!reply.content.isEmpty)
    }
    #expect(await chat.transcript.entries.count == 7)

    // Reasoning, for models that declare it.
    if model.supportsReasoning {
        let thought = try await AetherLanguageSession(model: model).respond(to: "What is 17 times 3? Answer with the number.",
                                                                            options: .init(maximumResponseTokens: 600, reasoning: .on))
        report.reasoning = .init(reasoning: thought.reasoning ?? "", content: thought.content, reason: "\(thought.stopReason)")
        #expect(!(thought.reasoning ?? "").isEmpty && !thought.content.contains("<think>") && !thought.content.contains("</think>"))
    }
    // The FoundationModels adapter: the same replies through LanguageModelSession.
    for (index, prompt) in prompts.enumerated() {
        let facade = try #require(greedy[prompt])
        let foundation = try await LanguageModelSession(model: model, instructions: instructions)
            .respond(to: prompt, options: GenerationOptions(maximumResponseTokens: 160))
        var streamed = ""
        for try await snapshot in LanguageModelSession(model: model, instructions: instructions)
            .streamResponse(to: prompt, options: GenerationOptions(maximumResponseTokens: 160)) { streamed = snapshot.content }
        report.greedy[index].foundationModelsMatches = foundation.content == facade.content && streamed == facade.content
        #expect(report.greedy[index].foundationModelsMatches, "LanguageModelSession gives the facade's reply")
    }
    if let reasoning = report.reasoning {
        let foundation = try await LanguageModelSession(model: model).respond(to: "What is 17 times 3? Answer with the number.",
            options: GenerationOptions(maximumResponseTokens: 600), contextOptions: ContextOptions(reasoningLevel: .light))
        let matches = foundation.content == reasoning.content
        report.foundationModelsReasoningMatches = matches
        #expect(matches, "a reasoning level gives the facade's reasoning-on answer")
    }
    await model.unload()

    // The runtime and tokenizer alone, on the same rendered prompts.
    let tokenizer = try await LanguageTokenizer.open(bundle: bundle)
    let runtime = try await LanguageGenerationSession.open(bundle: bundle, platform: .macos)
    for (index, prompt) in prompts.enumerated() {
        let rendered = try tokenizer.render([.system(instructions), .user(prompt)], thinking: false).tokenIDs
        let reply = try await runtime.generate(prompt: rendered, maximumTokens: 160)
        let decoded = tokenizer.decode(reply.tokens)
        let facade = try #require(greedy[prompt])
        report.greedy[index].runtimeDecoded = decoded
        report.greedy[index].runtimeOutputTokens = reply.tokens.count
        report.greedy[index].runtimeMillisecondsPerToken = reply.totalSeconds * 1000 / Double(max(reply.tokens.count, 1))
        report.greedy[index].matchesRuntime = decoded == facade.content && reply.tokens.count == facade.usage.outputTokens
            && rendered.count == facade.usage.promptTokens
        #expect(report.greedy[index].matchesRuntime, "the facade's greedy reply is the runtime's")
    }
    await runtime.unload()
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(report).write(to: output)
}

private func seconds(since start: ContinuousClock.Instant) -> Double {
    let elapsed = ContinuousClock.now - start
    return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
}

private struct FacadeReport: Encodable {
    struct Greedy: Encodable {
        let prompt: String, content: String, outputTokens: Int, promptTokens: Int, stopReason: String
        let facadeMillisecondsPerToken: Double, streamSnapshots: Int, streamDeltasEqualContent: Bool
        var runtimeDecoded = "", runtimeOutputTokens = 0, runtimeMillisecondsPerToken = 0.0, matchesRuntime = false
        var foundationModelsMatches = false
    }
    struct Stop: Encodable { let stop: String, content: String, expected: String, reason: String }
    struct Reasoning: Encodable { let reasoning: String, content: String, reason: String }
    var device = LanguageGenerationSession.deviceArchitectureName
    var model = ""
    var hostLoad = "Shared host: GPU jobs of cooperating agents are serialized by a lock, other host work is not; timings are observations"
    var prewarmSeconds = 0.0
    var greedy: [Greedy] = []
    var stopSequence: Stop?
    var seededRepeats = false
    var otherSeedDiffers = false
    var cancelledThenAnswered = false
    var conversation: [String] = []
    var reasoning: Reasoning?
    var foundationModelsReasoningMatches: Bool?
}
