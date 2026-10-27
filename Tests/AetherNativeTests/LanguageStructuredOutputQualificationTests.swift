import AetherKit
import AetherLanguage
import AetherLanguageCoreAI
import AetherLanguageFoundationModels
import AetherLanguageMacros
import AetherLanguageTokenizers
import Foundation
import FoundationModels
import Testing

// The same character under Kit's @AetherGenerable and FoundationModels' @Generable: nested objects, arrays, both
// kinds of enum, an optional and guides on every kind of property.

@AetherGenerable enum NativeMood: Equatable { case calm, cheerful, stern }
@AetherGenerable enum NativeColour: Equatable { case red, green, blue, black, white }
@AetherGenerable enum NativeAccessory: Equatable { case none, hat(colour: NativeColour), cane(length: Double) }
@AetherGenerable struct NativeLayer: Equatable {
    @AetherGuide(description: "The garment") var garment: String
    var warm: Bool
}
@AetherGenerable struct NativeOutfit: Equatable {
    @AetherGuide(description: "The main colour") var colour: NativeColour
    @AetherGuide(description: "One to three layers", .count(1...3)) var layers: [NativeLayer]
}
@AetherGenerable(description: "A fictional adult character for an image generator")
struct NativeCharacter: Equatable {
    @AetherGuide(description: "Full name") var name: String
    @AetherGuide(description: "Age in years", .range(18...80)) var age: Int
    var mood: NativeMood
    @AetherGuide(description: "Two to four distinctive features", .count(2...4)) var features: [String]
    var outfit: NativeOutfit
    var accessory: NativeAccessory
    @AetherGuide(description: "How sure you are of the description", .range(0.0...1.0)) var confidence: Double
    var nickname: String?
}

@Generable enum FMNativeMood { case calm, cheerful, stern }
@Generable enum FMNativeColour { case red, green, blue, black, white }
@Generable enum FMNativeAccessory { case none, hat(colour: FMNativeColour), cane(length: Double) }
@Generable struct FMNativeLayer {
    @Guide(description: "The garment") var garment: String
    var warm: Bool
}
@Generable struct FMNativeOutfit {
    @Guide(description: "The main colour") var colour: FMNativeColour
    @Guide(description: "One to three layers", .count(1...3)) var layers: [FMNativeLayer]
}
@Generable(description: "A fictional adult character for an image generator")
struct FMNativeCharacter {
    @Guide(description: "Full name") var name: String
    @Guide(description: "Age in years", .range(18...80)) var age: Int
    var mood: FMNativeMood
    @Guide(description: "Two to four distinctive features", .count(2...4)) var features: [String]
    var outfit: FMNativeOutfit
    var accessory: FMNativeAccessory
    @Guide(description: "How sure you are of the description", .range(0.0...1.0)) var confidence: Double
    var nickname: String?
}

/// Opt-in native qualification of structured output on a registered bundle:
/// - the facade's `respond(to:generating:)` decodes into a macro type, greedy and with seeded sampling (and after
///   thinking, for models that declare reasoning), with every guide satisfied;
/// - FoundationModels' `LanguageModelSession(model:)` decodes into the paired `@Generable` type, respond and stream;
/// - the constraint's cost per call against the decode step, and per-token time against unconstrained generation.
/// Run under the shared GPU lease (`with-gpu-lock.sh`), release build:
///
/// AETHER_NATIVE_LANGUAGE_STRUCTURED=1
/// AETHER_LANGUAGE_STRUCTURED_BUNDLE=<registered bundle with language-model.json>
/// AETHER_LANGUAGE_STRUCTURED_OUTPUT=<report .json>
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_LANGUAGE_STRUCTURED"] == "1"), .timeLimit(.minutes(30)))
func languageStructuredOutputQualification() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let bundle = URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_STRUCTURED_BUNDLE"]))
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_STRUCTURED_OUTPUT"]))
    var report = StructuredReport()
    report.model = try LanguageModelManifest.load(from: bundle).sourceRepository
    let model = AetherLanguageModel(bundle: bundle)
    try await model.prewarm()
    let instructions = "You invent fictional adult characters for a fantasy story."
    let prompt = "Invent a character who keeps a lighthouse."

    func check(_ character: NativeCharacter) -> Bool {
        (18...80).contains(character.age) && (2...4).contains(character.features.count) && (1...3).contains(character.outfit.layers.count)
            && (0...1).contains(character.confidence) && !character.name.isEmpty
    }

    // The facade: greedy, then seeded sampling.
    let session = AetherLanguageSession(model: model, instructions: instructions)
    let greedy = try await session.respond(to: prompt, generating: NativeCharacter.self, options: .init(maximumResponseTokens: 600))
    let greedyJSON = await lastResponse(session)
    report.facadeGreedy = .init(json: greedyJSON, outputTokens: greedy.usage.outputTokens, guidesHold: check(greedy.content))
    #expect(check(greedy.content) && greedy.stopReason == .endOfTurn)
    let sampling = AetherLanguageSession(model: model, instructions: instructions)
    let sampled = try await sampling.respond(to: prompt, generating: NativeCharacter.self,
        options: .init(samplingMode: .random(top: 40, seed: 7), temperature: 0.8, maximumResponseTokens: 600))
    report.facadeSampled = .init(json: await lastResponse(sampling), outputTokens: sampled.usage.outputTokens, guidesHold: check(sampled.content))
    #expect(check(sampled.content))
    // A second turn sees the first reply's JSON.
    let followUp = try await session.respond(to: "Give her a sibling.", generating: NativeCharacter.self,
                                             options: .init(maximumResponseTokens: 600))
    report.secondTurnDecodes = check(followUp.content)
    #expect(report.secondTurnDecodes)

    if model.supportsReasoning {
        let thinking = AetherLanguageSession(model: model, instructions: instructions)
        let thought = try await thinking.respond(to: prompt, generating: NativeCharacter.self,
                                                 options: .init(maximumResponseTokens: 1500, reasoning: .on))
        report.facadeReasoning = .init(json: await lastResponse(thinking), outputTokens: thought.usage.outputTokens, guidesHold: check(thought.content))
        report.reasoningCharacters = thought.reasoning?.count ?? 0
        #expect(check(thought.content) && !(thought.reasoning ?? "").isEmpty)
    }

    // FoundationModels through the adapter.
    let foundation = try await LanguageModelSession(model: model, instructions: instructions)
        .respond(to: prompt, generating: FMNativeCharacter.self, options: GenerationOptions(maximumResponseTokens: 600))
    let fm = foundation.content
    report.foundationModels = .init(json: foundation.rawContent.jsonString, outputTokens: 0,
        guidesHold: (18...80).contains(fm.age) && (2...4).contains(fm.features.count) && (1...3).contains(fm.outfit.layers.count)
            && (0...1).contains(fm.confidence))
    // The same greedy request through FoundationModels: the same JSON, modulo the Kit/FoundationModels type names.
    report.foundationModelsMatchesFacade = (try? JSONSerialization.jsonObject(with: Data(foundation.rawContent.jsonString.utf8)) as? NSDictionary)
        == (try? JSONSerialization.jsonObject(with: Data(greedyJSON.utf8)) as? NSDictionary)
    #expect(report.foundationModels!.guidesHold)
    var snapshots = 0
    for try await _ in LanguageModelSession(model: model, instructions: instructions)
        .streamResponse(to: prompt, generating: FMNativeCharacter.self, options: GenerationOptions(maximumResponseTokens: 600)) { snapshots += 1 }
    report.foundationModelsStreamSnapshots = snapshots
    #expect(snapshots > 1)

    // Cost: the constraint per call against the step, and structured against unconstrained per-token time.
    let messages: [LanguageChatMessage] = [.system(instructions), .user(prompt)]
    for round in 0..<3 {
        var started = ContinuousClock.now
        let structured = try await model.generate(messages: messages, options: .init(maximumResponseTokens: 600),
            structured: LanguageStructuredOutput(schema: NativeCharacter.generationSchema, includeSchemaInPrompt: true)) { _, _ in }
        let structuredSeconds = seconds(since: started)
        started = ContinuousClock.now
        let plain = try await model.generate(messages: messages + [], options: .init(maximumResponseTokens: 200)) { _, _ in }
        let plainSeconds = seconds(since: started)
        guard round > 0, let statistics = structured.constraintStatistics else { continue }
        report.cost.append(.init(
            structuredMillisecondsPerToken: structuredSeconds * 1000 / Double(structured.outputTokens),
            unconstrainedMillisecondsPerToken: plainSeconds * 1000 / Double(plain.outputTokens),
            constraintCalls: statistics.calls, constraintCacheHits: statistics.cacheHits,
            constraintMeanMicroseconds: statistics.totalSeconds * 1e6 / Double(max(statistics.calls, 1)),
            constraintMaximumMicroseconds: statistics.maximumSeconds * 1e6, structuredTokens: structured.outputTokens,
            unconstrainedTokens: plain.outputTokens))
    }
    await model.unload()
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(report).write(to: output)
}

/// The JSON of a session's last reply (the transcript keeps a structured reply's JSON).
private func lastResponse(_ session: AetherLanguageSession) async -> String {
    for entry in await session.transcript.entries.reversed() { if case .response(let json) = entry { return json } }
    return ""
}

private func seconds(since start: ContinuousClock.Instant) -> Double {
    let elapsed = ContinuousClock.now - start
    return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
}

private struct StructuredReport: Encodable {
    struct Reply: Encodable { let json: String; let outputTokens: Int; let guidesHold: Bool }
    struct Cost: Encodable {
        let structuredMillisecondsPerToken: Double, unconstrainedMillisecondsPerToken: Double
        let constraintCalls: Int, constraintCacheHits: Int
        let constraintMeanMicroseconds: Double, constraintMaximumMicroseconds: Double
        let structuredTokens: Int, unconstrainedTokens: Int
    }
    var device = LanguageGenerationSession.deviceArchitectureName
    var model = ""
    var hostLoad = "Shared host: GPU jobs of cooperating agents are serialized by a lock, other host work is not; timings are observations"
    var facadeGreedy: Reply?
    var facadeSampled: Reply?
    var secondTurnDecodes = false
    var facadeReasoning: Reply?
    var reasoningCharacters = 0
    var foundationModels: Reply?
    var foundationModelsMatchesFacade = false
    var foundationModelsStreamSnapshots = 0
    var cost: [Cost] = []
}
