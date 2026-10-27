import AetherKit
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import Darwin
import Foundation
import Metal
import Testing

/// Opt-in native check that a registered chat bundle runs end to end: `AetherLanguageTokenizers` renders the model's
/// own template, `LanguageGenerationSession.open(bundle:)` decodes greedily on the GPU, and the reply decodes to text
/// that stopped at the end of the turn. Two short instruction prompts (a sentence rewrite and a one-line look) are
/// synthetic and non-graphic; the first runs twice to show greedy decoding is repeatable. Records timings, the
/// process footprint and Metal's allocated size (the footprint does not count Core AI's model memory). Run under the shared GPU lease (`with-gpu-lock.sh`), release build:
///
/// AETHER_NATIVE_LANGUAGE_CHAT=1
/// AETHER_LANGUAGE_CHAT_BUNDLE=<registered bundle with language-model.json>
/// AETHER_LANGUAGE_CHAT_OUTPUT=<report .json>
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_LANGUAGE_CHAT"] == "1"), .timeLimit(.minutes(20)))
func languageChatBundleRendersGeneratesAndDecodes() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let bundle = URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_CHAT_BUNDLE"]))
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_CHAT_OUTPUT"]))
    let manifest = try LanguageModelManifest.load(from: bundle)
    let tokenizer = try await LanguageTokenizer.open(bundle: bundle)
    let metal = MTLCreateSystemDefaultDevice()
    let idle = chatFootprint(), idleMetal = metal?.currentAllocatedSize ?? 0
    let opened = ContinuousClock.now
    let session = try await LanguageGenerationSession.open(bundle: bundle, platform: .macos)
    let openSeconds = chatSeconds(since: opened)
    let cases: [(name: String, messages: [LanguageChatMessage], limit: Int)] = [
        ("rewrite", [
            .system("Rewrite the picture description as one vivid sentence for an image generator. Keep every name exactly as written. Reply with the sentence only."),
            .user("Mara, an adult fictional explorer with short dark hair, wearing a green field jacket, resting after a journey in a book-lined study, soft window light."),
        ], 360),
        ("look", [
            .system("You describe fictional adult characters for an image generator. Write one line of 2 to 12 words describing the character's look. Reply with the line only."),
            .user("Character: Aria Vale from Lanternfall, a cartographer in her thirties.\nCurrent look: silver braid, amber eyes"),
        ], 120),
    ]
    var runs: [ChatRun] = [], peak = idle, peakMetal = idleMetal
    for (index, entry) in (cases + [cases[0]]).enumerated() {
        let prompt = try tokenizer.render(entry.messages, thinking: false)
        let reply = try await session.generate(prompt: prompt.tokenIDs, maximumTokens: entry.limit)
        peak = max(peak, chatFootprint()); peakMetal = max(peakMetal, metal?.currentAllocatedSize ?? 0)
        let text = tokenizer.decode(reply.tokens)
        runs.append(ChatRun(name: index == cases.count ? entry.name + "-repeat" : entry.name, promptTokens: prompt.tokenIDs.count,
            replyTokens: reply.tokens.count, stopReason: reply.stopReason, timeToFirstTokenSeconds: reply.timeToFirstTokenSeconds,
            totalSeconds: reply.totalSeconds, tokens: reply.tokens, text: text))
        #expect(reply.stopReason == "eos", "\(entry.name) stopped by \(reply.stopReason)")
        #expect(!text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !text.contains("<|") && !text.contains("<think>"))
    }
    #expect(runs[0].tokens == runs[2].tokens, "greedy decoding repeats")
    let unloadStarted = ContinuousClock.now
    await session.unload()
    let report = ChatReport(device: LanguageGenerationSession.deviceArchitectureName, repository: manifest.sourceRepository,
        revision: manifest.sourceRevision, modelSHA256: manifest.modelSHA256, quantization: manifest.quantization,
        maximumContextLength: manifest.maximumContextLength, openSeconds: openSeconds, unloadSeconds: chatSeconds(since: unloadStarted),
        idleFootprintMiB: Int(idle >> 20), peakFootprintMiB: Int(peak >> 20), afterUnloadFootprintMiB: Int(chatFootprint() >> 20),
        idleMetalMiB: idleMetal >> 20, peakMetalMiB: peakMetal >> 20, afterUnloadMetalMiB: (metal?.currentAllocatedSize ?? 0) >> 20,
        greedyRepeatIdentical: runs[0].tokens == runs[2].tokens, runs: runs)
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(report).write(to: output)
}

private struct ChatRun: Encodable {
    let name: String, promptTokens: Int, replyTokens: Int, stopReason: String
    let timeToFirstTokenSeconds: Double, totalSeconds: Double
    let tokens: [Int], text: String
}

private struct ChatReport: Encodable {
    let device: String, repository: String, revision: String, modelSHA256: String, quantization: String
    let maximumContextLength: Int, openSeconds: Double, unloadSeconds: Double
    let idleFootprintMiB: Int, peakFootprintMiB: Int, afterUnloadFootprintMiB: Int
    let idleMetalMiB: Int, peakMetalMiB: Int, afterUnloadMetalMiB: Int
    let greedyRepeatIdentical: Bool
    let runs: [ChatRun]
}

private func chatSeconds(since start: ContinuousClock.Instant) -> Double {
    let elapsed = ContinuousClock.now - start
    return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
}

private func chatFootprint() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return result == KERN_SUCCESS ? info.phys_footprint : 0
}
