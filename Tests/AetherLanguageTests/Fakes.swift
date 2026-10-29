import AetherKit
@testable import AetherLanguage
import AetherLanguageRuntime
import AetherLanguageTokenizers
import Foundation

/// A byte-level vocabulary for CPU tests: IDs 0–255 are single bytes, then chat markers, then a padded head as Qwen3's.
enum FakeVocabulary {
    static let imStart = 256, imEnd = 257, thinkStart = 258, thinkEnd = 259
    static let tokenizerSize = 260
    /// Logits rows: four padding rows past the tokenizer, as a padded model head has.
    static let headRows = 264

    static func tokens(_ text: String) -> [Int] { text.utf8.map(Int.init) }

    static let bytes: LanguageTokenBytes = {
        let printable = Set(Array(33...126) + Array(161...172) + Array(174...255))
        var shifted: [Int: Int] = [:]
        var next = 256
        for byte in 0...255 where !printable.contains(byte) { shifted[byte] = next; next += 1 }
        var vocab: [String: Int] = [:]
        for byte in 0...255 { vocab[String(Unicode.Scalar(UInt32(shifted[byte] ?? byte))!)] = byte }
        let added: [[String: Any]] = [
            ["id": imStart, "content": "<|im_start|>", "special": true], ["id": imEnd, "content": "<|im_end|>", "special": true],
            ["id": thinkStart, "content": "<think>", "special": false], ["id": thinkEnd, "content": "</think>", "special": false],
        ]
        let json: [String: Any] = ["model": ["vocab": vocab], "added_tokens": added, "decoder": ["type": "ByteLevel"]]
        return try! LanguageTokenBytes(tokenizerJSON: try! JSONSerialization.data(withJSONObject: json))
    }()
}

/// Renders like a hybrid Qwen3 template: `<|im_start|>role\ncontent<|im_end|>\n` per message, then the generation prompt,
/// with an empty think block when thinking is off.
struct FakeCodec: LanguageTextCodec {
    var bytes: LanguageTokenBytes { FakeVocabulary.bytes }

    func render(_ messages: [LanguageChatMessage], thinking: Bool) throws -> [Int] {
        var out: [Int] = []
        for message in messages {
            guard case .text(let text)? = message.parts.last else { continue }
            out += [FakeVocabulary.imStart] + FakeVocabulary.tokens("\(message.role.rawValue)\n\(text)") + [FakeVocabulary.imEnd]
                + FakeVocabulary.tokens("\n")
        }
        out += [FakeVocabulary.imStart] + FakeVocabulary.tokens("assistant\n")
        if !thinking { out += [FakeVocabulary.thinkStart] + FakeVocabulary.tokens("\n\n") + [FakeVocabulary.thinkEnd] + FakeVocabulary.tokens("\n\n") }
        return out
    }

    func tokenID(_ token: String) -> Int? {
        ["<|im_start|>": FakeVocabulary.imStart, "<|im_end|>": FakeVocabulary.imEnd,
         "<think>": FakeVocabulary.thinkStart, "</think>": FakeVocabulary.thinkEnd][token]
    }
}

/// Replays a scripted reply for each prompt through the runtime's own decode loop: every step's logits peak at the
/// script's next token, so sampling, constraints, stops and delivery overlap behave as with a real backend.
actor FakeBackend: LanguageBackend {
    nonisolated let vocabularySize = FakeVocabulary.headRows
    nonisolated let maximumContextLength: Int
    nonisolated let stopTokenIDs: Set<Int> = [FakeVocabulary.imEnd]
    private let script: @Sendable ([Int]) -> [Int]
    private let stepDelay: Duration?
    private(set) var prompts: [[Int]] = []
    private(set) var requests: [LanguageGenerationRequest] = []
    private(set) var unloads = 0
    private(set) var stepsRun = 0

    init(context: Int = 2048, stepDelay: Duration? = nil, script: @escaping @Sendable ([Int]) -> [Int]) {
        maximumContextLength = context
        self.stepDelay = stepDelay
        self.script = script
    }

    func generate(_ request: LanguageGenerationRequest,
                  onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?) async throws
        -> LanguageGenerationResult {
        prompts.append(request.prompt)
        requests.append(request)
        let reply = script(request.prompt) + [FakeVocabulary.imEnd]
        func row(_ index: Int) -> [Float] {
            var logits = [Float](repeating: -10, count: FakeVocabulary.headRows)
            let next = reply[min(index, reply.count - 1)]
            logits[next] = 10
            logits[(next + 1) % FakeVocabulary.tokenizerSize] = 9
            return logits
        }
        let output = try await LanguageDecodeLoop.generate(prompt: request.prompt, maximumTokens: request.maximumTokens,
            stopTokenIDs: stopTokenIDs, vocabularySize: vocabularySize, sampling: request.sampling, processor: request.constraint,
            prefill: { row(0) }, step: { _, position in
                if let stepDelay = self.stepDelay { try await Task.sleep(for: stepDelay) }
                self.stepsRun += 1
                return row(position - request.prompt.count + 1)
            }, onToken: onToken)
        return LanguageGenerationResult(tokens: output.tokens, stopReason: output.stopReason, seed: output.seed,
                                        timeToFirstTokenSeconds: output.timeToFirstTokenSeconds, totalSeconds: output.totalSeconds)
    }

    func unload() async { unloads += 1 }
}

enum FakeModel {
    static func make(context: Int = 2048, supportsReasoning: Bool = true, backend: FakeBackend? = nil,
                     lifecycle: ModelLifecycle = ModelLifecycle(policy: .manual, monitor: MemoryPressureMonitor(systemSource: false)),
                     onLoad: @escaping @Sendable () async throws -> Void = {}) -> AetherLanguageModel {
        let backend = backend ?? FakeBackend(context: context) { _ in FakeVocabulary.tokens("Hello") }
        return AetherLanguageModel(bundle: URL(filePath: "/fake"), availability: .available, supportsReasoning: supportsReasoning,
                                   contextLength: context, lifecycle: lifecycle) {
            try await onLoad()
            return LoadedLanguageModel(backend: backend, codec: FakeCodec())
        }
    }
}
