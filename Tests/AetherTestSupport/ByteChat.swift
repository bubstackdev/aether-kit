@_spi(Testing) import AetherHost
public import AetherText
public import Foundation
import Synchronization

/// A byte-level vocabulary for the ported 0.0.x CPU tests (their `FakeVocabulary`): ids 0–255 are single bytes, then
/// chat markers and tool tags, then a padded head as Qwen3's. Built through a synthetic `tokenizer.json`, so the byte
/// table's reader is on the path.
public enum ByteVocab {
    public static let imStart: Int32 = 256, imEnd: Int32 = 257, thinkStart: Int32 = 258, thinkEnd: Int32 = 259
    public static let toolOpen: Int32 = 260, toolClose: Int32 = 261
    public static let tokenizerSize = 262
    /// Logits rows: two padding rows past the tokenizer, as a padded model head has.
    public static let headRows = 264

    public static func tokens(_ text: String) -> [Int32] { text.utf8.map(Int32.init) }

    /// GPT-2's byte-to-unicode spelling of raw bytes, written out independently of the implementation's inverse.
    public static func spelled(_ bytes: some Sequence<UInt8>) -> String {
        let printable = Set(Array(33...126) + Array(161...172) + Array(174...255))
        var shifted: [Int: Int] = [:]
        var next = 256
        for byte in 0...255 where !printable.contains(byte) {
            shifted[byte] = next
            next += 1
        }
        return String(String.UnicodeScalarView(bytes.map { Unicode.Scalar(UInt32(shifted[Int($0)] ?? Int($0)))! }))
    }

    public static func tokenizerJSON(vocab: [String: Int], added: [(Int, String, Bool)], decoder: [String: Any] = ["type": "ByteLevel"]) throws -> Data {
        try JSONSerialization.data(withJSONObject: [
            "model": ["type": "BPE", "vocab": vocab],
            "added_tokens": added.map { ["id": $0.0, "content": $0.1, "special": $0.2] },
            "decoder": decoder,
        ] as [String: Any])
    }

    public static let bytes: TokenBytes = {
        var vocab: [String: Int] = [:]
        for byte in 0...255 { vocab[spelled([UInt8(byte)])] = byte }
        let added: [(Int, String, Bool)] = [
            (Int(imStart), "<|im_start|>", true), (Int(imEnd), "<|im_end|>", true),
            (Int(thinkStart), "<think>", false), (Int(thinkEnd), "</think>", false),
            (Int(toolOpen), "<tool_call>", false), (Int(toolClose), "</tool_call>", false),
        ]
        return try! TokenBytes(tokenizerJSON: try! tokenizerJSON(vocab: vocab, added: added))
    }()

    public static let markers: [String: Int32] = [
        "<|im_start|>": imStart, "<|im_end|>": imEnd, "<think>": thinkStart, "</think>": thinkEnd,
        "<tool_call>": toolOpen, "</tool_call>": toolClose,
    ]

    public static let profile = OutputProfile(tags: [
        .init(open: "<think>", close: "</think>", kind: .thinking), .init(open: "<tool_call>", close: "</tool_call>", kind: .toolCall),
    ], toolDialect: "hermes-json")
}

public struct ByteDecoding: TokenDecoding {
    public init() {}
    public var tokenBytes: TokenBytes? { ByteVocab.bytes }
    public func tokenID(_ text: String) -> Int32? { ByteVocab.markers[text] }
    public func decode(_ ids: [Int32]) -> String {
        String(decoding: ByteVocab.bytes.bytes(of: ids.map(Int.init), skipSpecialTokens: false), as: UTF8.self)
    }
}

/// Renders like a hybrid Qwen3 template: `<|im_start|>role\ncontent<|im_end|>\n` per message (an assistant's tool
/// calls as Hermes JSON, a tool turn as `<tool_response>`), then the generation prompt, with an empty think block when
/// thinking is off. Tools are listed in the system turn.
public struct ByteRenderer: PromptRenderer {
    /// Like a template whose thinking can't be switched off (`toggle: none`): the generation prompt always ends
    /// with an open `<think>\n`.
    public var alwaysThinks = false

    public init(alwaysThinks: Bool = false) { self.alwaysThinks = alwaysThinks }
    public func render(_ messages: [ChatMessage], options: RenderOptions) throws -> RenderedPrompt {
        var out: [Int32] = []
        if !options.tools.isEmpty {
            let names = options.tools.map(\.name).joined(separator: ",")
            out += [ByteVocab.imStart] + ByteVocab.tokens("system\ntools: \(names)") + [ByteVocab.imEnd] + ByteVocab.tokens("\n")
        }
        for message in messages {
            var body = ByteVocab.tokens(message.content)
            for call in message.toolCalls {
                body += [ByteVocab.toolOpen] + ByteVocab.tokens("{\"name\": \"\(call.name)\", \"arguments\": \(call.arguments)}") + [ByteVocab.toolClose]
            }
            if message.role == .tool { body = ByteVocab.tokens("<tool_response>") + body + ByteVocab.tokens("</tool_response>") }
            out += [ByteVocab.imStart] + ByteVocab.tokens("\(message.role.rawValue)\n") + body + [ByteVocab.imEnd] + ByteVocab.tokens("\n")
        }
        guard options.addGenerationPrompt else { return RenderedPrompt(tokens: out) }
        out += [ByteVocab.imStart] + ByteVocab.tokens("assistant\n")
        if alwaysThinks { return RenderedPrompt(tokens: out + [ByteVocab.thinkStart] + ByteVocab.tokens("\n"), startsInThinking: true) }
        if options.thinking == false { out += [ByteVocab.thinkStart] + ByteVocab.tokens("\n\n") + [ByteVocab.thinkEnd] + ByteVocab.tokens("\n\n") }
        return RenderedPrompt(tokens: out)
    }
}

public enum ByteChat {
    /// The tokens after the last generation prompt (`<|im_start|>assistant\n`, and its empty think block if any).
    /// With `opened` (``ByteRenderer/alwaysThinks``), the prompt's own `<think>\n` isn't part of the reply.
    public static func replyStart(_ context: [Int32], opened alwaysOpened: Bool = false) -> Int? {
        let lead = [ByteVocab.imStart] + ByteVocab.tokens("assistant\n")
        var i = context.count - lead.count
        while i >= 0 {
            if Array(context[i..<(i + lead.count)]) == lead {
                var start = i + lead.count
                let empty = [ByteVocab.thinkStart] + ByteVocab.tokens("\n\n") + [ByteVocab.thinkEnd] + ByteVocab.tokens("\n\n")
                if context.count >= start + empty.count, Array(context[start..<(start + empty.count)]) == empty { start += empty.count }
                // A template that opened the block itself: the reply starts inside it.
                let opened = [ByteVocab.thinkStart] + ByteVocab.tokens("\n")
                if alwaysOpened, context.count >= start + opened.count, Array(context[start..<(start + opened.count)]) == opened { start += opened.count }
                return start
            }
            i -= 1
        }
        return nil
    }

    /// The prompt's text (bytes and markers), for scripts that branch on what they were asked.
    public static func text(_ prompt: [Int32]) -> String { ByteDecoding().decode(prompt) }

    /// A predictor that replies with `script(prompt)` and then `<|im_end|>`.
    public static func scripted(opened: Bool = false, _ script: @escaping @Sendable ([Int32]) -> [Int32]) -> @Sendable ([Int32]) -> Int32 {
        { context in
            guard let start = replyStart(context, opened: opened) else { return ByteVocab.imEnd }
            let reply = script(Array(context[..<start])) + [ByteVocab.imEnd]
            let n = context.count - start
            return reply[min(n, reply.count - 1)]
        }
    }

    public static func generator(
        capacity: Int = 4096, profile: OutputProfile = ByteVocab.profile, decoder: any TokenDecoding = ByteDecoding(),
        renderer: any PromptRenderer = ByteRenderer(), _ script: @escaping @Sendable ([Int32]) -> [Int32]
    ) async throws -> Generator {
        let opened = (renderer as? ByteRenderer)?.alwaysThinks ?? false
        let fake = FakeLM(FakeLM.Config(vocab: ByteVocab.headRows, predict: scripted(opened: opened, script)))
        let engine = try await DecoderEngine(
            step: fake, graph: fake.manifestGraph(maxContext: capacity), device: Device(platform: .macOS, architecture: "fake9", physicalMemory: 64 << 30),
            options: DecoderEngine.Options(capacity: capacity, memoryBudget: .bytes(1 << 30)))
        return Generator(engine: engine, decoder: decoder, renderer: renderer, stopTokens: [ByteVocab.imEnd],
                         profile: profile, gpu: .unadmitted)
    }
}

/// Prompts a scripted model saw, shared with the test (the fake model itself moves into the engine).
public final class PromptLog: Sendable {
    private let storage = Mutex<[[Int32]]>([])
    public init() {}
    public func append(_ prompt: [Int32]) { storage.withLock { $0.append(prompt) } }
    public var all: [[Int32]] { storage.withLock { $0 } }
}
