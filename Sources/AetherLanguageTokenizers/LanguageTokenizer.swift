import AetherKit
import Foundation
import Tokenizers

/// A language model's local Hugging Face tokenizer and chat template (`tokenizer.json`, `tokenizer_config.json`,
/// and `chat_template.jinja` or `chat_template.json` when the checkpoint keeps its template there). Never
/// downloads files. Rendering uses the model's own template, so a hybrid Qwen3 renders its empty think block when
/// thinking is off, an instruct-only Qwen3 renders none, and Qwen3-VL renders its image placeholders. Exactness against
/// upstream is verified for Qwen3-family templates; other templates render through swift-jinja unverified.
public struct LanguageTokenizer: Sendable {
    private let tokenizer: any Tokenizer
    /// The model's chat template as upstream Transformers renders it (see `template(in:)`), nil when it has none.
    private let chatTemplate: String?

    /// Opens the tokenizer of a registered language bundle after `LanguageModelManifest.verifyTokenizer(in:)` (every
    /// registered tokenizer file when the registration lists `tokenizerFiles`; the model asset is not hashed), and
    /// verifies it again after loading.
    public static func open(bundle: URL) async throws -> Self {
        let manifest = try LanguageModelManifest.load(from: bundle)
        let directory = try manifest.verifyTokenizer(in: bundle)
        let tokenizer = try await open(directory: directory)
        _ = try manifest.verifyTokenizer(in: bundle)
        return tokenizer
    }

    /// Opens a local tokenizer folder, for example a checkpoint snapshot or an export's `tokenizer/`.
    public static func open(directory: URL) async throws -> Self {
        guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("tokenizer.json").path) else {
            throw AetherError.invalidModel("no tokenizer.json in \(directory.lastPathComponent)")
        }
        let tokenizer: any Tokenizer
        do { tokenizer = try await AutoTokenizer.from(modelFolder: directory) }
        catch { throw AetherError.invalidModel("tokenizer did not load: \(error)") }
        return Self(tokenizer: tokenizer, chatTemplate: try template(in: directory))
    }

    /// The chat template from the same sources, in the same order, as the swift-transformers loader:
    /// `chat_template.jinja`, then `chat_template.json`, then `tokenizer_config.json` (a string, or the list's
    /// "default"). Upstream renders with Jinja's default `keep_trailing_newline=False`, which drops one trailing
    /// newline from the template source; swift-jinja keeps it, which would add a stray "\n" after Qwen3-VL's
    /// generation prompt. So that newline is dropped here.
    package static func template(in directory: URL) throws -> String? {
        func json(_ name: String) throws -> [String: Any]? {
            let url = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            return try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        }
        var source: String?
        let jinja = directory.appendingPathComponent("chat_template.jinja")
        if FileManager.default.fileExists(atPath: jinja.path) {
            source = try String(contentsOf: jinja, encoding: .utf8)
        } else if let value = try json("chat_template.json")?["chat_template"] as? String {
            source = value
        } else if let value = try json("tokenizer_config.json")?["chat_template"] {
            source = value as? String
                ?? (value as? [[String: Any]])?.first { $0["name"] as? String == "default" }?["template"] as? String
        }
        guard let template = source else { return nil }
        // Scalars, not Characters: "\r\n" is one Character. Jinja treats \r\n, \r and \n alike as a newline.
        var scalars = template.unicodeScalars
        if scalars.last == "\n" { scalars.removeLast(); if scalars.last == "\r" { scalars.removeLast() } }
        else if scalars.last == "\r" { scalars.removeLast() }
        return String(scalars)
    }

    /// Plain text to token IDs, adding no special tokens.
    public func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }

    /// Token IDs to text. Special tokens (for example a trailing `<|im_end|>`) are dropped by default. Decode the whole
    /// reply (or a growing prefix): one token from a streaming `onToken` can be part of a multi-byte character.
    public func decode(_ tokenIDs: [Int], skipSpecialTokens: Bool = true) -> String {
        tokenizer.decode(tokens: tokenIDs, skipSpecialTokens: skipSpecialTokens)
    }

    /// The ID of one token, for example `<|im_end|>` to stop on.
    public func tokenID(_ token: String) -> Int? { tokenizer.convertTokenToId(token) }

    public var hasChatTemplate: Bool { chatTemplate != nil }

    /// Renders `messages` through the model's chat template with the generation prompt. `thinking` sets the
    /// template's `enable_thinking`; templates without that switch ignore it.
    public func render(_ messages: [LanguageChatMessage], thinking: Bool = false) throws -> LanguageChatPrompt {
        guard !messages.isEmpty else { throw AetherError.invalidRequest("a chat needs at least one message") }
        guard let chatTemplate else { throw AetherError.invalidModel("the tokenizer has no chat template") }
        let ids: [Int]
        do {
            ids = try tokenizer.applyChatTemplate(messages: messages.map(\.templateMessage), chatTemplate: .literal(chatTemplate),
                addGenerationPrompt: true, truncation: false, maxLength: nil, tools: nil,
                additionalContext: ["enable_thinking": thinking])
        } catch { throw AetherError.invalidRequest("chat template failed: \(error)") }
        return LanguageChatPrompt(tokenIDs: ids, imageCount: messages.reduce(0) { $0 + $1.imageCount })
    }
}
