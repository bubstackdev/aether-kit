public import AetherRuntime
import CryptoKit
import Foundation

/// One chat turn.
public struct ChatMessage: Sendable, Hashable {
    public enum Role: String, Sendable, Hashable {
        case system, user, assistant, tool
    }

    public var role: Role
    public var content: String

    public init(_ role: Role, _ content: String) {
        self.role = role
        self.content = content
    }

    public static func system(_ text: String) -> ChatMessage { ChatMessage(.system, text) }
    public static func user(_ text: String) -> ChatMessage { ChatMessage(.user, text) }
    public static func assistant(_ text: String) -> ChatMessage { ChatMessage(.assistant, text) }
}

public struct RenderOptions: Sendable, Hashable {
    /// nil leaves the template's own default (the manifest's `thinking.defaultOn` says what that is).
    public var thinking: Bool?
    public var addGenerationPrompt: Bool

    public init(thinking: Bool? = nil, addGenerationPrompt: Bool = true) {
        self.thinking = thinking
        self.addGenerationPrompt = addGenerationPrompt
    }
}

/// A prompt as tokens, and whether generation starts inside a thinking block (the prompt ends with its open tag).
public struct RenderedPrompt: Sendable, Hashable {
    public var tokens: [Int32]
    public var startsInThinking: Bool

    public init(tokens: [Int32], startsInThinking: Bool = false) {
        self.tokens = tokens
        self.startsInThinking = startsInThinking
    }
}

/// Renders a conversation to prompt tokens (DESIGN.md §7). `JinjaChatTemplate` is the default; family renderers
/// register by the manifest's `tokenizer.renderer` id (``PromptRenderers``), never by catalog id.
public protocol PromptRenderer: Sendable {
    func render(_ messages: [ChatMessage], options: RenderOptions) throws -> RenderedPrompt
}

/// The bundle's Jinja chat template through swift-transformers (token-identical to HF `apply_chat_template`).
///
/// Thinking (manifest `tokenizer.thinking`):
/// - `toggle: {templateVar}`: an explicit request sets that variable; nil leaves the template's default.
///   **Fallback:** when thinking is turned off and the rendered prompt does not end with a closed block (the template
///   ignored the flag), an empty block `open\n\nclose\n\n` is appended, so the model does not open its own.
/// - `toggle: {emitEmpty}`: thinking off appends that text unless the prompt already ends with it.
/// - `toggle: none`: thinking cannot be switched off; the request is ignored.
public struct JinjaChatTemplate: PromptRenderer {
    public let tokenizer: TokenizerAdapter
    public let template: String
    public let thinking: Manifest.Thinking?

    public init(tokenizer: TokenizerAdapter, template: String, thinking: Manifest.Thinking?) {
        self.tokenizer = tokenizer
        self.template = template
        self.thinking = thinking
    }

    public func render(_ messages: [ChatMessage], options: RenderOptions) throws -> RenderedPrompt {
        var context: [String: any Sendable] = [:]
        if case .templateVar(let name)? = thinking?.toggle, let want = options.thinking { context[name] = want }
        var ids = try tokenizer.applyChatTemplate(
            messages, template: template, context: context, addGenerationPrompt: options.addGenerationPrompt)
        if let thinking, options.thinking == false, options.addGenerationPrompt {
            let tail = Self.trimmed(tokenizer.decode(Array(ids.suffix(12))))
            switch thinking.toggle {
            case .templateVar?, nil:
                if !tail.hasSuffix(thinking.close) {
                    ids += tokenizer.encode(thinking.open + "\n\n" + thinking.close + "\n\n", addSpecialTokens: false)
                }
            case .emitEmpty(let text)?:
                if !tail.hasSuffix(Self.trimmed(text)) { ids += tokenizer.encode(text, addSpecialTokens: false) }
            case .none?:
                break
            }
        }
        let tail = Self.trimmed(tokenizer.decode(Array(ids.suffix(12))))
        let inside = thinking.map { tail.hasSuffix(Self.trimmed($0.open)) } ?? false
        return RenderedPrompt(tokens: ids, startsInThinking: inside)
    }

    static func trimmed(_ s: String) -> String {
        var t = Substring(s)
        while let last = t.last, last.isWhitespace { t = t.dropLast() }
        return String(t)
    }
}

/// Renderer selection by `tokenizer.renderer` (default `jinja`).
public enum PromptRenderers {
    public static func make(manifest: Manifest, bundle: InstalledBundle, tokenizer: TokenizerAdapter) throws -> any PromptRenderer {
        guard let block = manifest.tokenizer else {
            throw AetherError.manifestInvalid(field: "tokenizer", reason: "the manifest has no tokenizer")
        }
        switch block.renderer ?? "jinja" {
        case "jinja":
            guard let path = block.chatTemplate, let url = bundle.url(of: path) else {
                throw AetherError.manifestInvalid(field: "tokenizer.chatTemplate", reason: "the jinja renderer needs a chat template")
            }
            let data: Data
            do { data = try Data(contentsOf: url) } catch { throw AetherError.runtime(underlying: error) }
            if let expected = block.chatTemplateSha256 {
                let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                guard actual == expected else { throw AetherError.integrity(file: path.description, expected: expected, actual: actual) }
            }
            return JinjaChatTemplate(tokenizer: tokenizer, template: String(decoding: data, as: UTF8.self), thinking: block.thinking)
        case "gemma4-turns":
            // TODO(M3): Gemma 4 has no Jinja template; a turn renderer over `tokenizer.turnFormat`
            // (`<|turn>`/`<turn|>`, role `model`) needs an HF reference to gate against first.
            throw AetherError.unsupportedOperation("the gemma4-turns renderer is not implemented yet")
        case let other:
            throw AetherError.unsupportedOperation("prompt renderer '\(other)' is not implemented")
        }
    }
}
