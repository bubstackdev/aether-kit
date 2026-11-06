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
    /// An assistant turn's tool calls (rendered as the template's `tool_calls`).
    public var toolCalls: [ToolCall]
    /// A tool turn's tool name (`tool` role only).
    public var toolName: String?

    public init(_ role: Role, _ content: String, toolCalls: [ToolCall] = [], toolName: String? = nil) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolName = toolName
    }

    public static func system(_ text: String) -> ChatMessage { ChatMessage(.system, text) }
    public static func user(_ text: String) -> ChatMessage { ChatMessage(.user, text) }
    public static func assistant(_ text: String, toolCalls: [ToolCall] = []) -> ChatMessage {
        ChatMessage(.assistant, text, toolCalls: toolCalls)
    }
    /// A tool's output, handed back to the model.
    public static func tool(_ output: String, name: String? = nil) -> ChatMessage { ChatMessage(.tool, output, toolName: name) }

    /// The template's message: `role` and `content`, plus `tool_calls` (OpenAI form) and a tool turn's `name`. Tool
    /// arguments go in as the JSON text the model wrote when the template writes a string verbatim (Qwen and Granite
    /// test `arguments is string`), else as an object, so a template that pipes them through `tojson` doesn't
    /// encode a string twice.
    func templateMessage(argumentsAsText: Bool = true) -> [String: any Sendable] {
        var message: [String: any Sendable] = ["role": role.rawValue, "content": content]
        if !toolCalls.isEmpty {
            message["tool_calls"] = toolCalls.map { call -> [String: any Sendable] in
                var arguments: any Sendable = call.arguments
                if !argumentsAsText, let object = try? JSONDecoder().decode(JSONValue.self, from: Data(call.arguments.utf8)) {
                    arguments = object.templateValue
                }
                return ["id": call.id, "type": "function", "function": ["name": call.name, "arguments": arguments] as [String: any Sendable]]
            }
        }
        if let toolName { message["name"] = toolName }
        return message
    }
}

public struct RenderOptions: Sendable, Hashable {
    /// nil leaves the template's own default (the manifest's `thinking.defaultOn` says what that is).
    public var thinking: Bool?
    public var addGenerationPrompt: Bool
    /// Tools the model may call, rendered into the template's `tools`.
    public var tools: [ToolDefinition]

    public init(thinking: Bool? = nil, addGenerationPrompt: Bool = true, tools: [ToolDefinition] = []) {
        self.thinking = thinking
        self.addGenerationPrompt = addGenerationPrompt
        self.tools = tools
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
            messages, template: template, context: context, addGenerationPrompt: options.addGenerationPrompt, tools: options.tools,
            toolArgumentsAsText: template.contains("arguments is string"))
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
    /// The bundle's chat template source (`tokenizer.chatTemplate`), checked against `tokenizer.chatTemplateSha256`
    /// when the manifest pins it.
    public static func chatTemplate(manifest: Manifest, bundle: InstalledBundle) throws -> String {
        guard let block = manifest.tokenizer, let path = block.chatTemplate, let url = bundle.url(of: path) else {
            throw AetherError.manifestInvalid(field: "tokenizer.chatTemplate", reason: "the renderer needs a chat template")
        }
        let data: Data
        do { data = try Data(contentsOf: url) } catch { throw AetherError.runtime(underlying: error) }
        if let expected = block.chatTemplateSha256 {
            let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard actual == expected else { throw AetherError.integrity(file: path.description, expected: expected, actual: actual) }
        }
        // The source goes in verbatim: swift-jinja drops one trailing newline as Jinja's default
        // `keep_trailing_newline=False` does (0.0.x had to strip it itself; TokenizerParityTests pins it).
        return String(decoding: data, as: UTF8.self)
    }

    public static func make(manifest: Manifest, bundle: InstalledBundle, tokenizer: TokenizerAdapter) throws -> any PromptRenderer {
        guard let block = manifest.tokenizer else {
            throw AetherError.manifestInvalid(field: "tokenizer", reason: "the manifest has no tokenizer")
        }
        switch block.renderer ?? "jinja" {
        case "jinja":
            return JinjaChatTemplate(tokenizer: tokenizer, template: try chatTemplate(manifest: manifest, bundle: bundle),
                                     thinking: block.thinking)
        case "gemma4-turns":
            // TODO(M3): Gemma 4 has no Jinja template; a turn renderer over `tokenizer.turnFormat`
            // (`<|turn>`/`<turn|>`, role `model`) needs an HF reference to gate against first.
            throw AetherError.unsupportedOperation("the gemma4-turns renderer is not implemented yet")
        case let other:
            throw AetherError.unsupportedOperation("prompt renderer '\(other)' is not implemented")
        }
    }
}
