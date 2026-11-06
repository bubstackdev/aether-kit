public import AetherRuntime
public import Foundation

/// A tool the model may call (DESIGN.md §7): rendered into the chat template's `tools` in OpenAI function form.
public struct ToolDefinition: Sendable, Hashable {
    public var name: String
    public var description: String
    /// The arguments' JSON Schema (an object schema).
    public var parameters: JSONValue

    public init(name: String, description: String, parameters: JSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    /// `{"type": "function", "function": {"name", "description", "parameters"}}` for the template.
    var templateSpec: [String: any Sendable] {
        ["type": "function", "function": ["name": name, "description": description, "parameters": parameters.templateValue] as [String: any Sendable]]
    }
}

/// One tool call the model made: a tool name and its arguments as JSON object text.
public struct ToolCall: Sendable, Hashable {
    public var id: String
    public var name: String
    /// The arguments as JSON text (an object), exactly as the model wrote them.
    public var arguments: String

    public init(id: String = UUID().uuidString, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

/// How a model family writes tool calls (DESIGN.md §7), chosen by manifest `tokenizer.toolDialect` and never probed.
/// The Generator cuts each call's body out of the reply with the dialect's tags (``OutputProfile``); the dialect
/// parses the body into one or more calls.
public protocol ToolDialect: Sendable {
    var id: String { get }
    /// Parses one call body (the text between the dialect's tags): one call, or several.
    func parse(_ body: String) throws(AetherError) -> [ToolCall]
}

/// `hermes-json` (Qwen3, Granite): `<tool_call>\n{"name": "…", "arguments": {…}}\n</tool_call>`. A body may also
/// hold several objects (one after another, or a JSON array), as small models sometimes write parallel calls.
/// The arguments are kept as the model wrote them, byte for byte, so rendering the call back reproduces the
/// generated tokens and the next turn reuses them.
public struct HermesJSONDialect: ToolDialect {
    public var id: String { "hermes-json" }

    public init() {}

    public func parse(_ body: String) throws(AetherError) -> [ToolCall] {
        let bytes = Array(body.utf8)
        guard var values = JSONScan.values(bytes, 0..<bytes.count), !values.isEmpty else {
            throw .generation(.toolCall("a hermes-json call is a JSON object: \(body.prefix(200))"))
        }
        if values.count == 1, bytes[values[0].lowerBound] == UInt8(ascii: "["),
           let elements = JSONScan.values(bytes, (values[0].lowerBound + 1)..<(values[0].upperBound - 1)) {
            values = elements
        }
        return try values.map { range throws(AetherError) in try Self.call(bytes, range) }
    }

    private static func call(_ bytes: [UInt8], _ range: Range<Int>) throws(AetherError) -> ToolCall {
        let text = String(decoding: bytes[range], as: UTF8.self)
        guard bytes[range.lowerBound] == UInt8(ascii: "{"), let members = JSONScan.members(bytes, range),
              let nameRange = members["name"],
              let name = try? JSONDecoder().decode(String.self, from: Data(bytes[nameRange])), !name.isEmpty
        else { throw .generation(.toolCall("a hermes-json call is a JSON object with a name: \(text.prefix(200))")) }
        guard let span = members["arguments"] ?? members["parameters"] else { return ToolCall(name: name, arguments: "{}") }
        var arguments = String(decoding: bytes[span], as: UTF8.self)
        if bytes[span.lowerBound] == UInt8(ascii: "\""), let inner = try? JSONDecoder().decode(String.self, from: Data(bytes[span])) {
            // Some models write the arguments as a JSON string; it must still hold an object.
            arguments = inner
        }
        guard (try? JSONSerialization.jsonObject(with: Data(arguments.utf8))) is [String: Any] else {
            throw .generation(.toolCall("the arguments of '\(name)' are not a JSON object"))
        }
        return ToolCall(name: name, arguments: arguments)
    }
}

/// Spans of JSON text, without parsing values: enough to keep a model's bytes as it wrote them.
enum JSONScan {
    static func isSpace(_ b: UInt8) -> Bool { b == 0x20 || b == 0x0A || b == 0x0D || b == 0x09 }

    /// The end of the JSON value starting at `i` (no leading whitespace), or nil if it doesn't end within `end`.
    static func end(of bytes: [UInt8], at start: Int, before end: Int) -> Int? {
        guard start < end else { return nil }
        var i = start
        switch bytes[i] {
        case UInt8(ascii: "\""):
            i += 1
            while i < end {
                if bytes[i] == UInt8(ascii: "\\") { i += 2; continue }
                if bytes[i] == UInt8(ascii: "\"") { return i + 1 }
                i += 1
            }
            return nil
        case UInt8(ascii: "{"), UInt8(ascii: "["):
            var depth = 0
            while i < end {
                switch bytes[i] {
                case UInt8(ascii: "\""):
                    guard let next = self.end(of: bytes, at: i, before: end) else { return nil }
                    i = next
                    continue
                case UInt8(ascii: "{"), UInt8(ascii: "["): depth += 1
                case UInt8(ascii: "}"), UInt8(ascii: "]"):
                    depth -= 1
                    if depth == 0 { return i + 1 }
                default: break
                }
                i += 1
            }
            return nil
        default:
            while i < end, !isSpace(bytes[i]), ![UInt8(ascii: ","), UInt8(ascii: "}"), UInt8(ascii: "]"), UInt8(ascii: ":")].contains(bytes[i]) { i += 1 }
            return i > start ? i : nil
        }
    }

    /// The top-level values in `range`, separated by whitespace or commas; nil if any is cut off or malformed.
    static func values(_ bytes: [UInt8], _ range: Range<Int>) -> [Range<Int>]? {
        var out: [Range<Int>] = []
        var i = range.lowerBound
        while i < range.upperBound {
            if isSpace(bytes[i]) || bytes[i] == UInt8(ascii: ",") { i += 1; continue }
            guard let end = end(of: bytes, at: i, before: range.upperBound) else { return nil }
            guard (try? JSONSerialization.jsonObject(with: Data(bytes[i..<end]), options: .fragmentsAllowed)) != nil else { return nil }
            out.append(i..<end)
            i = end
        }
        return out
    }

    /// An object's members: each key (decoded) and its value's raw span.
    static func members(_ bytes: [UInt8], _ range: Range<Int>) -> [String: Range<Int>]? {
        var out: [String: Range<Int>] = [:]
        var i = range.lowerBound + 1
        let end = range.upperBound - 1  // the closing brace
        func skip() { while i < end, isSpace(bytes[i]) || bytes[i] == UInt8(ascii: ",") { i += 1 } }
        skip()
        while i < end {
            guard bytes[i] == UInt8(ascii: "\""), let keyEnd = self.end(of: bytes, at: i, before: end),
                  let key = try? JSONDecoder().decode(String.self, from: Data(bytes[i..<keyEnd])) else { return nil }
            i = keyEnd
            while i < end, isSpace(bytes[i]) { i += 1 }
            guard i < end, bytes[i] == UInt8(ascii: ":") else { return nil }
            i += 1
            while i < end, isSpace(bytes[i]) { i += 1 }
            guard let valueEnd = self.end(of: bytes, at: i, before: end) else { return nil }
            out[key] = i..<valueEnd
            i = valueEnd
            skip()
        }
        return out
    }
}

public enum ToolDialects {
    /// The dialect a manifest names. Only `hermes-json` is implemented; others (`qwen-xml`, `lfm3-xml`) are refused
    /// until a gate covers them, and a model without a dialect takes no tools.
    public static func make(_ id: String?) throws(AetherError) -> any ToolDialect {
        switch id {
        case "hermes-json"?: return HermesJSONDialect()
        case nil: throw .unsupportedOperation("this model declares no tool dialect (tokenizer.toolDialect), so it takes no tools")
        case let other?: throw .unsupportedOperation("tool dialect '\(other)' is not implemented; only hermes-json is")
        }
    }
}

extension JSONValue {
    /// The value as the template's (Jinja's) plain Swift values.
    var templateValue: any Sendable {
        switch self {
        case .null: return Optional<String>.none as any Sendable
        case .bool(let b): return b
        case .int(let i): return i
        case .double(let d): return d
        case .string(let s): return s
        case .array(let a): return a.map(\.templateValue)
        case .object(let o): return o.mapValues(\.templateValue)
        }
    }
}
