import Foundation

/// A generation schema compiled into the JSON a constrained model may write: one canonical layout (`{"key": value,
/// "key2": value}`, a space after each colon and comma, no other whitespace), every property in `x-order` (an optional
/// property's value may be `null`), strings with JSON escapes and valid UTF-8, integers and numbers within their
/// bounds (no exponents; numbers up to 9 integer and 9 fraction digits, integers up to 18 digits), and arrays within
/// their counts. Anything outside the subset FoundationModels' `@Generable` emits is refused when compiling.
struct LanguageJSONGrammar: Sendable {
    enum Node: Sendable, Hashable {
        /// Members in order: the text before each value (`{"key": ` for the first, `, "key": ` after), then the value.
        case object(members: [Member])
        case string
        /// Exactly one of these byte strings (JSON string literals, `true`, `false`, `null`). None is a prefix of another.
        case choice([[UInt8]])
        case integer(minimum: Int64?, maximum: Int64?)
        /// Bounds in billionths (nine fraction digits).
        case number(minimum: Int64?, maximum: Int64?)
        case array(item: Int, minimum: Int, maximum: Int?)
        case anyOf([Int])
    }

    struct Member: Sendable, Hashable {
        let text: [UInt8]
        let value: Int
    }

    private(set) var nodes: [Node] = []
    private(set) var root = 0

    static let numberScale: Int64 = 1_000_000_000

    init(schema: AetherGenerationSchema) throws {
        var definitions: [String: JSONValue] = [:]
        if case .object(let found)? = schema.document["$defs"] { definitions = found }
        var compiler = Compiler(definitions: definitions)
        root = try compiler.compile(.object(schema.document))
        nodes = compiler.nodes
    }

    private struct Compiler {
        let definitions: [String: JSONValue]
        var nodes: [Node] = []
        var compiledDefinitions: [String: Int] = [:]
        var nullNode: Int?

        static let allowedKeys: Set<String> = [
            "type", "properties", "required", "x-order", "additionalProperties", "title", "description", "enum", "const",
            "minimum", "maximum", "items", "minItems", "maxItems", "anyOf", "$ref", "$defs",
        ]

        mutating func add(_ node: Node) -> Int {
            nodes.append(node)
            return nodes.count - 1
        }

        mutating func compile(_ value: JSONValue) throws -> Int {
            guard case .object(let schema) = value else { throw AetherLanguageError.unsupportedSchema("a schema node is a JSON object") }
            if let unknown = schema.keys.first(where: { !Self.allowedKeys.contains($0) }) {
                throw AetherLanguageError.unsupportedSchema("the keyword \"\(unknown)\" isn't supported")
            }
            if case .string(let reference)? = schema["$ref"] {
                let prefix = "#/$defs/"
                guard reference.hasPrefix(prefix), let target = definitions[String(reference.dropFirst(prefix.count))] else {
                    throw AetherLanguageError.unsupportedSchema("unresolved reference \(reference)")
                }
                let name = String(reference.dropFirst(prefix.count))
                if let index = compiledDefinitions[name] { return index }
                // Reserve the index first, so a recursive type refers to itself.
                let index = add(.anyOf([]))
                compiledDefinitions[name] = index
                let body = try compile(target)
                nodes[index] = .anyOf([body])
                return index
            }
            if case .array(let alternatives)? = schema["anyOf"] {
                return add(.anyOf(try alternatives.map { try compile($0) }))
            }
            if let constant = schema["const"] {
                guard case .string(let text) = constant else { throw AetherLanguageError.unsupportedSchema("const must be a string") }
                return add(.choice([Self.literal(text)]))
            }
            guard case .string(let type)? = schema["type"] else { throw AetherLanguageError.unsupportedSchema("a schema node needs a type") }
            switch type {
            case "object": return try object(schema)
            case "string":
                if case .array(let options)? = schema["enum"] {
                    let texts = try options.map { option -> [UInt8] in
                        guard case .string(let text) = option else { throw AetherLanguageError.unsupportedSchema("enum values must be strings") }
                        return Self.literal(text)
                    }
                    guard !texts.isEmpty else { throw AetherLanguageError.unsupportedSchema("an empty enum") }
                    return add(.choice(Array(Set(texts)).sorted { $0.lexicographicallyPrecedes($1) }))
                }
                return add(.string)
            case "integer":
                let minimum = try bound(schema["minimum"]).map { Int64(($0).rounded(.up)) }
                let maximum = try bound(schema["maximum"]).map { Int64(($0).rounded(.down)) }
                if let minimum, let maximum, minimum > maximum { throw AetherLanguageError.unsupportedSchema("an empty integer range") }
                return add(.integer(minimum: minimum, maximum: maximum))
            case "number":
                let scale = Double(LanguageJSONGrammar.numberScale)
                let minimum = try bound(schema["minimum"]).map { Int64(($0 * scale).rounded(.up)) }
                let maximum = try bound(schema["maximum"]).map { Int64(($0 * scale).rounded(.down)) }
                if let minimum, let maximum, minimum > maximum { throw AetherLanguageError.unsupportedSchema("an empty number range") }
                return add(.number(minimum: minimum, maximum: maximum))
            case "boolean": return add(.choice([Array("false".utf8), Array("true".utf8)]))
            case "null": return null()
            case "array":
                guard let items = schema["items"] else { throw AetherLanguageError.unsupportedSchema("an array needs items") }
                let minimum = try count(schema["minItems"]) ?? 0
                let maximum = try count(schema["maxItems"])
                if let maximum, maximum < minimum { throw AetherLanguageError.unsupportedSchema("an empty count range") }
                let item = try compile(items)
                return add(.array(item: item, minimum: minimum, maximum: maximum))
            default:
                throw AetherLanguageError.unsupportedSchema("the type \"\(type)\" isn't supported")
            }
        }

        private mutating func object(_ schema: [String: JSONValue]) throws -> Int {
            if let additional = schema["additionalProperties"], additional != .bool(false) {
                throw AetherLanguageError.unsupportedSchema("objects must not allow additional properties")
            }
            var properties: [String: JSONValue] = [:]
            if case .object(let found)? = schema["properties"] { properties = found }
            var required: Set<String> = []
            if case .array(let names)? = schema["required"] {
                required = Set(names.compactMap { if case .string(let name) = $0 { name } else { nil } })
            }
            var order: [String]
            if case .array(let names)? = schema["x-order"] {
                order = names.compactMap { if case .string(let name) = $0 { name } else { nil } }
            } else {
                order = properties.keys.sorted()
            }
            guard Set(order) == Set(properties.keys), order.count == properties.count else {
                throw AetherLanguageError.unsupportedSchema("x-order must list every property once")
            }
            let index = add(.object(members: []))
            var members: [Member] = []
            for (position, name) in order.enumerated() {
                var value = try compile(properties[name]!)
                if !required.contains(name) { value = add(.anyOf([value, null()])) }
                members.append(Member(text: Array((position == 0 ? "{" : ", ").utf8) + Self.literal(name) + Array(": ".utf8), value: value))
            }
            nodes[index] = .object(members: members)
            return index
        }

        private mutating func null() -> Int {
            if let nullNode { return nullNode }
            let index = add(.choice([Array("null".utf8)]))
            nullNode = index
            return index
        }

        private func bound(_ value: JSONValue?) throws -> Double? {
            guard let value else { return nil }
            guard case .number(let number) = value, number.isFinite else { throw AetherLanguageError.unsupportedSchema("bounds must be numbers") }
            return number
        }

        private func count(_ value: JSONValue?) throws -> Int? {
            guard let number = try bound(value) else { return nil }
            guard number >= 0, number == number.rounded() else { throw AetherLanguageError.unsupportedSchema("counts must be whole numbers") }
            return Int(number)
        }

        /// A JSON string literal's bytes.
        static func literal(_ text: String) -> [UInt8] {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .withoutEscapingSlashes
            return Array(try! encoder.encode(text))
        }
    }
}
