public import AetherRuntime
import Foundation

/// A generation schema compiled into the JSON a constrained model may write: one canonical layout (`{"key": value,
/// "key2": value}`, a space after each colon and comma, no other whitespace), properties in `x-order` (an optional
/// property, one not `required`, may be left out; its value is `null` only where its own schema allows `null`), strings with JSON escapes and valid UTF-8, integers and numbers within their
/// bounds (no exponents; numbers up to 9 integer and 9 fraction digits, integers up to 18 digits), and arrays within
/// their counts. Anything outside the subset FoundationModels' `@Generable` emits is refused when compiling.
struct JSONGrammar: Sendable {
    enum Node: Sendable, Hashable {
        /// Members in order, each its key and value; an optional member may be left out. Written `{`, then each
        /// member written as `"key": value` with `, ` before all but the first written, then `}`.
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
        /// `"key": `
        let key: [UInt8]
        let value: Int
        /// Not `required`: the member may be left out.
        let optional: Bool
    }

    private(set) var nodes: [Node] = []
    private(set) var root = 0

    static let numberScale: Int64 = 1_000_000_000

    init(schema: OutputSchema) throws {
        var definitions: [String: JSONValue] = [:]
        if case .object(let found)? = schema.document["$defs"] { definitions = found }
        var compiler = Compiler(definitions: definitions)
        root = try compiler.compile(.object(schema.document))
        nodes = compiler.nodes
        try Self.checkAlternatives(nodes)
    }

    /// `anyOf` must offer something, and must not reach itself through `anyOf`/`$ref` alone (a cycle that consumes no
    /// byte, which the matcher would expand forever).
    static func checkAlternatives(_ nodes: [Node]) throws {
        var state = [UInt8](repeating: 0, count: nodes.count)  // 0 new, 1 on the path, 2 done
        func visit(_ index: Int) throws {
            guard case .anyOf(let alternatives) = nodes[index] else { return }
            if state[index] == 2 { return }
            if state[index] == 1 { throw AetherError.generation(.unsupportedSchema("a reference cycle that writes nothing")) }
            guard !alternatives.isEmpty else { throw AetherError.generation(.unsupportedSchema("an empty anyOf")) }
            state[index] = 1
            for alternative in alternatives { try visit(alternative) }
            state[index] = 2
        }
        for index in nodes.indices { try visit(index) }
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

        /// The keywords each type holds its values to; any other constraint beside a type would be accepted and then
        /// not held to, so it is refused. (Annotations and `$defs` are allowed everywhere.)
        static let typeKeys: [String: Set<String>] = [
            "object": ["properties", "required", "x-order", "additionalProperties"],
            "string": ["enum", "const"],
            "integer": ["minimum", "maximum"],
            "number": ["minimum", "maximum"],
            "boolean": [],
            "null": [],
            "array": ["items", "minItems", "maxItems"],
        ]

        mutating func add(_ node: Node) -> Int {
            nodes.append(node)
            return nodes.count - 1
        }

        mutating func compile(_ value: JSONValue) throws -> Int {
            guard case .object(let schema) = value else { throw AetherError.generation(.unsupportedSchema("a schema node is a JSON object")) }
            if let unknown = schema.keys.first(where: { !Self.allowedKeys.contains($0) }) {
                throw AetherError.generation(.unsupportedSchema("the keyword \"\(unknown)\" isn't supported"))
            }
            // A `$ref` or `anyOf` node takes no constraint beside it: the grammar compiles the reference or the
            // alternatives alone, so a sibling (a `minimum`, a `type`) would be accepted and not held to.
            for combinator in ["$ref", "anyOf"] where schema[combinator] != nil {
                if let sibling = schema.keys.sorted().first(where: { !["$ref", "anyOf", "$defs", "title", "description"].contains($0) }) {
                    throw AetherError.generation(.unsupportedSchema("\"\(sibling)\" beside \"\(combinator)\" isn't supported"))
                }
                if schema["$ref"] != nil, schema["anyOf"] != nil {
                    throw AetherError.generation(.unsupportedSchema("\"$ref\" and \"anyOf\" in one node aren't supported"))
                }
            }
            if case .string(let reference)? = schema["$ref"] {
                let prefix = "#/$defs/"
                guard reference.hasPrefix(prefix), let target = definitions[String(reference.dropFirst(prefix.count))] else {
                    throw AetherError.generation(.unsupportedSchema("unresolved reference \(reference)"))
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
                guard case .string(let text) = constant, schema["type"].map({ $0 == .string("string") }) ?? true else {
                    throw AetherError.generation(.unsupportedSchema("const is supported on strings only"))
                }
                if let other = schema.keys.sorted().first(where: { !["const", "enum", "type", "title", "description", "$defs"].contains($0) }) {
                    throw AetherError.generation(.unsupportedSchema("\"\(other)\" beside \"const\" isn't supported"))
                }
                // `const` and `enum` together: the value must satisfy both.
                if schema["enum"] != nil, try !enumeration(schema["enum"]).contains(text) {
                    throw AetherError.generation(.unsupportedSchema("const \(Self.text(text)) isn't one of the enum's values"))
                }
                return add(.choice([Self.literal(text)]))
            }
            guard case .string(let type)? = schema["type"] else { throw AetherError.generation(.unsupportedSchema("a schema node needs a type")) }
            // Enforced on strings only: an `enum` on another type would be accepted and then not held to.
            if schema["enum"] != nil, type != "string" {
                throw AetherError.generation(.unsupportedSchema("enum is supported on strings only"))
            }
            if let accepted = Self.typeKeys[type],
               let other = schema.keys.sorted().first(where: { !accepted.contains($0) && !["type", "title", "description", "$defs"].contains($0) }) {
                throw AetherError.generation(.unsupportedSchema("\"\(other)\" isn't supported on the type \"\(type)\""))
            }
            switch type {
            case "object": return try object(schema)
            case "string":
                if schema["enum"] != nil {
                    let texts = try enumeration(schema["enum"]).map(Self.literal)
                    guard !texts.isEmpty else { throw AetherError.generation(.unsupportedSchema("an empty enum")) }
                    return add(.choice(Array(Set(texts)).sorted { $0.lexicographicallyPrecedes($1) }))
                }
                return add(.string)
            case "integer":
                let minimum = try limit(schema["minimum"], scale: 1, rounding: .up, kind: "integer")
                let maximum = try limit(schema["maximum"], scale: 1, rounding: .down, kind: "integer")
                if let minimum, let maximum, minimum > maximum { throw AetherError.generation(.unsupportedSchema("an empty integer range")) }
                return add(.integer(minimum: minimum, maximum: maximum))
            case "number":
                let scale = Double(JSONGrammar.numberScale)
                let minimum = try limit(schema["minimum"], scale: scale, rounding: .up, kind: "number")
                let maximum = try limit(schema["maximum"], scale: scale, rounding: .down, kind: "number")
                if let minimum, let maximum, minimum > maximum { throw AetherError.generation(.unsupportedSchema("an empty number range")) }
                return add(.number(minimum: minimum, maximum: maximum))
            case "boolean": return add(.choice([Array("false".utf8), Array("true".utf8)]))
            case "null": return null()
            case "array":
                guard let items = schema["items"] else { throw AetherError.generation(.unsupportedSchema("an array needs items")) }
                let minimum = try count(schema["minItems"], isMinimum: true) ?? 0
                let maximum = try count(schema["maxItems"], isMinimum: false)
                if let maximum, maximum < minimum { throw AetherError.generation(.unsupportedSchema("an empty count range")) }
                let item = try compile(items)
                return add(.array(item: item, minimum: minimum, maximum: maximum))
            default:
                throw AetherError.generation(.unsupportedSchema("the type \"\(type)\" isn't supported"))
            }
        }

        private mutating func object(_ schema: [String: JSONValue]) throws -> Int {
            if let additional = schema["additionalProperties"], additional != .bool(false) {
                throw AetherError.generation(.unsupportedSchema("objects must not allow additional properties"))
            }
            var properties: [String: JSONValue] = [:]
            if case .object(let found)? = schema["properties"] { properties = found }
            var required: Set<String> = []
            if let names = schema["required"] {
                guard case .array(let names) = names else { throw AetherError.generation(.unsupportedSchema("required must be an array of names")) }
                for name in names {
                    guard case .string(let name) = name else { throw AetherError.generation(.unsupportedSchema("required must be an array of names")) }
                    required.insert(name)
                }
            }
            // No additional properties: a required name the object doesn't declare could never be written.
            if let missing = required.subtracting(properties.keys).sorted().first {
                throw AetherError.generation(.unsupportedSchema(
                    "the required property \(Self.text(missing)) isn't declared, and additional properties aren't allowed"))
            }
            var order: [String]
            if case .array(let names)? = schema["x-order"] {
                order = names.compactMap { if case .string(let name) = $0 { name } else { nil } }
            } else {
                order = properties.keys.sorted()
            }
            guard Set(order) == Set(properties.keys), order.count == properties.count else {
                throw AetherError.generation(.unsupportedSchema("x-order must list every property once"))
            }
            let index = add(.object(members: []))
            var members: [Member] = []
            for name in order {
                // An optional property is left out, not written as `null` (unless its own schema allows null).
                let value = try compile(properties[name]!)
                members.append(Member(key: Self.literal(name) + Array(": ".utf8), value: value, optional: !required.contains(name)))
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
            guard let number = value.number, number.isFinite else { throw AetherError.generation(.unsupportedSchema("bounds must be numbers")) }
            return number
        }

        /// A bound in the grammar's fixed point (`scale` units per 1), rounded inward. Every value the grammar writes has
        /// a magnitude below 10^18 in those units (18 integer digits, or 9 + 9 for numbers), so a bound beyond that
        /// excludes nothing (`nil`) or everything (an empty range), and the rest convert to `Int64` exactly.
        private func limit(_ value: JSONValue?, scale: Double, rounding: FloatingPointRoundingRule, kind: String) throws -> Int64? {
            // An integer bound converts exactly: through Double it lost precision above 2^53 (9007199254740993 became
            // ...992 and the bound itself was refused).
            if case .int(let integer)? = value {
                let (scaled, overflow) = Int64(integer).multipliedReportingOverflow(by: Int64(scale))
                if !overflow, scaled.magnitude < 1_000_000_000_000_000_000 { return scaled }
            }
            guard let number = try bound(value) else { return nil }
            let producible = 1e18
            // Exact when the bound's shortest decimal form lands on the fixed-point grid: `number * scale` in binary
            // can fall just inside or outside it at large magnitudes and round the bound itself away.
            if let exact = Self.exactFixedPoint(number, scale: scale), Double(exact).magnitude < producible { return exact }
            let scaled = (number * scale).rounded(rounding)
            let isMinimum = rounding == .up
            if scaled >= producible {
                if isMinimum { throw AetherError.generation(.unsupportedSchema("an empty \(kind) range")) }
                return nil
            }
            if scaled <= -producible {
                if isMinimum { return nil }
                throw AetherError.generation(.unsupportedSchema("an empty \(kind) range"))
            }
            return Int64(scaled)
        }

        /// `number × scale` as an integer, when the decimal the number prints as is a whole multiple of 1/scale.
        static func exactFixedPoint(_ number: Double, scale: Double) -> Int64? {
            guard var decimal = Decimal(string: "\(number)", locale: Locale(identifier: "en_US_POSIX")) else { return nil }
            decimal *= Decimal(Int64(scale))
            var source = decimal, whole = Decimal()
            NSDecimalRound(&whole, &source, 0, .plain)
            guard whole == decimal else { return nil }
            return Int64(NSDecimalNumber(decimal: whole).stringValue)
        }

        private func count(_ value: JSONValue?, isMinimum: Bool) throws -> Int? {
            guard let number = try bound(value) else { return nil }
            guard number >= 0, number == number.rounded() else { throw AetherError.generation(.unsupportedSchema("counts must be whole numbers")) }
            // Past `Int` (`Int.max` itself arrives as 2^63): as a maximum it bounds nothing; no reply can reach it as a minimum.
            guard let count = Int(exactly: number) else {
                if isMinimum { throw AetherError.generation(.unsupportedSchema("an empty count range")) }
                return nil
            }
            return count
        }

        /// An `enum`'s string values.
        private func enumeration(_ value: JSONValue?) throws -> [String] {
            guard case .array(let options)? = value else { throw AetherError.generation(.unsupportedSchema("enum must be an array")) }
            return try options.map { option in
                guard case .string(let text) = option else { throw AetherError.generation(.unsupportedSchema("enum values must be strings")) }
                return text
            }
        }

        /// A name or value as it appears in an error: its JSON literal.
        static func text(_ value: String) -> String { String(decoding: literal(value), as: UTF8.self) }

        /// A JSON string literal's bytes.
        static func literal(_ text: String) -> [UInt8] {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .withoutEscapingSlashes
            // A String always encodes.
            return (try? encoder.encode(text)).map(Array.init) ?? Array(#""""#.utf8)
        }
    }
}
