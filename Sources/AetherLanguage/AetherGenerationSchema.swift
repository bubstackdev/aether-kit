import Foundation

/// A type a language model can generate: its JSON Schema, and decoding from the JSON the constrained model writes.
/// `@AetherGenerable` (in `AetherLanguageMacros`) writes the schema; it can also be written by hand.
public protocol AetherGenerable: Decodable, Sendable {
    static var generationSchema: AetherGenerationSchema { get }
}

/// A JSON Schema in the subset FoundationModels' `@Generable` emits, which the JSON-schema constraint decodes into:
/// objects with ordered properties (`x-order`), `string` (with `enum` or `const`), `integer` and `number` (with
/// `minimum` and `maximum`), `boolean`, `array` (with `minItems` and `maxItems`), `anyOf`, and definitions referenced
/// with `$ref`. Built for a Swift type, it matches what FoundationModels emits for the same type, so one constraint
/// serves both FoundationModels' and Kit's generable types.
public struct AetherGenerationSchema: Sendable, Hashable, Codable, CustomStringConvertible {
    /// The schema document: the root type, with every nested definition hoisted into its `$defs`.
    var document: [String: JSONValue]
    /// Nested objects and unions are referenced from a parent's `$defs`; primitives and string enumerations are inline.
    private let referenced: Bool

    init(document: [String: JSONValue], referenced: Bool) {
        self.document = document
        self.referenced = referenced
    }

    /// Reads a schema document (for example FoundationModels' `GenerationSchema`, JSON-encoded).
    public init(json: Data) throws {
        guard case .object(let document) = try JSONDecoder().decode(JSONValue.self, from: json) else {
            throw AetherLanguageError.unsupportedSchema("a schema is a JSON object")
        }
        self.init(document: document, referenced: document["anyOf"] != nil || document["type"] == .string("object"))
    }

    public init(from decoder: any Decoder) throws {
        guard case .object(let document) = try JSONValue(from: decoder) else {
            throw AetherLanguageError.unsupportedSchema("a schema is a JSON object")
        }
        self.init(document: document, referenced: document["anyOf"] != nil || document["type"] == .string("object"))
    }

    public func encode(to encoder: any Encoder) throws { try JSONValue.object(document).encode(to: encoder) }

    /// The schema as JSON text, keys sorted.
    public var json: String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try! encoder.encode(JSONValue.object(document)), as: UTF8.self)
    }

    public var description: String { json }

    /// The type's name (`title`), when it has one.
    public var name: String? {
        if case .string(let title)? = document["title"] { title } else { nil }
    }

    // MARK: Building schemas for Swift types

    /// One stored property of a generable struct, in declaration order.
    public struct Property: Sendable {
        let name: String
        let node: JSONValue
        let definitions: [String: JSONValue]
        let optional: Bool

        public init<Value: AetherGenerable>(name: String, description: String? = nil, type: Value.Type,
                                            guides: [AetherGenerationGuide<Value>] = []) {
            self.name = name
            var (node, definitions) = Value.generationSchema.embedded()
            for guide in guides { guide.apply(&node, &definitions) }
            if let description { node = node.setting("description", .string(description)) }
            self.node = node
            self.definitions = definitions
            optional = false
        }

        public init<Value: AetherGenerable>(name: String, description: String? = nil, type: Value?.Type,
                                            guides: [AetherGenerationGuide<Value>] = []) {
            self.init(name: name, description: description, type: Value.self, guides: guides)
            self = Property(name: self.name, node: node, definitions: definitions, optional: true)
        }

        private init(name: String, node: JSONValue, definitions: [String: JSONValue], optional: Bool) {
            self.name = name
            self.node = node
            self.definitions = definitions
            self.optional = optional
        }
    }

    /// A struct: an object whose properties appear in this order (`x-order`); optional properties aren't `required`.
    public static func object(name: String, description: String? = nil, properties: [Property]) -> Self {
        var document: [String: JSONValue] = [
            "type": .string("object"), "title": .string(name), "additionalProperties": .bool(false),
            "properties": .object(Dictionary(properties.map { ($0.name, $0.node) }, uniquingKeysWith: { $1 })),
            "required": .array(properties.filter { !$0.optional }.map { .string($0.name) }),
            "x-order": .array(properties.map { .string($0.name) }),
        ]
        if let description { document["description"] = .string(description) }
        let definitions = properties.reduce(into: [String: JSONValue]()) { $0.merge($1.definitions) { $1 } }
        if !definitions.isEmpty { document["$defs"] = .object(definitions) }
        return Self(document: document, referenced: true)
    }

    /// An enum without associated values: one of the case names, as a string.
    public static func enumeration(name: String, description: String? = nil, cases: [String]) -> Self {
        var document: [String: JSONValue] = ["type": .string("string"), "title": .string(name), "enum": .array(cases.map { .string($0) })]
        if let description { document["description"] = .string(description) }
        return Self(document: document, referenced: false)
    }

    /// One case of an enum with associated values: its name and its values (a lone unlabelled value is `value`).
    public struct Case: Sendable {
        let name: String
        let values: [Property]
        public init(name: String, values: [Property] = []) {
            self.name = name
            self.values = values
        }
    }

    /// An enum with associated values: any of its cases, each an object discriminated by `type`.
    public static func union(name: String, description: String? = nil, cases: [Case]) -> Self {
        var definitions: [String: JSONValue] = [:]
        var alternatives: [JSONValue] = []
        for item in cases {
            let caseName = "Discriminated" + item.name.prefix(1).uppercased() + item.name.dropFirst()
            let discriminant = Property(constant: item.name)
            let object = Self.object(name: caseName, properties: [discriminant] + item.values)
            var body = object.document
            if case .object(let nested)? = body.removeValue(forKey: "$defs") { definitions.merge(nested) { $1 } }
            definitions[caseName] = .object(body)
            alternatives.append(.object(["$ref": .string("#/$defs/\(caseName)")]))
        }
        var document: [String: JSONValue] = ["title": .string(name), "anyOf": .array(alternatives)]
        if let description { document["description"] = .string(description) }
        if !definitions.isEmpty { document["$defs"] = .object(definitions) }
        return Self(document: document, referenced: true)
    }

    /// The schema as a property's or element's node: a `$ref` to a definition for objects and unions (their own
    /// definitions hoisted alongside), or the node itself, without its type's title and description, for the rest.
    func embedded() -> (node: JSONValue, definitions: [String: JSONValue]) {
        var body = document
        var definitions: [String: JSONValue] = [:]
        if case .object(let nested)? = body.removeValue(forKey: "$defs") { definitions = nested }
        guard referenced, let name else {
            body.removeValue(forKey: "title")
            body.removeValue(forKey: "description")
            return (.object(body), definitions)
        }
        definitions[name] = .object(body)
        return (.object(["$ref": .string("#/$defs/\(name)")]), definitions)
    }

    /// Primitive schemas.
    static func primitive(_ type: String) -> Self { Self(document: ["type": .string(type)], referenced: false) }
}

extension AetherGenerationSchema.Property {
    /// A union case's discriminant: `"type": {"const": name}`.
    fileprivate init(constant: String) {
        self.init(name: "type", type: String.self, guides: [.constant(constant)])
    }
}

/// Narrows a property's or element's schema, as FoundationModels' `GenerationGuide` does: ranges, counts, choices.
public struct AetherGenerationGuide<Value>: Sendable {
    let apply: @Sendable (inout JSONValue, inout [String: JSONValue]) -> Void

    init(_ apply: @escaping @Sendable (inout JSONValue, inout [String: JSONValue]) -> Void) { self.apply = apply }

    fileprivate static func setting(_ pairs: [(String, JSONValue)]) -> Self {
        Self { node, _ in for (key, value) in pairs { node = node.setting(key, value) } }
    }
}

extension AetherGenerationGuide where Value == String {
    /// Exactly this string.
    public static func constant(_ value: String) -> Self { Self { node, _ in node = .object(["const": .string(value)]) } }
    /// One of these strings.
    public static func anyOf(_ values: [String]) -> Self { setting([("enum", .array(values.map { .string($0) }))]) }
}

extension AetherGenerationGuide where Value == Int {
    public static func minimum(_ value: Int) -> Self { setting([("minimum", .number(Double(value)))]) }
    public static func maximum(_ value: Int) -> Self { setting([("maximum", .number(Double(value)))]) }
    public static func range(_ range: ClosedRange<Int>) -> Self {
        setting([("minimum", .number(Double(range.lowerBound))), ("maximum", .number(Double(range.upperBound)))])
    }
}

extension AetherGenerationGuide where Value == Double {
    public static func minimum(_ value: Double) -> Self { setting([("minimum", .number(value))]) }
    public static func maximum(_ value: Double) -> Self { setting([("maximum", .number(value))]) }
    public static func range(_ range: ClosedRange<Double>) -> Self {
        setting([("minimum", .number(range.lowerBound)), ("maximum", .number(range.upperBound))])
    }
}

extension AetherGenerationGuide where Value == Float {
    public static func minimum(_ value: Float) -> Self { setting([("minimum", .number(Double(value)))]) }
    public static func maximum(_ value: Float) -> Self { setting([("maximum", .number(Double(value)))]) }
    public static func range(_ range: ClosedRange<Float>) -> Self {
        setting([("minimum", .number(Double(range.lowerBound))), ("maximum", .number(Double(range.upperBound)))])
    }
}

extension AetherGenerationGuide {
    public static func count<Element>(_ count: Int) -> Self where Value == [Element] {
        setting([("minItems", .number(Double(count))), ("maxItems", .number(Double(count)))])
    }
    public static func count<Element>(_ range: ClosedRange<Int>) -> Self where Value == [Element] {
        setting([("minItems", .number(Double(range.lowerBound))), ("maxItems", .number(Double(range.upperBound)))])
    }
    public static func minimumCount<Element>(_ count: Int) -> Self where Value == [Element] {
        setting([("minItems", .number(Double(count)))])
    }
    public static func maximumCount<Element>(_ count: Int) -> Self where Value == [Element] {
        setting([("maxItems", .number(Double(count)))])
    }

    /// Applies `guide` to every element.
    public static func element<Element>(_ guide: AetherGenerationGuide<Element>) -> Self where Value == [Element] {
        Self { node, definitions in
            guard case .object(var array) = node, var items = array["items"] else { return }
            guide.apply(&items, &definitions)
            array["items"] = items
            node = .object(array)
        }
    }
}

// So an `@AetherGuide(.count(…))` attribute type-checks without its property's element type, as FoundationModels'
// `@Guide` does: the attribute resolves to `[Never]`. `@AetherGenerable` copies the guides into code where the element
// type is known, so these are never applied.
extension AetherGenerationGuide where Value == [Never] {
    @_documentation(visibility: private) public static func count(_ count: Int) -> Self { Self { _, _ in } }
    @_documentation(visibility: private) public static func count(_ range: ClosedRange<Int>) -> Self { Self { _, _ in } }
    @_documentation(visibility: private) public static func minimumCount(_ count: Int) -> Self { Self { _, _ in } }
    @_documentation(visibility: private) public static func maximumCount(_ count: Int) -> Self { Self { _, _ in } }
}

extension String: AetherGenerable { public static var generationSchema: AetherGenerationSchema { .primitive("string") } }
extension Int: AetherGenerable { public static var generationSchema: AetherGenerationSchema { .primitive("integer") } }
extension Double: AetherGenerable { public static var generationSchema: AetherGenerationSchema { .primitive("number") } }
extension Float: AetherGenerable { public static var generationSchema: AetherGenerationSchema { .primitive("number") } }
extension Bool: AetherGenerable { public static var generationSchema: AetherGenerationSchema { .primitive("boolean") } }

extension Array: AetherGenerable where Element: AetherGenerable {
    public static var generationSchema: AetherGenerationSchema {
        let (items, definitions) = Element.generationSchema.embedded()
        var document: [String: JSONValue] = ["type": .string("array"), "items": items]
        if !definitions.isEmpty { document["$defs"] = .object(definitions) }
        return AetherGenerationSchema(document: document, referenced: false)
    }
}

/// A JSON value, for schema documents.
package enum JSONValue: Sendable, Hashable, Codable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    package init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([JSONValue].self) { self = .array(value) }
        else { self = .object(try container.decode([String: JSONValue].self)) }
    }

    package func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .number(let value):
            if value == value.rounded(), abs(value) < 1e15 { try container.encode(Int(value)) } else { try container.encode(value) }
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }

    func setting(_ key: String, _ value: JSONValue) -> JSONValue {
        guard case .object(var object) = self else { return self }
        object[key] = value
        return .object(object)
    }
}

/// A coding key from a string, for the decoders `@AetherGenerable` writes for enums with associated values.
public struct AetherGenerableCodingKey: CodingKey, ExpressibleByStringLiteral, Sendable {
    public let stringValue: String
    public var intValue: Int? { nil }
    public init(stringValue: String) { self.stringValue = stringValue }
    public init?(intValue: Int) { nil }
    public init(stringLiteral value: String) { stringValue = value }
}
