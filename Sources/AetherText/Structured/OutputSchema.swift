public import AetherRuntime
public import Foundation

/// A type a model can generate as JSON (structured output): its schema, and decoding from the JSON the constrained
/// model writes. Write the schema with ``OutputSchema``'s builders; FoundationModels' `@Generable` types go through
/// `AetherFoundationModels`, which hands their own schema over.
public protocol StructuredOutput: Decodable, Sendable {
    static var outputSchema: OutputSchema { get }
}

/// A JSON Schema in the subset FoundationModels' `@Generable` emits, which the JSON-schema constraint compiles:
/// objects with ordered properties (`x-order`), `string` (with `enum` or `const`), `integer` and `number` (with
/// `minimum` and `maximum`), `boolean`, `array` (with `minItems` and `maxItems`), `anyOf`, and definitions under
/// `$defs` referenced with `$ref`. Built for a Swift type, it matches what FoundationModels emits for the same type,
/// so one constraint serves both. Ported from 0.0.x `AetherGenerationSchema`.
public struct OutputSchema: Sendable, Hashable, Codable, CustomStringConvertible {
    /// The schema document: the root type, with every nested definition hoisted into its `$defs`.
    public internal(set) var document: [String: JSONValue]
    /// Nested objects and unions are referenced from a parent's `$defs`; primitives and string enumerations are inline.
    private let referenced: Bool

    init(document: [String: JSONValue], referenced: Bool) {
        self.document = document
        self.referenced = referenced
    }

    /// Reads a schema document (for example FoundationModels' `GenerationSchema`, JSON-encoded).
    public init(json: Data) throws(AetherError) {
        let value: JSONValue
        do { value = try JSONDecoder().decode(JSONValue.self, from: json) } catch {
            throw .generation(.unsupportedSchema("not JSON: \(error)"))
        }
        guard case .object(let document) = value else { throw .generation(.unsupportedSchema("a schema is a JSON object")) }
        self.init(document: document, referenced: document["anyOf"] != nil || document["type"] == .string("object"))
    }

    public init(from decoder: any Decoder) throws {
        guard case .object(let document) = try JSONValue(from: decoder) else {
            throw AetherError.generation(.unsupportedSchema("a schema is a JSON object"))
        }
        self.init(document: document, referenced: document["anyOf"] != nil || document["type"] == .string("object"))
    }

    public func encode(to encoder: any Encoder) throws { try JSONValue.object(document).encode(to: encoder) }

    /// The schema as JSON text, keys sorted.
    public var json: String { JSONValue.object(document).compactText }

    public var description: String { json }

    /// The type's name (`title`), when it has one.
    public var name: String? { document["title"]?.string }

    // MARK: Building schemas for Swift types

    /// One stored property of a struct, in declaration order.
    public struct Property: Sendable {
        let name: String
        let node: JSONValue
        let definitions: [String: JSONValue]
        let optional: Bool

        public init<Value: StructuredOutput>(name: String, description: String? = nil, type: Value.Type, guides: [OutputGuide<Value>] = []) {
            self.name = name
            var (node, definitions) = Value.outputSchema.embedded()
            for guide in guides { guide.apply(&node, &definitions) }
            if let description { node = node.setting("description", .string(description)) }
            self.node = node
            self.definitions = definitions
            optional = false
        }

        public init<Value: StructuredOutput>(name: String, description: String? = nil, type: Value?.Type, guides: [OutputGuide<Value>] = []) {
            self.init(name: name, description: description, type: Value.self, guides: guides)
            self = Property(name: self.name, node: node, definitions: definitions, optional: true)
        }

        private init(name: String, node: JSONValue, definitions: [String: JSONValue], optional: Bool) {
            self.name = name
            self.node = node
            self.definitions = definitions
            self.optional = optional
        }

        /// A union case's discriminant: `"type": {"const": name}`.
        fileprivate init(constant: String) {
            self.init(name: "type", type: String.self, guides: [.constant(constant)])
        }
    }

    /// A struct: an object whose properties appear in this order (`x-order`); optional properties aren't `required`.
    public static func object(name: String, description: String? = nil, properties: [Property]) -> Self {
        var definitions: [String: JSONValue] = [:]
        let nodes = properties.map { property in
            var nodes = [property.node]
            merge(property.definitions, into: &definitions, referencedFrom: &nodes)
            return (property.name, nodes[0])
        }
        var document: [String: JSONValue] = [
            "type": .string("object"), "title": .string(name), "additionalProperties": .bool(false),
            "properties": .object(Dictionary(nodes, uniquingKeysWith: { $1 })),
            "required": .array(properties.filter { !$0.optional }.map { .string($0.name) }),
            "x-order": .array(properties.map { .string($0.name) }),
        ]
        if let description { document["description"] = .string(description) }
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
            let (reference, caseDefinitions) = Self.object(name: caseName, properties: [discriminant] + item.values).embedded()
            var references = [reference]
            merge(caseDefinitions, into: &definitions, referencedFrom: &references)
            alternatives.append(references[0])
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
        var bodies = [JSONValue.object(body)]
        var hoisted = [name: bodies[0]]
        Self.merge(definitions, into: &hoisted, referencedFrom: &bodies)
        hoisted[name] = bodies[0]
        return (.object(["$ref": .string("#/$defs/\(name)")]), hoisted)
    }

    /// Adds `incoming` definitions to `definitions`. A name `definitions` already holds with a different schema (two
    /// enums with a `value` case are both `DiscriminatedValue`; two types can share a name) is renamed, in `incoming`
    /// and in `nodes`, which refer to it, so neither overwrites the other. Names that don't collide are kept, so the
    /// schema still matches FoundationModels' for the same types.
    private static func merge(_ incoming: [String: JSONValue], into definitions: inout [String: JSONValue],
                              referencedFrom nodes: inout [JSONValue]) {
        var renamed: [String: String] = [:]
        var taken = Set(definitions.keys).union(incoming.keys)
        // A definition that refers to a renamed one differs from its namesake too, even with identical text.
        var changed = true
        while changed {
            changed = false
            for (name, schema) in incoming.sorted(by: { $0.key < $1.key }) where renamed[name] == nil {
                guard let existing = definitions[name],
                      existing != schema || !schema.references.isDisjoint(with: renamed.keys) else { continue }
                var suffix = 2
                while taken.contains("\(name)\(suffix)") { suffix += 1 }
                renamed[name] = "\(name)\(suffix)"
                taken.insert("\(name)\(suffix)")
                changed = true
            }
        }
        for (name, schema) in incoming {
            let key = renamed[name] ?? name
            if definitions[key] == nil { definitions[key] = schema.renamingReferences(renamed) }
        }
        nodes = nodes.map { $0.renamingReferences(renamed) }
    }

    /// Primitive schemas.
    static func primitive(_ type: String) -> Self { Self(document: ["type": .string(type)], referenced: false) }
}

/// Narrows a property's or element's schema, as FoundationModels' `GenerationGuide` does: ranges, counts, choices.
public struct OutputGuide<Value>: Sendable {
    let apply: @Sendable (inout JSONValue, inout [String: JSONValue]) -> Void

    init(_ apply: @escaping @Sendable (inout JSONValue, inout [String: JSONValue]) -> Void) { self.apply = apply }

    fileprivate static func setting(_ pairs: [(String, JSONValue)]) -> Self {
        Self { node, _ in for (key, value) in pairs { node = node.setting(key, value) } }
    }
}

extension OutputGuide where Value == String {
    /// Exactly this string.
    public static func constant(_ value: String) -> Self { Self { node, _ in node = .object(["const": .string(value)]) } }
    /// One of these strings.
    public static func anyOf(_ values: [String]) -> Self { setting([("enum", .array(values.map { .string($0) }))]) }
}

extension OutputGuide where Value == Int {
    public static func minimum(_ value: Int) -> Self { setting([("minimum", .int(value))]) }
    public static func maximum(_ value: Int) -> Self { setting([("maximum", .int(value))]) }
    public static func range(_ range: ClosedRange<Int>) -> Self {
        setting([("minimum", .int(range.lowerBound)), ("maximum", .int(range.upperBound))])
    }
}

extension OutputGuide where Value == Double {
    public static func minimum(_ value: Double) -> Self { setting([("minimum", .double(value))]) }
    public static func maximum(_ value: Double) -> Self { setting([("maximum", .double(value))]) }
    public static func range(_ range: ClosedRange<Double>) -> Self {
        setting([("minimum", .double(range.lowerBound)), ("maximum", .double(range.upperBound))])
    }
}

extension OutputGuide where Value == Float {
    public static func minimum(_ value: Float) -> Self { setting([("minimum", .double(Double(value)))]) }
    public static func maximum(_ value: Float) -> Self { setting([("maximum", .double(Double(value)))]) }
    public static func range(_ range: ClosedRange<Float>) -> Self {
        setting([("minimum", .double(Double(range.lowerBound))), ("maximum", .double(Double(range.upperBound)))])
    }
}

extension OutputGuide {
    public static func count<Element>(_ count: Int) -> Self where Value == [Element] {
        setting([("minItems", .int(count)), ("maxItems", .int(count))])
    }
    public static func count<Element>(_ range: ClosedRange<Int>) -> Self where Value == [Element] {
        setting([("minItems", .int(range.lowerBound)), ("maxItems", .int(range.upperBound))])
    }
    public static func minimumCount<Element>(_ count: Int) -> Self where Value == [Element] { setting([("minItems", .int(count))]) }
    public static func maximumCount<Element>(_ count: Int) -> Self where Value == [Element] { setting([("maxItems", .int(count))]) }

    /// Applies `guide` to every element.
    public static func element<Element>(_ guide: OutputGuide<Element>) -> Self where Value == [Element] {
        Self { node, definitions in
            guard case .object(var array) = node, var items = array["items"] else { return }
            guide.apply(&items, &definitions)
            array["items"] = items
            node = .object(array)
        }
    }
}

extension String: StructuredOutput { public static var outputSchema: OutputSchema { .primitive("string") } }
extension Int: StructuredOutput { public static var outputSchema: OutputSchema { .primitive("integer") } }
extension Double: StructuredOutput { public static var outputSchema: OutputSchema { .primitive("number") } }
extension Float: StructuredOutput { public static var outputSchema: OutputSchema { .primitive("number") } }
extension Bool: StructuredOutput { public static var outputSchema: OutputSchema { .primitive("boolean") } }

extension Array: StructuredOutput where Element: StructuredOutput {
    public static var outputSchema: OutputSchema {
        let (items, definitions) = Element.outputSchema.embedded()
        var document: [String: JSONValue] = ["type": .string("array"), "items": items]
        if !definitions.isEmpty { document["$defs"] = .object(definitions) }
        return OutputSchema(document: document, referenced: false)
    }
}

extension JSONValue {
    /// The definition names this value's `$ref`s point to, at any depth.
    var references: Set<String> {
        switch self {
        case .object(let object):
            var names = object.values.reduce(into: Set<String>()) { $0.formUnion($1.references) }
            if case .string(let reference)? = object["$ref"], reference.hasPrefix("#/$defs/") { names.insert(String(reference.dropFirst(8))) }
            return names
        case .array(let values): return values.reduce(into: Set<String>()) { $0.formUnion($1.references) }
        default: return []
        }
    }

    /// This value with each `$ref` to a definition in `names` pointing at its new name.
    func renamingReferences(_ names: [String: String]) -> JSONValue {
        guard !names.isEmpty else { return self }
        switch self {
        case .object(let object):
            var result = object.mapValues { $0.renamingReferences(names) }
            if case .string(let reference)? = object["$ref"], reference.hasPrefix("#/$defs/"),
               let name = names[String(reference.dropFirst(8))] { result["$ref"] = .string("#/$defs/\(name)") }
            return .object(result)
        case .array(let values): return .array(values.map { $0.renamingReferences(names) })
        default: return self
        }
    }

    func setting(_ key: String, _ value: JSONValue) -> JSONValue {
        guard case .object(var object) = self else { return self }
        object[key] = value
        return .object(object)
    }

    /// A number, integer or not.
    var number: Double? {
        switch self {
        case .int(let i): Double(i)
        case .double(let d): d
        default: nil
        }
    }
}
