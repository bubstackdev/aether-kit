import Foundation

/// Any JSON value. The typed-but-loose manifest blocks (MANIFEST.md §2.5–§2.8) carry nested parameters this way
/// until their drivers arrive; the manifest decoder also uses it for its `x-` and `$ref` pass.
public enum JSONValue: Sendable, Hashable, Codable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let i = try? c.decode(Int.self) { self = .int(i) }
        else if let d = try? c.decode(Double.self) { self = .double(d) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .int(let i): try c.encode(i)
        case .double(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }

    public subscript(key: String) -> JSONValue? {
        if case .object(let o) = self { return o[key] }
        return nil
    }

    public var string: String? { if case .string(let s) = self { s } else { nil } }
    public var int: Int? { if case .int(let i) = self { i } else { nil } }
    public var object: [String: JSONValue]? { if case .object(let o) = self { o } else { nil } }
    public var array: [JSONValue]? { if case .array(let a) = self { a } else { nil } }

    /// Compact JSON text, keys sorted (for provenance strings such as a zoo `compression` object).
    public var compactText: String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return (try? String(decoding: encoder.encode(self), as: UTF8.self)) ?? ""
    }
}

extension JSONValue {
    /// Removes every `x-` key (MANIFEST.md §2.1.6) and rejects every `$ref` object (§2.1.5: the adapter inlines
    /// refs at install time, so the runtime never follows one). `provenance` may record refs as plain strings.
    func sanitized(at path: String = "") throws(AetherError) -> JSONValue {
        switch self {
        case .object(let object):
            if object["$ref"] != nil {
                throw .manifestInvalid(
                    field: path.isEmpty ? "$ref" : path,
                    reason: "unresolved $ref; the adapter must inline sidecar references at install time")
            }
            var out: [String: JSONValue] = [:]
            for (key, value) in object where !key.hasPrefix("x-") {
                out[key] = try value.sanitized(at: path.isEmpty ? key : "\(path).\(key)")
            }
            return .object(out)
        case .array(let array):
            var out: [JSONValue] = []
            for (i, value) in array.enumerated() { out.append(try value.sanitized(at: "\(path).\(i)")) }
            return .array(out)
        default:
            return self
        }
    }
}
