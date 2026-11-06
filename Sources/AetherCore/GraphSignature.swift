/// A graph function's declared I/O as plain data (DESIGN.md §6.1).
///
/// AetherRuntime builds it from Core AI's `InferenceFunctionDescriptor`; AetherCore only ever sees this value, so the
/// zoo adapter and manifest validation stay free of Core AI and testable anywhere. Order is the graph's own.
public struct GraphSignature: Sendable, Hashable, Codable {
    public var function: String
    public var inputs: [TensorSpec]
    public var states: [TensorSpec]
    public var outputs: [TensorSpec]

    public init(function: String, inputs: [TensorSpec], states: [TensorSpec], outputs: [TensorSpec]) {
        self.function = function
        self.inputs = inputs
        self.states = states
        self.outputs = outputs
    }

    public func input(_ name: String) -> TensorSpec? { inputs.first { $0.name == name } }
    public func state(_ name: String) -> TensorSpec? { states.first { $0.name == name } }
    public func output(_ name: String) -> TensorSpec? { outputs.first { $0.name == name } }

    /// True when any input or state has a dynamic dimension: the shape kind the specialization rule keys on.
    public var hasDynamicInputsOrStates: Bool {
        (inputs + states).contains { $0.isDynamic }
    }
}

/// One named value in a ``GraphSignature``.
public struct TensorSpec: Sendable, Hashable, Codable {
    public enum Kind: String, Sendable, Hashable, Codable {
        case ndArray
        case image
        case unknown
    }

    /// Marks a dynamic dimension in ``shape``.
    public static let dynamic = -1

    public var name: String
    public var kind: Kind
    public var scalarType: ScalarType
    /// Dimensions, with ``dynamic`` (-1) for a dimension resolved at bind time. Empty for non-array values.
    public var shape: [Int]

    public init(name: String, kind: Kind = .ndArray, scalarType: ScalarType, shape: [Int]) {
        self.name = name
        self.kind = kind
        self.scalarType = scalarType
        self.shape = shape.map { $0 < 0 ? Self.dynamic : $0 }
    }

    public var rank: Int { shape.count }
    public var isDynamic: Bool { shape.contains(Self.dynamic) }
    public var dynamicAxes: [Int] { shape.indices.filter { shape[$0] == Self.dynamic } }

    /// Replaces each dynamic dimension, in order, with the next value of `dimensions`.
    public func resolving(_ dimensions: [Int]) throws(AetherError) -> [Int] {
        var queue = dimensions[...]
        var out = shape
        for axis in out.indices where out[axis] == Self.dynamic {
            guard let value = queue.popFirst(), value > 0 else {
                throw AetherError.tensor(
                    name: name, reason: .shapeMismatch(expected: shape, found: dimensions))
            }
            out[axis] = value
        }
        guard queue.isEmpty else {
            throw AetherError.tensor(name: name, reason: .shapeMismatch(expected: shape, found: dimensions))
        }
        return out
    }

    /// Whether a concrete shape fits this spec: same rank, equal static dimensions, positive dynamic ones.
    public func accepts(_ concrete: [Int]) -> Bool {
        guard concrete.count == shape.count else { return false }
        return zip(shape, concrete).allSatisfy { declared, actual in
            declared == Self.dynamic ? actual > 0 : declared == actual
        }
    }

    /// `float16 [28×1×8×?×128]`
    public var summary: String {
        "\(scalarType) [\(shape.map { $0 == Self.dynamic ? "?" : String($0) }.joined(separator: "×"))]"
    }
}

/// An element type, by name. String-backed so that types this build does not know still round-trip through JSON.
public struct ScalarType: RawRepresentable, Sendable, Hashable, Codable, CustomStringConvertible,
    ExpressibleByStringLiteral
{
    public let rawValue: String
    public init(rawValue: String) { self.rawValue = rawValue }
    public init(stringLiteral value: String) { self.rawValue = value }

    public static let bool: ScalarType = "bool"
    public static let int8: ScalarType = "int8"
    public static let int16: ScalarType = "int16"
    public static let int32: ScalarType = "int32"
    public static let int64: ScalarType = "int64"
    public static let uint8: ScalarType = "uint8"
    public static let uint16: ScalarType = "uint16"
    public static let uint32: ScalarType = "uint32"
    public static let uint64: ScalarType = "uint64"
    public static let float16: ScalarType = "float16"
    public static let bfloat16: ScalarType = "bfloat16"
    public static let float32: ScalarType = "float32"
    public static let float64: ScalarType = "float64"
    public static let unknown: ScalarType = "unknown"

    /// Bytes per element, or nil for sub-byte, complex or unknown types.
    public var byteWidth: Int? {
        switch self {
        case .bool, .int8, .uint8: 1
        case .int16, .uint16, .float16, .bfloat16: 2
        case .int32, .uint32, .float32: 4
        case .int64, .uint64, .float64: 8
        default: nil
        }
    }

    public var isFloatingPoint: Bool { [.float16, .bfloat16, .float32, .float64].contains(self) }
    public var isInteger: Bool {
        [.int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64].contains(self)
    }

    /// The short form manifests use for precision (`fp16`, `int4`, …).
    public var precisionName: String {
        switch self {
        case .float16: "fp16"
        case .bfloat16: "bf16"
        case .float32: "fp32"
        case .float64: "fp64"
        default: rawValue
        }
    }

    public var description: String { rawValue }

    public init(from decoder: any Decoder) throws {
        rawValue = try decoder.singleValueContainer().decode(String.self)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}
