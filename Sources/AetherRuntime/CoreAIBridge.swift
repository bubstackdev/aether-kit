import CoreAI

// Conversions between Core AI's types and AetherCore's plain-data mirrors. Internal: nothing above AetherRuntime
// sees a Core AI type (DESIGN.md §3).

extension ScalarType {
    init(_ type: NDArray.ScalarType) {
        self = switch type {
        case .bool: .bool
        case .int8: .int8
        case .int16: .int16
        case .int32: .int32
        case .int64: .int64
        case .uint8: .uint8
        case .uint16: .uint16
        case .uint32: .uint32
        case .uint64: .uint64
        case .float16: .float16
        case .bfloat16: .bfloat16
        case .float32: .float32
        case .float64: .float64
        case .int2: "int2"
        case .int3: "int3"
        case .int4: "int4"
        case .int5: "int5"
        case .int6: "int6"
        case .int7: "int7"
        case .int128: "int128"
        case .uint1: "uint1"
        case .uint2: "uint2"
        case .uint3: "uint3"
        case .uint4: "uint4"
        case .uint5: "uint5"
        case .uint6: "uint6"
        case .uint7: "uint7"
        case .uint128: "uint128"
        case .float8e5m2: "float8e5m2"
        case .float8e4m3fn: "float8e4m3fn"
        case .float8e8m0fn: "float8e8m0fn"
        case .float4e2m1fn: "float4e2m1fn"
        case .cfloat16: "cfloat16"
        case .cfloat32: "cfloat32"
        case .cfloat64: "cfloat64"
        @unknown default: .unknown
        }
    }

    /// The Core AI type for the byte-addressable types Aether allocates.
    var coreAI: NDArray.ScalarType? {
        switch self {
        case .bool: .bool
        case .int8: .int8
        case .int16: .int16
        case .int32: .int32
        case .int64: .int64
        case .uint8: .uint8
        case .uint16: .uint16
        case .uint32: .uint32
        case .uint64: .uint64
        case .float16: .float16
        case .bfloat16: .bfloat16
        case .float32: .float32
        case .float64: .float64
        default: nil
        }
    }
}

extension TensorSpec {
    init(name: String, descriptor: InferenceValue.Descriptor?) {
        switch descriptor {
        case .ndArray(let d)?:
            self.init(name: name, kind: .ndArray, scalarType: ScalarType(d.scalarType), shape: d.shape)
        case .image?:
            self.init(name: name, kind: .image, scalarType: .unknown, shape: [])
        case nil:
            self.init(name: name, kind: .unknown, scalarType: .unknown, shape: [])
        @unknown default:
            self.init(name: name, kind: .unknown, scalarType: .unknown, shape: [])
        }
    }
}

extension GraphSignature {
    init(_ descriptor: InferenceFunctionDescriptor) {
        self.init(
            function: descriptor.name,
            inputs: descriptor.inputNames.map { TensorSpec(name: $0, descriptor: descriptor.inputDescriptor(of: $0)) },
            states: descriptor.stateNames.map { TensorSpec(name: $0, descriptor: descriptor.stateDescriptor(of: $0)) },
            outputs: descriptor.outputNames.map { TensorSpec(name: $0, descriptor: descriptor.outputDescriptor(of: $0)) })
    }
}

extension Device {
    /// This device, with Core AI's architecture name (`AIModel.deviceArchitectureName`).
    public static var current: Device {
        .host(architecture: AIModel.deviceArchitectureName)
    }
}
