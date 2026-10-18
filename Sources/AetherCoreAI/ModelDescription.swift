import CoreAI

/// A model value's declared storage. Dynamic dimensions are reported unchanged.
public struct ModelValueDescription: Sendable, Codable, Equatable {
    public let kind: String
    public let scalarType: String?
    public let shape: [Int]?

    init(_ descriptor: InferenceValue.Descriptor) {
        switch descriptor {
        case .ndArray(let array):
            kind = "tensor"
            scalarType = String(describing: array.scalarType)
            shape = array.shape
        case .image:
            kind = "image"
            scalarType = nil
            shape = nil
        @unknown default:
            kind = "unknown"
            scalarType = nil
            shape = nil
        }
    }
}

public struct ModelFunctionDescription: Sendable, Codable, Equatable {
    public let name: String
    public let inputs: [String: ModelValueDescription]
    public let outputs: [String: ModelValueDescription]
    public let states: [String: ModelValueDescription]

    init(_ descriptor: InferenceFunctionDescriptor) {
        name = descriptor.name
        inputs = Dictionary(uniqueKeysWithValues: descriptor.inputNames.compactMap { name in
            descriptor.inputDescriptor(of: name).map { (name, ModelValueDescription($0)) }
        })
        outputs = Dictionary(uniqueKeysWithValues: descriptor.outputNames.compactMap { name in
            descriptor.outputDescriptor(of: name).map { (name, ModelValueDescription($0)) }
        })
        states = Dictionary(uniqueKeysWithValues: descriptor.stateNames.compactMap { name in
            descriptor.stateDescriptor(of: name).map { (name, ModelValueDescription($0)) }
        })
    }
}
