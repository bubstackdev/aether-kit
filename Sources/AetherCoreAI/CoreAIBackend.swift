import AetherKit
import CoreAI
import Foundation

/// Specialization preferences for the reference executor.
public enum ModelComputePolicy: String, Sendable, CaseIterable {
    case automatic
    case cpuOnly
    case preferGPU

    var options: SpecializationOptions {
        switch self {
        case .automatic: .default
        case .cpuOnly: .cpuOnly
        case .preferGPU: SpecializationOptions(preferredComputeUnitKind: .gpu)
        }
    }
}

// Kept internal so lifecycle tests can exercise suspended native work without weights.
protocol ModelBackend: Sendable {
    func run(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue]
}

struct CoreAIBackend: ModelBackend {
    let model: AIModel

    static func load(at url: URL, compute: ModelComputePolicy, expectFrequentReshapes: Bool) async throws -> Self {
        guard url.isFileURL, FileManager.default.fileExists(atPath: url.path) else {
            throw AetherError.invalidModel("expected an existing local model URL")
        }
        try Task.checkCancellation()
        var options = compute.options
        options.expectFrequentReshapes = expectFrequentReshapes
        let model = try await AIModel(contentsOf: url, options: options)
        try Task.checkCancellation()
        return Self(model: model)
    }

    var functions: [ModelFunctionDescription] {
        model.functionNames.sorted().compactMap { name in
            model.functionDescriptor(for: name).map(ModelFunctionDescription.init)
        }
    }

    func run(function name: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue] {
        guard let descriptor = model.functionDescriptor(for: name) else {
            throw AetherError.invalidModel("function '\(name)' does not exist")
        }
        guard descriptor.stateNames.isEmpty else {
            throw AetherError.unsupported("the reference executor accepts stateless functions only")
        }
        guard Set(inputs.keys) == Set(descriptor.inputNames) else {
            throw AetherError.invalidTensor("input names must match \(descriptor.inputNames.sorted())")
        }
        // Validate the entire boundary before allocating native inputs or dispatching work.
        for (name, tensor) in inputs {
            guard case .ndArray(let declared) = descriptor.inputDescriptor(of: name),
                  declared.scalarType == tensor.scalarType else {
                throw AetherError.unsupported("input '\(name)' has an unsupported or mismatched scalar type")
            }
            guard declared.shape.count == tensor.shape.count,
                  zip(declared.shape, tensor.shape).allSatisfy({ $0 < 0 || $0 == $1 }) else {
                throw AetherError.invalidTensor("input '\(name)' expects shape \(declared.shape)")
            }
        }
        for name in descriptor.outputNames {
            guard case .ndArray(let declared) = descriptor.outputDescriptor(of: name),
                  [.float32, .float16, .int32].contains(declared.scalarType) else {
                throw AetherError.unsupported("output '\(name)' has an unsupported or mismatched scalar type")
            }
        }
        guard let function = try model.loadFunction(named: name) else {
            throw AetherError.invalidModel("function '\(name)' could not be loaded")
        }
        let nativeInputs = inputs.mapValues(\.nativeArray)
        try Task.checkCancellation()
        var outputs = try await function.run(inputs: nativeInputs)
        // Completion precedes copying. Returned tensors have independent Swift ownership.
        var result: [String: TensorValue] = [:]
        for name in descriptor.outputNames {
            guard let value = outputs.remove(name), let array = value.ndArray else {
                throw AetherError.invalidTensor("missing tensor output '\(name)'")
            }
            switch array.scalarType {
            case .float32: result[name] = .float32(try copyTensor(array, as: Float.self))
            case .float16: result[name] = .float16(try copyTensor(array, as: Float16.self))
            case .int32: result[name] = .int32(try copyTensor(array, as: Int32.self))
            default: throw AetherError.unsupported("output '\(name)' has an unsupported scalar type")
            }
        }
        return result
    }
}

private func copyTensor<Scalar: TensorScalar & BitwiseCopyable>(
    _ array: NDArray, as type: Scalar.Type
) throws -> Tensor<Scalar> {
    let view = array.view(as: type)
    guard let elements = view.contiguousElements else {
        throw AetherError.unsupported("noncontiguous tensor output")
    }
    _ = try Tensor<Scalar>.elementCount(for: array.shape)
    var values: [Scalar] = []
    values.reserveCapacity(elements.count)
    for index in 0..<elements.count { values.append(elements[index]) }
    return try Tensor(shape: array.shape, values: values)
}

extension TensorValue {
    fileprivate var scalarType: NDArray.ScalarType {
        switch self {
        case .float32: .float32
        case .float16: .float16
        case .int32: .int32
        }
    }

    var nativeArray: NDArray {
        switch self {
        case .float32(let tensor): copyNativeTensor(tensor, scalarType: .float32)
        case .float16(let tensor): copyNativeTensor(tensor, scalarType: .float16)
        case .int32(let tensor): copyNativeTensor(tensor, scalarType: .int32)
        }
    }
}

// NDArray(scalars:shape:) iterates through a generic Sequence inside the runtime.
// Copy contiguous storage in bulk while keeping the native allocation independent
// of the Swift tensor and its lifetime.
private func copyNativeTensor<Scalar: TensorScalar & BitwiseCopyable>(
    _ tensor: Tensor<Scalar>, scalarType: NDArray.ScalarType
) -> NDArray {
    var array = NDArray(shape: tensor.shape, scalarType: scalarType)
    tensor.values.withUnsafeBytes { source in
        array.mutableRawView().withUnsafeMutableBytes { destination, _, _ in
            // Validated tensor shapes always contain at least one scalar.
            destination.copyMemory(from: source.baseAddress!, byteCount: source.count)
        }
    }
    return array
}
