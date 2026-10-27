import AetherKit
import AetherH3
import AetherCoreAI
import CoreAI
import Foundation
import Metal

/// Internal runtime-input function. Its actor and one-call admission retain all
/// shared storage through joined completion. Variable shapes are checked against
/// the real native descriptor on every call, before any buffer mutation.
actor NativeH3Function {
    private let model: AIModel
    private let function: InferenceFunction
    private let descriptor: ModelFunctionDescription
    private let stream: SerialComputeStream
    private let device: any MTLDevice
    private var inputs: [String: any MTLBuffer] = [:]
    private var outputs: [String: any MTLBuffer] = [:]
    private var busy = false
    static func open(_ asset: H3ModelAsset) async throws -> NativeH3Function { try await NativeH3Function(asset) }
    private init(_ asset: H3ModelAsset) async throws {
        try asset.verify(); try Task.checkCancellation()
        guard let device = MTLCreateSystemDefaultDevice() else { throw AetherError.unsupported("H3 requires a Metal device") }
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu); options.expectFrequentReshapes = true
        let model = try await AIModel(contentsOf: asset.url, options: options)
        try Task.checkCancellation()
        guard let description = model.functionDescriptor(for: asset.function),
              let function = try model.loadFunction(named: asset.function) else { throw AetherError.invalidModel("missing H3 runtime function") }
        self.model = model; self.function = function; self.descriptor = ModelFunctionDescription(description)
        self.device = device; self.stream = SerialComputeStream()
    }
    func run(inputs data: [String: H3TensorData], outputs specs: [String: H3TensorSpec],
             observer: (@Sendable (H3BlockEvent) -> Void)? = nil) async throws -> [String: H3TensorData] {
        try Task.checkCancellation()
        guard !busy else { throw AetherError.sessionBusy }
        try H3RuntimeBindings.validate(descriptor, inputs: data.mapValues(\.spec), outputs: specs)
        busy = true; defer { busy = false }
        func buffer(_ old: (any MTLBuffer)?, _ spec: H3TensorSpec) throws -> any MTLBuffer {
            guard spec.byteCount <= device.maxBufferLength else { throw AetherError.unsupported("H3 runtime buffer exceeds device limit") }
            if let old, old.length == spec.byteCount { return old }
            guard let value = device.makeBuffer(length: spec.byteCount, options: .storageModeShared) else { throw AetherError.unsupported("could not allocate H3 runtime buffer") }
            return value
        }
        for (name, tensor) in data {
            try Task.checkCancellation()
            let target = try buffer(inputs[name], tensor.spec)
            tensor.bytes.withUnsafeBytes { target.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            guard h3SHA256(Data(bytesNoCopy: target.contents(), count: tensor.spec.byteCount, deallocator: .none)) == tensor.sha256 else {
                throw AetherError.invalidTensor("H3 runtime input copy changed bytes")
            }
            inputs[name] = target
        }
        for (name, spec) in specs { outputs[name] = try buffer(outputs[name], spec) }
        let values = Dictionary(uniqueKeysWithValues: data.map { name, tensor in
            (name, InferenceFunction.AsyncValue(unsafeBuffer: inputs[name]!, scalarType: tensor.spec.scalarType.native, shape: tensor.spec.shape))
        })
        let names = specs.keys.sorted(), first = names[0], firstSpec = specs[first]!
        do {
            var a = InferenceFunction.AsyncMutableValue(unsafeBuffer: outputs[first]!, scalarType: firstSpec.scalarType.native, shape: firstSpec.shape)
            if names.count == 2 {
                let second = names[1], secondSpec = specs[second]!
                var b = InferenceFunction.AsyncMutableValue(unsafeBuffer: outputs[second]!, scalarType: secondSpec.scalarType.native, shape: secondSpec.shape)
                var views = InferenceFunction.AsyncMutableViews(); views.insert(&a, for: first); views.insert(&b, for: second)
                _ = try stream.encode(function, inputs: values, outputViews: views)
            } else {
                var views = InferenceFunction.AsyncMutableViews(); views.insert(&a, for: first)
                _ = try stream.encode(function, inputs: values, outputViews: views)
            }
        } catch { await stream.currentWorkCompleted(); observer?(.nativeCompleted); throw error }
        observer?(.nativeSubmitted); await stream.currentWorkCompleted(); observer?(.nativeCompleted)
        try Task.checkCancellation()
        var result: [String: H3TensorData] = [:]
        for (name, spec) in specs {
            result[name] = try H3TensorData(spec: spec, bytes: Data(bytes: outputs[name]!.contents(), count: spec.byteCount))
        }
        return result
    }
}

/// The same pre-dispatch descriptor and byte-admission gate is usable in CPU tests.
public enum H3RuntimeBindings {
    public static func validate(_ descriptor: ModelFunctionDescription, inputs: [String: H3TensorSpec], outputs: [String: H3TensorSpec]) throws {
        guard descriptor.states.isEmpty, !inputs.isEmpty, (1...2).contains(outputs.count),
              Set(inputs.keys) == Set(descriptor.inputs.keys), Set(outputs.keys) == Set(descriptor.outputs.keys),
              (Array(inputs.values) + Array(outputs.values)).reduce(0, { $0 + $1.byteCount }) <= 2 << 30 else {
            throw AetherError.invalidModel("H3 runtime bindings exceed descriptor or buffer admission")
        }
        for (specs, descriptions) in [(inputs, descriptor.inputs), (outputs, descriptor.outputs)] {
            for (name, spec) in specs {
                guard let value = descriptions[name], value.kind == "tensor", value.scalarType == spec.scalarType.rawValue,
                      let shape = value.shape, shape.count == spec.shape.count,
                      zip(shape, spec.shape).allSatisfy({ $0 == -1 || $0 == $1 }) else {
                    throw AetherError.invalidModel("H3 runtime binding dimensions or precision differ")
                }
            }
        }
    }
}
private extension H3ScalarType {
    var native: NDArray.ScalarType {
        switch self { case .bfloat16: .bfloat16; case .float32: .float32; case .int32: .int32 }
    }
}
