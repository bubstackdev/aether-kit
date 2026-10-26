import AetherKit
import AetherH3
import AetherCoreAI
@preconcurrency import CoreAI
import Foundation
import Metal

/// Fixed outer-phase binding. Embedded graph weights remain in their exported
/// precision; BF16 inputs/outputs use direct shared buffers without fp16 copies.
actor NativeH3PhaseBackend: H3PhaseBackend {
    private struct Binding {
        let name: String
        let buffer: any MTLBuffer
        let spec: H3TensorSpec
    }
    private let model: AIModel
    private let function: InferenceFunction
    private let inputs: [Binding]
    private let outputs: [Binding]
    private let stream: ComputeStream
    private let observer: (@Sendable (H3BlockEvent) -> Void)?
    static func open(_ probe: H3PhaseProbePlan.Case,
                     observer: (@Sendable (H3BlockEvent) -> Void)?) async throws -> NativeH3PhaseBackend {
        try await NativeH3PhaseBackend(probe: probe, observer: observer)
    }
    private init(probe: H3PhaseProbePlan.Case, observer: (@Sendable (H3BlockEvent) -> Void)?) async throws {
        try probe.model.verify(); try probe.verifyInputs(); try Task.checkCancellation()
        guard probe.boundBytes <= 2 << 30, (1...2).contains(probe.references.count),
              let device = MTLCreateSystemDefaultDevice(),
              (probe.inputs + probe.references).allSatisfy({ $0.spec.byteCount <= device.maxBufferLength }) else {
            throw AetherError.unsupported("H3 outer phase requires admitted buffers and a Metal device")
        }
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model = try await AIModel(contentsOf: probe.model.url, options: options)
        try Task.checkCancellation()
        guard let descriptor = model.functionDescriptor(for: probe.model.function) else {
            throw AetherError.invalidModel("missing H3 outer function")
        }
        try H3PhaseBindings.validate(ModelFunctionDescription(descriptor), probe: probe)
        guard let function = try model.loadFunction(named: probe.model.function) else {
            throw AetherError.invalidModel("could not load H3 outer function")
        }
        var inputs: [Binding] = [], outputs: [Binding] = []
        for file in probe.inputs {
            let data = try file.read()
            guard let buffer = device.makeBuffer(length: data.spec.byteCount, options: .storageModeShared) else {
                throw AetherError.unsupported("could not allocate H3 outer input")
            }
            data.bytes.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            guard h3SHA256(Data(bytesNoCopy: buffer.contents(), count: data.spec.byteCount, deallocator: .none)) == file.sha256 else {
                throw AetherError.invalidTensor("H3 outer shared input copy changed bytes")
            }
            inputs.append(Binding(name: file.name, buffer: buffer, spec: file.spec))
        }
        for file in probe.references {
            try Task.checkCancellation()
            guard let buffer = device.makeBuffer(length: file.spec.byteCount, options: .storageModeShared) else {
                throw AetherError.unsupported("could not allocate H3 outer output")
            }
            outputs.append(Binding(name: file.name, buffer: buffer, spec: file.spec))
        }
        self.model = model; self.function = function; self.inputs = inputs; self.outputs = outputs
        self.stream = ComputeStream(); self.observer = observer
    }
    func run() async throws -> [String: H3TensorData] {
        try Task.checkCancellation()
        let values = Dictionary(uniqueKeysWithValues: inputs.map {
            ($0.name, InferenceFunction.AsyncValue(unsafeBuffer: $0.buffer, scalarType: $0.spec.scalarType.native, shape: $0.spec.shape))
        })
        // The opaque SDK stream is not Sendable. This compatibility import is
        // confined here; the session retains admission and buffers until joined
        // completion, even after cancellation or a partial encode failure.
        let stream = self.stream
        do {
            let first = outputs[0]
            var a = InferenceFunction.AsyncMutableValue(unsafeBuffer: first.buffer, scalarType: first.spec.scalarType.native, shape: first.spec.shape)
            if outputs.count == 2 {
                let second = outputs[1]
                var b = InferenceFunction.AsyncMutableValue(unsafeBuffer: second.buffer, scalarType: second.spec.scalarType.native, shape: second.spec.shape)
                var views = InferenceFunction.AsyncMutableViews()
                views.insert(&a, for: first.name); views.insert(&b, for: second.name)
                _ = try function.encode(inputs: values, outputViews: views, to: stream)
            } else {
                var views = InferenceFunction.AsyncMutableViews()
                views.insert(&a, for: first.name)
                _ = try function.encode(inputs: values, outputViews: views, to: stream)
            }
        }
        catch { await stream.currentWorkCompleted(); observer?(.nativeCompleted); throw error }
        observer?(.nativeSubmitted)
        await stream.currentWorkCompleted(); observer?(.nativeCompleted)
        try Task.checkCancellation()
        var result: [String: H3TensorData] = [:]
        for output in outputs {
            try Task.checkCancellation()
            result[output.name] = try H3TensorData(spec: output.spec, bytes: Data(bytes: output.buffer.contents(), count: output.spec.byteCount))
        }
        return result
    }
}
private extension H3ScalarType {
    var native: NDArray.ScalarType {
        switch self { case .bfloat16: .bfloat16; case .float32: .float32; case .int32: .int32 }
    }
}
