import AetherKit
import AetherH3
import AetherCoreAI
@preconcurrency import CoreAI
import Foundation
import Metal

actor NativeH3BlockBackend: H3BlockBackend {
    private struct Binding {
        let name: String
        let buffer: any MTLBuffer
        let spec: H3TensorSpec
    }
    private let model: AIModel
    private let function: InferenceFunction
    private let inputs: [Binding]
    private let output: any MTLBuffer
    private let outputSpec: H3TensorSpec
    private let stream: ComputeStream
    private let observer: (@Sendable (H3BlockEvent) -> Void)?

    static func open(_ probe: H3BlockProbePlan.Case,
                     observer: (@Sendable (H3BlockEvent) -> Void)?) async throws -> NativeH3BlockBackend {
        try await NativeH3BlockBackend(probe: probe, observer: observer)
    }
    private init(probe: H3BlockProbePlan.Case, observer: (@Sendable (H3BlockEvent) -> Void)?) async throws {
        // Do CPU verification before the first native acquisition. No NDArray
        // parameter arrays or fp16 conversion is involved in this path.
        try probe.model.verify()
        try probe.verifyInputs()
        try Task.checkCancellation()
        guard let device = MTLCreateSystemDefaultDevice(), probe.boundBytes <= 2 << 30,
              probe.inputs.allSatisfy({ $0.spec.byteCount <= device.maxBufferLength }),
              probe.output.byteCount <= device.maxBufferLength else {
            throw AetherError.unsupported("H3 block needs a Metal device and admitted buffer extents")
        }
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model = try await AIModel(contentsOf: probe.model.url, options: options)
        try Task.checkCancellation()
        guard let descriptor = model.functionDescriptor(for: probe.model.function) else { throw AetherError.invalidModel("missing H3 block function") }
        try H3BlockBindings.validate(ModelFunctionDescription(descriptor), probe: probe)
        guard let function = try model.loadFunction(named: probe.model.function) else { throw AetherError.invalidModel("could not load H3 block function") }
        var inputs: [Binding] = []
        for input in probe.inputs {
            try Task.checkCancellation()
            let data = try input.read()
            guard let buffer = device.makeBuffer(length: data.spec.byteCount, options: .storageModeShared) else { throw AetherError.unsupported("could not allocate H3 shared input") }
            data.bytes.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            let copied = Data(bytesNoCopy: buffer.contents(), count: data.spec.byteCount, deallocator: .none)
            guard h3SHA256(copied) == input.sha256 else { throw AetherError.invalidTensor("H3 shared weight/input copy changed bytes") }
            inputs.append(Binding(name: input.name, buffer: buffer, spec: input.spec))
        }
        guard let output = device.makeBuffer(length: probe.output.byteCount, options: .storageModeShared) else { throw AetherError.unsupported("could not allocate H3 shared output") }
        self.model = model; self.function = function; self.inputs = inputs
        self.output = output; self.outputSpec = probe.output
        self.stream = ComputeStream()
        self.observer = observer
    }

    func run() async throws -> H3TensorData {
        try Task.checkCancellation()
        // Buffers stay owned by this backend across success, cancellation and
        // encode failure. Even a partial submission is joined before release.
        // Core AI's opaque stream is not Sendable in this SDK. The compatibility
        // import is confined here; this actor owns buffers, and H3BlockSession
        // admits one call until stream completion, including cancellation/errors.
        // The session serializes calls and joins each one before this stream
        // is reused, including cancellation and partial encode failures.
        let stream = self.stream
        let values = Dictionary(uniqueKeysWithValues: inputs.map { binding in
            (binding.name, InferenceFunction.AsyncValue(unsafeBuffer: binding.buffer,
                scalarType: binding.spec.scalarType.native, shape: binding.spec.shape))
        })
        var result = InferenceFunction.AsyncMutableValue(unsafeBuffer: output, scalarType: outputSpec.scalarType.native, shape: outputSpec.shape)
        var views = InferenceFunction.AsyncMutableViews()
        views.insert(&result, for: "hidden_out")
        do {
            _ = try function.encode(inputs: values, outputViews: views, to: stream)
        } catch {
            await stream.currentWorkCompleted()
            observer?(.nativeCompleted)
            throw error
        }
        observer?(.nativeSubmitted)
        await stream.currentWorkCompleted()
        observer?(.nativeCompleted)
        try Task.checkCancellation()
        let data = Data(bytes: output.contents(), count: outputSpec.byteCount)
        return try H3TensorData(spec: outputSpec, bytes: data)
    }
}

private extension H3ScalarType {
    var native: NDArray.ScalarType {
        switch self { case .bfloat16: .bfloat16; case .float32: .float32; case .int32: .int32 }
    }
}
