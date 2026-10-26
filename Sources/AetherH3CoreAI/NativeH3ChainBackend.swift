import AetherKit
import AetherH3
import AetherCoreAI
@preconcurrency import CoreAI
import Foundation
import Metal

actor NativeH3ChainBackend: H3ChainBackend {
    private struct Binding {
        let name: String
        let spec: H3TensorSpec
        let buffer: any MTLBuffer
    }
    private let model: AIModel
    private let function: InferenceFunction
    private let stream: ComputeStream
    private let fixed: [Binding]
    private let steps: [[Binding]]
    private let weightFiles: [[H3TensorFile]]
    private let weightSlots: [Binding]
    private let initial: Binding
    private let initialFile: H3TensorFile
    private let fixedFiles: [H3TensorFile]
    private let tableFiles: [H3TensorFile]
    private var runtimeDirty = false
    private let outputs: [any MTLBuffer]
    private let observer: (@Sendable (H3ChainEvent) -> Void)?

    static func open(_ probe: H3ChainProbePlan.Case,
                     observer: (@Sendable (H3ChainEvent) -> Void)?) async throws -> NativeH3ChainBackend {
        try await NativeH3ChainBackend(probe: probe, observer: observer)
    }
    private init(probe: H3ChainProbePlan.Case, observer: (@Sendable (H3ChainEvent) -> Void)?) async throws {
        try probe.verify()
        try Task.checkCancellation()
        guard let device = MTLCreateSystemDefaultDevice(), probe.boundBytes <= 4 << 30,
              probe.base.output.byteCount <= device.maxBufferLength else {
            throw AetherError.unsupported("H3 chain requires an admitted Metal workspace")
        }
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model = try await AIModel(contentsOf: probe.base.model.url, options: options)
        try Task.checkCancellation()
        guard let descriptor = model.functionDescriptor(for: probe.base.model.function) else { throw AetherError.invalidModel("missing H3 chain function") }
        try H3BlockBindings.validate(ModelFunctionDescription(descriptor), probe: probe.base)
        guard let function = try model.loadFunction(named: probe.base.model.function) else { throw AetherError.invalidModel("could not load H3 chain function") }
        let fixed = try probe.base.inputs.filter { ["tags", "cos", "sin"].contains($0.name) }.map { try Self.load($0, device: device) }
        guard let hidden = probe.base.inputs.first(where: { $0.name == "hidden" }) else { throw AetherError.invalidModel("H3 chain lacks initial hidden") }
        let initial = try Self.load(hidden, device: device)
        let steps = try probe.steps.map { step in
            try ([step.table] + (probe.streamsWeights ? [] : step.weights)).map { try Self.load($0, device: device) }
        }
        var slots: [Binding] = []
        if probe.streamsWeights {
            for file in probe.steps[0].weights {
                guard file.spec.byteCount <= device.maxBufferLength,
                      let buffer = device.makeBuffer(length: file.spec.byteCount, options: .storageModeShared) else {
                    throw AetherError.unsupported("could not allocate H3 resident weight slot")
                }
                slots.append(Binding(name: file.name, spec: file.spec, buffer: buffer))
            }
        }
        var outputs: [any MTLBuffer] = []
        for _ in 0..<2 {
            guard let buffer = device.makeBuffer(length: probe.base.output.byteCount, options: .storageModeShared) else {
                throw AetherError.unsupported("could not allocate distinct H3 chain outputs")
            }
            outputs.append(buffer)
        }
        self.model = model; self.function = function; self.stream = ComputeStream()
        self.fixed = fixed; self.steps = steps; self.initial = initial
        self.initialFile = hidden
        self.fixedFiles = probe.base.inputs.filter { ["tags", "cos", "sin"].contains($0.name) }
        self.tableFiles = probe.steps.map(\.table)
        self.weightSlots = slots
        self.weightFiles = probe.streamsWeights ? probe.steps.map(\.weights) : []
        self.outputs = outputs; self.observer = observer
    }
    private static func load(_ file: H3TensorFile, device: any MTLDevice) throws -> Binding {
        try Task.checkCancellation()
        let data = try file.read()
        guard data.spec.byteCount <= device.maxBufferLength,
              let buffer = device.makeBuffer(length: data.spec.byteCount, options: .storageModeShared) else {
            throw AetherError.unsupported("could not allocate H3 chain binding")
        }
        data.bytes.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let copy = Data(bytesNoCopy: buffer.contents(), count: data.spec.byteCount, deallocator: .none)
        guard h3SHA256(copy) == file.sha256 else { throw AetherError.invalidTensor("H3 chain binding copy changed bytes") }
        return Binding(name: file.name, spec: file.spec, buffer: buffer)
    }

    func run() async throws -> [H3TensorData] {
        guard weightFiles.isEmpty else { throw AetherError.unsupported("fifty-block snapshots require run(onOutput:)") }
        return try await execute(delivery: nil)
    }
    func run(onOutput: @escaping H3ChainOutputHandler) async throws { _ = try await execute(delivery: onOutput) }
    /// Internal trajectory route. New contexts are checked for extent/precision,
    /// copied exactly, and never compared to unrelated fixed-probe hashes.
    func run(initial hidden: H3TensorData, tables: [H3TensorData], tags: H3TensorData,
             executionObserver: (@Sendable (H3ChainEvent) -> Void)? = nil,
             onOutput: @escaping H3ChainOutputHandler) async throws {
        guard weightFiles.count == 50, tables.count == steps.count,
              hidden.spec == initial.spec, let tagSlot = fixed.first(where: { $0.name == "tags" }), tags.spec == tagSlot.spec,
              zip(tables, steps).allSatisfy({ $0.spec == $1[0].spec }) else {
            throw AetherError.invalidTensor("H3 runtime backbone input shape, precision or block count differs")
        }
        let validTags = tags.bytes.withUnsafeBytes { raw in
            stride(from: 0, to: raw.count, by: 4).allSatisfy {
                let index = Int(Int32(bitPattern: raw.loadUnaligned(fromByteOffset: $0, as: UInt32.self).littleEndian))
                return (0..<tables[0].spec.shape[1]).contains(index)
            }
        }
        guard validTags else { throw AetherError.invalidTensor("H3 runtime modulation index is out of range") }
        runtimeDirty = true // Set before copying so partial cancellation restores on fixed-probe retry.
        try Self.copy(hidden, to: initial); try Self.copy(tags, to: tagSlot)
        for (table, step) in zip(tables, steps) { try Self.copy(table, to: step[0]) }
        _ = try await execute(delivery: onOutput, restoreOriginal: false, executionObserver: executionObserver)
    }
    private static func copy(_ value: H3TensorData, to binding: Binding) throws {
        try Task.checkCancellation()
        guard value.spec == binding.spec else { throw AetherError.invalidTensor("H3 runtime copy shape differs") }
        value.bytes.withUnsafeBytes { binding.buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        guard h3SHA256(Data(bytesNoCopy: binding.buffer.contents(), count: binding.spec.byteCount, deallocator: .none)) == value.sha256 else {
            throw AetherError.invalidTensor("H3 runtime backbone copy changed bytes")
        }
    }
    private func execute(delivery: H3ChainOutputHandler?, restoreOriginal: Bool = true,
                         executionObserver: (@Sendable (H3ChainEvent) -> Void)? = nil) async throws -> [H3TensorData] {
        let observer = executionObserver ?? self.observer
        if restoreOriginal && runtimeDirty {
            try Self.copy(initialFile.read(), to: initial)
            for (file, binding) in zip(fixedFiles, fixed) { try Self.copy(file.read(), to: binding) }
            for (file, step) in zip(tableFiles, steps) { try Self.copy(file.read(), to: step[0]) }
            runtimeDirty = false
        }
        var hidden = initial.buffer
        var snapshots: [H3TensorData] = []
        for (block, step) in steps.enumerated() {
            try Task.checkCancellation()
            // Every previous encode is joined before shared weight slots change.
            // A failed/cancelled reload submits nothing; the next run reloads 0.
            if !weightFiles.isEmpty {
                for (file, slot) in zip(weightFiles[block], weightSlots) {
                    guard file.name == slot.name, file.spec == slot.spec else { throw AetherError.invalidModel("H3 weight slot layout changed") }
                    let value = try file.read()
                    value.bytes.withUnsafeBytes { slot.buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
                    let copy = Data(bytesNoCopy: slot.buffer.contents(), count: slot.spec.byteCount, deallocator: .none)
                    guard h3SHA256(copy) == file.sha256 else { throw AetherError.invalidTensor("H3 streamed weight copy changed bytes") }
                }
            }
            try Task.checkCancellation()
            let output = outputs[block % 2]
            // The initial buffer is immutable; alternating outputs never alias
            // the previous hidden state. Every step joins before buffer reuse.
            let values = Dictionary(uniqueKeysWithValues: (fixed + step + weightSlots).map {
                ($0.name, InferenceFunction.AsyncValue(unsafeBuffer: $0.buffer,
                    scalarType: $0.spec.scalarType.native, shape: $0.spec.shape))
            }).merging(["hidden": InferenceFunction.AsyncValue(unsafeBuffer: hidden,
                scalarType: initial.spec.scalarType.native, shape: initial.spec.shape)]) { _, new in new }
            let stream = self.stream
            var result = InferenceFunction.AsyncMutableValue(unsafeBuffer: output,
                scalarType: initial.spec.scalarType.native, shape: initial.spec.shape)
            var views = InferenceFunction.AsyncMutableViews()
            views.insert(&result, for: "hidden_out")
            do {
                _ = try function.encode(inputs: values, outputViews: views, to: stream)
            } catch {
                await stream.currentWorkCompleted()
                observer?(H3ChainEvent(block: block, event: .nativeCompleted))
                throw error
            }
            observer?(H3ChainEvent(block: block, event: .nativeSubmitted))
            await stream.currentWorkCompleted()
            observer?(H3ChainEvent(block: block, event: .nativeCompleted))
            try Task.checkCancellation()
            let snapshot = try H3TensorData(spec: initial.spec,
                bytes: Data(bytes: output.contents(), count: initial.spec.byteCount))
            if let delivery { try await delivery(block, snapshot) } else { snapshots.append(snapshot) }
            try Task.checkCancellation()
            hidden = output
        }
        return snapshots
    }
}

private extension H3ScalarType {
    var native: NDArray.ScalarType {
        switch self { case .bfloat16: .bfloat16; case .float32: .float32; case .int32: .int32 }
    }
}
