import AetherKit
import AetherH3
import AetherCoreAI
@preconcurrency import CoreAI
import Foundation
import Metal

/// One profile graph with reusable named input/output slots. Activations are copied in; weights are read from their
/// (in-place) files straight into their slots, hashed the first time each file is read by this graph and trusted
/// afterwards (the files are immutable checkpoint blobs). Shapes are checked against the native descriptor before any
/// slot changes. One call at a time; every call joins its GPU work before returning.
actor NativeH3Graph {
    enum Input: Sendable {
        case data(H3TensorData)
        case file(H3TensorFile)
        var spec: H3TensorSpec { switch self { case .data(let d): d.spec; case .file(let f): f.spec } }
    }
    private let model: AIModel
    private let function: InferenceFunction
    private let descriptor: ModelFunctionDescription
    private let stream = ComputeStream()
    private let device: any MTLDevice
    private var slots: [String: any MTLBuffer] = [:]
    private var outputs: [String: any MTLBuffer] = [:]
    /// sha256 of each file already hashed into a slot, so later steps reuse the check instead of re-hashing.
    private var verified: Set<String> = []
    /// The file each weight slot currently holds, so an unchanged binding is not even re-read.
    private var resident: [String: String] = [:]
    private var busy = false
    /// Two weight banks: `load(_:into:)` fills one while a `run(..., bank:)` on the other waits for the GPU (actor
    /// reentrancy at the GPU join), so streamed weights load behind the previous call's GPU work.
    private var banks: [[String: any MTLBuffer]] = [[:], [:]]
    private var bankResident: [[String: String]] = [[:], [:]]

    static func open(_ asset: H3ModelAsset) async throws -> NativeH3Graph { try await NativeH3Graph(asset) }
    private init(_ asset: H3ModelAsset) async throws {
        try asset.verify(); try Task.checkCancellation()
        guard let device = MTLCreateSystemDefaultDevice() else { throw AetherError.unsupported("H3 requires a Metal device") }
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu); options.expectFrequentReshapes = true
        let model = try await AIModel(contentsOf: asset.url, options: options)
        try Task.checkCancellation()
        guard let description = model.functionDescriptor(for: asset.function),
              let function = try model.loadFunction(named: asset.function) else { throw AetherError.invalidModel("missing H3 profile function") }
        self.model = model; self.function = function; self.descriptor = ModelFunctionDescription(description); self.device = device
    }

    private func buffer(_ old: (any MTLBuffer)?, _ spec: H3TensorSpec) throws -> any MTLBuffer {
        if let old, old.length == spec.byteCount { return old }
        guard spec.byteCount <= device.maxBufferLength,
              let value = device.makeBuffer(length: spec.byteCount, options: .storageModeShared) else {
            throw AetherError.unsupported("could not allocate H3 profile buffer")
        }
        return value
    }

    /// Reads `files` into weight bank `bank` (0 or 1), skipping bindings it already holds.
    func load(_ files: [String: H3TensorFile], into bank: Int) throws {
        for (name, file) in files.sorted(by: { $0.key < $1.key }) {
            try Task.checkCancellation()
            let slot = try buffer(banks[bank][name], file.spec)
            if slot !== banks[bank][name] { bankResident[bank][name] = nil }
            banks[bank][name] = slot
            guard bankResident[bank][name] != file.sha256 else { continue }
            bankResident[bank][name] = nil   // A failed or cancelled read leaves the slot unowned.
            try file.read(into: UnsafeMutableRawBufferPointer(start: slot.contents(), count: file.spec.byteCount), verify: !verified.contains(file.sha256))
            verified.insert(file.sha256); bankResident[bank][name] = file.sha256
        }
    }

    /// `bank`: take the `.file` inputs from that weight bank (loading any it does not hold yet) instead of the
    /// per-name slots. `prefetch`: once this call's GPU work is encoded, load those files into that (other) bank while
    /// the GPU runs; a failed prefetch is dropped (the next call loads what it lacks).
    func run(_ inputs: [String: Input], outputs specs: [String: H3TensorSpec], bank: Int? = nil,
             prefetch: (files: [String: H3TensorFile], bank: Int)? = nil) async throws -> [String: H3TensorData] {
        try Task.checkCancellation()
        guard !busy else { throw AetherError.sessionBusy }
        try H3RuntimeBindings.validate(descriptor, inputs: inputs.mapValues(\.spec), outputs: specs)
        busy = true; defer { busy = false }
        if let bank {
            var files: [String: H3TensorFile] = [:]
            for (name, input) in inputs { if case .file(let file) = input { files[name] = file } }
            try load(files, into: bank)
        }
        var bound: [String: any MTLBuffer] = [:]
        for (name, input) in inputs.sorted(by: { $0.key < $1.key }) {
            if let bank, case .file = input { bound[name] = banks[bank][name]!; continue }
            try Task.checkCancellation()
            let slot = try buffer(slots[name], input.spec)
            if slot !== slots[name] { resident[name] = nil }
            slots[name] = slot; bound[name] = slot
            switch input {
            case .data(let tensor):
                resident[name] = nil
                tensor.bytes.withUnsafeBytes { slot.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            case .file(let file):
                guard resident[name] != file.sha256 else { continue }
                resident[name] = nil   // A failed or cancelled read leaves the slot unowned.
                try file.read(into: UnsafeMutableRawBufferPointer(start: slot.contents(), count: file.spec.byteCount),
                              verify: !verified.contains(file.sha256))
                verified.insert(file.sha256); resident[name] = file.sha256
            }
        }
        for (name, spec) in specs { outputs[name] = try buffer(outputs[name], spec) }
        let values = Dictionary(uniqueKeysWithValues: inputs.map { name, input in
            (name, InferenceFunction.AsyncValue(unsafeBuffer: bound[name]!, scalarType: input.spec.scalarType.native, shape: input.spec.shape))
        })
        let names = specs.keys.sorted()
        func view(_ name: String) -> InferenceFunction.AsyncMutableValue {
            InferenceFunction.AsyncMutableValue(unsafeBuffer: outputs[name]!, scalarType: specs[name]!.scalarType.native, shape: specs[name]!.shape)
        }
        do {
            var a = view(names[0])
            if names.count == 2 {
                var b = view(names[1])
                var views = InferenceFunction.AsyncMutableViews(); views.insert(&a, for: names[0]); views.insert(&b, for: names[1])
                _ = try function.encode(inputs: values, outputViews: views, to: stream)
            } else {
                var views = InferenceFunction.AsyncMutableViews(); views.insert(&a, for: names[0])
                _ = try function.encode(inputs: values, outputViews: views, to: stream)
            }
        } catch {
            await stream.currentWorkCompleted(); throw error
        }
        if let prefetch, prefetch.bank != bank { try? load(prefetch.files, into: prefetch.bank) }
        await stream.currentWorkCompleted()
        try Task.checkCancellation()
        var result: [String: H3TensorData] = [:]
        for (name, spec) in specs {
            result[name] = try H3TensorData(spec: spec, bytes: Data(bytes: outputs[name]!.contents(), count: spec.byteCount))
        }
        return result
    }
}

private extension H3ScalarType {
    var native: NDArray.ScalarType {
        switch self { case .bfloat16: .bfloat16; case .float32: .float32; case .int32: .int32 }
    }
}
