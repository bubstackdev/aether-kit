import AetherKit
import AetherH3
import AetherCoreAI

/// Diagnostic boundaries for submitted work and resource ownership. Completion
/// means the stream has been joined, not that cancellation preempted the GPU.
@available(macOS 27, iOS 27, *)
public enum H3BlockEvent: String, Sendable, Codable {
    case nativeSubmitted, nativeCompleted, slotReleased, unloadStarted, unloaded
}

/// Experimental fixed-input, explicit-weight block session for qualification.
/// Owns verified immutable shared buffers and one native function. It does not
/// implement a generation trajectory or reuse Krea's fp16 host-copy path.
@available(macOS 27, iOS 27, *)
public actor H3BlockSession {
    public enum State: String, Sendable { case ready, running, closing, closed }
    public private(set) var state: State = .ready
    private var backend: (any H3BlockBackend)?
    private var active: Task<H3TensorData, any Error>?
    private let observer: (@Sendable (H3BlockEvent) -> Void)?
    init(backend: any H3BlockBackend, observer: (@Sendable (H3BlockEvent) -> Void)? = nil) {
        self.backend = backend; self.observer = observer
    }

    /// The optional diagnostic observer runs synchronously and must not block.
    public static func open(probe: H3BlockProbePlan.Case,
                            executionObserver: (@Sendable (H3BlockEvent) -> Void)? = nil) async throws -> H3BlockSession {
        try Task.checkCancellation()
        let backend = try await NativeH3BlockBackend.open(probe, observer: executionObserver)
        try Task.checkCancellation()
        return H3BlockSession(backend: backend, observer: executionObserver)
    }

    public func run() async throws -> H3TensorData {
        try Task.checkCancellation()
        guard state != .closing, state != .closed, let backend else { throw AetherError.sessionClosed }
        guard active == nil else { throw AetherError.sessionBusy }
        let task = Task { try Task.checkCancellation(); return try await backend.run() }
        active = task; state = .running
        defer { active = nil; if state == .running { state = .ready }; observer?(.slotReleased) }
        return try await withTaskCancellationHandler {
            let value = try await task.value
            try Task.checkCancellation()
            return value
        } onCancel: { task.cancel() }
    }
    /// Terminal close. Ownership remains until all submitted stream work returns.
    public func unload() async {
        guard state != .closed else { return }
        observer?(.unloadStarted)
        state = .closing
        if let active { _ = await active.result }
        backend = nil; state = .closed
        observer?(.unloaded)
    }
}

@available(macOS 27, iOS 27, *)
protocol H3BlockBackend: Sendable { func run() async throws -> H3TensorData }

/// Public descriptor gate, usable without loading a native model in CPU tests.
@available(macOS 27, iOS 27, *)
public enum H3BlockBindings {
    public static func validate(_ function: ModelFunctionDescription, probe: H3BlockProbePlan.Case) throws {
        guard function.name == probe.model.function, function.states.isEmpty,
              Set(function.inputs.keys) == Set(probe.inputs.map(\.name)),
              Set(function.outputs.keys) == ["hidden_out"] else {
            throw AetherError.invalidModel("H3 function names/states differ from the verified binding plan")
        }
        for input in probe.inputs { try validate(function.inputs[input.name], spec: input.spec) }
        try validate(function.outputs["hidden_out"], spec: probe.output)
    }
    private static func validate(_ value: ModelValueDescription?, spec: H3TensorSpec) throws {
        guard let value, value.kind == "tensor", value.scalarType == spec.scalarType.rawValue,
              let shape = value.shape, shape.count == spec.shape.count,
              zip(shape, spec.shape).allSatisfy({ $0 == -1 || $0 == $1 }) else {
            throw AetherError.invalidModel("H3 tensor descriptor differs from planned scalar type/shape")
        }
    }
}
