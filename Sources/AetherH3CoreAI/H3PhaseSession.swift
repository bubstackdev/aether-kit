import AetherKit
import AetherH3
import AetherCoreAI

/// Experimental fixed-input, outer-module session for qualification.
/// Owns verified immutable shared buffers and one native function. Supports
/// prepare/time/heads with complete named outputs; fixed inputs restart each run. This does not implement a denoising trajectory.
@available(macOS 27, iOS 27, *)
public actor H3PhaseSession {
    public enum State: String, Sendable { case ready, running, closing, closed }
    public private(set) var state: State = .ready
    private var backend: (any H3PhaseBackend)?
    private var active: Task<[String: H3TensorData], any Error>?
    private let observer: (@Sendable (H3BlockEvent) -> Void)?
    init(backend: any H3PhaseBackend, observer: (@Sendable (H3BlockEvent) -> Void)? = nil) {
        self.backend = backend; self.observer = observer
    }

    /// The optional diagnostic observer runs synchronously and must not block.
    public static func open(probe: H3PhaseProbePlan.Case,
                            executionObserver: (@Sendable (H3BlockEvent) -> Void)? = nil) async throws -> H3PhaseSession {
        try Task.checkCancellation()
        let backend = try await NativeH3PhaseBackend.open(probe, observer: executionObserver)
        try Task.checkCancellation()
        return H3PhaseSession(backend: backend, observer: executionObserver)
    }

    public func run() async throws -> [String: H3TensorData] {
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
protocol H3PhaseBackend: Sendable { func run() async throws -> [String: H3TensorData] }

/// CPU descriptor validation before native dispatch.
@available(macOS 27, iOS 27, *)
public enum H3PhaseBindings {
    public static func validate(_ function: ModelFunctionDescription, probe: H3PhaseProbePlan.Case) throws {
        guard function.name == probe.model.function, function.states.isEmpty,
              Set(function.inputs.keys) == Set(probe.inputs.map(\.name)),
              Set(function.outputs.keys) == Set(probe.references.map(\.name)) else {
            throw AetherError.invalidModel("H3 outer function names or states differ")
        }
        for input in probe.inputs { try validate(function.inputs[input.name], spec: input.spec) }
        for output in probe.references { try validate(function.outputs[output.name], spec: output.spec) }
    }
    private static func validate(_ value: ModelValueDescription?, spec: H3TensorSpec) throws {
        guard let value, value.kind == "tensor", value.scalarType == spec.scalarType.rawValue,
              let shape = value.shape, shape.count == spec.shape.count,
              zip(shape, spec.shape).allSatisfy({ $0 == -1 || $0 == $1 }) else {
            throw AetherError.invalidModel("H3 outer descriptor precision or dimensions differ")
        }
    }
}
