import AetherKit
import AetherH3

/// An owned boundary snapshot. The awaited consumer may persist selected values;
/// retaining snapshots is caller-owned memory, outside native buffer admission.
public struct H3DenoisingOutput: Sendable {
    public let stage: String
    public let step: Int
    public let block: Int?
    public let tensors: [String: H3TensorData]
    public init(stage: String, step: Int, block: Int? = nil, tensors: [String: H3TensorData]) {
        self.stage = stage; self.step = step; self.block = block; self.tensors = tensors
    }
}
public struct H3DenoisingExecutionEvent: Sendable {
    public let stage: String
    public let step: Int
    public let block: Int?
    public let event: H3BlockEvent
    public init(stage: String, step: Int, block: Int? = nil, event: H3BlockEvent) {
        self.stage = stage; self.step = step; self.block = block; self.event = event
    }
}
public typealias H3DenoisingOutputHandler = @Sendable (H3DenoisingOutput) async throws -> Void

/// Experimental eight-step, provided-conditioning trajectory. No tokenizer or
/// decoder is acquired. Every run restarts its supplied initial latent state.
public actor H3DenoisingSession {
    public typealias State = H3BlockSession.State
    public private(set) var state: State = .ready
    private var backend: (any H3DenoisingBackend)?
    private var active: Task<H3Latents, any Error>?
    private let observer: (@Sendable (H3DenoisingExecutionEvent) -> Void)?
    init(backend: any H3DenoisingBackend, observer: (@Sendable (H3DenoisingExecutionEvent) -> Void)? = nil) {
        self.backend = backend; self.observer = observer
    }
    public static func open(plan: H3DenoisingPlan,
                            executionObserver: (@Sendable (H3DenoisingExecutionEvent) -> Void)? = nil) async throws -> H3DenoisingSession {
        try plan.verify(); try Task.checkCancellation()
        return H3DenoisingSession(backend: NativeH3DenoisingBackend(plan: plan, observer: executionObserver), observer: executionObserver)
    }
    public func run(initial: H3Latents, conditioning: H3TensorData,
                    onOutput: @escaping H3DenoisingOutputHandler = { _ in }) async throws -> H3Latents {
        try Task.checkCancellation()
        guard state != .closing, state != .closed, let backend else { throw AetherError.sessionClosed }
        guard active == nil else { throw AetherError.sessionBusy }
        let task = Task { try Task.checkCancellation(); return try await backend.run(initial: initial, conditioning: conditioning, onOutput: onOutput) }
        active = task; state = .running
        defer {
            active = nil; if state == .running { state = .ready }
            observer?(H3DenoisingExecutionEvent(stage: "session", step: -1, event: .slotReleased))
        }
        return try await withTaskCancellationHandler {
            let result = try await task.value; try Task.checkCancellation(); return result
        } onCancel: { task.cancel() }
    }
    public func unload() async {
        guard state != .closed else { return }
        state = .closing; observer?(H3DenoisingExecutionEvent(stage: "session", step: -1, event: .unloadStarted))
        if let active { _ = await active.result }
        backend = nil; state = .closed
        observer?(H3DenoisingExecutionEvent(stage: "session", step: -1, event: .unloaded))
    }
}
protocol H3DenoisingBackend: Sendable {
    func run(initial: H3Latents, conditioning: H3TensorData, onOutput: @escaping H3DenoisingOutputHandler) async throws -> H3Latents
}
