import AetherKit
import AetherH3

public struct H3ChainEvent: Sendable {
    public let block: Int?
    public let event: H3BlockEvent
    public init(block: Int? = nil, event: H3BlockEvent) { self.block = block; self.event = event }
}

public typealias H3ChainOutputHandler = @Sendable (Int, H3TensorData) async throws -> Void

/// Experimental bounded chain session. Fixed inputs restart each run; intermediate
/// values pass through distinct Metal ping-pong buffers, without host narrowing.
public actor H3ChainSession {
    public typealias State = H3BlockSession.State
    public private(set) var state: State = .ready
    private var backend: (any H3ChainBackend)?
    private var active: Task<[H3TensorData], any Error>?
    private let observer: (@Sendable (H3ChainEvent) -> Void)?
    init(backend: any H3ChainBackend, observer: (@Sendable (H3ChainEvent) -> Void)? = nil) {
        self.backend = backend; self.observer = observer
    }
    /// Diagnostic callbacks are synchronous and must not block.
    public static func open(probe: H3ChainProbePlan.Case,
                            executionObserver: (@Sendable (H3ChainEvent) -> Void)? = nil) async throws -> H3ChainSession {
        try Task.checkCancellation()
        let backend = try await NativeH3ChainBackend.open(probe, observer: executionObserver)
        try Task.checkCancellation()
        return H3ChainSession(backend: backend, observer: executionObserver)
    }
    /// Returns owned snapshots for the three-block profile only.
    public func run() async throws -> [H3TensorData] { try await perform(delivery: nil) }
    /// Delivers one owned block snapshot at a time. Consumers control any data
    /// they retain; retained consumer data is outside the native buffer bound.
    public func run(onOutput: @escaping H3ChainOutputHandler) async throws {
        _ = try await perform(delivery: onOutput)
    }
    private func perform(delivery: H3ChainOutputHandler?) async throws -> [H3TensorData] {
        try Task.checkCancellation()
        guard state != .closing, state != .closed, let backend else { throw AetherError.sessionClosed }
        guard active == nil else { throw AetherError.sessionBusy }
        let task = Task {
            try Task.checkCancellation()
            if let delivery { try await backend.run(onOutput: delivery); return [H3TensorData]() }
            return try await backend.run()
        }
        active = task; state = .running
        defer { active = nil; if state == .running { state = .ready }; observer?(H3ChainEvent(event: .slotReleased)) }
        return try await withTaskCancellationHandler {
            let values = try await task.value
            try Task.checkCancellation(); return values
        } onCancel: { task.cancel() }
    }
    public func unload() async {
        guard state != .closed else { return }
        observer?(H3ChainEvent(event: .unloadStarted)); state = .closing
        if let active { _ = await active.result }
        backend = nil; state = .closed
        observer?(H3ChainEvent(event: .unloaded))
    }
}
protocol H3ChainBackend: Sendable {
    func run() async throws -> [H3TensorData]
    func run(onOutput: @escaping H3ChainOutputHandler) async throws
}
extension H3ChainBackend {
    func run(onOutput: @escaping H3ChainOutputHandler) async throws {
        for (index, value) in try await run().enumerated() {
            try Task.checkCancellation(); try await onOutput(index, value)
        }
    }
}
