import AetherKit
import Foundation

/// Experimental local-model reference executor, separate from generation APIs.
///
/// Executes one stateless float32, float16 or int32 call at a time. Concurrent calls fail with
/// `sessionBusy`; `unload()` rejects new calls and waits for the active call.
/// Native calls are awaited before their buffers and model can be released.
public actor ModelSession {
    public enum State: String, Sendable { case ready, running, closing, closed }

    public nonisolated let functions: [ModelFunctionDescription]
    public private(set) var state: State = .ready
    private var backend: (any ModelBackend)?
    private var active: Task<[String: TensorValue], any Error>?

    init(backend: any ModelBackend, functions: [ModelFunctionDescription] = []) {
        self.backend = backend
        self.functions = functions
    }

    public static func open(
        at url: URL, compute: ModelComputePolicy = .automatic, expectFrequentReshapes: Bool = false
    ) async throws -> ModelSession {
        let backend = try await CoreAIBackend.load(at: url, compute: compute, expectFrequentReshapes: expectFrequentReshapes)
        return ModelSession(backend: backend, functions: backend.functions)
    }

    /// Float32 convenience API. Mixed scalar types use `runTensors`.
    public func run(
        function: String = "main", inputs: [String: FloatTensor]
    ) async throws -> [String: FloatTensor] {
        let outputs = try await runTensors(function: function, inputs: inputs.mapValues(TensorValue.float32))
        return try outputs.mapValues { value in
            guard case .float32(let tensor) = value else {
                throw AetherError.unsupported("use runTensors for non-float32 outputs")
            }
            return tensor
        }
    }

    public func runTensors(
        function: String = "main", inputs: [String: TensorValue]
    ) async throws -> [String: TensorValue] {
        try Task.checkCancellation()
        guard state != .closed, state != .closing, let backend else {
            throw AetherError.sessionClosed
        }
        guard active == nil else { throw AetherError.sessionBusy }
        let task = Task {
            try Task.checkCancellation()
            return try await backend.run(function: function, inputs: inputs)
        }
        active = task
        state = .running
        defer {
            active = nil
            if state == .running { state = .ready }
        }
        return try await withTaskCancellationHandler {
            let result = try await task.value
            try Task.checkCancellation()
            return result
        } onCancel: {
            task.cancel()
        }
    }

    /// Waits for active inference, then releases this session's model ownership.
    /// This does not evict the OS specialization cache or promise page eviction.
    public func unload() async {
        guard state != .closed else { return }
        state = .closing
        if let active { _ = await active.result }
        backend = nil
        state = .closed
    }
}
