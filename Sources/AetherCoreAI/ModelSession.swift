import AetherKit
import AetherAssets
import Foundation

/// Explicit caller estimates for model residency and peak per-call temporary storage.
/// Unknown sizes must be qualified before using budgeted execution; these are not measured memory.
@available(macOS 27, iOS 27, *)
public struct ModelResourceAdmission: Sendable {
    public let coordinator: ResourceCoordinator
    public let residentBytes: UInt64
    public let runScratchBytes: UInt64
    public init(coordinator: ResourceCoordinator, residentBytes: UInt64, runScratchBytes: UInt64) throws {
        guard residentBytes > 0, runScratchBytes > 0 else { throw ResourceAdmissionError.invalidByteCount }
        self.coordinator = coordinator; self.residentBytes = residentBytes; self.runScratchBytes = runScratchBytes
    }
}

/// Experimental local-model reference executor, separate from generation APIs.
///
/// Executes one stateless float32, float16 or int32 call at a time. Concurrent calls fail with
/// `sessionBusy`; `unload()` rejects new calls and waits for the active call.
/// Native calls are awaited before their buffers and model can be released.
@available(macOS 27, iOS 27, *)
public actor ModelSession {
    public enum State: String, Sendable { case ready, running, closing, closed }

    public nonisolated let functions: [ModelFunctionDescription]
    public private(set) var state: State = .ready
    private var backend: (any ModelBackend)?
    private var active: Task<[String: TensorValue], any Error>?
    private let admission: ModelResourceAdmission?
    private var residency: ResourceReservation?
    private var installedAssets: AssetReadLease?

    init(backend: any ModelBackend, functions: [ModelFunctionDescription] = [],
         admission: ModelResourceAdmission? = nil, residency: ResourceReservation? = nil,
         installedAssets: AssetReadLease? = nil) {
        self.backend = backend
        self.functions = functions
        self.admission = admission
        self.residency = residency
        self.installedAssets = installedAssets
    }

    public static func open(
        at url: URL, compute: ModelComputePolicy = .automatic, expectFrequentReshapes: Bool = false,
        admission: ModelResourceAdmission? = nil
    ) async throws -> ModelSession {
        try await load(at: url, compute: compute, expectFrequentReshapes: expectFrequentReshapes,
                       admission: admission, installedAssets: nil)
    }

    /// Opens a declared model directory/file and keeps the installation read-protected until unload.
    public static func open(
        modelPath: String, in installation: AssetReadLease, compute: ModelComputePolicy = .automatic,
        expectFrequentReshapes: Bool = false, admission: ModelResourceAdmission? = nil
    ) async throws -> ModelSession {
        let components = modelPath.split(separator: "/", omittingEmptySubsequences: false)
        guard !components.isEmpty, components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
              !modelPath.contains("\\"), !modelPath.utf8.contains(0),
              installation.manifest.files.contains(where: { $0.path == modelPath || $0.path.hasPrefix(modelPath + "/") }) else {
            throw AetherError.invalidModel("model path is not declared in the installation")
        }
        let url = installation.directory.appendingPathComponent(modelPath)
        guard url.standardizedFileURL.path == url.resolvingSymlinksInPath().path else {
            throw AetherError.invalidModel("installed model path contains a symlink")
        }
        return try await load(at: url, compute: compute, expectFrequentReshapes: expectFrequentReshapes,
                              admission: admission, installedAssets: installation)
    }

    private static func load(
        at url: URL, compute: ModelComputePolicy, expectFrequentReshapes: Bool,
        admission: ModelResourceAdmission?, installedAssets: AssetReadLease?
    ) async throws -> ModelSession {
        try Task.checkCancellation()
        let residency = try admission.map { try $0.coordinator.reserve(bytes: $0.residentBytes) }
        do {
            let backend = try await CoreAIBackend.load(at: url, compute: compute, expectFrequentReshapes: expectFrequentReshapes)
            return ModelSession(backend: backend, functions: backend.functions, admission: admission, residency: residency, installedAssets: installedAssets)
        } catch {
            residency?.release()
            throw error
        }
    }

    deinit { backend = nil; installedAssets = nil; residency?.release() }

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
        guard state != .closed, state != .closing, backend != nil else {
            throw AetherError.sessionClosed
        }
        guard active == nil else { throw AetherError.sessionBusy }
        let task = try startRun(function: function, inputs: inputs)
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

    // Keep the backend's local reference out of the suspended caller frame: once the task joins,
    // unload can release the model without a separate caller continuation retaining its weights.
    private func startRun(function: String, inputs: [String: TensorValue]) throws -> Task<[String: TensorValue], any Error> {
        guard let backend else { throw AetherError.sessionClosed }
        let scratch = try admission.map { try $0.coordinator.reserve(bytes: $0.runScratchBytes) }
        return Task {
            defer { scratch?.release() }
            try Task.checkCancellation()
            return try await backend.run(function: function, inputs: inputs)
        }
    }

    /// Waits for active inference, then releases this session's model ownership.
    /// This does not evict the OS specialization cache or promise page eviction.
    public func unload() async {
        guard state != .closed else { return }
        state = .closing
        if let active { _ = await active.result }
        backend = nil
        installedAssets = nil
        residency?.release()
        residency = nil
        state = .closed
    }
}
