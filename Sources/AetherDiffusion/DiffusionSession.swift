import AetherKit
import Foundation

/// Backend seam for spatial latent diffusion. Tensors use float32 NCHW, with batch size one.
/// The decoder owns the model's latent scaling and returns planar RGB on [-1,1].
public protocol EpsilonDiffusionBackend: Sendable {
    func encode(_ tokenIDs: Int32Tensor) async throws -> FloatTensor
    func denoise(sample: FloatTensor, timestep: Float, conditioning: FloatTensor) async throws -> FloatTensor
    func decode(_ latents: FloatTensor) async throws -> GeneratedImage
    /// Joined, reusable release of all retained native stages.
    func unload() async
}

/// Serial reference orchestration, independent of Core AI and of a particular model family.
/// Owned host tensors cross each stage boundary; no device-resident performance claim.
public actor DiffusionSession {
    private let backend: any EpsilonDiffusionBackend
    private var active: Task<GeneratedImage, any Error>?
    private var closing = false
    private var closeTask: Task<Void, Never>?
    private let gpu: GPUAdmissionOwner?
    private var pending: GPUAdmissionAttempt?

    /// The session requires exclusive ownership of its backend for the entire generation, including between stages.
    public init(backend: any EpsilonDiffusionBackend, gpu: GPUAdmissionOwner? = nil) {
        self.backend = backend; self.gpu = gpu
    }

    /// Exact caller noise makes independent framework comparisons possible. Seeded noise belongs to the family adapter.
    public func generate(tokenIDs: Int32Tensor, noise: FloatTensor, schedule: EpsilonSchedule,
                         progress: @escaping @Sendable (Int, FloatTensor) throws -> Void = { _, _ in }) async throws -> GeneratedImage {
        try Task.checkCancellation()
        guard !closing else { throw AetherError.sessionClosed }
        guard active == nil else { throw AetherError.sessionBusy }
        guard noise.shape.count == 4, noise.shape[0] == 1 else { throw AetherError.invalidTensor("noise must be NCHW with batch one") }
        let backend = backend, attempt = gpu.map(GPUAdmissionAttempt.init)
        let task = Task {
            let grant = try await attempt?.admit(orThrow: { AetherError.sessionClosed })
            defer { grant?.release() }
            let result: GeneratedImage
            do {
                let conditioning = try await backend.encode(tokenIDs)
                var x = try FloatTensor(shape: noise.shape, values: noise.values.map { $0 * schedule.initialNoiseSigma })
                for step in schedule.timesteps.indices {
                    try Task.checkCancellation()
                    let epsilon = try await backend.denoise(sample: schedule.scaledInput(x, step: step),
                                                            timestep: schedule.timesteps[step], conditioning: conditioning)
                    x = try schedule.advance(x, epsilon: epsilon, step: step)
                    try progress(step, x)
                }
                try Task.checkCancellation()
                result = try await backend.decode(x)
            } catch {
                await backend.unload()
                throw error
            }
            await backend.unload()
            try Task.checkCancellation()
            return result
        }
        active = task; pending = attempt
        defer { active = nil; pending = nil }
        return try await withTaskCancellationHandler {
            let result = await task.result
            // The task has joined every submitted backend call before releasing the operation's resources.
            try Task.checkCancellation()
            return try result.get()
        } onCancel: { task.cancel() }
    }

    /// Terminal close. Rejects new work and joins submitted native calls before releasing stages.
    public func unload() async {
        if let closeTask { await closeTask.value; return }
        closing = true; pending?.withdraw()
        let operation = active, backend = backend
        let task = Task {
            if let operation { _ = await operation.result }
            await backend.unload()
        }
        closeTask = task
        await task.value
    }
}
