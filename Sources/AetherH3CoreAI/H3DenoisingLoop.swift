import AetherKit
import AetherH3

protocol H3DenoisingKernels: Sendable {
    func prepare(step: Int, latents: H3Latents, conditioning: H3TensorData) async throws -> H3TensorData
    func backbone(step: Int, hidden: H3TensorData, onOutput: @escaping H3DenoisingOutputHandler) async throws -> H3TensorData
    func heads(step: Int, hidden: H3TensorData, latents: H3Latents) async throws -> H3Latents
}

/// Shared native/CPU-test control flow. State advances only after both velocity
/// outputs validate and the complete callback returns; requests remain immutable.
enum H3DenoisingLoop {
    static func run(initial: H3Latents, conditioning: H3TensorData, schedule: H3Schedule,
                    kernels: any H3DenoisingKernels, onOutput: @escaping H3DenoisingOutputHandler) async throws -> H3Latents {
        var latents = initial
        for step in schedule.steps {
            try Task.checkCancellation()
            try await onOutput(H3DenoisingOutput(stage: "input", step: step.index, tensors: ["video": latents.video,"audio": latents.audio]))
            try Task.checkCancellation()
            let prepared = try await kernels.prepare(step: step.index, latents: latents, conditioning: conditioning)
            try await onOutput(H3DenoisingOutput(stage: "prepare", step: step.index, tensors: ["hidden": prepared]))
            try Task.checkCancellation()
            let hidden = try await kernels.backbone(step: step.index, hidden: prepared, onOutput: onOutput)
            try Task.checkCancellation()
            let velocity = try await kernels.heads(step: step.index, hidden: hidden, latents: latents)
            try await onOutput(H3DenoisingOutput(stage: "heads", step: step.index, tensors: ["video": velocity.video,"audio": velocity.audio]))
            try Task.checkCancellation()
            latents = try latents.advanced(schedule: schedule, step: step.index, videoVelocity: velocity.video, audioVelocity: velocity.audio)
            try await onOutput(H3DenoisingOutput(stage: "latents", step: step.index, tensors: ["video": latents.video,"audio": latents.audio]))
        }
        try Task.checkCancellation(); return latents
    }
}
