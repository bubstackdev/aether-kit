import AetherKit
import AetherCoreAI
import AetherKrea

/// Experimental resident encoder using the prototype's `encode` entry point.
/// Native qualification is pending. Concurrent requests fail with `sessionBusy`.
@available(macOS 27, iOS 27, *)
public struct KreaTextEncoder: Sendable {
    private let session: ModelSession

    public static func open(
        assets: KreaAssets, compute: ModelComputePolicy = .preferGPU
    ) async throws -> Self {
        let session = try await ModelSession.open(
            at: assets.textEncoder.url, compute: compute, expectFrequentReshapes: true
        )
        return Self(session: session)
    }

    public func encode(_ prompt: KreaPrompt) async throws -> KreaConditioning {
        let result = try await session.runTensors(function: "encode", inputs: prompt.encoderInputs)
        guard result.count == 1, case .float16(let tensor) = result["text_embeds"] else {
            throw AetherError.invalidTensor("Krea encoder must return float16 text_embeds")
        }
        return try KreaConditioning(encoderOutput: tensor, prompt: prompt)
    }

    /// Terminal close. Open a new encoder to load again.
    public func unload() async { await session.unload() }
}
