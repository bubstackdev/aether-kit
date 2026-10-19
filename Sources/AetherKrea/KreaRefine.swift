import AetherKit
import Foundation

/// Image-to-image start for a refine pass: the encoded picture's packed latents `x0` ([imageTokens, 64], the VAE
/// encoder's output) and the schedule step to start from. Steps `startStep…steps−1` run; the start latents are
/// `(1 − σ) · x0 + σ · noise` at that step's sigma (flow matching). `startStep` 0 would ignore the picture, so it is
/// refused; the last step is the gentlest (Turbo, 8 steps: σ ≈ 0.31 at step 7, 0.51 at step 6, 0.66 at step 5).
public struct KreaRefineStart: Sendable {
    public let latents: FloatTensor
    public let startStep: Int
    public init(latents: FloatTensor, startStep: Int) throws {
        guard latents.shape.count == 2, latents.shape[1] == 64, latents.shape[0] > 0, startStep >= 1,
              latents.values.allSatisfy(\.isFinite) else {
            throw AetherError.invalidRequest("refine start needs finite [imageTokens,64] latents and a step >= 1")
        }
        self.latents = latents; self.startStep = startStep
    }
}

public extension KreaSchedule {
    /// The flow-matching start for `step`: `(1 − σ_step) · x0 + σ_step · noise`, elementwise.
    func refineStart(x0: [Float], noise: [Float], step: Int) throws -> [Float] {
        guard timesteps.indices.contains(step), step >= 1, x0.count == noise.count else {
            throw AetherError.invalidRequest("invalid refine step or mismatched latent/noise counts")
        }
        let sigma = sigmas[step]
        return zip(x0, noise).map { roundedProduct(1 - sigma, $0) + roundedProduct(sigma, $1) }
    }
}

public extension KreaModelAsset {
    /// An asset outside the bundle manifest (the refine pass's VAE encoder), accepted only when its declared stream hash
    /// is the pinned one. Opening reads the declaration and metadata, not the stream; call `verify()` before use.
    static func external(at url: URL, expectedSHA256: String) throws -> Self {
        guard url.isFileURL else { throw AetherError.invalidModel("asset must be local") }
        let directory = url.standardizedFileURL.resolvingSymlinksInPath()
        let declaration = try Data(contentsOf: directory.appendingPathComponent("main.hash"))
        guard declaration.count == 32, declaration.hex == expectedSHA256 else {
            throw AetherError.invalidModel("external asset declaration differs from its pin: \(directory.lastPathComponent)")
        }
        return Self(url: directory, declaredSHA256: declaration.hex,
                    metadataSHA256: try hashFile(directory.appendingPathComponent("metadata.json")))
    }
}
