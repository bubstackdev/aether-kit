import AetherKit

/// Geometry, step count and seed for one FLUX.2 [klein] generation.
/// Sizes are refused, never rounded: both sides must be positive multiples of 16.
/// This is an execution contract of the dynamic export, not a claim of device qualification.
public struct Flux2Configuration: Sendable, Equatable {
    /// Packed latent channels per token: 32 VAE channels × a 2×2 patch.
    public static let latentChannels = 128
    /// The export's dynamic image-token bound (generated plus reference tokens), 1024² for text-only use.
    public static let maximumImageTokens = 4096
    public let width: Int
    public let height: Int
    public let steps: Int
    public let seed: UInt64

    public var rows: Int { height / 16 }
    public var columns: Int { width / 16 }
    public var imageTokenCount: Int { rows * columns }

    /// Klein is step-distilled for four steps without guidance.
    public init(width: Int, height: Int, steps: Int = 4, seed: UInt64 = 0) throws {
        guard width >= 32, height >= 32, width.isMultiple(of: 16), height.isMultiple(of: 16) else {
            throw AetherError.invalidRequest("FLUX.2 dimensions must be multiples of 16, at least 32")
        }
        let tokens = (width / 16).multipliedReportingOverflow(by: height / 16)
        guard !tokens.overflow, tokens.partialValue <= Self.maximumImageTokens else {
            throw AetherError.invalidRequest("FLUX.2 export supports at most \(Self.maximumImageTokens) image tokens")
        }
        guard (1...50).contains(steps) else { throw AetherError.invalidRequest("FLUX.2 steps must be in 1...50") }
        self.width = width; self.height = height; self.steps = steps; self.seed = seed
    }

    /// Initial packed noise `[imageTokens, 128]` from the documented seeded generator.
    public func initialNoise() throws -> FloatTensor {
        try FloatTensor(shape: [imageTokenCount, Self.latentChannels],
                        values: SeededGaussianNoise.generate(seed: seed, count: imageTokenCount * Self.latentChannels))
    }
}
