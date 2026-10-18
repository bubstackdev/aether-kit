import AetherKit

/// The supported geometry and schedule of the prototype's dynamic Krea export.
/// This is an execution contract, not a claim of device qualification.
public struct KreaConfiguration: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let steps: Int
    public let seed: UInt64

    public var imageTokenCount: Int { (width / 16) * (height / 16) }

    public init(width: Int, height: Int, steps: Int = 8, seed: UInt64 = 0) throws {
        guard width > 0, height > 0, width.isMultiple(of: 16), height.isMultiple(of: 16) else {
            throw AetherError.invalidRequest("Krea dimensions must be positive multiples of 16")
        }
        let tokens = (width / 16).multipliedReportingOverflow(by: height / 16)
        guard !tokens.overflow, tokens.partialValue <= 4096 else {
            throw AetherError.invalidRequest("Krea export supports at most 4096 image tokens")
        }
        guard (1...50).contains(steps) else {
            throw AetherError.invalidRequest("Krea steps must be in 1...50")
        }
        self.width = width
        self.height = height
        self.steps = steps
        self.seed = seed
    }
}
