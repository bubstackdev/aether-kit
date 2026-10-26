// Derived from the Apache-2.0 Aether prototype. See NOTICE.
import AetherKit

/// Experimental H3 geometry contract; acceptance does not establish device support.
/// Requested frames snap upward to the source VAE's 17n+5 frame grid at 24 fps.
public struct H3Geometry: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let requestedFrames: Int
    public let frames: Int
    public let latentFrames: Int
    public let latentHeight: Int
    public let latentWidth: Int
    public let targetVideoRows: Int
    public var durationSeconds: Double { Double(frames) / 24 }
    public var videoLatentShape: [Int] { [24, latentFrames, latentHeight, latentWidth] }
    public var audioLatentsPerChannel: Int { Int((Double(frames) / 24 * 40).rounded(.toNearestOrEven)) }

    public init(width: Int = 960, height: Int = 544, requestedFrames: Int = 120, diagnostic: Bool = false) throws {
        guard width > 0, height > 0, width.isMultiple(of: 32), height.isMultiple(of: 32) else {
            throw AetherError.invalidRequest("H3 dimensions must be positive multiples of 32")
        }
        let area = width.multipliedReportingOverflow(by: height)
        guard !area.overflow, area.partialValue <= 1_032_192, (1...360).contains(requestedFrames) else {
            throw AetherError.invalidRequest("H3 requires at most 1,032,192 pixels and 1...360 requested frames")
        }
        let n = (max(0, requestedFrames - 5) + 16) / 17
        let frames = 17 * n + 5
        guard frames <= 360, diagnostic || frames >= 120 else {
            throw AetherError.invalidRequest("H3 snapped duration must be 5...15 seconds; shorter clips require diagnostic mode")
        }
        self.width = width; self.height = height; self.requestedFrames = requestedFrames
        self.frames = frames; self.latentFrames = 5 * n + 2
        self.latentHeight = height / 16; self.latentWidth = width / 16
        self.targetVideoRows = (5 * n + 2) * (height / 32) * (width / 32)
    }

    /// Packs source [24,T,H,W] into [T*H/2*W/2,96], preserving channel/patch order.
    public func packVideo(_ latents: FloatTensor) throws -> FloatTensor {
        guard latents.shape == videoLatentShape else { throw AetherError.invalidTensor("H3 video noise must be [24,T,H,W]") }
        var result: [Float] = []; result.reserveCapacity(latents.values.count)
        for f in 0..<latentFrames {
            for y in stride(from: 0, to: latentHeight, by: 2) {
                for x in stride(from: 0, to: latentWidth, by: 2) {
                    for channel in 0..<24 {
                        for dy in 0..<2 {
                            for dx in 0..<2 {
                                result.append(latents.values[((channel * latentFrames + f) * latentHeight + y + dy) * latentWidth + x + dx])
                            }
                        }
                    }
                }
            }
        }
        return try FloatTensor(shape: [targetVideoRows, 96], values: result)
    }
}
