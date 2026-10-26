// Host policy from the Apache-2.0 Aether prototype (11 §B14). See NOTICE.
import AetherKit
import CoreGraphics
import Foundation

/// One conditioning picture for reference-image generation, as planar RGB float32 `[3, H, W]` on the
/// VAE's [−1, 1] scale with both sides positive multiples of 16. References are encoded once by the
/// bundle's VAE encoder into immutable packed latents; they are conditioning, not an img2img start.
/// No bytes, paths or contents are logged or cached by the session.
public struct Flux2ReferenceImage: Sendable, Equatable {
    public let pixels: FloatTensor
    public var height: Int { pixels.shape[1] }
    public var width: Int { pixels.shape[2] }
    public var rows: Int { height / 16 }
    public var columns: Int { width / 16 }
    public var tokens: Int { rows * columns }

    /// Exact planar pixels, e.g. a fixture's recorded encoder input. Values must be finite.
    public init(pixels: FloatTensor) throws {
        guard pixels.shape.count == 3, pixels.shape[0] == 3, pixels.shape[1] >= 16, pixels.shape[2] >= 16,
              pixels.shape[1].isMultiple(of: 16), pixels.shape[2].isMultiple(of: 16),
              pixels.shape[1] <= 4096, pixels.shape[2] <= 4096 else {
            throw AetherError.invalidRequest("FLUX.2 reference pixels must be [3, 16h, 16w]")
        }
        self.pixels = pixels
    }

    /// The bounded preparation size: fit inside `maxSide` without upscaling, then floor each side to a
    /// multiple of 16 (at least 16).
    public static func preparedSize(width: Int, height: Int, maxSide: Int) throws -> (width: Int, height: Int) {
        guard width > 0, height > 0, maxSide >= 16, maxSide.isMultiple(of: 16) else {
            throw AetherError.invalidRequest("invalid reference image or side bound")
        }
        let scale = min(1, Double(maxSide) / Double(max(width, height)))
        let w = max(16, Int(Double(width) * scale) / 16 * 16)
        let h = max(16, Int(Double(height) * scale) / 16 * 16)
        return (w, h)
    }

    /// Resamples a decoded image into sRGB with Core Graphics' high-quality interpolation and maps each
    /// 8-bit channel `v` to `v/127.5 − 1`. Alpha is composited onto black; callers needing a different
    /// background should flatten first. This resampler is not the reference library's Lanczos filter, so
    /// references prepared from the same file differ slightly from an upstream pipeline's tensors.
    public init(image: CGImage, maxSide: Int) throws {
        let size = try Self.preparedSize(width: image.width, height: image.height, maxSide: maxSide)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: size.width, height: size.height, bitsPerComponent: 8,
                                      bytesPerRow: size.width * 4, space: space,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw AetherError.invalidRequest("could not prepare reference image")
        }
        context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        guard let data = context.data else { throw AetherError.invalidRequest("could not read reference image") }
        let bytes = data.bindMemory(to: UInt8.self, capacity: size.width * size.height * 4)
        let plane = size.width * size.height
        var values = [Float](repeating: 0, count: plane * 3)
        for pixel in 0..<plane {
            for channel in 0..<3 { values[channel * plane + pixel] = Float(bytes[pixel * 4 + channel]) / 127.5 - 1 }
        }
        try self.init(pixels: FloatTensor(shape: [3, size.height, size.width], values: values))
    }
}

/// Validates a reference request against the bundle's bounds before any model loads.
public enum Flux2ReferencePolicy {
    public static func check(_ references: [(rows: Int, columns: Int)], generatedTokens: Int,
                             limits: Flux2ReferenceLimits?) throws {
        guard !references.isEmpty else { return }
        guard let limits else { throw AetherError.unsupported("this FLUX.2 bundle declares no reference images") }
        guard references.count <= limits.maxImages else {
            throw AetherError.invalidRequest("at most \(limits.maxImages) reference images are supported")
        }
        guard references.allSatisfy({ $0.rows * 16 <= limits.maxSide && $0.columns * 16 <= limits.maxSide }) else {
            throw AetherError.invalidRequest("reference sides must not exceed \(limits.maxSide) pixels")
        }
        let total = references.reduce(generatedTokens) { $0 + $1.rows * $1.columns }
        guard total <= limits.maxTotalTokens else {
            throw AetherError.invalidRequest("\(total) combined image tokens exceed the bundle limit \(limits.maxTotalTokens)")
        }
    }
}
