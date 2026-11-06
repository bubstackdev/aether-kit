public import CoreGraphics
public import Foundation
import ImageIO
import UniformTypeIdentifiers

/// An image a generative driver produced: planar float32 RGB `[3, height, width]` on the model's `[-1, 1]` scale,
/// kept as decoded (unclipped), so a parity check reads the decoder's own values. 8-bit forms are derived from it.
public struct GeneratedImage: Sendable, Hashable {
    public let width: Int
    public let height: Int
    /// Planar RGB, `[3, height, width]` row-major.
    public let planarRGB: [Float]

    /// Fails unless `planarRGB` holds `3 · width · height` finite values.
    public init(width: Int, height: Int, planarRGB: [Float]) throws(AetherError) {
        guard width > 0, height > 0, case let (plane, overflow) = width.multipliedReportingOverflow(by: height), !overflow,
              plane <= Int.max / 3, planarRGB.count == 3 * plane else {
            throw .tensor(name: "image", reason: .shapeMismatch(expected: [3, height, width], found: [planarRGB.count]))
        }
        guard planarRGB.allSatisfy(\.isFinite) else {
            throw .tensor(name: "image", reason: .unsupported("the decoded image has a non-finite value"))
        }
        self.width = width
        self.height = height
        self.planarRGB = planarRGB
    }

    /// Interleaved 8-bit RGB: `clamp(x / 2 + 0.5, 0, 1) · 255` in Float, rounded ties-to-even (0.0.x's and the lab's
    /// quantization, so its hashes carry over).
    public func rgb8() -> [UInt8] {
        let plane = width * height
        return [UInt8](unsafeUninitializedCapacity: plane * 3) { bytes, initialized in
            planarRGB.withUnsafeBufferPointer { values in
                for pixel in 0..<plane {
                    for channel in 0..<3 {
                        let scaled = min(max(values[channel * plane + pixel] / 2 + 0.5, 0), 1) * 255
                        bytes[pixel * 3 + channel] = UInt8(scaled.rounded(.toNearestOrEven))
                    }
                }
            }
            initialized = plane * 3
        }
    }

    /// The image as an sRGB `CGImage` (8 bits per channel, no alpha).
    public func cgImage() throws(AetherError) -> CGImage {
        guard let provider = CGDataProvider(data: Data(rgb8()) as CFData),
              let space = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: width * 3,
                space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider,
                decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else {
            throw .runtime(underlying: CocoaError(.coderInvalidValue))
        }
        return image
    }

    /// PNG bytes, encoded in memory.
    public func pngData() throws(AetherError) -> Data {
        let image = try cgImage()
        let encoded = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(encoded, UTType.png.identifier as CFString, 1, nil) else {
            throw .runtime(underlying: CocoaError(.fileWriteUnknown))
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw .runtime(underlying: CocoaError(.fileWriteUnknown)) }
        return encoded as Data
    }
}
