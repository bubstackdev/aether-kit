import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Finite float32 planar RGB on the model's [-1,1] scale, retained before clipping.
public struct GeneratedImage: Sendable {
    public let pixelsCHW: FloatTensor
    public var width: Int { pixelsCHW.shape[2] }
    public var height: Int { pixelsCHW.shape[1] }

    public init(pixelsCHW: FloatTensor) throws {
        guard pixelsCHW.shape.count == 3, pixelsCHW.shape[0] == 3 else {
            throw AetherError.invalidTensor("image pixels must be [3,height,width]")
        }
        self.pixelsCHW = pixelsCHW
    }

    /// Matches the lab's float32 scaling and ties-to-even quantization.
    public func rgb8() -> [UInt8] {
        let plane = width * height
        var bytes: [UInt8] = []
        bytes.reserveCapacity(plane * 3)
        for pixel in 0..<plane {
            for channel in 0..<3 {
                let scaled = min(max(pixelsCHW.values[channel * plane + pixel] / 2 + 0.5, 0), 1) * 255
                bytes.append(UInt8(scaled.rounded(.toNearestOrEven)))
            }
        }
        return bytes
    }

    public func savePNG(to url: URL) throws {
        guard url.isFileURL else { throw AetherError.invalidRequest("PNG destination must be local") }
        try pngData().write(to: url, options: .atomic)
    }

    /// Encodes owned image pixels without publishing an intermediate file.
    public func pngData() throws -> Data {
        let encoded = NSMutableData()
        let data = Data(rgb8())
        guard let provider = CGDataProvider(data: data as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                  bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider,
                  decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let destination = CGImageDestinationCreateWithData(encoded, UTType.png.identifier as CFString, 1, nil) else {
            throw AetherError.invalidRequest("could not create PNG destination")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw AetherError.invalidRequest("could not write PNG")
        }
        return encoded as Data
    }
}
