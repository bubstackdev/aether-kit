import CoreGraphics
import Foundation
import ImageIO

/// An 8-bit interleaved RGB image for vision preprocessing.
public struct VisionRGBImage: Sendable, Equatable {
    public let width: Int
    public let height: Int
    /// Row-major RGB, `width · height · 3` bytes.
    public let rgb: [UInt8]

    public enum Failure: Error, Equatable { case invalidPixels, undecodable }

    public init(rgb: [UInt8], width: Int, height: Int) throws {
        guard width > 0, height > 0, rgb.count == width * height * 3 else { throw Failure.invalidPixels }
        self.width = width
        self.height = height
        self.rgb = rgb
    }

    /// Draws `image` into 8-bit RGB in its own RGB colour space when it has one (no colour
    /// management, like the reference processor's decoding), otherwise in sRGB.
    public init(cgImage image: CGImage) throws {
        let width = image.width, height = image.height
        guard width > 0, height > 0 else { throw Failure.invalidPixels }
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width * 4, space: space,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw Failure.undecodable }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            rgb[3 * i] = rgba[4 * i]; rgb[3 * i + 1] = rgba[4 * i + 1]; rgb[3 * i + 2] = rgba[4 * i + 2]
        }
        try self.init(rgb: rgb, width: width, height: height)
    }

    /// Decodes an image file with ImageIO, applying its EXIF orientation.
    public init(contentsOf url: URL) throws {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let pixelWidth = properties[kCGImagePropertyPixelWidth] as? Int,
              let pixelHeight = properties[kCGImagePropertyPixelHeight] as? Int else { throw Failure.undecodable }
        let orientation = properties[kCGImagePropertyOrientation] as? UInt32 ?? 1
        let image: CGImage?
        if orientation == 1 {
            image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        } else {
            // A full-size thumbnail with the orientation transform applied.
            image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: max(pixelWidth, pixelHeight)] as CFDictionary)
        }
        guard let image else { throw Failure.undecodable }
        try self.init(cgImage: image)
    }
}
