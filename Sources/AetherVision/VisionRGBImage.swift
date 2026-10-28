import CoreGraphics
import Foundation
import ImageIO

/// What decoding does with an image's alpha channel.
public enum VisionAlphaPolicy: Sendable, Equatable {
    /// Drops alpha and keeps each pixel's stored colour, as the reference processor's `convert("RGB")` does (the
    /// default, for parity with upstream preprocessing). Exact for 8- and 16-bit images with straight (unpremultiplied)
    /// alpha, which is how ImageIO returns PNG, TIFF and similar files. A source that arrives premultiplied has already
    /// lost the stored colour of transparent pixels: its colours are unpremultiplied where alpha > 0, and fully
    /// transparent pixels become black.
    case dropAlpha
    /// Composites over an opaque colour (Core Graphics source-over in the image's own RGB colour space). Kit 0.0.2 always
    /// composited over black: `.composite(red: 0, green: 0, blue: 0)` keeps that behaviour.
    case composite(red: UInt8, green: UInt8, blue: UInt8)
}

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

    /// Converts `image` to 8-bit RGB in its own RGB colour space when it has one (no colour management, like the
    /// reference processor's decoding), otherwise in sRGB, treating alpha by `alpha`.
    public init(cgImage image: CGImage, alpha: VisionAlphaPolicy = .dropAlpha) throws {
        guard image.width > 0, image.height > 0 else { throw Failure.invalidPixels }
        let hasAlpha = ![.none, .noneSkipLast, .noneSkipFirst].contains(image.alphaInfo)
        if hasAlpha, alpha == .dropAlpha, let rgb = Self.storedRGB(of: image) {
            try self.init(rgb: rgb, width: image.width, height: image.height)
            return
        }
        let width = image.width, height = image.height
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
        let premultiplied = hasAlpha && alpha == .dropAlpha
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width * 4, space: space,
                                          bitmapInfo: (premultiplied ? CGImageAlphaInfo.premultipliedLast
                                                                     : CGImageAlphaInfo.noneSkipLast).rawValue) else { return false }
            context.interpolationQuality = .none
            if case .composite(let red, let green, let blue) = alpha, red | green | blue != 0 {
                context.setFillColor(CGColor(colorSpace: space, components: [CGFloat(red) / 255, CGFloat(green) / 255,
                                                                             CGFloat(blue) / 255, 1])!)
                context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw Failure.undecodable }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            let a = Int(rgba[4 * i + 3])
            for c in 0..<3 {
                let value = Int(rgba[4 * i + c])
                // A premultiplied source has lost its transparent pixels' stored colour: unpremultiply what remains.
                rgb[3 * i + c] = premultiplied ? (a == 0 ? 0 : UInt8(min(255, (value * 255 + a / 2) / a))) : UInt8(value)
            }
        }
        try self.init(rgb: rgb, width: width, height: height)
    }

    /// The stored colour of every pixel of an image with straight alpha, read from its own bytes (alpha dropped, 16-bit
    /// values reduced to their high byte, grey repeated on three channels), or nil for any other layout. Components are
    /// read in memory order; a 16-bit component may be in either byte order (ImageIO returns 16-bit PNG little-endian).
    /// Layouts that reorder whole 32-bit pixels take the drawing path.
    private static func storedRGB(of image: CGImage) -> [UInt8]? {
        let straight: [CGImageAlphaInfo] = [.last, .first]
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        guard straight.contains(image.alphaInfo), [8, 16].contains(image.bitsPerComponent),
              !image.bitmapInfo.contains(.floatComponents),
              [CGBitmapInfo.byteOrderDefault, .byteOrder16Big, .byteOrder32Big].contains(order)
                || (order == .byteOrder16Little && image.bitsPerComponent == 16),
              let model = image.colorSpace?.model, model == .rgb || model == .monochrome,
              let data = image.dataProvider?.data, let bytes = CFDataGetBytePtr(data) else { return nil }
        let channels = model == .rgb ? 4 : 2
        let componentBytes = image.bitsPerComponent / 8
        guard image.bitsPerPixel == channels * image.bitsPerComponent else { return nil }
        let width = image.width, height = image.height, rowBytes = image.bytesPerRow
        guard CFDataGetLength(data) >= rowBytes * (height - 1) + width * channels * componentBytes else { return nil }
        // Colour components in memory order: alpha last (R, G, B, A / grey, A) or first (A, R, G, B / A, grey).
        let colours = image.alphaInfo == .first ? Array(1..<channels) : Array(0..<(channels - 1))
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                let pixel = y * rowBytes + x * channels * componentBytes
                // The high byte of a 16-bit component is its first in big-endian order, its second in little-endian.
                let high = order == .byteOrder16Little ? 1 : 0
                let values = colours.map { bytes[pixel + $0 * componentBytes + (componentBytes == 2 ? high : 0)] }
                let d = 3 * (y * width + x)
                rgb[d] = values[0]
                rgb[d + 1] = values[model == .rgb ? 1 : 0]
                rgb[d + 2] = values[model == .rgb ? 2 : 0]
            }
        }
        return rgb
    }

    /// Decodes an image file with ImageIO, applying its EXIF orientation, and treating alpha by `alpha`. The image is
    /// decoded at full size and oriented by moving pixels, never resampled: ImageIO's oriented thumbnails are
    /// premultiplied, which would lose transparent pixels' stored colours.
    public init(contentsOf url: URL, alpha: VisionAlphaPolicy = .dropAlpha) throws {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw Failure.undecodable }
        let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
        let stored = try VisionRGBImage(cgImage: image, alpha: alpha)
        self = (2...8).contains(orientation) ? stored.oriented(orientation) : stored
    }

    /// The image displayed under EXIF `orientation` (2–8), by moving pixels.
    func oriented(_ orientation: Int) -> VisionRGBImage {
        let swapped = orientation >= 5
        let outWidth = swapped ? height : width, outHeight = swapped ? width : height
        var out = [UInt8](repeating: 0, count: rgb.count)
        for y in 0..<outHeight {
            for x in 0..<outWidth {
                let source: (x: Int, y: Int)
                switch orientation {
                case 2: source = (width - 1 - x, y)
                case 3: source = (width - 1 - x, height - 1 - y)
                case 4: source = (x, height - 1 - y)
                case 5: source = (y, x)
                case 6: source = (y, height - 1 - x)
                case 7: source = (width - 1 - y, height - 1 - x)
                default: source = (width - 1 - y, x)                // 8
                }
                let s = 3 * (source.y * width + source.x), d = 3 * (y * outWidth + x)
                out[d] = rgb[s]; out[d + 1] = rgb[s + 1]; out[d + 2] = rgb[s + 2]
            }
        }
        return VisionRGBImage(validated: out, width: outWidth, height: outHeight)
    }

    private init(validated rgb: [UInt8], width: Int, height: Int) {
        self.width = width
        self.height = height
        self.rgb = rgb
    }
}
