public import CoreGraphics
public import Foundation
import ImageIO

// The vision drivers' image input (DESIGN.md §8): what an app hands over, and its decode to 8-bit RGB. Here, one layer
// below the task modules, because AetherVision, AetherEmbeddings (page retrieval) and AetherDecide (vision towers) all
// take images and never depend on each other (§3), as the generative foundation's `GeneratedImage` is (§6.4).

/// An image for a vision model: a `CGImage` or an image file. Nothing is decoded until an ``ImagePreprocessor``
/// prepares it, in the colour space the model's manifest names.
public struct ImageInput: @unchecked Sendable {
    // @unchecked: CGImage is immutable and thread-safe, but not annotated Sendable.

    /// What decoding does with an alpha channel.
    public enum AlphaPolicy: Sendable, Hashable {
        /// Drops alpha and keeps each pixel's stored colour, as Pillow's `convert("RGB")` does: the default, for parity
        /// with reference preprocessing. Exact for 8- and 16-bit images with straight (unpremultiplied) alpha, which is
        /// how ImageIO returns PNG, TIFF and similar files. A source that arrives premultiplied has already lost the
        /// stored colour of transparent pixels: its colours are unpremultiplied where alpha > 0, and fully transparent
        /// pixels become black.
        case dropAlpha
        /// Composites over an opaque colour (Core Graphics source-over in the decode's colour space).
        case composite(red: UInt8, green: UInt8, blue: UInt8)
    }

    enum Source {
        case image(CGImage)
        case file(URL)
    }

    let source: Source
    public var alpha: AlphaPolicy

    /// An image already in memory. It is taken as displayed: there is no orientation to apply.
    public init(_ cgImage: CGImage, alpha: AlphaPolicy = .dropAlpha) {
        source = .image(cgImage)
        self.alpha = alpha
    }

    /// An image file in any format ImageIO reads. It is decoded at full size and its EXIF orientation applied by moving
    /// pixels, never resampled.
    public init(contentsOf url: URL, alpha: AlphaPolicy = .dropAlpha) {
        source = .file(url)
        self.alpha = alpha
    }

    /// Decodes to 8-bit RGB, upright, in `colorSpace`.
    func decode(colorSpace: Manifest.ImagePreprocess.ColorSpace) throws(AetherError) -> RGB8Image {
        switch source {
        case .image(let image):
            return try RGB8Image(cgImage: image, alpha: alpha, colorSpace: colorSpace)
        case .file(let url):
            guard let file = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(file, 0, nil) else {
                throw .invalidInput("'\(url.lastPathComponent)' is not an image ImageIO can decode")
            }
            let properties = CGImageSourceCopyPropertiesAtIndex(file, 0, nil) as? [CFString: Any]
            let orientation = properties?[kCGImagePropertyOrientation] as? Int ?? 1
            let stored = try RGB8Image(cgImage: image, alpha: alpha, colorSpace: colorSpace)
            // Oriented after the full-size decode: ImageIO's oriented thumbnails are premultiplied, which would lose
            // transparent pixels' stored colours.
            return (2...8).contains(orientation) ? stored.oriented(orientation) : stored
        }
    }
}

/// An 8-bit interleaved RGB image, upright: the preprocessor's working form.
struct RGB8Image: Sendable, Equatable {
    let width: Int
    let height: Int
    /// Row-major RGB, `width · height · 3` bytes.
    let rgb: [UInt8]

    init(rgb: [UInt8], width: Int, height: Int) throws(AetherError) {
        guard width > 0, height > 0, rgb.count == width * height * 3 else {
            throw .invalidInput("\(rgb.count) bytes are not a \(width)x\(height) RGB image")
        }
        self.width = width
        self.height = height
        self.rgb = rgb
    }

    private init(validated rgb: [UInt8], width: Int, height: Int) {
        self.width = width
        self.height = height
        self.rgb = rgb
    }

    /// Converts `image` to 8-bit RGB, treating alpha by `alpha`. `deviceRGB` keeps an RGB image's stored values (no
    /// colour management, as Pillow decodes) and converts any other to sRGB; `sRGB` converts everything to sRGB.
    init(cgImage image: CGImage, alpha: ImageInput.AlphaPolicy, colorSpace: Manifest.ImagePreprocess.ColorSpace) throws(AetherError) {
        guard image.width > 0, image.height > 0 else { throw .invalidInput("the image has no pixels") }
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
        let own = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil }
        let space: CGColorSpace = switch colorSpace {
        case .deviceRGB: own ?? sRGB
        case .sRGB: sRGB
        }
        // The stored values are what's asked for unless an image in another space must be converted to sRGB.
        let storedValuesWanted = colorSpace == .deviceRGB || own?.name == CGColorSpace.sRGB
        let hasAlpha = ![.none, .noneSkipLast, .noneSkipFirst].contains(image.alphaInfo)
        if hasAlpha, alpha == .dropAlpha, storedValuesWanted, let rgb = Self.storedRGB(of: image) {
            try self.init(rgb: rgb, width: image.width, height: image.height)
            return
        }
        let width = image.width, height = image.height
        let premultiplied = hasAlpha && alpha == .dropAlpha
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: space,
                bitmapInfo: (premultiplied ? CGImageAlphaInfo.premultipliedLast : CGImageAlphaInfo.noneSkipLast).rawValue)
            else { return false }
            context.interpolationQuality = .none
            if case .composite(let red, let green, let blue) = alpha, red | green | blue != 0 {
                context.setFillColor(CGColor(colorSpace: space, components: [CGFloat(red) / 255, CGFloat(green) / 255,
                                                                             CGFloat(blue) / 255, 1])!)
                context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw .invalidInput("the image could not be drawn as 8-bit RGB") }
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
        // The high byte of a 16-bit component is its first in big-endian order, its second in little-endian.
        let high = componentBytes == 2 && order == .byteOrder16Little ? 1 : 0
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                let pixel = y * rowBytes + x * channels * componentBytes
                let d = 3 * (y * width + x)
                for c in 0..<3 {
                    rgb[d + c] = bytes[pixel + colours[model == .rgb ? c : 0] * componentBytes + high]
                }
            }
        }
        return rgb
    }

    /// The image displayed under EXIF `orientation` (2–8), by moving pixels.
    func oriented(_ orientation: Int) -> RGB8Image {
        let swapped = orientation >= 5
        let outWidth = swapped ? height : width, outHeight = swapped ? width : height
        var out = [UInt8](repeating: 0, count: rgb.count)
        for y in 0..<outHeight {
            for x in 0..<outWidth {
                let source: (x: Int, y: Int) = switch orientation {
                case 2: (width - 1 - x, y)
                case 3: (width - 1 - x, height - 1 - y)
                case 4: (x, height - 1 - y)
                case 5: (y, x)
                case 6: (y, height - 1 - x)
                case 7: (width - 1 - y, height - 1 - x)
                default: (width - 1 - y, x)                // 8
                }
                let s = 3 * (source.y * width + source.x), d = 3 * (y * outWidth + x)
                out[d] = rgb[s]
                out[d + 1] = rgb[s + 1]
                out[d + 2] = rgb[s + 2]
            }
        }
        return RGB8Image(validated: out, width: outWidth, height: outHeight)
    }
}
