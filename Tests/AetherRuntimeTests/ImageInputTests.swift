import AetherCore
import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers
@testable import AetherRuntime

/// `ImageInput`'s decode, ported with 0.0.x's `VisionRGBImageTests`: tiny lossless PNGs with known pixels, JPEGs and
/// PNGs carrying each EXIF orientation, and the alpha fixtures (`Fixtures/vision/alpha`, written with Transformers'
/// `load_image` answers by `Forge/vision/fixtures_preprocess.py`).
@Suite("Image input")
struct ImageInputTests {
    private static func write(_ bytes: [UInt8], width: Int, height: Int, space: CGColorSpace, bitsPerPixel: Int = 32,
                              alpha: CGImageAlphaInfo = .noneSkipLast, type: UTType = .png,
                              properties: [CFString: Any] = [:]) throws -> URL {
        let image = try cgImage(bytes, width: width, height: height, space: space, bitsPerPixel: bitsPerPixel, alpha: alpha)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("aether-image-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("image").appendingPathExtension(type.preferredFilenameExtension ?? "img")
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
            remove(url)
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))
        return url
    }

    private static func cgImage(_ bytes: [UInt8], width: Int, height: Int, space: CGColorSpace, bitsPerPixel: Int = 32,
                                alpha: CGImageAlphaInfo = .noneSkipLast) throws -> CGImage {
        let provider = try #require(CGDataProvider(data: Data(bytes) as CFData))
        return try #require(CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: bitsPerPixel,
                                    bytesPerRow: width * bitsPerPixel / 8, space: space,
                                    bitmapInfo: CGBitmapInfo(rawValue: alpha.rawValue), provider: provider, decode: nil,
                                    shouldInterpolate: false, intent: .defaultIntent))
    }

    /// Deletes a fixture's temporary folder.
    private static func remove(_ url: URL) { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

    private static let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

    private static func decode(_ url: URL, alpha: ImageInput.AlphaPolicy = .dropAlpha,
                               _ space: Manifest.ImagePreprocess.ColorSpace = .deviceRGB) throws -> RGB8Image {
        try ImageInput(contentsOf: url, alpha: alpha).decode(colorSpace: space)
    }

    /// R, G and B land in that order, with the stored values unchanged.
    @Test func losslessRGBKeepsChannelOrderAndValues() throws {
        let url = try Self.write([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 10, 20, 30, 255], width: 2, height: 2, space: Self.sRGB)
        defer { Self.remove(url) }
        let image = try Self.decode(url)
        #expect(image.width == 2 && image.height == 2)
        #expect(image.rgb == [255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30])
    }

    /// `deviceRGB` does no colour management, like Pillow: a Display P3 image keeps its stored values. `sRGB` converts
    /// it, and leaves an sRGB image's values alone.
    @Test func deviceRGBKeepsStoredValuesAndSRGBConverts() throws {
        let p3 = try #require(CGColorSpace(name: CGColorSpace.displayP3))
        let url = try Self.write([200, 100, 50, 255, 10, 220, 130, 255], width: 2, height: 1, space: p3)
        defer { Self.remove(url) }
        #expect(try Self.decode(url).rgb == [200, 100, 50, 10, 220, 130])
        let converted = try Self.decode(url, .sRGB).rgb
        #expect(converted != [200, 100, 50, 10, 220, 130])
        // P3 green is outside sRGB: converted, it saturates the green channel and moves red.
        #expect(converted[4] == 255 || converted[3] < 10)
        let srgb = try Self.write([200, 100, 50, 255, 10, 220, 130, 255], width: 2, height: 1, space: Self.sRGB)
        defer { Self.remove(srgb) }
        #expect(try Self.decode(srgb, .sRGB).rgb == [200, 100, 50, 10, 220, 130])
    }

    /// Grayscale expands to equal R, G and B, as Pillow's `convert("RGB")` does.
    @Test func grayscaleExpandsToEqualChannels() throws {
        let url = try Self.write([0, 64, 128, 255], width: 4, height: 1, space: CGColorSpaceCreateDeviceGray(), bitsPerPixel: 8,
                                 alpha: .none)
        defer { Self.remove(url) }
        #expect(try Self.decode(url).rgb == [0, 0, 0, 64, 64, 64, 128, 128, 128, 255, 255, 255])
    }

    /// By default alpha is dropped and each pixel's stored colour kept, as Pillow's `convert("RGB")` does; `.composite`
    /// draws over an opaque colour.
    @Test func alphaIsDroppedByDefaultAndCompositedOnRequest() throws {
        let url = try Self.write([255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 10, 20, 30, 255], width: 2, height: 2,
                                 space: Self.sRGB, alpha: .last)
        defer { Self.remove(url) }
        #expect(try Self.decode(url).rgb == [255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30])
        let black = try Self.decode(url, alpha: .composite(red: 0, green: 0, blue: 0)).rgb
        #expect(Array(black[0..<3]) == [255, 0, 0])
        #expect(black[3] == 0 && abs(Int(black[4]) - 128) <= 1 && black[5] == 0)
        #expect(Array(black[6..<9]) == [0, 0, 0])
        #expect(Array(black[9..<12]) == [10, 20, 30])
        let white = try Self.decode(url, alpha: .composite(red: 255, green: 255, blue: 255)).rgb
        #expect(Array(white[6..<9]) == [255, 255, 255])
        #expect(abs(Int(white[3]) - 127) <= 1 && white[4] == 255 && abs(Int(white[5]) - 127) <= 1)
    }

    /// The default matches upstream byte for byte on PNGs with transparent pixels in every layout an app may pass:
    /// 8-bit RGBA, 8-bit grey plus alpha, a palette with a transparency table and 16-bit RGBA.
    @Test func defaultAlphaMatchesUpstreamOnEveryLayout() throws {
        struct Expected: Decodable {
            struct Fixture: Decodable { let file: String; let upstreamRGB: [UInt8] }
            let size: [Int]
            let fixtures: [String: Fixture]
        }
        let directory = try #require(Bundle.module.url(forResource: "alpha", withExtension: nil, subdirectory: "Fixtures/vision"))
        let expected = try JSONDecoder().decode(Expected.self, from: Data(contentsOf: directory.appendingPathComponent("expected.json")))
        #expect(expected.fixtures.count == 4)
        for (name, fixture) in expected.fixtures.sorted(by: { $0.key < $1.key }) {
            let image = try Self.decode(directory.appendingPathComponent(fixture.file))
            #expect(image.width == expected.size[0] && image.height == expected.size[1], "\(name)")
            #expect(image.rgb == fixture.upstreamRGB, "\(name): \(image.rgb) is not upstream's \(fixture.upstreamRGB)")
        }
    }

    /// Every EXIF orientation, mirrored ones included, is applied: the decoded size is the displayed size, and each
    /// quadrant of a 32x16 image (red, green over blue, white) lands where the orientation puts it.
    @Test(arguments: [
        (1, 32, 16, ["red", "green", "blue", "white"]),
        (2, 32, 16, ["green", "red", "white", "blue"]),
        (3, 32, 16, ["white", "blue", "green", "red"]),
        (4, 32, 16, ["blue", "white", "red", "green"]),
        (5, 16, 32, ["red", "blue", "green", "white"]),
        (6, 16, 32, ["blue", "red", "white", "green"]),
        (7, 16, 32, ["white", "green", "blue", "red"]),
        (8, 16, 32, ["green", "white", "red", "blue"]),
    ])
    func exifOrientationIsApplied(orientation: Int, width: Int, height: Int, quadrants: [String]) throws {
        let colours: [String: [UInt8]] = ["red": [255, 0, 0], "green": [0, 255, 0], "blue": [0, 0, 255], "white": [255, 255, 255]]
        var rgba = [UInt8](repeating: 255, count: 32 * 16 * 4)
        for y in 0..<16 {
            for x in 0..<32 {
                let name = y < 8 ? (x < 16 ? "red" : "green") : (x < 16 ? "blue" : "white")
                let i = 4 * (y * 32 + x)
                rgba.replaceSubrange(i..<(i + 3), with: colours[name]!)
            }
        }
        let url = try Self.write(rgba, width: 32, height: 16, space: Self.sRGB, type: .jpeg,
                                 properties: [kCGImagePropertyOrientation: orientation, kCGImageDestinationLossyCompressionQuality: 1.0])
        defer { Self.remove(url) }
        let image = try Self.decode(url)
        #expect(image.width == width && image.height == height)
        // Quadrant centres; JPEG is lossy, so allow a few levels.
        let centres = [(width / 4, height / 4), (3 * width / 4, height / 4), (width / 4, 3 * height / 4), (3 * width / 4, 3 * height / 4)]
        for ((x, y), name) in zip(centres, quadrants) {
            let i = 3 * (y * image.width + x)
            let pixel = Array(image.rgb[i..<(i + 3)])
            #expect(zip(pixel, colours[name]!).allSatisfy { abs(Int($0) - Int($1)) <= 8 }, "orientation \(orientation): \(pixel) is not \(name)")
        }
    }

    /// The stored pixel shown at displayed (x, y) under EXIF `orientation`, for a stored image of `width` x `height`.
    private static func storedPixel(orientation: Int, x: Int, y: Int, width: Int, height: Int) -> (x: Int, y: Int) {
        switch orientation {
        case 2: (width - 1 - x, y)
        case 3: (width - 1 - x, height - 1 - y)
        case 4: (x, height - 1 - y)
        case 5: (y, x)
        case 6: (y, height - 1 - x)
        case 7: (width - 1 - y, height - 1 - x)
        case 8: (width - 1 - y, x)
        default: (x, y)
        }
    }

    /// The orientation transform is exact: a lossless PNG of seeded random pixels, tagged with each orientation,
    /// decodes to precisely the stored pixels moved where the orientation puts them, at an odd photo-like size.
    @Test(arguments: 1...8) func exifOrientationMovesPixelsWithoutResampling(orientation: Int) throws {
        let width = 301, height = 197
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15 &+ UInt64(orientation)
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            rgba[4 * i] = UInt8(truncatingIfNeeded: state >> 24)
            rgba[4 * i + 1] = UInt8(truncatingIfNeeded: state >> 32)
            rgba[4 * i + 2] = UInt8(truncatingIfNeeded: state >> 40)
        }
        let url = try Self.write(rgba, width: width, height: height, space: Self.sRGB, properties: [kCGImagePropertyOrientation: orientation])
        defer { Self.remove(url) }
        let image = try Self.decode(url)
        let swapped = orientation >= 5
        #expect(image.width == (swapped ? height : width) && image.height == (swapped ? width : height))
        var mismatches = 0
        for y in 0..<image.height {
            for x in 0..<image.width {
                let source = Self.storedPixel(orientation: orientation, x: x, y: y, width: width, height: height)
                let s = 4 * (source.y * width + source.x), d = 3 * (y * image.width + x)
                if image.rgb[d] != rgba[s] || image.rgb[d + 1] != rgba[s + 1] || image.rgb[d + 2] != rgba[s + 2] { mismatches += 1 }
            }
        }
        #expect(mismatches == 0, "orientation \(orientation): \(mismatches) pixels differ from an exact transform")
    }

    /// For opaque images, orienting by moving pixels equals ImageIO's own oriented decode (a full-size thumbnail with
    /// its transform), byte for byte.
    @Test(arguments: 2...8) func orientationEqualsImageIOsTransform(orientation: Int) throws {
        let width = 37, height = 23
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) { rgba[4 * i] = UInt8(i % 251); rgba[4 * i + 1] = UInt8((i * 7) % 253); rgba[4 * i + 2] = UInt8((i * 13) % 255) }
        let url = try Self.write(rgba, width: width, height: height, space: Self.sRGB, properties: [kCGImagePropertyOrientation: orientation])
        defer { Self.remove(url) }
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let thumbnail = try #require(CGImageSourceCreateThumbnailAtIndex(source, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(width, height)] as CFDictionary))
        #expect(try Self.decode(url) == ImageInput(thumbnail).decode(colorSpace: .deviceRGB))
    }

    /// A rotated file with transparent pixels keeps their stored colours: decoded at full size with alpha dropped,
    /// then oriented, where ImageIO's oriented thumbnail would have premultiplied them away.
    @Test(arguments: [3, 6, 8]) func rotatedTransparentPixelsKeepTheirStoredColour(orientation: Int) throws {
        let width = 5, height = 3
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            rgba[4 * i] = UInt8(40 + i); rgba[4 * i + 1] = UInt8(90 + i); rgba[4 * i + 2] = UInt8(200 - i); rgba[4 * i + 3] = UInt8((i % 3) * 120)
        }
        let url = try Self.write(rgba, width: width, height: height, space: Self.sRGB, alpha: .last,
                                 properties: [kCGImagePropertyOrientation: orientation])
        defer { Self.remove(url) }
        let image = try Self.decode(url)
        for y in 0..<image.height {
            for x in 0..<image.width {
                let source = Self.storedPixel(orientation: orientation, x: x, y: y, width: width, height: height)
                let s = 4 * (source.y * width + source.x), d = 3 * (y * image.width + x)
                #expect(Array(image.rgb[d..<(d + 3)]) == Array(rgba[s..<(s + 3)]), "orientation \(orientation) at (\(x), \(y))")
            }
        }
    }

    /// Third review, item 13: an image past `maximumPixels` is refused before its pixels are decoded (a file by its
    /// header); the default leaves room for a 48 MP photo.
    @Test func imagesPastThePixelLimitAreInvalidInput() throws {
        #expect(ImageInput.defaultMaximumPixels >= 8064 * 6048)
        let bytes = [UInt8](repeating: 128, count: 4 * 4 * 4)
        let url = try Self.write(bytes, width: 4, height: 4, space: Self.sRGB)
        defer { Self.remove(url) }
        let image = try Self.cgImage(bytes, width: 4, height: 4, space: Self.sRGB)
        func refused(_ input: ImageInput) -> Bool {
            do { _ = try input.decode(colorSpace: .deviceRGB); return false } catch {
                if case .invalidInput(let reason) = error { return reason.contains("4x4") }
                return false
            }
        }
        #expect(refused(ImageInput(contentsOf: url, maximumPixels: 15)) && refused(ImageInput(image, maximumPixels: 15)))
        #expect(try ImageInput(contentsOf: url, maximumPixels: 16).decode(colorSpace: .deviceRGB).width == 4)
        var input = ImageInput(image)
        input.maximumPixels = 16
        #expect(try input.decode(colorSpace: .deviceRGB).height == 4)
    }

    /// A file that isn't an image is `invalidInput`, raised when prepared; a CGImage input decodes as given.
    @Test func undecodableFilesAreInvalidInput() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("aether-image-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("not-an-image.png")
        try Data("hello".utf8).write(to: url)
        let preprocessor = try ImagePreprocessor(.init())
        #expect {
            try preprocessor.prepare(ImageInput(contentsOf: url))
        } throws: { error in
            if case .invalidInput = error as? AetherError { true } else { false }
        }
        let image = try Self.cgImage([1, 2, 3, 255, 4, 5, 6, 255], width: 2, height: 1, space: Self.sRGB)
        #expect(try preprocessor.prepare(ImageInput(image)).floats() == [1, 4, 2, 5, 3, 6])
    }
}
