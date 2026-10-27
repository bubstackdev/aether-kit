import AetherVision
import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

/// Hermetic checks of Kit's image decoding (`VisionRGBImage`), which the Qwen3-VL qualification otherwise only exercises
/// against an upstream fed the same decode. Fixtures are generated here: tiny lossless PNGs with known pixels, and JPEGs
/// carrying each EXIF orientation.
struct VisionRGBImageTests {
    private static func write(_ bytes: [UInt8], width: Int, height: Int, space: CGColorSpace, bitsPerPixel: Int = 32,
                              alpha: CGImageAlphaInfo = .noneSkipLast, type: UTType = .png,
                              properties: [CFString: Any] = [:]) throws -> URL {
        let provider = try #require(CGDataProvider(data: Data(bytes) as CFData))
        let image = try #require(CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: bitsPerPixel,
                                         bytesPerRow: width * bitsPerPixel / 8, space: space,
                                         bitmapInfo: CGBitmapInfo(rawValue: alpha.rawValue), provider: provider, decode: nil,
                                         shouldInterpolate: false, intent: .defaultIntent))
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("vision-rgb-\(UUID().uuidString)")
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

    /// Deletes a fixture's temporary folder.
    private static func remove(_ url: URL) { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

    private static let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

    /// R, G and B land in that order, with the stored values unchanged.
    @Test func losslessRGBKeepsChannelOrderAndValues() throws {
        let url = try Self.write([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 10, 20, 30, 255], width: 2, height: 2, space: Self.sRGB)
        defer { Self.remove(url) }
        let image = try VisionRGBImage(contentsOf: url)
        #expect(image.width == 2 && image.height == 2)
        #expect(image.rgb == [255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30])
    }

    /// No colour management, like the reference's PIL decoding: a Display P3 image keeps its stored values.
    @Test func taggedColourSpaceIsNotConverted() throws {
        let p3 = try #require(CGColorSpace(name: CGColorSpace.displayP3))
        let url = try Self.write([200, 100, 50, 255, 10, 220, 130, 255], width: 2, height: 1, space: p3)
        defer { Self.remove(url) }
        #expect(try VisionRGBImage(contentsOf: url).rgb == [200, 100, 50, 10, 220, 130])
    }

    /// Grayscale expands to equal R, G and B, as PIL's `convert("RGB")` does.
    @Test func grayscaleExpandsToEqualChannels() throws {
        let url = try Self.write([0, 64, 128, 255], width: 4, height: 1, space: CGColorSpaceCreateDeviceGray(), bitsPerPixel: 8,
                                 alpha: .none)
        defer { Self.remove(url) }
        #expect(try VisionRGBImage(contentsOf: url).rgb == [0, 0, 0, 64, 64, 64, 128, 128, 128, 255, 255, 255])
    }

    /// Alpha is composited over black. This differs from the reference's PIL `convert("RGB")`, which drops alpha and keeps
    /// the stored colour (documented in docs/qwen3-vl.md); opaque pixels are unaffected.
    @Test func alphaIsCompositedOverBlack() throws {
        let url = try Self.write([255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 10, 20, 30, 255], width: 2, height: 2,
                                 space: Self.sRGB, alpha: .last)
        defer { Self.remove(url) }
        let rgb = try VisionRGBImage(contentsOf: url).rgb
        #expect(Array(rgb[0..<3]) == [255, 0, 0])
        #expect(rgb[3] == 0 && abs(Int(rgb[4]) - 128) <= 1 && rgb[5] == 0)
        #expect(Array(rgb[6..<9]) == [0, 0, 0])
        #expect(Array(rgb[9..<12]) == [10, 20, 30])
    }

    /// Every EXIF orientation, mirrored ones included, is applied: the decoded size is the displayed size, and each quadrant of a 32x16 image (red, green
    /// over blue, white) lands where the orientation puts it.
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
        let image = try VisionRGBImage(contentsOf: url)
        #expect(image.width == width && image.height == height)
        // Top-left, top-right, bottom-left, bottom-right quadrant centres; JPEG is lossy, so allow a few levels.
        let centres = [(width / 4, height / 4), (3 * width / 4, height / 4), (width / 4, 3 * height / 4), (3 * width / 4, 3 * height / 4)]
        for ((x, y), name) in zip(centres, quadrants) {
            let i = 3 * (y * image.width + x)
            let pixel = Array(image.rgb[i..<(i + 3)])
            #expect(zip(pixel, colours[name]!).allSatisfy { abs(Int($0) - Int($1)) <= 8 }, "orientation \(orientation): \(pixel) is not \(name)")
        }
    }
}
