// Writes Kit's own ImageIO decode of each photo (`VisionRGBImage(contentsOf:)`) as a lossless PNG without a colour
// profile, for an upstream reference that sees exactly the RGB Kit's file path sees (development tooling only).
// Each PNG is decoded again by `VisionRGBImage` and must be byte-identical to the first decode.
//
//     xcrun swiftc -O -o .build/write-kit-decoded-photos Sources/AetherVision/VisionRGBImage.swift \
//         Scripts/write-kit-decoded-photos.swift
//     .build/write-kit-decoded-photos <new output dir> <photo.jpg> [...]
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

@main
struct WriteKitDecodedPhotos {
    static func main() throws {
        let arguments = CommandLine.arguments.dropFirst()
        guard let output = arguments.first, arguments.count > 1 else {
            print("usage: write-kit-decoded-photos <new output dir> <photo> [...]")
            exit(2)
        }
        let directory = URL(filePath: output)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        for path in arguments.dropFirst() {
            let input = URL(filePath: path)
            let image = try VisionRGBImage(contentsOf: input)
            var rgba = [UInt8](repeating: 255, count: image.width * image.height * 4)
            for i in 0..<(image.width * image.height) { for c in 0..<3 { rgba[4 * i + c] = image.rgb[3 * i + c] } }
            guard let provider = CGDataProvider(data: Data(rgba) as CFData),
                  let cgImage = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                                        bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
            else { throw VisionRGBImage.Failure.invalidPixels }
            let url = directory.appendingPathComponent(input.deletingPathExtension().lastPathComponent + ".png")
            guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
            else { throw VisionRGBImage.Failure.undecodable }
            CGImageDestinationAddImage(destination, cgImage, nil)
            guard CGImageDestinationFinalize(destination) else { throw VisionRGBImage.Failure.undecodable }
            guard try VisionRGBImage(contentsOf: url) == image else {
                print("\(url.lastPathComponent): the PNG does not decode to the same bytes")
                exit(1)
            }
            print("\(url.lastPathComponent) \(image.width)x\(image.height) round trip identical")
        }
    }
}
