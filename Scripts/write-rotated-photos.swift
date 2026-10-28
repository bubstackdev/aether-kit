// Writes rotated variants of lossless photos for the rotated-photo parity run (development tooling only): each variant
// stores the photo's pixels moved by the inverse of an EXIF orientation and carries that orientation, so the image
// displayed after the transform is the original exactly. Each variant must decode through `VisionRGBImage` to the
// original's bytes. `provenance.json` records the sources, orientations and hashes.
//
//     xcrun swiftc -O -o .build/write-rotated-photos Sources/AetherVision/VisionRGBImage.swift Scripts/write-rotated-photos.swift
//     .build/write-rotated-photos <new output dir> <orientation>:<photo.png> [...]
import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

@main
struct WriteRotatedPhotos {
    /// The stored pixel shown at displayed (x, y) under EXIF `orientation`, for a stored image of `width` x `height`.
    static func storedPixel(orientation: Int, x: Int, y: Int, width: Int, height: Int) -> (x: Int, y: Int) {
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

    static func sha256(_ url: URL) throws -> String {
        SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
    }

    static func main() throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard arguments.count > 1 else {
            print("usage: write-rotated-photos <new output dir> <orientation>:<photo.png> [...]")
            exit(2)
        }
        let directory = URL(filePath: arguments[0])
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        var records: [[String: Any]] = []
        for spec in arguments.dropFirst() {
            let parts = spec.split(separator: ":", maxSplits: 1)
            guard parts.count == 2, let orientation = Int(parts[0]), (2...8).contains(orientation) else {
                print("\(spec): expected <orientation 2-8>:<photo>")
                exit(2)
            }
            let input = URL(filePath: String(parts[1]))
            let original = try VisionRGBImage(contentsOf: input)
            let swapped = orientation >= 5
            let width = swapped ? original.height : original.width, height = swapped ? original.width : original.height
            // Place each displayed pixel where the orientation will read it from.
            var rgba = [UInt8](repeating: 255, count: width * height * 4)
            for y in 0..<original.height {
                for x in 0..<original.width {
                    let source = storedPixel(orientation: orientation, x: x, y: y, width: width, height: height)
                    let s = 4 * (source.y * width + source.x), d = 3 * (y * original.width + x)
                    for c in 0..<3 { rgba[s + c] = original.rgb[d + c] }
                }
            }
            guard let provider = CGDataProvider(data: Data(rgba) as CFData),
                  let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                      provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
            else { throw VisionRGBImage.Failure.invalidPixels }
            let name = input.deletingPathExtension().lastPathComponent + "-orientation\(orientation).png"
            let url = directory.appendingPathComponent(name)
            guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
            else { throw VisionRGBImage.Failure.undecodable }
            CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: orientation] as CFDictionary)
            guard CGImageDestinationFinalize(destination) else { throw VisionRGBImage.Failure.undecodable }
            let decoded = try VisionRGBImage(contentsOf: url)
            guard decoded == original else {
                print("\(name): the variant does not decode to the original's bytes")
                exit(1)
            }
            records.append(["file": name, "sha256": try sha256(url), "orientation": orientation,
                            "storedSize": [width, height], "displayedSize": [original.width, original.height],
                            "source": ["file": input.lastPathComponent, "sha256": try sha256(input)],
                            "decodesToSourceBytes": true])
            print("\(name): orientation \(orientation), stored \(width)x\(height), decodes to the source's bytes")
        }
        let provenance: [String: Any] = [
            "kind": "qwen3-vl-rotated-photos",
            "method": "each variant stores its source's pixels moved by the inverse of an EXIF orientation and carries that orientation, as a lossless PNG without a colour profile; VisionRGBImage(contentsOf:) of every variant equals its source's decode byte for byte",
            "photos": records,
        ]
        let data = try JSONSerialization.data(withJSONObject: provenance, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: directory.appendingPathComponent("provenance.json"))
    }
}
