import Foundation
import ImageIO
import Testing
@testable import AetherKit

@Test func imageQuantizationAndPNGMatchReferenceConvention() throws {
    let image = try GeneratedImage(pixelsCHW: FloatTensor(shape: [3, 1, 2], values: [-1, 1, 0, 2, -2, 0]))
    #expect(image.rgb8() == [0, 128, 0, 255, 255, 128])
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
    defer { try? FileManager.default.removeItem(at: url) }
    try image.savePNG(to: url)
    #expect(try image.pngData() == Data(contentsOf: url))
    let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
    let decoded = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    #expect(decoded.width == 2)
    #expect(decoded.height == 1)
    let data = try #require(decoded.dataProvider?.data)
    // ImageIO may expand RGB to RGBA; compare the first three bytes of each pixel.
    let bytes = data as Data
    let stride = decoded.bitsPerPixel / 8
    #expect(Array(bytes[0..<3]) == [0, 128, 0])
    #expect(Array(bytes[stride..<(stride + 3)]) == [255, 255, 128])
    #expect(throws: AetherError.self) {
        try GeneratedImage(pixelsCHW: FloatTensor(shape: [1, 1, 2], values: [0, 0]))
    }
}
