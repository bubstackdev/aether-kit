@testable import AetherVision
import Foundation
import Testing

private struct PreprocessFixture: Decodable {
    struct Size: Decodable { let width: Int; let height: Int; let targetWidth: Int?; let targetHeight: Int?; let refused: Bool? }
    struct Resize: Decodable { let width: Int; let height: Int; let targetWidth: Int; let targetHeight: Int; let input: [UInt8]; let output: [UInt8] }
    let sizes: [Size]
    let resizes: [Resize]
}

private func fixture() throws -> PreprocessFixture {
    let url = try #require(Bundle.module.url(forResource: "qwen3vl-preprocess", withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode(PreprocessFixture.self, from: Data(contentsOf: url))
}

struct Qwen3VLImagePreprocessorTests {
    @Test func targetSizesMatchTheReferenceSmartResize() throws {
        let f = try fixture()
        #expect(f.sizes.count >= 40)
        for s in f.sizes {
            if s.refused == true {
                #expect(throws: Qwen3VLImagePreprocessor.Failure.self) { try Qwen3VLImagePreprocessor.targetSize(width: s.width, height: s.height) }
            } else {
                let t = try Qwen3VLImagePreprocessor.targetSize(width: s.width, height: s.height)
                #expect(t.width == s.targetWidth && t.height == s.targetHeight, "\(s.width)x\(s.height)")
            }
        }
    }

    /// Antialiased bicubic resampling against torchvision's uint8 path: within one level everywhere
    /// (fixed-point rounding differs), and nearly always identical.
    @Test func resamplingMatchesTorchvisionWithinOneLevel() throws {
        for r in try fixture().resizes {
            let out = Qwen3VLImagePreprocessor.resize(rgb: r.input, width: r.width, height: r.height,
                                                      to: (r.targetWidth, r.targetHeight))
            #expect(out.count == r.output.count)
            let differences = zip(out, r.output).map { abs(Int($0) - Int($1)) }
            #expect(differences.max() ?? 0 <= 1, "\(r.width)x\(r.height) -> \(r.targetWidth)x\(r.targetHeight)")
            let exact = Double(differences.filter { $0 == 0 }.count) / Double(differences.count)
            #expect(exact >= 0.95, "\(r.width)x\(r.height): \(exact) exact")
        }
    }

    @Test func patchRowsAreNormalizedInMergeWindowOrder() throws {
        // A coordinate-coded 64x96 image: red = row, green = column, blue = 0.
        let (w, h) = (96, 64)
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0..<h { for x in 0..<w { rgb[(y * w + x) * 3] = UInt8(y); rgb[(y * w + x) * 3 + 1] = UInt8(x) } }
        let (pixels, grid) = try Qwen3VLImagePreprocessor.patchify(rgb: rgb, width: w, height: h)
        #expect(grid == (1, 4, 6) && pixels.count == 24 * 1536)
        // Row 5: window 1 (columns 2-3), patch (0, 1) -> grid (0, 3): pixel rows 0..15, columns 48..63.
        let row = Array(pixels[(5 * 1536)..<(6 * 1536)])
        let denormalize = { (v: Float) -> Int in Int((v * 0.5 + 0.5) * 255 + 0.5) }
        #expect(denormalize(row[0]) == 0 && denormalize(row[16 * 15]) == 15)                    // red, frame 0: rows
        #expect(denormalize(row[2 * 256]) == 48 && denormalize(row[2 * 256 + 15]) == 63)        // green, frame 0: columns
        #expect(Array(row[0..<256]) == Array(row[256..<512]))                                     // frames repeat
        #expect(abs(row[4 * 256] + 1) < 1e-6)                                                     // blue 0 -> -1
        // An exact-size image skips the resize.
        let processed = try Qwen3VLImagePreprocessor.process(rgb: rgb, width: w, height: h,
            configuration: { var c = Qwen3VLImagePreprocessor.Configuration(); c.minimumPixels = 1; return c }())
        #expect(processed.pixels == pixels)
    }
}
