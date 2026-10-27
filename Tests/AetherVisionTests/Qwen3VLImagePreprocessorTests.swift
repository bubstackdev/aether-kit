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
    /// Grid snapping against Forge's `nearest_grid` (expectations computed by it for the five
    /// shipped grids): nearest aspect ratio in log space, the first listed winning a tie.
    @Test func imagesSnapToTheNearestAspectGridLikeForge() throws {
        let grids: [Qwen3VLImagePreprocessor.Grid] = [.init(height: 44, width: 44), .init(height: 36, width: 48),
                                                      .init(height: 48, width: 36), .init(height: 32, width: 56),
                                                      .init(height: 56, width: 32)]
        let cases: [(Int, Int, Int, Int)] = [(1200, 1412, 48, 36), (1000, 1000, 44, 44), (640, 480, 36, 48), (480, 640, 48, 36),
                                             (1920, 1080, 32, 56), (1080, 1920, 56, 32), (4032, 2268, 32, 56), (810, 608, 36, 48),
                                             (320, 256, 36, 48), (256, 416, 56, 32), (1, 200, 56, 32), (200, 1, 32, 56),
                                             (1150, 1000, 44, 44), (1160, 1000, 36, 48), (3000, 2000, 36, 48)]
        var configuration = Qwen3VLImagePreprocessor.Configuration()
        configuration.imageGrids = grids
        for (w, h, gh, gw) in cases {
            let grid = try Qwen3VLImagePreprocessor.nearestGrid(width: w, height: h, in: grids)
            #expect(grid == .init(height: gh, width: gw), "\(w)x\(h)")
            let target = try Qwen3VLImagePreprocessor.targetSize(width: w, height: h, configuration: configuration)
            #expect(target == (gw * 16, gh * 16), "\(w)x\(h)")
        }
        // A tie goes to the first listed grid; no grids is refused.
        #expect(try Qwen3VLImagePreprocessor.nearestGrid(width: 10, height: 10, in: [.init(height: 2, width: 2), .init(height: 4, width: 4)])
                == .init(height: 2, width: 2))
        #expect(throws: Qwen3VLImagePreprocessor.Failure.noGrid) { try Qwen3VLImagePreprocessor.nearestGrid(width: 1, height: 1, in: []) }
        // Resized exactly to the grid (no padding): the patch grid is the snapped one.
        let small = try Qwen3VLImagePreprocessor.process(rgb: Array(repeating: 200, count: 40 * 30 * 3), width: 40, height: 30,
                                                         configuration: configuration)
        #expect(small.grid == (1, 36, 48))
        #expect(small.pixels.count == 36 * 48 * configuration.rowLength)
    }

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
