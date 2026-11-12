import AetherCore
import Foundation
import Testing
@testable import AetherImage

@Suite("Krea 2 stitching")
struct Krea2StitchingTests {
    /// The original scalar accumulator, kept as an independent normalization oracle and benchmark baseline.
    struct Scalar {
        let width: Int, height: Int
        let x: Krea2VAETiling.Axis, y: Krea2VAETiling.Axis
        var pixels: [Float], weights: [Float]

        init(width: Int, height: Int, tiling: Krea2VAETiling) {
            self.width = width
            self.height = height
            x = tiling.axis(width)
            y = tiling.axis(height)
            pixels = .init(repeating: 0, count: 3 * width * height)
            weights = .init(repeating: 0, count: width * height)
        }

        mutating func add(_ tile: [Float], row: Int, column: Int) {
            let plane = width * height, tilePlane = x.size * y.size
            for dy in 0..<y.size {
                let destination = (y.starts[row] + dy) * width + x.starts[column]
                let source = dy * x.size
                for dx in 0..<x.size {
                    let weight = y.weights[row][dy] * x.weights[column][dx]
                    weights[destination + dx] += weight
                    for channel in 0..<3 {
                        pixels[channel * plane + destination + dx] += tile[channel * tilePlane + source + dx] * weight
                    }
                }
            }
        }

        mutating func finish() -> [Float] {
            let plane = width * height
            for i in 0..<plane {
                for channel in 0..<3 { pixels[channel * plane + i] /= weights[i] }
            }
            return pixels
        }
    }

    static func tiles(width: Int, height: Int, tiling: Krea2VAETiling) -> [[Float]] {
        let x = tiling.axis(width), y = tiling.axis(height)
        return (0..<x.starts.count * y.starts.count).map { tile in
            // Different local values per tile expose normalization errors that coordinate-identical tiles hide.
            (0..<3 * x.size * y.size).map { i in Float((i * 13 + tile * 79) % 2003) / 1001 - 1 }
        }
    }

    @Test(arguments: [(80, 48, 32, 16), (80, 48, 32, 0), (96, 64, 64, 48),
                      (80, 16, 32, 16), (16, 16, 32, 16), (1008, 768, 512, 128)])
    func matchesScalarForDifferentTileValues(shape: (Int, Int, Int, Int)) throws {
        let (width, height, tile, overlap) = shape
        let tiling = try Krea2VAETiling(tileSize: tile, overlap: overlap)
        var scalar = Scalar(width: width, height: height, tiling: tiling)
        var accelerated = Krea2VAETiling.Accumulator(width: width, height: height, tiling: tiling)
        let tiles = Self.tiles(width: width, height: height, tiling: tiling)
        for row in scalar.y.starts.indices {
            for column in scalar.x.starts.indices {
                let pixels = tiles[row * scalar.x.starts.count + column]
                scalar.add(pixels, row: row, column: column)
                try accelerated.add(pixels, row: row, column: column)
            }
        }
        let expected = scalar.finish(), actual = try accelerated.finish()
        #expect(zip(expected, actual).allSatisfy { abs($0 - $1) < 0.000001 })
    }

    @Test func missingOrRepeatedTilesAreRefused() throws {
        let tiling = try Krea2VAETiling(tileSize: 32, overlap: 16)
        var accumulator = Krea2VAETiling.Accumulator(width: 48, height: 32, tiling: tiling)
        let tile = [Float](repeating: 1, count: 3 * 32 * 32)
        #expect(throws: AetherError.self) { try accumulator.finish() }
        try accumulator.add(tile, row: 0, column: 0)
        #expect(throws: AetherError.self) { try accumulator.finish() }
        #expect(throws: AetherError.self) { try accumulator.add(tile, row: 0, column: 0) }
        #expect(throws: AetherError.self) { try accumulator.add(tile, row: -1, column: 0) }
        #expect(throws: AetherError.self) { try accumulator.add(tile, row: 0, column: 2) }
    }

    /// Run with AETHER_KREA2_STITCH_BENCH=1 swift test -c release --filter Krea2StitchingTests/benchmark.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["AETHER_KREA2_STITCH_BENCH"] == "1"))
    func benchmark() throws {
        let tiling = try Krea2VAETiling(tileSize: 512, overlap: 256)
        let width = 1024, height = 1024
        let tiles = Self.tiles(width: width, height: height, tiling: tiling)
        let clock = ContinuousClock()
        var oldSeconds: [Double] = [], newSeconds: [Double] = []
        var checksum: Float = 0
        for run in 0..<12 {
            // Alternate order to reduce bias from cache warmth and machine load.
            for accelerated in (run.isMultiple(of: 2) ? [false, true] : [true, false]) {
                let start = clock.now
                let output: [Float]
                if accelerated {
                    var accumulator = Krea2VAETiling.Accumulator(width: width, height: height, tiling: tiling)
                    for row in 0..<3 {
                        for column in 0..<3 { try accumulator.add(tiles[row * 3 + column], row: row, column: column) }
                    }
                    output = try accumulator.finish()
                } else {
                    var accumulator = Scalar(width: width, height: height, tiling: tiling)
                    for row in 0..<3 {
                        for column in 0..<3 { accumulator.add(tiles[row * 3 + column], row: row, column: column) }
                    }
                    output = accumulator.finish()
                }
                let elapsed = start.duration(to: clock.now).components
                let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
                checksum += output[run]
                if run > 1 {
                    if accelerated { newSeconds.append(seconds) } else { oldSeconds.append(seconds) }
                }
            }
        }
        print("stitch benchmark: scalar=\(oldSeconds) accelerated=\(newSeconds) checksum=\(checksum)")
    }
}
