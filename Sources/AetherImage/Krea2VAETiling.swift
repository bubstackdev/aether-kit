import AetherCore
import Foundation

/// Fixed-shape overlapping decoder tiles. Edge tiles shift back rather than introducing another specialization.
struct Krea2VAETiling: Sendable, Hashable {
    let tileSize: Int
    let overlap: Int

    init(tileSize: Int, overlap: Int) throws(AetherError) {
        guard tileSize >= 32, tileSize % 16 == 0, overlap >= 0, overlap < tileSize, overlap % 16 == 0 else {
            throw .generation(.invalidParameters(
                "decoder tileSize must be at least 32 and divisible by 16; overlap must be a nonnegative multiple of 16 smaller than tileSize"))
        }
        self.tileSize = tileSize
        self.overlap = overlap
    }

    var profile: String { "vae-tiles-\(tileSize)-\(overlap)" }

    struct Axis {
        let size: Int
        let starts: [Int]
        let weights: [[Float]]
    }

    func axis(_ extent: Int) -> Axis {
        let size = min(tileSize, extent)
        var starts = [0]
        if extent > size {
            starts = Array(stride(from: 0, through: extent - size, by: size - overlap))
            if starts.last != extent - size { starts.append(extent - size) }
        }
        let weights = starts.indices.map { index in
            var values = [Float](repeating: 1, count: size)
            if index > 0 {
                let count = starts[index - 1] + size - starts[index]
                for i in 0..<count {
                    let sine = sin((Float(i) + 0.5) / Float(count) * (.pi / 2))
                    values[i] *= sine * sine
                }
            }
            if index + 1 < starts.count {
                let count = starts[index] + size - starts[index + 1]
                for i in 0..<count {
                    let cosine = cos((Float(i) + 0.5) / Float(count) * (.pi / 2))
                    values[size - count + i] *= cosine * cosine
                }
            }
            return values
        }
        return Axis(size: size, starts: starts, weights: weights)
    }

    struct Accumulator {
        let width: Int
        let height: Int
        let x: Axis
        let y: Axis
        private var pixels: [Float]
        private var weights: [Float]

        init(width: Int, height: Int, tiling: Krea2VAETiling) {
            self.width = width
            self.height = height
            x = tiling.axis(width)
            y = tiling.axis(height)
            pixels = [Float](repeating: 0, count: 3 * width * height)
            weights = [Float](repeating: 0, count: width * height)
        }

        mutating func add(_ tile: [Float], row: Int, column: Int) throws(AetherError) {
            guard tile.count == 3 * x.size * y.size else {
                throw .tensor(name: "decoder tile", reason: .shapeMismatch(expected: [3, y.size, x.size], found: [tile.count]))
            }
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

        mutating func finish() throws(AetherError) -> [Float] {
            let plane = width * height
            for i in 0..<plane {
                guard weights[i] > 0 else {
                    throw .tensor(name: "decoder tiles", reason: .unsupported("tiles leave an uncovered pixel"))
                }
                for channel in 0..<3 { pixels[channel * plane + i] /= weights[i] }
            }
            return pixels
        }
    }
}
