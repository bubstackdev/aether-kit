import AetherCore
import Accelerate
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
        private var rowWeights: [Float]
        private var added: [Bool]

        init(width: Int, height: Int, tiling: Krea2VAETiling) {
            self.width = width
            self.height = height
            x = Self.normalized(tiling.axis(width), extent: width)
            y = Self.normalized(tiling.axis(height), extent: height)
            pixels = [Float](repeating: 0, count: 3 * width * height)
            rowWeights = [Float](repeating: 0, count: x.size)
            added = [Bool](repeating: false, count: x.starts.count * y.starts.count)
        }

        /// The 2D weight sum factors into the product of the two axis sums, including shifted edge tiles and
        /// overlaps spanning more than two tiles. Normalize here to avoid a full weight image and final division.
        private static func normalized(_ axis: Axis, extent: Int) -> Axis {
            var sums = [Float](repeating: 0, count: extent)
            for (i, start) in axis.starts.enumerated() {
                for j in 0..<axis.size { sums[start + j] += axis.weights[i][j] }
            }
            let weights = axis.starts.enumerated().map { i, start in
                (0..<axis.size).map { axis.weights[i][$0] / sums[start + $0] }
            }
            return Axis(size: axis.size, starts: axis.starts, weights: weights)
        }

        mutating func add(_ tile: [Float], row: Int, column: Int) throws(AetherError) {
            guard tile.count == 3 * x.size * y.size else {
                throw .tensor(name: "decoder tile", reason: .shapeMismatch(expected: [3, y.size, x.size], found: [tile.count]))
            }
            guard y.starts.indices.contains(row), x.starts.indices.contains(column),
                  !added[row * x.starts.count + column] else {
                throw .tensor(name: "decoder tile", reason: .unsupported("tile origin is invalid or already added"))
            }
            let plane = width * height, tilePlane = x.size * y.size
            let tileWidth = x.size, tileHeight = y.size, startX = x.starts[column], startY = y.starts[row]
            let wx = x.weights[column], wy = y.weights[row], width = width
            tile.withUnsafeBufferPointer { source in
                pixels.withUnsafeMutableBufferPointer { destination in
                    rowWeights.withUnsafeMutableBufferPointer { weights in
                        wx.withUnsafeBufferPointer { horizontal in
                            for dy in 0..<tileHeight {
                                var vertical = wy[dy]
                                vDSP_vsmul(horizontal.baseAddress!, 1, &vertical, weights.baseAddress!, 1, vDSP_Length(tileWidth))
                                let target = (startY + dy) * width + startX
                                for channel in 0..<3 {
                                    let input = source.baseAddress! + channel * tilePlane + dy * tileWidth
                                    let output = destination.baseAddress! + channel * plane + target
                                    vDSP_vma(input, 1, weights.baseAddress!, 1, output, 1, output, 1, vDSP_Length(tileWidth))
                                }
                            }
                        }
                    }
                }
            }
            added[row * x.starts.count + column] = true
        }

        mutating func finish() throws(AetherError) -> [Float] {
            guard added.allSatisfy({ $0 }) else {
                throw .tensor(name: "decoder tiles", reason: .unsupported("not all decoder tiles were added"))
            }
            return pixels
        }
    }
}
