// Derived from the Apache-2.0 Aether prototype. See NOTICE.
import AetherKit

/// The source VAE's temporal boundaries: seven latent frames per call at stride five,
/// 22 decoded frames per window, 17 emitted plus a five-frame cross-fade into the next.
/// Blends normalized decoder samples, before ImageNet denormalization or quantization.
public struct H3VideoStitcher: Sendable {
    public static let windowFrames = 22
    private let area: Int
    public let expectedWindows: Int
    public private(set) var windows = 0
    private var tail: [Float]?

    public init(width: Int, height: Int, frames: Int) throws {
        guard frames >= 22, (frames - 5).isMultiple(of: 17) else {
            throw AetherError.invalidRequest("H3 streaming decode requires 17n+5 frames, n >= 1")
        }
        let area = width.multipliedReportingOverflow(by: height)
        guard width > 0, height > 0, !area.overflow, !area.partialValue.multipliedReportingOverflow(by: 66).overflow else {
            throw AetherError.invalidRequest("H3 decoder canvas overflow")
        }
        self.area = area.partialValue; expectedWindows = (frames - 5) / 17
    }

    public init(geometry: H3Geometry) throws {
        try self.init(width: geometry.width, height: geometry.height, frames: geometry.frames)
    }

    /// Input is CHW float32 `[3,22,H,W]`; output is frame-major `[17,3,H,W]`.
    public mutating func push(_ window: [Float]) throws -> [Float] {
        guard windows < expectedWindows, window.count == area * 66, window.allSatisfy(\.isFinite) else {
            throw AetherError.invalidTensor("H3 decoded window count, shape or values differ")
        }
        var frames: [Float] = []; frames.reserveCapacity(area * 51)
        for f in 0..<17 {
            for c in 0..<3 {
                for p in 0..<area {
                    let value = window[(c * 22 + f) * area + p]
                    if f < 5, let tail {
                        let weight = Float(f) / 5
                        frames.append(product(tail[(c * 5 + f) * area + p], 1 - weight) + product(value, weight))
                    } else {
                        frames.append(value)
                    }
                }
            }
        }
        var next: [Float] = []; next.reserveCapacity(area * 15)
        for c in 0..<3 { next.append(contentsOf: window[((c * 22 + 17) * area)..<((c * 22 + 22) * area)]) }
        tail = next; windows += 1
        return frames
    }

    /// The final five frames, frame-major `[5,3,H,W]`.
    public func finish() throws -> [Float] {
        guard windows == expectedWindows, let tail else {
            throw AetherError.invalidTensor("H3 decoder did not supply every temporal window")
        }
        var frames: [Float] = []; frames.reserveCapacity(tail.count)
        for f in 0..<5 { for c in 0..<3 { frames.append(contentsOf: tail[((c * 5 + f) * area)..<((c * 5 + f + 1) * area)]) } }
        return frames
    }
}

/// Source VAE spatial tiling outside the native graph. Pixel-space boundaries are
/// recorded by the original VAE, including offsets on odd latent rows.
public struct H3SpatialTiling: Sendable, Equatable, Codable {
    public let tileHeight: Int
    public let tileWidth: Int
    public let yStarts: [Int]
    public let xStarts: [Int]
    public let yOverlaps: [Int]
    public let xOverlaps: [Int]
    private enum CodingKeys: String, CodingKey {
        case tileHeight = "tile_height", tileWidth = "tile_width", yStarts = "y_starts", xStarts = "x_starts"
        case yOverlaps = "y_overlaps", xOverlaps = "x_overlaps"
    }
    public var tileCount: Int { yStarts.count * xStarts.count }

    public init(tileHeight: Int, tileWidth: Int, yStarts: [Int], xStarts: [Int], yOverlaps: [Int], xOverlaps: [Int]) {
        self.tileHeight = tileHeight; self.tileWidth = tileWidth; self.yStarts = yStarts; self.xStarts = xStarts
        self.yOverlaps = yOverlaps; self.xOverlaps = xOverlaps
    }

    public func validate(width: Int, height: Int) throws {
        let area = width.multipliedReportingOverflow(by: height)
        guard width > 0, height > 0, width.isMultiple(of: 32), height.isMultiple(of: 32), !area.overflow,
              area.partialValue <= 1_032_192 else {
            throw AetherError.invalidRequest("H3 spatial canvas exceeds the profile contract")
        }
        guard tileHeight == 256, tileWidth == 256 else {
            throw AetherError.invalidModel("H3 spatial tile export must retain the source 256px tile")
        }
        for (extent, tile, starts, overlaps) in [(height, tileHeight, yStarts, yOverlaps), (width, tileWidth, xStarts, xOverlaps)] {
            guard !starts.isEmpty, starts.count <= 64, starts[0] == 0, starts.count == overlaps.count + 1,
                  starts[starts.count - 1] + tile == extent else {
                throw AetherError.invalidModel("H3 spatial tile grid does not cover the canvas")
            }
            guard starts.allSatisfy({ $0.isMultiple(of: 16) }),
                  overlaps.allSatisfy({ $0 > 0 && $0 < tile && $0.isMultiple(of: 16) }),
                  zip(zip(starts, starts.dropFirst()), overlaps).allSatisfy({ $0.0 + tile - $1 == $0.1 }) else {
                throw AetherError.invalidModel("H3 spatial tile offsets/overlaps differ")
            }
        }
    }

    /// Extracts seven frames of one tile from globally packed `[7*H/32*W/32, 96]` latents.
    /// Tile starts need not align with the global 2×2 spatial patches.
    public func extract(width: Int, height: Int, packed: [Float], index: Int) throws -> [Float] {
        try validate(width: width, height: height)
        let (rowsH, rowsW) = (height / 32, width / 32)
        guard (0..<tileCount).contains(index), packed.count == 7 * rowsH * rowsW * 96 else {
            throw AetherError.invalidTensor("H3 spatial packed input differs")
        }
        let y0 = yStarts[index / xStarts.count] / 16, x0 = xStarts[index % xStarts.count] / 16
        var tile: [Float] = []; tile.reserveCapacity(7 * (tileHeight / 32) * (tileWidth / 32) * 96)
        for f in 0..<7 {
            for y in 0..<(tileHeight / 32) {
                for x in 0..<(tileWidth / 32) {
                    for c in 0..<24 {
                        for dy in 0..<2 {
                            for dx in 0..<2 {
                                let gy = y0 + y * 2 + dy, gx = x0 + x * 2 + dx
                                tile.append(packed[((f * rowsH + gy / 2) * rowsW + gx / 2) * 96 + c * 4 + (gy % 2) * 2 + gx % 2])
                            }
                        }
                    }
                }
            }
        }
        return tile
    }
}

/// Blends decoded spatial tiles into one `[3,frames,H,W]` window: vertically first, then
/// horizontally, using the original (unblended) neighbouring tiles before cropping.
public struct H3SpatialStitcher: Sendable {
    private let tiling: H3SpatialTiling
    private let width: Int, height: Int, frames: Int
    public private(set) var next = 0
    private var above: [[Float]]
    private var output: [Float]

    public init(tiling: H3SpatialTiling, width: Int, height: Int, frames: Int = H3VideoStitcher.windowFrames) throws {
        try tiling.validate(width: width, height: height)
        guard frames == H3VideoStitcher.windowFrames else {
            throw AetherError.invalidRequest("H3 spatial window must contain 22 decoded frames")
        }
        self.tiling = tiling; self.width = width; self.height = height; self.frames = frames
        above = Array(repeating: [], count: tiling.xStarts.count)
        output = Array(repeating: 0, count: 3 * frames * height * width)
    }

    /// Original NCTHW tile data with singleton N removed, in row-major tile order.
    public mutating func push(_ original: [Float]) throws {
        let t = tiling, h = t.tileHeight, w = t.tileWidth
        guard next < t.tileCount, original.count == 3 * frames * h * w, original.allSatisfy(\.isFinite) else {
            throw AetherError.invalidTensor("H3 spatial decoded tile count/shape/values differ")
        }
        let iy = next / t.xStarts.count, ix = next % t.xStarts.count
        var pixels = original
        if iy > 0 {
            let overlap = t.yOverlaps[iy - 1], prior = above[ix]
            for plane in 0..<(3 * frames) {
                for y in 0..<overlap {
                    let weight = Float(y) / Float(overlap)
                    for x in 0..<w {
                        let at = (plane * h + y) * w + x
                        pixels[at] = product(prior[(plane * h + h - overlap + y) * w + x], 1 - weight) + product(pixels[at], weight)
                    }
                }
            }
        }
        if ix > 0 {
            let overlap = t.xOverlaps[ix - 1], prior = above[ix - 1]
            for plane in 0..<(3 * frames) {
                for y in 0..<h {
                    for x in 0..<overlap {
                        let at = (plane * h + y) * w + x
                        let weight = Float(x) / Float(overlap)
                        pixels[at] = product(prior[(plane * h + y) * w + w - overlap + x], 1 - weight) + product(pixels[at], weight)
                    }
                }
            }
        }
        let cropH = h - (iy < t.yOverlaps.count ? t.yOverlaps[iy] : 0)
        let cropW = w - (ix < t.xOverlaps.count ? t.xOverlaps[ix] : 0)
        for plane in 0..<(3 * frames) {
            for y in 0..<cropH {
                let to = (plane * height + t.yStarts[iy] + y) * width + t.xStarts[ix], from = (plane * h + y) * w
                output.replaceSubrange(to..<(to + cropW), with: pixels[from..<(from + cropW)])
            }
        }
        above[ix] = original; next += 1
    }

    public func finish() throws -> [Float] {
        guard next == tiling.tileCount else { throw AetherError.invalidTensor("H3 spatial decoder did not supply every tile") }
        return output
    }
}

extension H3Geometry {
    /// Packed rows per latent frame (`H/32 * W/32`).
    public var rowsPerLatentFrame: Int { (height / 32) * (width / 32) }
    /// Temporal decoder windows: seven latent frames at stride five.
    public var videoWindows: Int { (frames - 5) / 17 }

    /// The packed `[7*H/32*W/32, 96]` rows for decoder window `index` of packed `[rows, 96]` video latents.
    public func videoWindowRows(_ packed: [Float], index: Int) throws -> [Float] {
        guard packed.count == targetVideoRows * 96, (0..<videoWindows).contains(index) else {
            throw AetherError.invalidTensor("H3 packed video latents or window index differ")
        }
        let start = index * 5 * rowsPerLatentFrame * 96
        return Array(packed[start..<(start + 7 * rowsPerLatentFrame * 96)])
    }
}

/// Frame-major `[n,3,H,W]` ImageNet-normalized samples → interleaved RGB8, clipped,
/// rounded ties-to-even, with separately rounded products (the prototype's writer).
public enum H3Pixels {
    public static func rgb8(frames: [Float], width: Int, height: Int) throws -> [[UInt8]] {
        let area = width * height
        guard area > 0, frames.count.isMultiple(of: area * 3), frames.allSatisfy(\.isFinite) else {
            throw AetherError.invalidTensor("invalid stitched H3 frames")
        }
        let std: [Float] = [0.229, 0.224, 0.225], mean: [Float] = [0.485, 0.456, 0.406]
        return stride(from: 0, to: frames.count, by: area * 3).map { base in
            var rgb = [UInt8](repeating: 0, count: area * 3)
            for p in 0..<area {
                for c in 0..<3 {
                    let normalized = min(max(product(frames[base + c * area + p], std[c]) + mean[c], 0), 1)
                    rgb[p * 3 + c] = UInt8((normalized * 255).rounded(.toNearestOrEven))
                }
            }
            return rgb
        }
    }
}

@inline(never) private func product(_ lhs: Float, _ rhs: Float) -> Float { lhs * rhs }
