public import AetherCore
import AetherRuntime
import Foundation

// MiniMax H3's host recipe (MANIFEST.md §5.10): the packed layout and rotary tables, the row timesteps, the seeded
// latents, the scheduler arms, and the video and audio decode around the graphs. Plain Swift over arrays, ported from
// the 0.0.x prototype (whose pipeline-captured fixtures it matched) and checked against Diffusers' own layout
// (Tests/AetherVideoTests/Fixtures/h3-layout.json, Forge/h3/layout_fixture_h3.py).

/// The frame grid of one qualified profile: frames on the source VAE's 17n + 5 grid at 24 fps, 16× spatial, the
/// (1, 2, 2) patch, 40 audio latents a second.
public struct H3Geometry: Sendable, Hashable {
    public let width: Int
    public let height: Int
    public let frames: Int
    public let latentFrames: Int
    public var latentHeight: Int { height / 16 }
    public var latentWidth: Int { width / 16 }
    /// Packed rows per latent frame (`H/32 · W/32`).
    public var rowsPerLatentFrame: Int { (height / 32) * (width / 32) }
    public var videoRows: Int { latentFrames * rowsPerLatentFrame }
    /// Audio latents per channel (stereo: twice as many rows).
    public let audioLatents: Int
    public var audioRows: Int { audioLatents * 2 }
    /// Temporal decoder windows: seven latent frames at stride five, 17 new frames each.
    public var videoWindows: Int { (frames - 5) / 17 }

    public init(_ profile: Manifest.H3.Profile) {
        width = profile.width
        height = profile.height
        frames = profile.frames
        latentFrames = profile.latentFrames
        audioLatents = profile.audioLatents
    }

    init(width: Int, height: Int, frames: Int, latentFrames: Int, audioLatents: Int) {
        self.width = width
        self.height = height
        self.frames = frames
        self.latentFrames = latentFrames
        self.audioLatents = audioLatents
    }
}

/// The packed sequence of a text-to-video-and-audio step (`MiniMaxH3PrepareLayoutStep.build_packed_sequence`):
/// `[text | audio (channel-major) | video]`, with `(t, h, w)` positions computed in Double and narrowed once to Float.
struct H3Layout: Sendable {
    /// `[rows, 3]` row-major.
    let positions: [Float]
    /// Per row: text 1, video 0, audio 2.
    let tags: [Int32]
    let textIndices: [Int32]
    let audioIndices: [Int32]
    let videoIndices: [Int32]
    var rows: Int { tags.count }

    /// `latentHeight`, `latentWidth` are the latent grid (H/16, W/16); `audioLatents` per channel.
    init(textTokens n: Int, latentFrames: Int, latentHeight h: Int, latentWidth w: Int, audioLatents: Int) {
        let frameRows = (h / 2) * (w / 2)
        let audioStart = n, videoStart = n + audioLatents * 2
        let rows = videoStart + latentFrames * frameRows
        var positions = [Float](repeating: 0, count: rows * 3)
        for i in 0..<n { positions[i * 3] = Float(i) }
        let sqrtArea = Double(h * w).squareRoot()
        func grid(_ dimension: Int) -> [Double] {
            let ratio = Double(dimension) / sqrtArea
            let left = (1 - ratio) / 2
            let step = ((left + ratio) - left) / Double(dimension / 2)
            return (0..<(dimension / 2)).map { (left + Double($0) * step) * 32 }
        }
        let hg = grid(h), wg = grid(w)
        var time = Double(n)
        for f in 0..<latentFrames {
            for (y, hy) in hg.enumerated() {
                for (x, wx) in wg.enumerated() {
                    let at = (videoStart + f * frameRows + y * wg.count + x) * 3
                    positions[at] = Float(time)
                    positions[at + 1] = Float(hy)
                    positions[at + 2] = Float(wx)
                }
            }
            time += (5.0 / 3.0) * (f % 5 == 0 ? 1.0 : 4.0)
        }
        for channel in 0..<2 {
            for t in 0..<audioLatents {
                let at = (audioStart + channel * audioLatents + t) * 3
                positions[at] = Float(n + t)
                positions[at + 2] = Float(channel == 0 ? wg[0] : wg[wg.count - 1])
            }
        }
        var tags = [Int32](repeating: 1, count: n) + [Int32](repeating: 0, count: rows - n)
        for i in audioStart..<videoStart { tags[i] = 2 }
        self.positions = positions
        self.tags = tags
        textIndices = (0..<n).map(Int32.init)
        audioIndices = (audioStart..<videoStart).map(Int32.init)
        videoIndices = (videoStart..<rows).map(Int32.init)
    }

    init(textTokens: Int, geometry g: H3Geometry) {
        self.init(textTokens: textTokens, latentFrames: g.latentFrames, latentHeight: g.latentHeight,
                  latentWidth: g.latentWidth, audioLatents: g.audioLatents)
    }

    /// `gather[r]`: the source row of packed row `r` in `[text; video; audio]` (the pack graph's input order).
    var gather: [Int32] {
        var gather = [Int32](repeating: 0, count: rows)
        let t = textIndices.count, v = videoIndices.count
        for (j, r) in textIndices.enumerated() { gather[Int(r)] = Int32(j) }
        for (j, r) in videoIndices.enumerated() { gather[Int(r)] = Int32(t + j) }
        for (j, r) in audioIndices.enumerated() { gather[Int(r)] = Int32(t + v + j) }
        return gather
    }

    /// One step's row timesteps (`build_row_timesteps`): text and video rows take the video timestep, audio rows the
    /// audio one; the distinct values ascending, each row's index into them, and the AdaLN row `index · 3 + tag`.
    func timesteps(video: Float, audio: Float) -> (values: [Float], indices: [Int32], adaln: [Int32]) {
        var perRow = [Float](repeating: video, count: rows)
        for i in audioIndices { perRow[Int(i)] = audio }
        let values = Array(Set(perRow)).sorted()
        let indices = perRow.map { value in Int32(values.firstIndex(of: value)!) }
        return (values, indices, zip(indices, tags).map { $0 * 3 + $1 })
    }

    /// The transformer's rotary tables `[rows, 96]` (`MiniMaxH3RotaryPosEmbed`: 16 frequencies at θ = 10000 shared by
    /// the t, h and w axes, their angle blocks concatenated and then repeated for `rotate_half`), float32 throughout.
    func rotary() -> (cos: [Float], sin: [Float]) {
        let inverse = (0..<16).map { Float(1) / powf(10000, Float(2 * $0) / 32) }
        var cos = [Float](repeating: 0, count: rows * 96), sin = cos
        for row in 0..<rows {
            for axis in 0..<3 {
                let position = positions[row * 3 + axis]
                for f in 0..<16 {
                    let angle = position * inverse[f]
                    let c = cosf(angle), s = sinf(angle)
                    for copy in 0..<2 {
                        cos[row * 96 + copy * 48 + axis * 16 + f] = c
                        sin[row * 96 + copy * 48 + axis * 16 + f] = s
                    }
                }
            }
        }
        return (cos, sin)
    }
}

extension H3Geometry {
    /// Seeded initial latents (0.0.x's stream): Kit's ``SeededGaussianNoise`` read once, the source `[24, T, H, W]`
    /// video tensor first, packed to `[videoRows, 96]`, then the channel-major audio rows `[audioRows, 32]`.
    func initialLatents(seed: UInt64) throws(AetherError) -> (video: [Float], audio: [Float]) {
        let videoCount = videoRows * 96, audioCount = audioRows * 32
        let noise = try SeededGaussianNoise.generate(seed: seed, count: videoCount + audioCount)
        return (packFrames(Array(noise[..<videoCount])), Array(noise[videoCount...]))
    }

    /// `[24, latentFrames, H, W]` → rows frame-major then row-major, each `(channel, dy, dx)` (`patchify_video_latents`
    /// with patch (1, 2, 2)).
    func packFrames(_ values: [Float]) -> [Float] {
        let lh = latentHeight, lw = latentWidth, lf = latentFrames
        var result: [Float] = []
        result.reserveCapacity(values.count)
        for f in 0..<lf {
            for y in stride(from: 0, to: lh, by: 2) {
                for x in stride(from: 0, to: lw, by: 2) {
                    for channel in 0..<24 {
                        for dy in 0..<2 {
                            for dx in 0..<2 {
                                result.append(values[((channel * lf + f) * lh + y + dy) * lw + x + dx])
                            }
                        }
                    }
                }
            }
        }
        return result
    }

    /// Decoder window `index`'s packed rows `[7 · rowsPerLatentFrame, 96]` of packed `[videoRows, 96]` latents.
    func windowRows(_ packed: [Float], index: Int) -> ArraySlice<Float> {
        let start = index * 5 * rowsPerLatentFrame * 96
        return packed[start..<(start + 7 * rowsPerLatentFrame * 96)]
    }
}

/// One scheduler arm of a recipe (0.0.x's `H3Schedule`): the data-ward x0 blend, float32, products rounded before the
/// adds. `timesteps[i] = 1 − σᵢ`.
struct H3Arm: Sendable {
    let sigmas: [Float]
    let timesteps: [Float]

    init(_ arm: Manifest.H3.Arm) {
        sigmas = arm.sigmas
        timesteps = arm.timesteps
    }

    /// `x ← r·x + (1 − r)·(x + σ·v)` with `σ = 1 − t[step]` and `r = σ[step+1] / σ[step]`, in place.
    func advance(step: Int, sample: inout [Float], velocity: [Float], from start: Int = 0) {
        let sigma = 1 - timesteps[step]
        let ratio = sigmas[step + 1] / sigmas[step]
        for i in start..<sample.count {
            let value = sample[i]
            let denoised = value + h3Product(sigma, velocity[i])
            sample[i] = h3Product(ratio, value) + h3Product(1 - ratio, denoised)
        }
    }
}

@inline(never) func h3Product(_ lhs: Float, _ rhs: Float) -> Float { lhs * rhs }

/// Pads a step's modulation inputs to the trajectory's largest cardinality, so the body and heads see one shape for
/// the whole render (0.0.x's `H3ModulationPadding`). Exact: the block reads its table only at the AdaLN rows
/// (`tags` < 3k) and the heads read `temb` only at `indices` (< k), so the appended copies are never selected.
/// Without it a cardinality change mid-render specializes the body a second time (about 2.5 GiB at 960×544×124).
enum H3Padding {
    /// `[6, 3k, D]` bf16 bits → `[6, 3c, D]`, each slice's rows followed by copies of them.
    static func table(_ bytes: [UInt8], k: Int, cardinality c: Int, width: Int) -> [UInt8] {
        repeatRows(bytes, slices: 6, rows: 3 * k, target: 3 * c, rowBytes: width * 2)
    }

    /// `[k, C]` float32 → `[c, C]`.
    static func embedding(_ values: [Float], k: Int, cardinality c: Int) -> [Float] {
        let width = values.count / k
        return (0..<c).flatMap { values[(($0 % k) * width)..<(($0 % k + 1) * width)] }
    }

    static func repeatRows(_ bytes: [UInt8], slices: Int, rows: Int, target: Int, rowBytes: Int) -> [UInt8] {
        if rows == target { return bytes }
        var out = [UInt8]()
        out.reserveCapacity(slices * target * rowBytes)
        for slice in 0..<slices {
            let base = slice * rows * rowBytes
            for row in 0..<target {
                let at = base + (row % rows) * rowBytes
                out.append(contentsOf: bytes[at..<(at + rowBytes)])
            }
        }
        return out
    }
}

/// The source VAE's spatial tiling (`_split_tiles`): 256 px tiles, at least 64 px of overlap, the slack added
/// round-robin in 16 px steps so every boundary stays latent-aligned.
struct H3Tiling: Sendable, Equatable {
    let yStarts: [Int], xStarts: [Int], yOverlaps: [Int], xOverlaps: [Int]
    static let side = 256
    var count: Int { yStarts.count * xStarts.count }

    init(width: Int, height: Int) {
        func split(_ length: Int) -> (starts: [Int], overlaps: [Int]) {
            var count = (length + 255) / 256
            while 256 * count - 64 * (count - 1) - length < 0 { count += 1 }
            var overlaps = [Int](repeating: 64, count: count - 1)
            for i in 0..<((256 * count - overlaps.reduce(0, +) - length) / 16) { overlaps[i % (count - 1)] += 16 }
            var starts = [0]
            for overlap in overlaps { starts.append(starts[starts.count - 1] + 256 - overlap) }
            return (starts, overlaps)
        }
        let y = split(height), x = split(width)
        yStarts = y.starts
        xStarts = x.starts
        yOverlaps = y.overlaps
        xOverlaps = x.overlaps
    }

    /// Tile `index`'s seven frames, `[7 · 8 · 8, 96]`, out of a window's globally packed `[7 · H/32 · W/32, 96]` rows.
    /// Tile starts need not align with the global 2×2 patches.
    func extract(_ packed: ArraySlice<Float>, width: Int, height: Int, index: Int) -> [Float] {
        let rowsH = height / 32, rowsW = width / 32, base = packed.startIndex
        let y0 = yStarts[index / xStarts.count] / 16, x0 = xStarts[index % xStarts.count] / 16
        var tile: [Float] = []
        tile.reserveCapacity(7 * 64 * 96)
        for f in 0..<7 {
            for y in 0..<8 {
                for x in 0..<8 {
                    for c in 0..<24 {
                        for dy in 0..<2 {
                            for dx in 0..<2 {
                                let gy = y0 + y * 2 + dy, gx = x0 + x * 2 + dx
                                tile.append(packed[base + ((f * rowsH + gy / 2) * rowsW + gx / 2) * 96 + c * 4 + (gy % 2) * 2 + gx % 2])
                            }
                        }
                    }
                }
            }
        }
        return tile
    }
}

/// Blends a window's decoded tiles into `[3, 22, H, W]`: vertically, then horizontally, against the original
/// (unblended) neighbours, then crops (the source VAE's blend).
struct H3SpatialStitcher {
    let tiling: H3Tiling
    let width: Int, height: Int
    static let frames = 22
    private var next = 0
    private var above: [[Float]]
    private(set) var output: [Float]

    init(tiling: H3Tiling, width: Int, height: Int) {
        self.tiling = tiling
        self.width = width
        self.height = height
        above = Array(repeating: [], count: tiling.xStarts.count)
        output = Array(repeating: 0, count: 3 * Self.frames * height * width)
    }

    /// One tile's `[3, 22, 256, 256]`, in row-major tile order.
    mutating func push(_ original: [Float]) {
        let t = tiling, side = H3Tiling.side, planes = 3 * Self.frames
        let iy = next / t.xStarts.count, ix = next % t.xStarts.count
        var pixels = original
        if iy > 0 {
            let overlap = t.yOverlaps[iy - 1], prior = above[ix]
            for plane in 0..<planes {
                for y in 0..<overlap {
                    let weight = Float(y) / Float(overlap)
                    for x in 0..<side {
                        let at = (plane * side + y) * side + x
                        pixels[at] = h3Product(prior[(plane * side + side - overlap + y) * side + x], 1 - weight) + h3Product(pixels[at], weight)
                    }
                }
            }
        }
        if ix > 0 {
            let overlap = t.xOverlaps[ix - 1], prior = above[ix - 1]
            for plane in 0..<planes {
                for y in 0..<side {
                    for x in 0..<overlap {
                        let at = (plane * side + y) * side + x
                        let weight = Float(x) / Float(overlap)
                        pixels[at] = h3Product(prior[(plane * side + y) * side + side - overlap + x], 1 - weight) + h3Product(pixels[at], weight)
                    }
                }
            }
        }
        let cropH = side - (iy < t.yOverlaps.count ? t.yOverlaps[iy] : 0)
        let cropW = side - (ix < t.xOverlaps.count ? t.xOverlaps[ix] : 0)
        for plane in 0..<planes {
            for y in 0..<cropH {
                let to = (plane * height + t.yStarts[iy] + y) * width + t.xStarts[ix], from = (plane * side + y) * side
                output.replaceSubrange(to..<(to + cropW), with: pixels[from..<(from + cropW)])
            }
        }
        above[ix] = original
        next += 1
    }
}

/// The source VAE's temporal boundaries: each 22-frame window emits 17 frames, its first five cross-faded with the
/// previous window's last five; the final five frames close the clip. Works on normalized samples `[3, 22, H, W]`.
struct H3TemporalStitcher {
    let area: Int
    private var tail: [Float]?

    init(width: Int, height: Int) { area = width * height }

    /// A window `[3, 22, H, W]` → its 17 frames `[17, 3, H, W]`.
    mutating func push(_ window: [Float]) -> [Float] {
        var frames: [Float] = []
        frames.reserveCapacity(area * 51)
        for f in 0..<17 {
            for c in 0..<3 {
                let at = (c * 22 + f) * area
                if f < 5, let tail {
                    let weight = Float(f) / 5, from = (c * 5 + f) * area
                    for p in 0..<area { frames.append(h3Product(tail[from + p], 1 - weight) + h3Product(window[at + p], weight)) }
                } else {
                    frames.append(contentsOf: window[at..<(at + area)])
                }
            }
        }
        var next: [Float] = []
        next.reserveCapacity(area * 15)
        for c in 0..<3 { next.append(contentsOf: window[((c * 22 + 17) * area)..<((c * 22 + 22) * area)]) }
        tail = next
        return frames
    }

    /// The clip's last five frames `[5, 3, H, W]`.
    func finish() -> [Float] {
        guard let tail else { return [] }
        var frames: [Float] = []
        for f in 0..<5 { for c in 0..<3 { frames.append(contentsOf: tail[((c * 5 + f) * area)..<((c * 5 + f + 1) * area)]) } }
        return frames
    }
}

/// Frame-major `[n, 3, H, W]` ImageNet-normalized samples → interleaved RGB8 per frame: `x·std + mean`, clipped to
/// [0, 1], `·255` rounded ties to even, products rounded separately (0.0.x's writer).
enum H3Pixels {
    static func rgb8(_ frames: [Float], width: Int, height: Int) -> [[UInt8]] {
        let area = width * height
        let std: [Float] = [0.229, 0.224, 0.225], mean: [Float] = [0.485, 0.456, 0.406]
        return stride(from: 0, to: frames.count, by: area * 3).map { base in
            var rgb = [UInt8](repeating: 0, count: area * 3)
            for c in 0..<3 {
                for p in 0..<area {
                    let normalized = min(max(h3Product(frames[base + c * area + p], std[c]) + mean[c], 0), 1)
                    rgb[p * 3 + c] = UInt8((normalized * 255).rounded(.toNearestOrEven))
                }
            }
            return rgb
        }
    }
}
