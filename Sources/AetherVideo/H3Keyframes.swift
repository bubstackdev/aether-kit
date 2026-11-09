public import AetherCore
import AetherRuntime
import Foundation

// MiniMax H3's FL2VA keyframes (MANIFEST.md §5.10): a clip anchored to a picture at its first frame, its last frame or
// both. Each keyframe conditions the clip twice, as Diffusers' pipeline does: its VAE encoding becomes condition rows of
// the packed sequence, held at t = 0.999 through every step, and the conditioner sees it as a labelled vision block
// (Qwen3-VL's vision tower, M-RoPE positions and DeepStack) before the prompt. The rules are the 0.0.5 presentation's
// (the 0.0.x `H3Presentation` and `H3VisionInputs`), checked against Diffusers' own step
// (Tests/AetherVideoTests/Fixtures/h3-keyframes.json, Forge/h3/keyframe_fixture_h3.py).

/// One keyframe: interleaved RGB8 at the clip's canvas (`width · height · 3` bytes, rows top to bottom). The pipeline
/// doesn't resize: a keyframe whose size isn't the profile's canvas is refused.
public struct VideoKeyframe: Sendable, Hashable {
    public let width: Int
    public let height: Int
    public let rgb: [UInt8]

    /// Fails unless `rgb` holds `width · height · 3` bytes for positive sides.
    public init(rgb: [UInt8], width: Int, height: Int) throws(AetherError) {
        guard width > 0, height > 0, rgb.count == width * height * 3 else {
            throw .generation(.invalidParameters("a keyframe's RGB bytes don't match \(width)x\(height)"))
        }
        self.width = width
        self.height = height
        self.rgb = rgb
    }
}

/// The keyframes a clip is anchored to (the FL2VA task): its first frame, its last frame, or both.
public enum VideoKeyframes: Sendable, Hashable {
    case first(VideoKeyframe)
    case last(VideoKeyframe)
    case firstAndLast(VideoKeyframe, VideoKeyframe)

    /// The keyframes in packed order: first, then last.
    public var frames: [VideoKeyframe] {
        switch self {
        case .first(let frame), .last(let frame): [frame]
        case .firstAndLast(let first, let last): [first, last]
        }
    }

    /// Which end each of ``frames`` is anchored to.
    var anchors: [H3Anchor] {
        switch self {
        case .first: [.first]
        case .last: [.last]
        case .firstAndLast: [.first, .last]
        }
    }
}

/// Where a keyframe's condition rows sit in the clip's rotary time: the first latent frame, or the last.
enum H3Anchor: Sendable, Hashable { case first, last }

/// The manifest's FL2VA rules: `h3.keyframe` (the condition encode and its noise level) and `h3.vision` (the tower's
/// patching and the presentation's special ids), plus the conditioner's M-RoPE section.
struct H3KeyframeRules: Sendable, Hashable {
    struct Keyframe: Decodable, Hashable {
        var tileSize: Int
        var conditionTimestep: Float
    }
    struct Vision: Decodable, Hashable {
        struct Presentation: Decodable, Hashable {
            var label: String
            var visionStart: Int32
            var imagePad: Int32
            var visionEnd: Int32
        }
        var patchSize: Int
        var mergeSize: Int
        var temporalPatchSize: Int
        var tableSide: Int
        var deepstackLevels: Int
        var presentation: Presentation
    }
    struct Rope: Decodable, Hashable { var mropeSection: [Int] }

    var keyframe: Keyframe
    var vision: Vision
    var mropeSection: [Int]

    /// The modality tags of a packed row (Diffusers' `MINIMAX_H3_*_TAG`).
    static let videoTag: Int32 = 0, textTag: Int32 = 1, audioTag: Int32 = 2
    /// The processor's pixel bounds: a canvas outside them would be resized before patching, which the driver doesn't do.
    static let pixelBounds = 65_536...16_777_216

    /// Nil when the export doesn't carry FL2VA (no `keyframe` or `vision`).
    init?(_ h3: Manifest.H3) throws(AetherError) {
        guard h3.tasks.contains("fl2va"), let keyframe = h3.keyframe, let vision = h3.vision else { return nil }
        self.keyframe = try Self.decode(Keyframe.self, keyframe, field: "h3.keyframe")
        self.vision = try Self.decode(Vision.self, vision, field: "h3.vision")
        if let rope = h3.conditioner.rope {
            mropeSection = try Self.decode(Rope.self, rope, field: "h3.conditioner.rope").mropeSection
        } else {
            mropeSection = [24, 20, 20]
        }
        guard self.vision.presentation.label.contains("{index}"), self.vision.patchSize > 0, self.vision.mergeSize > 0,
              self.vision.temporalPatchSize > 0, self.vision.tableSide > 1, mropeSection.count == 3 else {
            throw .manifestInvalid(field: "h3.vision", reason: "needs a {index} label, positive patching and a three-axis M-RoPE section")
        }
    }

    private static func decode<T: Decodable>(_ type: T.Type, _ value: JSONValue, field: String) throws(AetherError) -> T {
        do { return try JSONDecoder().decode(T.self, from: JSONEncoder().encode(value)) } catch {
            throw .manifestInvalid(field: field, reason: "\(error)")
        }
    }

    func label(_ index: Int) -> String { vision.presentation.label.replacingOccurrences(of: "{index}", with: "\(index)") }

    /// A canvas's patch grid (rows, columns); both sides must be multiples of `patch · merge` within the processor's
    /// bounds, so its `smart_resize` keeps the canvas as it is.
    func grid(width: Int, height: Int) throws(AetherError) -> (height: Int, width: Int) {
        let unit = vision.patchSize * vision.mergeSize
        guard width % unit == 0, height % unit == 0, Self.pixelBounds.contains(width * height) else {
            throw .generation(.invalidParameters("a \(width)x\(height) keyframe can't be presented as it is"))
        }
        return (height / vision.patchSize, width / vision.patchSize)
    }
}

/// Upstream's FL2VA presentation (`MiniMaxH3FL2VATextEncoderStep`): each keyframe, in packed order, contributes its
/// label (tagged text) and a vision block, `<|vision_start|>`, one `<|image_pad|>` per merged image token and
/// `<|vision_end|>` (all tagged video); the prompt follows, tagged text. Every piece is tokenized on its own.
///
/// `positions` are Qwen3-VL's M-RoPE positions (`get_rope_index`): a run of other tokens counts p, p+1, … on all three
/// axes; an image's pads take (p, p + row, p + column) over its merged grid in raster order, and p then advances by the
/// larger merged side, not by the token count.
struct H3Presentation: Sendable, Hashable {
    /// One keyframe's image-pad rows `start..<start + tokens`, over a merged grid `rows × columns`.
    struct Image: Sendable, Hashable {
        let start: Int
        let rows: Int
        let columns: Int
        var tokens: Int { rows * columns }
    }

    let ids: [Int32]
    let tags: [Int32]
    /// `positions[axis][row]`: time, height, width.
    let positions: [[Int32]]
    let images: [Image]
    var count: Int { ids.count }

    /// `prompt`: the prompt's ids; `canvases`: each keyframe's size, in packed order; `tokenize` a label's ids.
    init(prompt: [Int32], canvases: [(width: Int, height: Int)], rules: H3KeyframeRules,
         tokenize: (String) throws(AetherError) -> [Int32]) throws(AetherError) {
        let p = rules.vision.presentation, merge = rules.vision.mergeSize
        var ids: [Int32] = [], tags: [Int32] = [], images: [Image] = []
        for (index, canvas) in canvases.enumerated() {
            let grid = try rules.grid(width: canvas.width, height: canvas.height)
            let label = try tokenize(rules.label(index + 1))
            ids += label
            tags += Array(repeating: H3KeyframeRules.textTag, count: label.count)
            ids.append(p.visionStart)
            let image = Image(start: ids.count, rows: grid.height / merge, columns: grid.width / merge)
            images.append(image)
            ids += Array(repeating: p.imagePad, count: image.tokens)
            ids.append(p.visionEnd)
            tags += Array(repeating: H3KeyframeRules.videoTag, count: image.tokens + 2)
        }
        ids += prompt
        tags += Array(repeating: H3KeyframeRules.textTag, count: prompt.count)
        self.ids = ids
        self.tags = tags
        self.images = images
        positions = Self.positions(count: ids.count, images: images)
    }

    static func positions(count: Int, images: [Image]) -> [[Int32]] {
        var axes = [[Int32]](repeating: [], count: 3)
        for a in 0..<3 { axes[a].reserveCapacity(count) }
        var p: Int32 = 0, row = 0
        for image in images + [Image(start: count, rows: 0, columns: 0)] {
            while row < image.start {
                for a in 0..<3 { axes[a].append(p) }
                p += 1
                row += 1
            }
            for y in 0..<image.rows {
                for x in 0..<image.columns {
                    axes[0].append(p)
                    axes[1].append(p + Int32(y))
                    axes[2].append(p + Int32(x))
                }
            }
            p += Int32(max(image.rows, image.columns))
            row += image.tokens
        }
        return axes
    }

    /// The axis (0 time, 1 height, 2 width) of each rotary frequency: Qwen3-VL's interleaved layout
    /// (`recomposition_frequencies`), frequency j on height when j % 3 == 1 and j < 3 · section[1], on width when
    /// j % 3 == 2 and j < 3 · section[2], else on time.
    static func rotaryAxes(section: [Int]) -> [Int] {
        (0..<section.reduce(0, +)).map { j in
            if j % 3 == 1, j < 3 * section[1] { return 1 }
            if j % 3 == 2, j < 3 * section[2] { return 2 }
            return 0
        }
    }

    /// cos or sin `[T, width]` bf16 bits at these positions, gathered from the conditioner's text table (`[N, width]`
    /// bf16, row p = position p on every axis): column j takes the row of its axis's position. Upstream computes each
    /// axis's angles with the same multiply and cast and only selects among them, so the gather is exact.
    func rotary(_ table: [UInt16], width: Int, section: [Int]) throws(AetherError) -> [UInt16] {
        let half = width / 2, axes = Self.rotaryAxes(section: section)
        guard axes.count == half, width == 2 * half, table.count % width == 0 else {
            throw .manifestMismatch(field: "h3.conditioner.rope.mropeSection", expected: "\(half) frequencies", found: "\(axes.count)")
        }
        let limit = table.count / width
        guard positions.allSatisfy({ $0.allSatisfy { Int($0) < limit } }) else {
            throw .generation(.invalidParameters("the presentation's positions exceed the conditioner's \(limit) rotary rows"))
        }
        var out = [UInt16](repeating: 0, count: count * width)
        for row in 0..<count {
            for j in 0..<width {
                out[row * width + j] = table[Int(positions[axes[j % half]][row]) * width + j]
            }
        }
        return out
    }
}

/// The vision tower's inputs for one canvas-sized keyframe, as upstream's processor and tower derive them, every row in
/// merge-window order (merge × merge windows in raster order, the patches inside a window in raster order).
struct H3VisionInputs: Sendable {
    /// `[n, 3 · T · p²]` bf16 bits: (x − 127.5) / 127.5 in float32 (the processor's fused rescale and normalize), the
    /// frame repeated over the T temporal slices, each row `[C][T][p][p]`, rounded to bf16 as the conditioner call casts.
    let pixels: [UInt16]
    /// `[n, 4]`: the bilinear taps into the learned position table and their weights, in float32 exactly as
    /// `get_vision_interpolation_indices_and_weights` (corners aligned).
    let positionIndex: [Int32]
    let positionWeight: [Float]
    /// `[n, 2]`: each patch's (row, column), the tower's 2-D rotary.
    let patchCoordinates: [Int32]
    let gridHeight: Int
    let gridWidth: Int
    var patches: Int { gridHeight * gridWidth }

    init(_ keyframe: VideoKeyframe, rules: H3KeyframeRules) throws(AetherError) {
        let (gh, gw) = try rules.grid(width: keyframe.width, height: keyframe.height)
        let v = rules.vision, p = v.patchSize, m = v.mergeSize, t = v.temporalPatchSize, side = v.tableSide
        let rowLength = 3 * t * p * p, n = gh * gw, width = keyframe.width
        let levels = (0..<256).map { h3BFloat16Bits((Float($0) - 127.5) / 127.5) }
        var pixels = [UInt16](repeating: 0, count: n * rowLength)
        var index: [Int32] = [], weight: [Float] = [], coordinates: [Int32] = []
        index.reserveCapacity(n * 4)
        weight.reserveCapacity(n * 4)
        coordinates.reserveCapacity(n * 2)
        let ys = Self.taps(gh, side: side), xs = Self.taps(gw, side: side)
        keyframe.rgb.withUnsafeBufferPointer { rgb in
            pixels.withUnsafeMutableBufferPointer { out in
                var row = 0
                for wy in 0..<(gh / m) {
                    for wx in 0..<(gw / m) {
                        for i in 0..<m {
                            for j in 0..<m {
                                let r = wy * m + i, c = wx * m + j
                                var offset = row * rowLength
                                for channel in 0..<3 {
                                    for _ in 0..<t {
                                        for y in 0..<p {
                                            let base = ((r * p + y) * width + c * p) * 3 + channel
                                            for x in 0..<p {
                                                out[offset] = levels[Int(rgb[base + x * 3])]
                                                offset += 1
                                            }
                                        }
                                    }
                                }
                                let (ty, tx) = (ys[r], xs[c])
                                index += [ty.lower * side + tx.lower, ty.lower * side + tx.upper,
                                          ty.upper * side + tx.lower, ty.upper * side + tx.upper].map { Int32($0) }
                                weight += [ty.lowerWeight * tx.lowerWeight, ty.lowerWeight * tx.upperWeight,
                                           ty.upperWeight * tx.lowerWeight, ty.upperWeight * tx.upperWeight]
                                coordinates += [Int32(r), Int32(c)]
                                row += 1
                            }
                        }
                    }
                }
            }
        }
        self.pixels = pixels
        positionIndex = index
        positionWeight = weight
        patchCoordinates = coordinates
        gridHeight = gh
        gridWidth = gw
    }

    /// Per position along an axis of `size` patches: the two taps and their weights, in float32 with upstream's order
    /// of operations: src = i · (side − 1) / max(size − 1, 1), weight = max(1 − |src − floor(src) − k|, 0).
    static func taps(_ size: Int, side: Int) -> [(lower: Int, upper: Int, lowerWeight: Float, upperWeight: Float)] {
        (0..<size).map { i in
            let src = Float(i) * Float(side - 1) / Float(max(size - 1, 1))
            let floor = src.rounded(.down), lower = Int(floor)
            return (min(max(lower, 0), side - 1), min(max(lower + 1, 0), side - 1),
                    max(1 - abs(src - floor - 0), 0), max(1 - abs(src - floor - 1), 0))
        }
    }
}

/// The presentation's image rows on the host, in bf16 as upstream computes them: the tower's merger rows replace the
/// image-pad embeddings (`masked_scatter`), and DeepStack level k is added at the image rows after decoder layer k, in
/// float32 rounded to nearest-even bf16 (torch's bf16 add).
enum H3ImageRows {
    /// `hidden` `[T, width]` bf16 bits; `embeddings[i]` `[images[i].tokens, width]`.
    static func scatter(_ embeddings: [[UInt8]], into hidden: inout [UInt8], width: Int, at images: [H3Presentation.Image]) {
        let rowBytes = width * 2
        for (rows, image) in zip(embeddings, images) {
            hidden.replaceSubrange((image.start * rowBytes)..<((image.start + image.tokens) * rowBytes), with: rows)
        }
    }

    /// `deepstack[i]` `[levels, images[i].tokens, width]` bf16 bits; adds level `level`.
    static func add(_ deepstack: [[UInt8]], level: Int, into hidden: inout [UInt8], width: Int, at images: [H3Presentation.Image]) {
        hidden.withUnsafeMutableBytes { target in
            let out = target.bindMemory(to: UInt16.self)
            for (features, image) in zip(deepstack, images) {
                features.withUnsafeBytes { source in
                    let add = source.bindMemory(to: UInt16.self)
                    let from = level * image.tokens * width, to = image.start * width
                    for i in 0..<(image.tokens * width) {
                        let sum = Float(bitPattern: UInt32(out[to + i]) << 16) + Float(bitPattern: UInt32(add[from + i]) << 16)
                        out[to + i] = h3BFloat16Bits(sum)
                    }
                }
            }
        }
    }
}

/// `value` rounded to bf16 (to nearest, ties to even), as its 16 bits.
@inline(__always) func h3BFloat16Bits(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    if value.isNaN { return UInt16(truncatingIfNeeded: bits >> 16) | 0x40 }
    return UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
}

/// numpy's float64 `sum` of a contiguous array (pairwise summation, `pairwise_sum_DOUBLE`, blocks of 128 with eight
/// accumulators): the order Diffusers sums the last anchor's rotary span in. A sequential sum differs from it in the last
/// ulp from 16 latent frames on.
func h3NumpySum(_ values: ArraySlice<Double>) -> Double {
    let n = values.count, a = values.startIndex
    if n < 8 {
        var sum = -0.0
        for value in values { sum += value }
        return sum
    }
    if n <= 128 {
        var r = (0..<8).map { values[a + $0] }
        var i = 8
        while i < n - n % 8 {
            for j in 0..<8 { r[j] += values[a + i + j] }
            i += 8
        }
        var sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n {
            sum += values[a + i]
            i += 1
        }
        return sum
    }
    var half = n / 2
    half -= half % 8
    return h3NumpySum(values[a..<(a + half)]) + h3NumpySum(values[(a + half)...])
}
