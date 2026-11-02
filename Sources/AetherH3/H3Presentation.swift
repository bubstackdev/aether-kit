import AetherKit
import Foundation

/// Upstream's FL2VA conditioner presentation (diffusers `MiniMaxH3FL2VATextEncoderStep`): each keyframe, in packed order
/// (first, then last), contributes a `"<Picture i>: "` label tagged as text and a vision block, `<|vision_start|>`, one
/// `<|image_pad|>` per merged image token and `<|vision_end|>`, tagged as video; the prompt follows, tagged as text.
/// Every piece is tokenized on its own. Pure: no model is touched.
///
/// `positions` are Qwen3-VL's M-RoPE positions (`get_rope_index`): a run of non-pad tokens counts p, p+1, ... on all
/// three axes; an image's pads take (p, p + row, p + column) over its merged grid in raster order, and p then advances
/// by the larger merged side, not by the token count.
public struct H3Presentation: Sendable, Equatable {
    /// The constants a profile's `conditioner_vision` entry declares (the defaults are the pinned checkpoint's).
    public struct Vision: Sendable, Equatable {
        public var visionStart: Int32 = 151_652
        public var imagePad: Int32 = 151_655
        public var visionEnd: Int32 = 151_653
        public var textTag: Int32 = 1
        public var videoTag: Int32 = 0
        public var patchSize = 16
        public var mergeSize = 2
        public var temporalPatchSize = 2
        /// Side of the tower's learned square position table.
        public var tableSide = 48
        /// Rotary frequencies per axis (time, height, width), interleaved.
        public var mropeSection = [24, 20, 20]
        /// The processor's pixel bounds: a canvas outside them would be resized before patching, which Kit does not do.
        public var minimumPixels = 65_536
        public var maximumPixels = 16_777_216
        public init() {}
        public static func label(_ index: Int) -> String { "<Picture \(index)>: " }
        /// Patch rows and columns of a canvas, which must be a multiple of `patchSize · mergeSize` on both sides.
        public func grid(width: Int, height: Int) throws -> (height: Int, width: Int) {
            let unit = patchSize * mergeSize
            guard width > 0, height > 0, width % unit == 0, height % unit == 0 else {
                throw AetherError.invalidRequest("H3 keyframe \(width)x\(height) is not a multiple of \(unit)")
            }
            // Within the bounds and on the factor grid, upstream's `smart_resize` keeps the canvas as it is.
            guard (minimumPixels...maximumPixels).contains(width * height) else {
                throw AetherError.unsupported("H3 keyframe \(width)x\(height) is outside the processor's pixel bounds")
            }
            return (height / patchSize, width / patchSize)
        }
    }
    /// One keyframe's image-pad rows: `start..<start + tokens`, over a patch grid of `gridHeight` x `gridWidth`.
    public struct Image: Sendable, Equatable {
        public let start: Int
        public let tokens: Int
        public let gridHeight: Int
        public let gridWidth: Int
    }

    public let ids: [Int32]
    /// The per-row modality tags the denoiser's layout takes.
    public let tags: [Int32]
    /// M-RoPE positions: `positions[axis][row]`, axes time, height, width.
    public let positions: [[Int32]]
    public let images: [Image]
    public var count: Int { ids.count }

    /// `prompt`: the prompt's ids (tokenized alone). `keyframes`: each keyframe's canvas size, in packed order.
    /// `tokenize` tokenizes a label without special tokens.
    public init(prompt: [Int32], keyframes: [(width: Int, height: Int)], vision: Vision = .init(),
                tokenize: (String) throws -> [Int32]) throws {
        var ids: [Int32] = [], tags: [Int32] = [], images: [Image] = []
        for (index, size) in keyframes.enumerated() {
            let grid = try vision.grid(width: size.width, height: size.height)
            let label = try tokenize(Vision.label(index + 1))
            let tokens = grid.height * grid.width / (vision.mergeSize * vision.mergeSize)
            ids += label; tags += Array(repeating: vision.textTag, count: label.count)
            ids.append(vision.visionStart)
            images.append(Image(start: ids.count, tokens: tokens, gridHeight: grid.height, gridWidth: grid.width))
            ids += Array(repeating: vision.imagePad, count: tokens)
            ids.append(vision.visionEnd)
            tags += Array(repeating: vision.videoTag, count: tokens + 2)
        }
        ids += prompt; tags += Array(repeating: vision.textTag, count: prompt.count)
        self.ids = ids; self.tags = tags; self.images = images
        self.positions = Self.positions(count: ids.count, images: images, merge: vision.mergeSize)
    }

    static func positions(count: Int, images: [Image], merge: Int) -> [[Int32]] {
        var rows = [[Int32]](repeating: [], count: 3)
        for r in 0..<3 { rows[r].reserveCapacity(count) }
        var p = 0, row = 0
        for image in images + [Image(start: count, tokens: 0, gridHeight: 0, gridWidth: 0)] {
            while row < image.start {
                for r in 0..<3 { rows[r].append(Int32(p)) }
                p += 1; row += 1
            }
            let h = image.gridHeight / max(merge, 1), w = image.gridWidth / max(merge, 1)
            for hi in 0..<h { for wi in 0..<w {
                rows[0].append(Int32(p)); rows[1].append(Int32(p + hi)); rows[2].append(Int32(p + wi))
            } }
            p += max(h, w); row += image.tokens
        }
        return rows
    }

    /// The axis (0 time, 1 height, 2 width) of each of the `half` rotary frequencies: Qwen3-VL's interleaved layout,
    /// frequency j on height when j % 3 == 1 and j < 3 · section[1], on width when j % 3 == 2 and j < 3 · section[2].
    public static func rotaryAxes(section: [Int]) -> [Int] {
        (0..<section.reduce(0, +)).map { j in
            if j % 3 == 1, j < 3 * section[1] { return 1 }
            if j % 3 == 2, j < 3 * section[2] { return 2 }
            return 0
        }
    }

    /// cos or sin `[1, T, 128]` bf16 for these positions, gathered from the conditioner's 1-D table (`[N, 128]` bf16,
    /// row p = position p on every axis). Each column j takes the row of its axis's position: upstream's
    /// recomposition selects per-frequency values from per-axis tables computed by the same arithmetic, so this is
    /// exact, with no angle recomputed.
    public func rotary(_ table: H3TensorData, section: [Int]) throws -> H3TensorData {
        let width = table.spec.shape.last ?? 0, half = width / 2
        let axes = Self.rotaryAxes(section: section)
        guard table.spec.scalarType == .bfloat16, table.spec.shape.count == 2, axes.count == half, width == 2 * half else {
            throw AetherError.invalidModel("H3 rotary table does not match the M-RoPE section")
        }
        let limit = table.spec.shape[0], t = count
        guard positions.allSatisfy({ $0.allSatisfy { Int($0) < limit } }) else {
            throw AetherError.invalidRequest("H3 presentation positions exceed the rotary table")
        }
        var out = Data(count: t * width * 2)
        table.bytes.withUnsafeBytes { source in
            out.withUnsafeMutableBytes { target in
                let src = source.bindMemory(to: UInt16.self), dst = target.bindMemory(to: UInt16.self)
                for row in 0..<t {
                    for j in 0..<width {
                        let position = Int(positions[axes[j % half]][row])
                        dst[row * width + j] = src[position * width + j]
                    }
                }
            }
        }
        return try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, t, width]), bytes: out)
    }
}

/// The vision tower's inputs for one canvas-sized RGB8 keyframe, as upstream's processor and tower derive them. All
/// rows are in merge-window order (merge x merge windows in raster order, the patches inside a window in raster order).
public struct H3VisionInputs: Sendable {
    /// `[n, 3 · T · p²]` bf16: (x − 127.5) / 127.5 in float32 (the processor's fused rescale and normalize), the frame
    /// repeated over the T temporal slices, each row `[C][T][p][p]`, rounded to bf16 as the step feeds the tower.
    public let pixels: H3TensorData
    /// `[n, 4]` i32 and `[n, 4]` f32: the bilinear taps into the learned position table and their weights, computed in
    /// float32 exactly as `get_vision_interpolation_indices_and_weights` (corners aligned).
    public let positionIndex: H3TensorData
    public let positionWeight: H3TensorData
    /// `[n, 2]` i32: each patch's (row, column), driving the tower's 2-D rotary.
    public let patchCoordinates: H3TensorData
    public let gridHeight: Int
    public let gridWidth: Int

    public init(rgb: [UInt8], width: Int, height: Int, vision: H3Presentation.Vision = .init()) throws {
        let (gh, gw) = try vision.grid(width: width, height: height)
        guard rgb.count == width * height * 3 else { throw AetherError.invalidRequest("H3 keyframe must be interleaved RGB8 at its size") }
        let p = vision.patchSize, m = vision.mergeSize, t = vision.temporalPatchSize, side = vision.tableSide
        let rowLength = 3 * t * p * p, n = gh * gw
        // Only 256 input levels: their bf16 values, computed once.
        let levels = (0..<256).map { h3BFloat16Bits((Float($0) - 127.5) / 127.5) }
        var pixels = [UInt16](repeating: 0, count: n * rowLength)
        var index = [Int32](), weight = [Float](), coordinates = [Int32]()
        index.reserveCapacity(n * 4); weight.reserveCapacity(n * 4); coordinates.reserveCapacity(n * 2)
        let ys = Self.taps(gh, side: side), xs = Self.taps(gw, side: side)
        var row = 0
        for windowRow in 0..<(gh / m) { for windowColumn in 0..<(gw / m) { for i in 0..<m { for j in 0..<m {
            let r = windowRow * m + i, c = windowColumn * m + j
            var offset = row * rowLength
            for channel in 0..<3 {
                for _ in 0..<t {
                    for y in 0..<p {
                        let base = ((r * p + y) * width + c * p) * 3 + channel
                        for x in 0..<p { pixels[offset] = levels[Int(rgb[base + x * 3])]; offset += 1 }
                    }
                }
            }
            let (y, x) = (ys[r], xs[c])
            index += [y.0 * side + x.0, y.0 * side + x.1, y.1 * side + x.0, y.1 * side + x.1].map { Int32($0) }
            weight += [y.2 * x.2, y.2 * x.3, y.3 * x.2, y.3 * x.3]
            coordinates += [Int32(r), Int32(c)]
            row += 1
        } } } }
        self.pixels = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [n, rowLength]),
                                       bytes: pixels.withUnsafeBytes { Data($0) })
        positionIndex = try H3TensorData(spec: H3TensorSpec(scalarType: .int32, shape: [n, 4]), bytes: index.withUnsafeBytes { Data($0) })
        positionWeight = try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [n, 4]), bytes: weight.withUnsafeBytes { Data($0) })
        patchCoordinates = try H3TensorData(spec: H3TensorSpec(scalarType: .int32, shape: [n, 2]), bytes: coordinates.withUnsafeBytes { Data($0) })
        gridHeight = gh; gridWidth = gw
    }

    /// Per position along an axis of `size` patches: (lower tap, upper tap, lower weight, upper weight), in float32
    /// with upstream's operation order: src = i · (side − 1) / max(size − 1, 1), weight = max(1 − |src − floor − k|, 0).
    static func taps(_ size: Int, side: Int) -> [(Int, Int, Float, Float)] {
        (0..<size).map { i in
            let src = Float(i) * Float(side - 1) / Float(max(size - 1, 1))
            let floor = src.rounded(.down)
            let lower = Int(floor)
            return (min(max(lower, 0), side - 1), min(max(lower + 1, 0), side - 1),
                    max(1 - abs(src - floor - 0), 0), max(1 - abs(src - floor - 1), 0))
        }
    }
}

/// `value` rounded to bf16 (to nearest, ties to even), as its 16 bits.
@inline(__always)
public func h3BFloat16Bits(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    if value.isNaN { return UInt16(truncatingIfNeeded: bits >> 16) | 0x40 }
    return UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
}
