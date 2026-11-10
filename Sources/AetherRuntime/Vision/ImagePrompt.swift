public import AetherCore
import CryptoKit
import Foundation

// The host half of `vlm-static-inject@1` (MANIFEST.md §1.2, §2.6; docs/VLM.md): what an image becomes before the GPU
// (``PreparedPromptImage``), how a rendered prompt places its images (``ImagePromptPlan``), and the engine that binds
// them (``ImageConditionedEngine``). Pure and CPU-only, so the planning is tested without a model.

/// What a vision-language engine takes: its image tokens, its slots and its preprocessing (`image` and
/// `preprocess.image`, checked against the graphs when the engine loads).
public struct ImageInputSpec: Sendable {
    /// The template's image placeholder and its markers.
    public var padID: Int32
    public var openID: Int32?
    public var closeID: Int32?
    /// Image token `slot` is fed as `idBase + slot`.
    public var idBase: Int32
    /// The decoder's static image rows: every image token of a context, packed in prompt order.
    public var slots: Int
    /// The most images a context holds (the decoder's rope-shift entries).
    public var maxImages: Int
    /// The vision tower's patch rows: one image's patches, zero-padded to this.
    public var patchRows: Int
    /// The patch grid's merge: an image of `h × w` patches is `(h / merge) · (w / merge)` tokens.
    public var merge: Int
    public let preprocessor: ImagePreprocessor

    public init(
        padID: Int32, openID: Int32?, closeID: Int32?, idBase: Int32, slots: Int, maxImages: Int, patchRows: Int,
        merge: Int, preprocessor: ImagePreprocessor
    ) {
        self.padID = padID
        self.openID = openID
        self.closeID = closeID
        self.idBase = idBase
        self.slots = slots
        self.maxImages = maxImages
        self.patchRows = patchRows
        self.merge = merge
        self.preprocessor = preprocessor
    }

    /// The most tokens one image takes (the tower's rows over merge²).
    public var tokensPerImage: Int { patchRows / (merge * merge) }

    /// The CPU half of an image: decode, resize, normalize and patchify (``ImagePreprocessor``), then its grid and
    /// content digest. No GPU, no engine state: call it before the GPU grant, on any thread.
    package func prepare(_ input: ImageInput) throws(AetherError) -> PreparedPromptImage {
        let prepared = try preprocessor.prepare(input)
        guard prepared.shape.count == 2, let patch = preprocessor.configuration.patchify?.patch else {
            throw .manifestInvalid(field: "preprocess.image.layout", reason: "a vision-language model takes patch rows")
        }
        let gh = prepared.geometry.height / patch, gw = prepared.geometry.width / patch
        guard gh % merge == 0, gw % merge == 0, gh * gw == prepared.shape[0] else {
            throw .invalidInput("a \(prepared.geometry.width)x\(prepared.geometry.height) image doesn't merge into \(merge)x\(merge) patch blocks")
        }
        guard gh * gw <= patchRows else {
            throw .invalidInput("a \(gh)x\(gw)-patch image exceeds the vision tower's \(patchRows) rows")
        }
        return PreparedPromptImage(values: prepared.floats(), rowLength: prepared.shape[1], gridHeight: gh, gridWidth: gw,
                                   merge: merge, geometry: prepared.geometry)
    }
}

/// One image as the vision tower takes it: patch rows `[gridHeight · gridWidth, rowLength]` (unpadded), its patch grid,
/// and a digest of the rows and grid that names the image's content (two images with the same digest encode the same).
/// The kit's text stack prepares it (an engine's ``ImageConditionedEngine/bindImages(_:prepared:)`` receives it); its
/// rows are package-only.
public struct PreparedPromptImage: Sendable, Hashable {
    package let values: [Float]
    package let rowLength: Int
    /// The patch grid (rows, columns).
    public let gridHeight: Int
    public let gridWidth: Int
    package let merge: Int
    package let geometry: ImageGeometry
    /// SHA-256 (hex) of the grid and the rows' bytes.
    public let digest: String

    package init(values: [Float], rowLength: Int, gridHeight: Int, gridWidth: Int, merge: Int, geometry: ImageGeometry) {
        self.values = values
        self.rowLength = rowLength
        self.gridHeight = gridHeight
        self.gridWidth = gridWidth
        self.merge = merge
        self.geometry = geometry
        var hash = SHA256()
        withUnsafeBytes(of: (Int64(gridHeight), Int64(gridWidth), Int64(rowLength))) { hash.update(bufferPointer: $0) }
        values.withUnsafeBytes { hash.update(bufferPointer: $0) }
        digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// The image's tokens: one per merge × merge block.
    public var tokens: Int { (gridHeight / merge) * (gridWidth / merge) }

    public static func == (a: Self, b: Self) -> Bool { a.digest == b.digest }
    public func hash(into hasher: inout Hasher) { hasher.combine(digest) }
}

/// A rendered prompt with its images placed (Qwen3-VL's `qwen3vl-mrope` rule, MANIFEST.md §2.6): each image's single
/// placeholder expanded to its tokens, fed as extension ids over slots packed in prompt order, and the decoder's
/// static rotary inputs.
public struct ImagePromptPlan: Sendable, Hashable {
    /// Where one image sits: what the decoder's statics hold for it.
    public struct Image: Sendable, Hashable {
        public var digest: String
        /// The position of its first token.
        public var start: Int
        /// Its first slot.
        public var base: Int
        public var count: Int
        /// The merged grid (tokens are in raster order over it).
        public var gridHeight: Int
        public var gridWidth: Int

        /// Its rope shift: text after it resumes `count − max(gridHeight, gridWidth)` positions earlier.
        package var shiftAmount: Int { count - max(gridHeight, gridWidth) }
        public var end: Int { start + count }
    }

    /// The prompt as fed: each image's tokens are `idBase + slot`.
    public let tokens: [Int32]
    public let images: [Image]
    public let idBase: Int32
    public let padID: Int32

    init(tokens: [Int32], images: [Image], idBase: Int32, padID: Int32) {
        self.tokens = tokens
        self.images = images
        self.idBase = idBase
        self.padID = padID
    }

    /// The text-only rope-shift start (no entry applies to any position).
    package static let unusedShiftStart: Int32 = 1 << 30

    /// Plans `rendered` (one placeholder per image, as the chat template renders it) with `images` in order. Refuses
    /// (`invalidParameters`) a placeholder outside its markers, a count of placeholders other than the images', more
    /// images or tokens than the decoder holds, and a rendered id already in the extension range.
    package init(rendered: [Int32], images prepared: [PreparedPromptImage], spec: ImageInputSpec) throws(AetherError) {
        guard prepared.count <= spec.maxImages else {
            throw .generation(.invalidParameters("\(prepared.count) images; this model takes at most \(spec.maxImages) per context"))
        }
        var tokens: [Int32] = []
        tokens.reserveCapacity(rendered.count + prepared.reduce(0) { $0 + $1.tokens })
        var images: [Image] = []
        var base = 0
        for (index, id) in rendered.enumerated() {
            guard id < spec.idBase else {
                throw .generation(.invalidParameters("rendered token \(id) at \(index) is an image token id"))
            }
            guard id == spec.padID else {
                tokens.append(id)
                continue
            }
            if let open = spec.openID, index == 0 || rendered[index - 1] != open {
                throw .generation(.invalidParameters("the image placeholder at \(index) doesn't follow its open marker"))
            }
            if let close = spec.closeID, index + 1 >= rendered.count || rendered[index + 1] != close {
                throw .generation(.invalidParameters("the image placeholder at \(index) isn't followed by its close marker"))
            }
            guard images.count < prepared.count else {
                throw .generation(.invalidParameters("more image placeholders than images (\(prepared.count))"))
            }
            let image = prepared[images.count]
            let count = image.tokens
            guard count <= spec.tokensPerImage else {
                throw .generation(.invalidParameters("an image of \(count) tokens; the vision tower takes \(spec.tokensPerImage)"))
            }
            guard base + count <= spec.slots else {
                throw .generation(.invalidParameters("\(base + count) image tokens; this model holds \(spec.slots) per context"))
            }
            images.append(Image(digest: image.digest, start: tokens.count, base: base, count: count,
                                gridHeight: image.gridHeight / image.merge, gridWidth: image.gridWidth / image.merge))
            for i in 0..<count { tokens.append(spec.idBase + Int32(base + i)) }
            base += count
        }
        guard images.count == prepared.count else {
            throw .generation(.invalidParameters("\(prepared.count) images for \(images.count) image placeholders"))
        }
        self.tokens = tokens
        self.images = images
        idBase = spec.idBase
        padID = spec.padID
    }

    /// The prompt with every image token as the placeholder id (what a sampler's history and a constraint see, as
    /// the reference processor's `input_ids` have them).
    package var placeholderTokens: [Int32] { tokens.map { $0 >= idBase ? padID : $0 } }

    /// The decoder's `imagePositions` `[slots, 3]` row-major: each slot's (t, h, w) rotary positions; unused slots zero.
    package func imagePositions(slots: Int) -> [Int32] {
        var out = [Int32](repeating: 0, count: slots * 3)
        var shift = 0
        for image in images {
            let s0 = image.start - shift
            for i in 0..<image.count {
                let row = (image.base + i) * 3
                out[row] = Int32(s0)
                out[row + 1] = Int32(s0 + i / image.gridWidth)
                out[row + 2] = Int32(s0 + i % image.gridWidth)
            }
            shift += image.shiftAmount
        }
        return out
    }

    /// The decoder's `ropeShiftStart` and `ropeShiftAmount` `[entries]`: image k's end and amount; unused entries
    /// ``unusedShiftStart`` and 0.
    package func ropeShifts(entries: Int) -> (start: [Int32], amount: [Int32]) {
        var start = [Int32](repeating: Self.unusedShiftStart, count: entries)
        var amount = [Int32](repeating: 0, count: entries)
        for (k, image) in images.enumerated() where k < entries {
            start[k] = Int32(image.end)
            amount[k] = Int32(image.shiftAmount)
        }
        return (start, amount)
    }

    /// How much of a state holding `bound` (and some prompt) this plan can reuse: the start of the first image whose
    /// content, slots or position differ, `Int.max` when none does. Two images with the same grid render identical
    /// token ids, so token equality alone would reuse another image's state.
    package func reusableLimit(bound: [Image]) -> Int {
        for k in 0..<max(bound.count, images.count) {
            let old = k < bound.count ? bound[k] : nil, new = k < images.count ? images[k] : nil
            if old == new { continue }
            return min(old?.start ?? .max, new?.start ?? .max)
        }
        return .max
    }
}

/// A ``LanguageEngine`` that also takes images (a `vlm-static-inject@1` decoder): the Generator binds a prompt's
/// images before it prefills. `LanguageEngine` itself stays token-level (BACKENDS.md §1.2).
public protocol ImageConditionedEngine: LanguageEngine {
    /// The image contract; nil for a model that takes no images.
    nonisolated var imageInput: ImageInputSpec? { get }
    /// The images the static inputs hold, in prompt order.
    var boundImages: [ImagePromptPlan.Image] { get }
    /// Binds `plan`'s images before its prompt is fed: encodes each image the statics don't already hold at its slots
    /// (`prepared[k]` is plan image k), then writes the rotary inputs. The kit's Generator calls it after rewinding the
    /// state to where the bound images still match the plan's. The bind is the engine's operation until it returns: a feed, decode,
    /// rewind, reset or second bind meanwhile is refused. A failure leaves no image bound; if the state then still
    /// holds tokens fed with images bound before (its position is past the first one's start), feeding, decoding and
    /// snapshots are refused until a rewind to at most that start, a reset or a restore.
    func bindImages(_ plan: ImagePromptPlan, prepared: [PreparedPromptImage]) async throws(AetherError)
}
