// MANIFEST.md §2.5 and §2.8: the vision and embedding drivers' typed blocks. `preprocess.image` (what the host does to
// an image before the graph: decode, resize, normalize, lay out), `postprocess` (what the driver does with the graph's
// outputs, one variant per kind) and `labels`. Closed vocabularies: a value outside them is rejected at decode time,
// naming its field. Each driver checks that the kind it gets is its own.

extension Manifest {
    /// `preprocess.image`: how ``ImagePreprocessor`` (AetherRuntime) turns a decoded image into the graph's input.
    /// Steps, in order: decode in ``colorSpace`` (EXIF orientation applied), ``resize`` on 8-bit RGB, then
    /// `(x · scale − mean) / std` per channel in Float, in ``channelOrder``, laid out as ``layout``, stored as
    /// ``hostDType``.
    public struct ImagePreprocess: Sendable, Hashable, Codable {
        /// `[width, height]` in the JSON.
        public struct Size: Sendable, Hashable, Codable {
            public var width: Int
            public var height: Int

            public init(width: Int, height: Int) {
                self.width = width
                self.height = height
            }

            public init(from decoder: any Decoder) throws {
                let c = try decoder.singleValueContainer()
                let sides = try c.decode([Int].self)
                guard sides.count == 2 else {
                    throw DecodingError.dataCorruptedError(in: c, debugDescription: "size is [width, height]")
                }
                width = sides[0]
                height = sides[1]
            }

            public func encode(to encoder: any Encoder) throws {
                var c = encoder.singleValueContainer()
                try c.encode([width, height])
            }
        }

        public struct Resize: Sendable, Hashable, Codable {
            public enum Mode: String, Sendable, Hashable, Codable {
                /// To ``Resize/size`` exactly; the aspect ratio is not kept.
                case stretch
                /// Scaled to fit inside ``Resize/size`` (aspect kept, the scaled side floored), placed at ``Resize/anchor``
                /// on a canvas of ``Resize/padValue``.
                case letterbox
                /// Scaled to cover ``Resize/size`` (aspect kept, the scaled side floored), or with the shorter side to
                /// ``Resize/shortestSide`` when set, then the centre cut out, its offset `round((scaled − size) / 2)` with
                /// ties to even (torchvision's `Resize` + `CenterCrop`).
                case centerCrop
                /// Scaled down so the longer side is ``Resize/maxSide`` (aspect kept, the other side floored, at least 1);
                /// an image already within it is left as is. The tensor's size follows the image.
                case fitLongSide
            }

            /// Where a letterboxed image sits on its canvas.
            public enum Anchor: String, Sendable, Hashable, Codable {
                /// Padding split evenly, the odd pixel on the right or bottom.
                case center
                /// Padding only on the right and bottom (YOLOX).
                case topLeft
            }

            /// The resampling filter. Parity filters reproduce a reference library bit for bit; others are named as a
            /// bundle needs them (MANIFEST.md §2.5).
            public enum Filter: String, Sendable, Hashable, Codable {
                /// Accelerate's `vImageScale` (its own Lanczos kernel): fast, not a parity filter.
                case vImage
                /// Pillow's `Image.resize(…, BICUBIC)`: Keys a = −0.5, the kernel widened when downscaling, 22-bit
                /// fixed-point weights. Bit-exact.
                case pillowBicubic
                /// torch's uint8 `interpolate(…, "bicubic", antialias=True)` (torchvision's `resize` of a uint8 tensor,
                /// as Hugging Face's fast image processors call it): the same kernel with int16 weights at torch's
                /// precision. Bit-exact.
                case torchUint8Bicubic
            }

            public var mode: Mode
            /// The output size; required by `stretch`, `letterbox` and `centerCrop`.
            public var size: Size?
            /// The longer side's limit; required by `fitLongSide`.
            public var maxSide: Int?
            /// `centerCrop` only: the shorter side's length before the crop (torchvision's `Resize(n)`, the longer side
            /// floored), at least both of ``size``'s sides. Absent: the smallest scale that covers ``size``, which for a
            /// square size is the same.
            public var shortestSide: Int?
            /// `letterbox` only; default `center`.
            public var anchor: Anchor?
            /// `letterbox` only: the canvas's 8-bit value, one for every channel or one per channel (R, G, B);
            /// default 0. A bare number in the JSON is the one-value form (`114` is `[114]`).
            public var padValue: [Int]?
            public var filter: Filter

            public init(mode: Mode, size: Size? = nil, maxSide: Int? = nil, shortestSide: Int? = nil, anchor: Anchor? = nil,
                        padValue: [Int]? = nil, filter: Filter) {
                self.mode = mode
                self.size = size
                self.maxSide = maxSide
                self.shortestSide = shortestSide
                self.anchor = anchor
                self.padValue = padValue
                self.filter = filter
            }

            public init(from decoder: any Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                mode = try c.decode(Mode.self, forKey: .mode)
                size = try c.decodeIfPresent(Size.self, forKey: .size)
                maxSide = try c.decodeIfPresent(Int.self, forKey: .maxSide)
                shortestSide = try c.decodeIfPresent(Int.self, forKey: .shortestSide)
                anchor = try c.decodeIfPresent(Anchor.self, forKey: .anchor)
                if let one = try? c.decodeIfPresent(Int.self, forKey: .padValue) {
                    padValue = [one]
                } else {
                    padValue = try c.decodeIfPresent([Int].self, forKey: .padValue)
                }
                filter = try c.decode(Filter.self, forKey: .filter)
            }
        }

        public enum ColorSpace: String, Sendable, Hashable, Codable {
            /// The stored values, without colour management, as Pillow decodes (an RGB image in its own space; any
            /// other, such as grey, converted to sRGB).
            case deviceRGB
            /// Converted to sRGB.
            case sRGB
        }

        public enum ChannelOrder: String, Sendable, Hashable, Codable {
            case RGB
            case BGR
        }

        public enum Layout: String, Sendable, Hashable, Codable {
            /// `[1, 3, H, W]`.
            case NCHW
            /// `[1, V, 3, H, W]`: the image repeated in each of ``ImagePreprocess/views`` views.
            case NVCHW
            /// `[rows, dim]` patch rows (``ImagePreprocess/patchify``).
            case patches
        }

        /// `patches` layout: the image cut into `patch × patch` squares, each frame repeated `temporal` times.
        public struct Patchify: Sendable, Hashable, Codable {
            public enum RowOrder: String, Sendable, Hashable, Codable {
                /// Each `merge × merge` block of patches contiguous, blocks row-major (Qwen-VL).
                case blockMajor
                /// Patches row-major over the patch grid.
                case rowMajor
            }

            /// The order of one row's `3 · temporal · patch²` values.
            public enum WithinPatch: String, Sendable, Hashable, Codable {
                /// Channel, then frame, then pixel row and column (Qwen-VL).
                case channelFramePixel = "C,T,py,px"
                /// Pixel row and column, then channel; `temporal` must be 1 (SigLIP 2 NaFlex).
                case pixelChannel = "py,px,C"
            }

            public var patch: Int
            /// Default 1.
            public var temporal: Int?
            /// Default 1. The image's sides must be multiples of `patch · merge`.
            public var merge: Int?
            public var rowOrder: RowOrder
            public var withinPatch: WithinPatch

            public init(patch: Int, temporal: Int? = nil, merge: Int? = nil, rowOrder: RowOrder, withinPatch: WithinPatch) {
                self.patch = patch
                self.temporal = temporal
                self.merge = merge
                self.rowOrder = rowOrder
                self.withinPatch = withinPatch
            }
        }

        /// Only `exif` (the decoded image is displayed upright); the field is accepted for completeness.
        public enum Orientation: String, Sendable, Hashable, Codable {
            case exif
        }

        /// Absent: the decoded size, unchanged.
        public var resize: Resize?
        /// Default `deviceRGB`.
        public var colorSpace: ColorSpace?
        /// Default `RGB`.
        public var channelOrder: ChannelOrder?
        /// Multiplies each 8-bit value first; default 1 (`1/255` maps to [0, 1]).
        public var scale: Double?
        /// Per channel, in RGB order; default 0.
        public var mean: [Double]?
        /// Per channel, in RGB order; default 1.
        public var std: [Double]?
        /// Default `NCHW`.
        public var layout: Layout?
        /// `NVCHW` only; default 1.
        public var views: Int?
        /// Required by the `patches` layout, refused otherwise.
        public var patchify: Patchify?
        /// What the host prepares, `float32` or `float16` (rounded to nearest even after normalizing in Float). Binding
        /// converts it to the graph input's own type (read from the descriptor), so an fp16 graph may be fed from an
        /// fp32 host. Default `float32`.
        public var hostDType: ScalarType?
        public var orientation: Orientation?

        public init(
            resize: Resize? = nil, colorSpace: ColorSpace? = nil, channelOrder: ChannelOrder? = nil, scale: Double? = nil,
            mean: [Double]? = nil, std: [Double]? = nil, layout: Layout? = nil, views: Int? = nil, patchify: Patchify? = nil,
            hostDType: ScalarType? = nil, orientation: Orientation? = nil
        ) {
            self.resize = resize
            self.colorSpace = colorSpace
            self.channelOrder = channelOrder
            self.scale = scale
            self.mean = mean
            self.std = std
            self.layout = layout
            self.views = views
            self.patchify = patchify
            self.hostDType = hostDType
            self.orientation = orientation
        }

        /// Checks the block on its own; errors name the field under `prefix`.
        public func validate(at prefix: String = "preprocess.image") throws(AetherError) {
            if let resize {
                let p = "\(prefix).resize"
                switch resize.mode {
                case .stretch, .letterbox, .centerCrop:
                    guard let size = resize.size else { throw .manifestInvalid(field: "\(p).size", reason: "required by \(resize.mode.rawValue)") }
                    try require(size.width > 0 && size.height > 0, "\(p).size", "must be positive")
                    try require(resize.maxSide == nil, "\(p).maxSide", "only fitLongSide takes maxSide")
                case .fitLongSide:
                    guard let maxSide = resize.maxSide else { throw .manifestInvalid(field: "\(p).maxSide", reason: "required by fitLongSide") }
                    try require(maxSide > 0, "\(p).maxSide", "must be positive")
                    try require(resize.size == nil, "\(p).size", "fitLongSide takes maxSide, not size")
                }
                if resize.mode != .letterbox {
                    try require(resize.anchor == nil && resize.padValue == nil, "\(p).anchor", "only letterbox takes anchor and padValue")
                }
                if let side = resize.shortestSide {
                    try require(resize.mode == .centerCrop, "\(p).shortestSide", "only centerCrop takes shortestSide")
                    let size = resize.size!
                    try require(side >= max(size.width, size.height), "\(p).shortestSide", "must be at least both sides of size")
                }
                if let pad = resize.padValue {
                    try require((pad.count == 1 || pad.count == 3) && pad.allSatisfy { (0...255).contains($0) }, "\(p).padValue",
                                "must be one or three values in 0...255")
                }
            }
            if let scale { try require(scale.isFinite && scale > 0, "\(prefix).scale", "must be finite and > 0") }
            if let mean { try require(mean.count == 3 && mean.allSatisfy(\.isFinite), "\(prefix).mean", "must be three finite values") }
            if let std { try require(std.count == 3 && std.allSatisfy { $0.isFinite && $0 > 0 }, "\(prefix).std", "must be three finite values > 0") }
            let layout = layout ?? .NCHW
            if let views { try require(layout == .NVCHW && views > 0, "\(prefix).views", "only NVCHW takes views, at least 1") }
            switch (layout, patchify) {
            case (.patches, nil):
                throw .manifestInvalid(field: "\(prefix).patchify", reason: "required by the patches layout")
            case (.patches, let patchify?):
                let p = "\(prefix).patchify"
                try require(patchify.patch > 0, "\(p).patch", "must be positive")
                try require((patchify.temporal ?? 1) > 0, "\(p).temporal", "must be positive")
                try require((patchify.merge ?? 1) > 0, "\(p).merge", "must be positive")
                if patchify.withinPatch == .pixelChannel {
                    try require((patchify.temporal ?? 1) == 1, "\(p).temporal", "py,px,C rows hold one frame")
                }
            case (_, .some):
                throw .manifestInvalid(field: "\(prefix).patchify", reason: "only the patches layout takes patchify")
            default:
                break
            }
            if let hostDType {
                try require(hostDType == .float32 || hostDType == .float16, "\(prefix).hostDType", "must be float32 or float16")
            }
        }
    }

    /// `postprocess` (MANIFEST.md §2.8): what the driver does with the graph's outputs. One variant per `kind`. The
    /// detectors' fields sit beside `kind`; every other kind's sit in its own sub-object (`depth`, `embedding`,
    /// `gliner`, `probs`, `retrieval`, `sr`). Thresholds are defaults an app can override per request.
    public enum Postprocess: Sendable, Hashable, Codable {
        public enum Kind: String, Sendable, Hashable, Codable, CaseIterable {
            case depth, denseNms, detrSigmoid, embedding, gliner, yesNo, maxSim, superRes
        }

        /// `depth-estimator@1`.
        public struct Depth: Sendable, Hashable, Codable {
            public enum Polarity: String, Sendable, Hashable, Codable {
                /// Disparity or inverse depth: a larger value is nearer the camera (Depth Anything V1 and V2, MiDaS).
                case largerIsNearer
                /// Depth or distance, relative or metric: a larger value is farther from the camera (Depth Anything 3).
                case largerIsFarther
            }

            public var polarity: Polarity
            /// Whether the driver resamples the map back to the source image's size; default false (the map stays at
            /// the graph's resolution, with the geometry to place it).
            public var resizeBack: Bool?

            public init(polarity: Polarity, resizeBack: Bool? = nil) {
                self.polarity = polarity
                self.resizeBack = resizeBack
            }
        }

        public enum BoxFormat: String, Sendable, Hashable, Codable {
            case cxcywh, xyxy
        }

        public enum BoxSpace: String, Sendable, Hashable, Codable {
            /// Pixels of the graph's input, letterboxed or not: the driver maps them back through the geometry.
            case letterboxPixels
            /// [0, 1] of the graph's input.
            case normalized
        }

        /// How a class column maps to a label.
        public enum ClassIndex: String, Sendable, Hashable, Codable {
            /// Column `i` is label `i` of the label set's contiguous list (YOLOX's 80 COCO classes).
            case contiguous
            /// Column `i` is COCO category id `i`; unused ids are not labels (RF-DETR, column 0 unused).
            case cocoId
        }

        /// Default thresholds; a request may override them.
        public struct DetectionDefaults: Sendable, Hashable, Codable {
            public var scoreThreshold: Double
            public var maxDetections: Int

            public init(scoreThreshold: Double, maxDetections: Int) {
                self.scoreThreshold = scoreThreshold
                self.maxDetections = maxDetections
            }

            func validate(at prefix: String) throws(AetherError) {
                try require((0...1).contains(scoreThreshold), "\(prefix).scoreThreshold", "must be in 0...1")
                try require(maxDetections > 0, "\(prefix).maxDetections", "must be at least 1")
            }
        }

        public struct NMS: Sendable, Hashable, Codable {
            public enum Kind: String, Sendable, Hashable, Codable {
                /// Boxes suppress only boxes of their own class.
                case perClass
                /// Boxes suppress any overlapping box.
                case classAgnostic
            }

            public var type: Kind
            /// Boxes overlapping a kept box by more than this IoU are dropped.
            public var iou: Double

            public init(type: Kind, iou: Double) {
                self.type = type
                self.iou = iou
            }
        }

        /// `detector-dense@1`: one row per anchor, `[box(4), objectness, classes…]`, then non-maximum suppression.
        public struct DenseNMS: Sendable, Hashable, Codable {
            public enum Activation: String, Sendable, Hashable, Codable {
                /// The graph already applied it (YOLOX's in-graph sigmoid).
                case applied
                /// The host applies a sigmoid to objectness and classes.
                case sigmoid
            }

            public var boxFormat: BoxFormat
            public var boxSpace: BoxSpace
            /// Default `applied`.
            public var activation: Activation?
            public var nms: NMS
            public var defaults: DetectionDefaults
            /// Default `contiguous`.
            public var classIndex: ClassIndex?

            public init(boxFormat: BoxFormat, boxSpace: BoxSpace, activation: Activation? = nil, nms: NMS,
                        defaults: DetectionDefaults, classIndex: ClassIndex? = nil) {
                self.boxFormat = boxFormat
                self.boxSpace = boxSpace
                self.activation = activation
                self.nms = nms
                self.defaults = defaults
                self.classIndex = classIndex
            }
        }

        /// `detector-detr@1`: a set prediction scored by the sigmoid of each query's class logits; no NMS.
        public struct DetrSigmoid: Sendable, Hashable, Codable {
            /// Which (query, class) scores become detections.
            public enum Selection: String, Sendable, Hashable, Codable {
                /// The sigmoid of every query's every class logit is one candidate; the `maxDetections` best, by score
                /// descending then flat index `query · classes + class` ascending, are kept when their score is strictly
                /// above `scoreThreshold`. One query can yield several classes (RF-DETR's, LW-DETR's and Deformable
                /// DETR's `PostProcess`).
                case topKPairs
            }

            /// Instance masks from the optional `masks` output: one logit plane per query over the graph's whole input,
            /// resampled bilinearly to the source image (pixel centres, `align_corners=False`) and thresholded.
            public struct Masks: Sendable, Hashable, Codable {
                /// A mask pixel is foreground strictly above this logit; default 0.
                public var logitThreshold: Double?

                public init(logitThreshold: Double? = nil) { self.logitThreshold = logitThreshold }
            }

            public var boxFormat: BoxFormat
            public var boxSpace: BoxSpace
            public var defaults: DetectionDefaults
            /// Default `contiguous`.
            public var classIndex: ClassIndex?
            /// Default `topKPairs`.
            public var selection: Selection?
            /// Present exactly when the detector graph binds a `masks` output.
            public var masks: Masks?

            public init(boxFormat: BoxFormat, boxSpace: BoxSpace, defaults: DetectionDefaults, classIndex: ClassIndex? = nil,
                        selection: Selection? = nil, masks: Masks? = nil) {
                self.boxFormat = boxFormat
                self.boxSpace = boxSpace
                self.defaults = defaults
                self.classIndex = classIndex
                self.selection = selection
                self.masks = masks
            }
        }

        /// `clip-joint@1`, `text-embedder@1`.
        public struct Embedding: Sendable, Hashable, Codable {
            public enum Normalization: String, Sendable, Hashable, Codable {
                /// The graph's output is already L2-normalized.
                case inGraph
                /// The host L2-normalizes it.
                case host
                /// Not normalized.
                case none
            }

            public enum Pooling: String, Sendable, Hashable, Codable {
                /// The graph's output is one vector per input.
                case inGraph
                /// The host averages the token vectors under the attention mask.
                case mean
                /// The host takes the first token's vector.
                case cls
            }

            public enum Similarity: String, Sendable, Hashable, Codable {
                case cosine, dot
            }

            public var normalized: Normalization
            /// Default `inGraph`.
            public var pooling: Pooling?
            public var similarity: Similarity
            /// The embedding's length, checked against the graphs' outputs when they load.
            public var dimension: Int?
            /// `clip-joint@1`: the zero-shot temperature, the scores being `softmax(logitScale · similarity)` over the
            /// labels (CLIP's `logit_scale.exp()`, 100). Required by `clip-joint@1`.
            public var logitScale: Double?

            public init(normalized: Normalization, pooling: Pooling? = nil, similarity: Similarity, dimension: Int? = nil,
                        logitScale: Double? = nil) {
                self.normalized = normalized
                self.pooling = pooling
                self.similarity = similarity
                self.dimension = dimension
                self.logitScale = logitScale
            }
        }

        /// `gliner-classifier@1`: label logits, a softmax for one label, a sigmoid per label for several.
        public struct Gliner: Sendable, Hashable, Codable {
            /// Multi-label: a label is chosen above this probability; default 0.5.
            public var threshold: Double?

            public init(threshold: Double? = nil) { self.threshold = threshold }
        }

        /// `yes-no-classifier@1`: the graph's `probs` over (no, yes) in some order.
        public struct YesNo: Sendable, Hashable, Codable {
            /// The column of "yes".
            public var positiveIndex: Int
            /// Default 0.5.
            public var threshold: Double?

            public init(positiveIndex: Int, threshold: Double? = nil) {
                self.positiveIndex = positiveIndex
                self.threshold = threshold
            }
        }

        /// `late-interaction-retriever@1`: MaxSim over token vectors.
        public struct MaxSim: Sendable, Hashable, Codable {
            /// The document encoder's image-token rows, `[start, end)`.
            public var imageTokenSpan: [Int]
            /// `[rows, columns]` of those tokens on the page; `rows · columns` equals the span's length.
            public var patchGrid: [Int]

            public init(imageTokenSpan: [Int], patchGrid: [Int]) {
                self.imageTokenSpan = imageTokenSpan
                self.patchGrid = patchGrid
            }
        }

        /// `image-to-image-tiled@1`.
        public struct SuperRes: Sendable, Hashable, Codable {
            public enum ColorMatch: String, Sendable, Hashable, Codable {
                /// The host matches each channel's mean and standard deviation to the input's.
                case host
                case none
            }

            /// The graph's output range, `[low, high]`.
            public var range: [Double]
            /// Default `none`.
            public var colorMatch: ColorMatch?

            public init(range: [Double], colorMatch: ColorMatch? = nil) {
                self.range = range
                self.colorMatch = colorMatch
            }
        }

        case depth(Depth)
        case denseNms(DenseNMS)
        case detrSigmoid(DetrSigmoid)
        case embedding(Embedding)
        case gliner(Gliner)
        case yesNo(YesNo)
        case maxSim(MaxSim)
        case superRes(SuperRes)

        public var kind: Kind {
            switch self {
            case .depth: .depth
            case .denseNms: .denseNms
            case .detrSigmoid: .detrSigmoid
            case .embedding: .embedding
            case .gliner: .gliner
            case .yesNo: .yesNo
            case .maxSim: .maxSim
            case .superRes: .superRes
            }
        }

        enum CodingKeys: String, CodingKey { case kind, depth, embedding, gliner, probs, retrieval, sr }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            switch try c.decode(Kind.self, forKey: .kind) {
            case .depth: self = .depth(try c.decode(Depth.self, forKey: .depth))
            case .denseNms: self = .denseNms(try DenseNMS(from: decoder))
            case .detrSigmoid: self = .detrSigmoid(try DetrSigmoid(from: decoder))
            case .embedding: self = .embedding(try c.decode(Embedding.self, forKey: .embedding))
            case .gliner: self = .gliner(try c.decodeIfPresent(Gliner.self, forKey: .gliner) ?? Gliner())
            case .yesNo: self = .yesNo(try c.decode(YesNo.self, forKey: .probs))
            case .maxSim: self = .maxSim(try c.decode(MaxSim.self, forKey: .retrieval))
            case .superRes: self = .superRes(try c.decode(SuperRes.self, forKey: .sr))
            }
        }

        public func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(kind, forKey: .kind)
            switch self {
            case .depth(let v): try c.encode(v, forKey: .depth)
            case .denseNms(let v): try v.encode(to: encoder)
            case .detrSigmoid(let v): try v.encode(to: encoder)
            case .embedding(let v): try c.encode(v, forKey: .embedding)
            case .gliner(let v): try c.encode(v, forKey: .gliner)
            case .yesNo(let v): try c.encode(v, forKey: .probs)
            case .maxSim(let v): try c.encode(v, forKey: .retrieval)
            case .superRes(let v): try c.encode(v, forKey: .sr)
            }
        }

        func validate() throws(AetherError) {
            switch self {
            case .depth:
                break
            case .denseNms(let v):
                try require(v.nms.iou > 0 && v.nms.iou <= 1, "postprocess.nms.iou", "must be in (0, 1]")
                try v.defaults.validate(at: "postprocess.defaults")
            case .detrSigmoid(let v):
                try v.defaults.validate(at: "postprocess.defaults")
                if let t = v.masks?.logitThreshold { try require(t.isFinite, "postprocess.masks.logitThreshold", "must be finite") }
            case .embedding(let v):
                if let d = v.dimension { try require(d > 0, "postprocess.embedding.dimension", "must be at least 1") }
                if let t = v.logitScale { try require(t.isFinite && t > 0, "postprocess.embedding.logitScale", "must be finite and > 0") }
            case .gliner(let v):
                if let t = v.threshold { try require((0...1).contains(t), "postprocess.gliner.threshold", "must be in 0...1") }
            case .yesNo(let v):
                try require(v.positiveIndex >= 0, "postprocess.probs.positiveIndex", "must be a column index (>= 0)")
                if let t = v.threshold { try require((0...1).contains(t), "postprocess.probs.threshold", "must be in 0...1") }
            case .maxSim(let v):
                try require(v.imageTokenSpan.count == 2 && v.imageTokenSpan[0] >= 0 && v.imageTokenSpan[0] < v.imageTokenSpan[1],
                            "postprocess.retrieval.imageTokenSpan", "must be [start, end) with 0 <= start < end")
                try require(v.patchGrid.count == 2 && v.patchGrid.allSatisfy { $0 > 0 }, "postprocess.retrieval.patchGrid",
                            "must be [rows, columns], both positive")
                try require(v.patchGrid[0] * v.patchGrid[1] == v.imageTokenSpan[1] - v.imageTokenSpan[0], "postprocess.retrieval.patchGrid",
                            "rows · columns must equal the image-token span's length")
            case .superRes(let v):
                try require(v.range.count == 2 && v.range.allSatisfy(\.isFinite) && v.range[0] < v.range[1], "postprocess.sr.range",
                            "must be [low, high] with low < high")
            }
        }
    }

    /// `labels`: a detector's class names, `{set}` (a set built into the kit) or `{file}` (a JSON array of strings in
    /// an asset, which our own exports ship).
    public enum Labels: Sendable, Hashable, Codable {
        public enum BuiltIn: String, Sendable, Hashable, Codable {
            /// The 91 COCO category ids with their 80 names.
            case coco91 = "coco-91"
        }

        case set(BuiltIn)
        case file(AssetPath)

        enum CodingKeys: String, CodingKey { case set, file }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let set = try c.decodeIfPresent(BuiltIn.self, forKey: .set), file = try c.decodeIfPresent(AssetPath.self, forKey: .file)
            switch (set, file) {
            case (let set?, nil): self = .set(set)
            case (nil, let file?): self = .file(file)
            default:
                throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "exactly one of set, file"))
            }
        }

        public func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .set(let set): try c.encode(set, forKey: .set)
            case .file(let file): try c.encode(file, forKey: .file)
            }
        }
    }
}
