public import AetherCore
public import AetherHost
public import AetherRuntime
public import CoreGraphics

/// The types every detector driver shares (`detector-dense@1`, `detector-detr@1`): what a detection is, what a
/// request may override, and the class tables (MANIFEST.md §2.8, `labels`).

/// One detected object.
public struct Detection: Sendable, Hashable {
    /// A class: its id and name in the model's label set.
    public struct Label: Sendable, Hashable {
        /// For `coco-91`, the COCO category id (1...90, ten of them unused); for a label file, the name's index in it.
        public let id: Int
        public let name: String

        public init(id: Int, name: String) {
            self.id = id
            self.name = name
        }
    }

    public var label: Label
    /// The model's confidence, on [0, 1].
    public var score: Float
    /// The box in the decoded, upright source image, normalized to [0, 1] on each axis from the top-left, clipped to
    /// the image. Multiply by the source's width and height for pixels.
    public var box: CGRect
    /// The instance mask, from a model that makes one (`detector-detr@1` with `masks`); nil otherwise.
    public var mask: DetectionMask?

    public init(label: Label, score: Float, box: CGRect, mask: DetectionMask? = nil) {
        self.label = label
        self.score = score
        self.box = box
        self.mask = mask
    }
}

/// An instance mask at the source image's resolution, in the same space as ``Detection/box``, kept compact: the
/// smallest rectangle holding its foreground, one bit per pixel.
public struct DetectionMask: Sendable, Hashable {
    /// The decoded, upright source image's size: the mask's full extent.
    public let width: Int
    public let height: Int
    /// The foreground's bounding rectangle in source pixels; zero-sized when the mask is empty.
    public let bounds: Bounds
    /// `bounds.width · bounds.height` bits, row-major inside ``bounds``: pixel `i` is bit `i & 7` (least significant
    /// first) of byte `i >> 3`.
    public let bits: [UInt8]

    public struct Bounds: Sendable, Hashable {
        public let x: Int, y: Int, width: Int, height: Int

        public init(x: Int, y: Int, width: Int, height: Int) {
            self.x = x
            self.y = y
            self.width = width
            self.height = height
        }
    }

    /// A mask from `width · height` foreground flags, row-major from the top-left.
    public init(width: Int, height: Int, foreground: [Bool]) {
        precondition(foreground.count == width * height, "a mask holds width · height flags")
        var x0 = width, x1 = -1, y0 = height, y1 = -1
        for y in 0..<height {
            for x in 0..<width where foreground[y * width + x] {
                x0 = min(x0, x)
                x1 = max(x1, x)
                y0 = min(y0, y)
                y1 = max(y1, y)
            }
        }
        guard x1 >= 0 else {
            self.init(width: width, height: height, bounds: Bounds(x: 0, y: 0, width: 0, height: 0), bits: [])
            return
        }
        let bounds = Bounds(x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1)
        var bits = [UInt8](repeating: 0, count: (bounds.width * bounds.height + 7) / 8)
        for y in y0...y1 {
            for x in x0...x1 where foreground[y * width + x] {
                let i = (y - y0) * bounds.width + (x - x0)
                bits[i >> 3] |= 1 << UInt8(i & 7)
            }
        }
        self.init(width: width, height: height, bounds: bounds, bits: bits)
    }

    init(width: Int, height: Int, bounds: Bounds, bits: [UInt8]) {
        precondition(bits.count == (bounds.width * bounds.height + 7) / 8)
        self.width = width
        self.height = height
        self.bounds = bounds
        self.bits = bits
    }

    /// Whether source pixel (x, y) is foreground; false outside the image.
    public subscript(x x: Int, y y: Int) -> Bool {
        let column = x - bounds.x, row = y - bounds.y
        guard column >= 0, column < bounds.width, row >= 0, row < bounds.height else { return false }
        let i = row * bounds.width + column
        return bits[i >> 3] & (1 << UInt8(i & 7)) != 0
    }

    public var isEmpty: Bool { bounds.width == 0 }

    /// The number of foreground pixels.
    public var area: Int { bits.reduce(0) { $0 + $1.nonzeroBitCount } }

    /// `width · height` flags, row-major from the top-left.
    public var foreground: [Bool] {
        var out = [Bool](repeating: false, count: width * height)
        for row in 0..<bounds.height {
            for column in 0..<bounds.width where self[x: bounds.x + column, y: bounds.y + row] {
                out[(bounds.y + row) * width + bounds.x + column] = true
            }
        }
        return out
    }

    /// The whole mask as 8-bit values, row-major, 255 for foreground and 0 elsewhere (an alpha or
    /// `CGImage(maskWidth:…)` plane).
    public func alpha() -> [UInt8] { foreground.map { $0 ? 255 : 0 } }

    /// Intersection over union with a mask of the same image; 1 when both are empty.
    public func intersectionOverUnion(_ other: DetectionMask) -> Double {
        precondition(width == other.width && height == other.height)
        let a = bounds, b = other.bounds
        let x0 = max(a.x, b.x), x1 = min(a.x + a.width, b.x + b.width)
        let y0 = max(a.y, b.y), y1 = min(a.y + a.height, b.y + b.height)
        var intersection = 0
        if x0 < x1 && y0 < y1 {
            for y in y0..<y1 { for x in x0..<x1 where self[x: x, y: y] && other[x: x, y: y] { intersection += 1 } }
        }
        let union = area + other.area - intersection
        return union == 0 ? 1 : Double(intersection) / Double(union)
    }
}

/// One detection request's settings. Absent values are the manifest's `postprocess.defaults`.
public struct DetectionRequest: Sendable, Hashable {
    /// Detections scoring below this are dropped (on [0, 1]).
    public var scoreThreshold: Double?
    /// At most this many detections, the highest scoring first (at least 1).
    public var maxDetections: Int?
    /// The request's GPU priority (absent: `.medium`). Keep the job to change it while the request waits.
    public var job: GPUJob?

    public init(scoreThreshold: Double? = nil, maxDetections: Int? = nil, job: GPUJob? = nil) {
        self.scoreThreshold = scoreThreshold
        self.maxDetections = maxDetections
        self.job = job
    }

    /// The request over `defaults`, checked.
    func resolved(_ defaults: Manifest.Postprocess.DetectionDefaults) throws(AetherError) -> (scoreThreshold: Double, maxDetections: Int) {
        let threshold = scoreThreshold ?? defaults.scoreThreshold, count = maxDetections ?? defaults.maxDetections
        guard threshold.isFinite, (0...1).contains(threshold) else {
            throw .invalidInput("scoreThreshold \(threshold) is not in 0...1")
        }
        guard count > 0 else { throw .invalidInput("maxDetections \(count) is not at least 1") }
        return (threshold, count)
    }
}

/// A detector's class table: the labels in the order of the label set's contiguous list, and how a class column of the
/// graph's output names one (`postprocess.classIndex`).
public struct DetectionLabels: Sendable, Hashable {
    /// In contiguous order: column `i` of a `contiguous` model is `labels[i]`.
    public let labels: [Detection.Label]
    private let byID: [Int: Int]

    public init(_ labels: [Detection.Label]) {
        self.labels = labels
        byID = Dictionary(labels.enumerated().map { ($1.id, $0) }, uniquingKeysWith: { first, _ in first })
    }

    /// The label class column `column` names: `labels[column]` for `contiguous`, the label whose id is `column` for
    /// `cocoId`; nil for a column that names none (an unused COCO id, or past the table).
    public func label(column: Int, classIndex: Manifest.Postprocess.ClassIndex) -> Detection.Label? {
        switch classIndex {
        case .contiguous: labels.indices.contains(column) ? labels[column] : nil
        case .cocoId: byID[column].map { labels[$0] }
        }
    }

    /// The manifest's `labels`. Only the built-in sets so far; a label file is refused when the driver loads.
    init(_ labels: Manifest.Labels?) throws(AetherError) {
        switch labels {
        case .set(.coco91)?: self = .coco91
        case .file?: throw .manifestInvalid(field: "labels.file", reason: "label files are not supported yet; name a built-in set")
        case nil: throw .manifestInvalid(field: "labels", reason: "a detector requires labels")
        }
    }

    /// `coco-91`: COCO's 80 detection categories, ids 1...90 without 12, 26, 29, 30, 45, 66, 68, 69, 71 and 83, in
    /// ascending id order (YOLOX's contiguous 0...79), with upstream YOLOX's `COCO_CLASSES` names.
    public static let coco91 = DetectionLabels([
        (1, "person"), (2, "bicycle"), (3, "car"), (4, "motorcycle"), (5, "airplane"), (6, "bus"), (7, "train"),
        (8, "truck"), (9, "boat"), (10, "traffic light"), (11, "fire hydrant"), (13, "stop sign"), (14, "parking meter"),
        (15, "bench"), (16, "bird"), (17, "cat"), (18, "dog"), (19, "horse"), (20, "sheep"), (21, "cow"),
        (22, "elephant"), (23, "bear"), (24, "zebra"), (25, "giraffe"), (27, "backpack"), (28, "umbrella"),
        (31, "handbag"), (32, "tie"), (33, "suitcase"), (34, "frisbee"), (35, "skis"), (36, "snowboard"),
        (37, "sports ball"), (38, "kite"), (39, "baseball bat"), (40, "baseball glove"), (41, "skateboard"),
        (42, "surfboard"), (43, "tennis racket"), (44, "bottle"), (46, "wine glass"), (47, "cup"), (48, "fork"),
        (49, "knife"), (50, "spoon"), (51, "bowl"), (52, "banana"), (53, "apple"), (54, "sandwich"), (55, "orange"),
        (56, "broccoli"), (57, "carrot"), (58, "hot dog"), (59, "pizza"), (60, "donut"), (61, "cake"), (62, "chair"),
        (63, "couch"), (64, "potted plant"), (65, "bed"), (67, "dining table"), (70, "toilet"), (72, "tv"),
        (73, "laptop"), (74, "mouse"), (75, "remote"), (76, "keyboard"), (77, "cell phone"), (78, "microwave"),
        (79, "oven"), (80, "toaster"), (81, "sink"), (82, "refrigerator"), (84, "book"), (85, "clock"), (86, "vase"),
        (87, "scissors"), (88, "teddy bear"), (89, "hair drier"), (90, "toothbrush"),
    ].map { Detection.Label(id: $0.0, name: $0.1) })
}

/// A detector's boxes as the manifest describes them (`boxFormat`, `boxSpace`), mapped to the source image through the
/// prepared image's geometry.
struct DetectionBoxes: Sendable {
    let format: Manifest.Postprocess.BoxFormat
    let space: Manifest.Postprocess.BoxSpace
    let geometry: ImageGeometry

    /// Four box values in the manifest's format and space, as a rectangle in prepared pixels.
    func prepared(_ a: Double, _ b: Double, _ c: Double, _ d: Double) -> CGRect {
        let (sx, sy) = space == .normalized ? (Double(geometry.width), Double(geometry.height)) : (1, 1)
        switch format {
        case .cxcywh: return CGRect(x: (a - c / 2) * sx, y: (b - d / 2) * sy, width: c * sx, height: d * sy)
        case .xyxy: return CGRect(x: a * sx, y: b * sy, width: (c - a) * sx, height: (d - b) * sy)
        }
    }

    /// The same box in the source image, normalized and clipped to [0, 1] (``Detection/box``).
    func source(_ a: Double, _ b: Double, _ c: Double, _ d: Double) -> CGRect {
        let unit = CGRect(x: 0, y: 0, width: 1, height: 1)
        let raw = geometry.normalizedSourceRect(fromPrepared: prepared(a, b, c, d)).standardized
        let box = raw.intersection(unit)
        // Wholly outside the image: an empty box at the nearest edge point.
        return box.isNull ? CGRect(x: min(max(raw.minX, 0), 1), y: min(max(raw.minY, 0), 1), width: 0, height: 0) : box
    }
}
