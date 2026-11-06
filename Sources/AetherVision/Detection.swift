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

/// An instance mask: `width · height` foreground flags, row-major from the top-left, covering the whole source image
/// (the same normalized space as ``Detection/box``).
public struct DetectionMask: Sendable, Hashable {
    public let width: Int
    public let height: Int
    public let foreground: [Bool]

    public init(width: Int, height: Int, foreground: [Bool]) {
        precondition(foreground.count == width * height, "a mask holds width · height flags")
        self.width = width
        self.height = height
        self.foreground = foreground
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
