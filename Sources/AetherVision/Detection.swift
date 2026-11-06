public import AetherCore
public import AetherHost
public import AetherRuntime
public import CoreGraphics

// Object detection (MANIFEST.md §1.3, §2.8): the shared result and request types, the COCO label set and the
// `ObjectDetector` actor. Each detector driver supplies a head, the interpreter of its `postprocess` kind (`detrSigmoid`:
// DetrHead.swift); the detector dispatches on the manifest's kind.

/// One detected object.
public struct Detection: Sendable, Hashable {
    /// A class: its id in the manifest's label set (COCO category ids for `coco-91`) and its name.
    public struct Label: Sendable, Hashable {
        public let id: Int
        public let name: String

        public init(id: Int, name: String) {
            self.id = id
            self.name = name
        }
    }

    public let label: Label
    /// The model's confidence, in [0, 1].
    public let score: Float
    /// The object's box as a fraction of the decoded, upright source image (origin top-left), clamped to [0, 1].
    public let box: CGRect
    /// The instance mask, when the model predicts masks.
    public let mask: DetectionMask?

    public init(label: Label, score: Float, box: CGRect, mask: DetectionMask? = nil) {
        self.label = label
        self.score = score
        self.box = box
        self.mask = mask
    }
}

/// An instance mask at the source image's resolution, kept compact: the smallest rectangle holding its foreground, one
/// bit per pixel.
public struct DetectionMask: Sendable, Hashable {
    /// The decoded, upright source image's size.
    public let sourceWidth: Int
    public let sourceHeight: Int
    /// The foreground's bounding rectangle in source pixels; zero-sized when the mask is empty.
    public let left: Int
    public let top: Int
    public let width: Int
    public let height: Int
    /// `width · height` bits, row-major inside the rectangle: pixel `i` is bit `i & 7` (least significant first) of
    /// byte `i >> 3`.
    public let bits: [UInt8]

    init(sourceWidth: Int, sourceHeight: Int, left: Int, top: Int, width: Int, height: Int, bits: [UInt8]) {
        precondition(bits.count == (width * height + 7) / 8)
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.bits = bits
    }

    /// Whether source pixel (x, y) is foreground; false outside the image.
    public subscript(x x: Int, y y: Int) -> Bool {
        let column = x - left, row = y - top
        guard column >= 0, column < width, row >= 0, row < height else { return false }
        let i = row * width + column
        return bits[i >> 3] & (1 << UInt8(i & 7)) != 0
    }

    public var isEmpty: Bool { width == 0 }

    /// The number of foreground pixels.
    public var area: Int { bits.reduce(0) { $0 + $1.nonzeroBitCount } }

    /// The whole mask at the source's size, row-major, 255 for foreground and 0 elsewhere (an 8-bit alpha or
    /// `CGImage(maskWidth:…)` plane).
    public func alpha() -> [UInt8] {
        var out = [UInt8](repeating: 0, count: sourceWidth * sourceHeight)
        for row in 0..<height {
            for column in 0..<width where self[x: left + column, y: top + row] {
                out[(top + row) * sourceWidth + left + column] = 255
            }
        }
        return out
    }

    /// Intersection over union with another mask of the same source; 1 when both are empty.
    public func intersectionOverUnion(_ other: DetectionMask) -> Double {
        precondition(sourceWidth == other.sourceWidth && sourceHeight == other.sourceHeight)
        let x0 = max(left, other.left), x1 = min(left + width, other.left + other.width)
        let y0 = max(top, other.top), y1 = min(top + height, other.top + other.height)
        var intersection = 0
        if x0 < x1 && y0 < y1 {
            for y in y0..<y1 { for x in x0..<x1 where self[x: x, y: y] && other[x: x, y: y] { intersection += 1 } }
        }
        let union = area + other.area - intersection
        return union == 0 ? 1 : Double(intersection) / Double(union)
    }
}

/// One detection request's settings; each absent value is the manifest's `postprocess.defaults`.
public struct DetectionRequest: Sendable, Hashable {
    /// Detections score strictly above it.
    public var scoreThreshold: Double?
    /// At most this many detections, the best first.
    public var maxDetections: Int?
    /// The request's GPU priority (absent: `.medium`). Keep the job to change it while the request waits.
    public var job: GPUJob?

    public init(scoreThreshold: Double? = nil, maxDetections: Int? = nil, job: GPUJob? = nil) {
        self.scoreThreshold = scoreThreshold
        self.maxDetections = maxDetections
        self.job = job
    }
}

/// The COCO label set (`labels: {set: coco-91}`): the 91 category ids, 80 of them named.
enum COCO91 {
    /// Index = COCO category id; nil for the ids COCO never assigned (0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83).
    static let names: [String?] = [
        nil, "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", nil, "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", nil, "backpack", "umbrella", nil, nil, "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", nil, "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", nil, "dining table", nil, nil, "toilet", nil, "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", nil, "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    /// The labels of a detector's class columns: under `cocoId`, column `i` is category id `i` (nil when unassigned);
    /// under `contiguous`, column `i` is the `i`-th assigned id (80 columns).
    static func columns(_ classIndex: Manifest.Postprocess.ClassIndex) -> [Detection.Label?] {
        let labels = names.enumerated().map { id, name in name.map { Detection.Label(id: id, name: $0) } }
        switch classIndex {
        case .cocoId: return labels
        case .contiguous: return labels.compactMap { $0 }
        }
    }
}

extension ObjectDetector {
    /// The labels of a detector's class columns, from the manifest's `labels` (only the built-in `coco-91` so far).
    static func columnLabels(_ manifest: Manifest, classIndex: Manifest.Postprocess.ClassIndex) throws(AetherError) -> [Detection.Label?] {
        switch manifest.labels {
        case .set(.coco91)?: return COCO91.columns(classIndex)
        case .file?: throw .manifestInvalid(field: "labels.file", reason: "\(manifest.driver) takes only the built-in label sets so far")
        case nil: throw .manifestInvalid(field: "labels", reason: "\(manifest.driver) requires labels")
        }
    }
}

/// The graph's raw output, as a head reads it.
struct DetectorOutput: Sendable {
    var shape: [Int]
    var values: [Float]
}

/// A `postprocess` kind's interpreter (`detrSigmoid`: ``DetrHead``): the outputs it reads, checked against the graph
/// before it runs, and the detections it makes of them on the host.
protocol DetectorHead: Sendable {
    /// The contract the manifest was checked against (by the head's initializer, first).
    var contract: DriverContract { get }
    /// The detector graph's outputs to bind, by graph name.
    var outputs: [String] { get }
    /// Throws when the loaded graph's outputs are not the shapes the head reads.
    func check(_ signature: GraphSignature) throws(AetherError)
    /// The detections, best first.
    func detections(_ outputs: [String: DetectorOutput], geometry: ImageGeometry, request: DetectionRequest) -> [Detection]
}

/// The detector drivers: `detector-detr@1` (``DetrDetectorDriver``). Each loads an ``ObjectDetectorModel``: set its
/// options (a shared coordinator, GPU admission, a ledger), then make the ``ObjectDetector``.
public struct ObjectDetectorModel: Sendable {
    public let bundle: InstalledBundle
    public let manifest: Manifest
    public let device: Device
    public var options: PipelineOptions

    public func detector() throws -> ObjectDetector {
        try ObjectDetector(bundle: bundle, manifest: manifest, device: device, options: options)
    }
}

/// One loaded detector (DESIGN.md §8): ``ImageInput`` → ``ImagePreprocessor`` (CPU) → the `detector` graph (GPU, on
/// AetherHost's ``StatelessPipeline``) → the manifest's `postprocess` (CPU) → ``Detection``s, best first, boxes (and
/// masks) in the source image's space.
///
/// - **Postprocess** by kind: `detrSigmoid` (``DetrHead``: sigmoid scores, top-k over (query, class) pairs, no NMS,
///   optional instance masks). The request's threshold and limit override the manifest's defaults.
/// - **Memory and the GPU:** the graph stays loaded until ``unload()``. Each request is admitted against the measured
///   footprint (``FootprintLedger``) or the file's size, with the shared `ResourceCoordinator` when the options set
///   one, and through the options' GPU admission (by default ``GPUAdmission/shared``), one grant per request, held only
///   for the graph call; the postprocess runs after it.
/// - **One at a time:** requests run in GPU priority order (the request's `job`), in the order made within a priority;
///   ``unload()`` waits for the running one.
/// - **Cancellation** stops after the graph call with `CancellationError`. Any failure releases the graph; the
///   detector stays usable.
/// - **Real time:** ``prepare(_:)`` is the CPU half and ``detect(_:_:)-(PreparedImage,_)`` the GPU half.
public actor ObjectDetector {
    public typealias Options = PipelineOptions
    public typealias Request = DetectionRequest

    /// What the detector reads from a manifest, checked before anything loads.
    struct Configuration: Sendable {
        var image: String
        var head: any DetectorHead
        var postprocess: Manifest.Postprocess
        var preprocessor: ImagePreprocessor

        init(_ manifest: Manifest) throws(AetherError) {
            switch manifest.postprocess {
            case .detrSigmoid(let p)?: head = try DetrHead(manifest, p)
            default:
                throw .manifestInvalid(field: "postprocess", reason: "\(manifest.driver) requires kind detrSigmoid")
            }
            postprocess = manifest.postprocess!
            preprocessor = try ImagePreprocessor(manifest: manifest)
            guard (preprocessor.configuration.layout ?? .NCHW) == .NCHW else {
                throw .manifestInvalid(field: "preprocess.image.layout", reason: "\(manifest.driver) takes NCHW")
            }
            image = manifest.graphs["detector"]!.inputs["image"]!.name
        }
    }

    public nonisolated let manifest: Manifest
    public nonisolated let options: Options
    /// The manifest's `preprocess.image`, as a transform.
    public nonisolated let preprocessor: ImagePreprocessor
    /// The manifest's `postprocess`.
    public nonisolated let postprocess: Manifest.Postprocess
    let configuration: Configuration
    private let pipeline: StatelessPipeline

    /// Opens an installed bundle; the graph loads on the first request. `options.memory` absent means
    /// ``MemoryBudget/platformDefault``.
    public init(bundle: InstalledBundle, manifest: Manifest, device: Device, options: Options = Options()) throws(AetherError) {
        configuration = try Configuration(manifest)
        pipeline = try StatelessPipeline(
            bundle: bundle, manifest: manifest, device: device, contract: configuration.head.contract, options: options)
        self.manifest = manifest
        self.options = options
        preprocessor = configuration.preprocessor
        postprocess = configuration.postprocess
    }

    /// The detector over any graph loader (tests drive it with fakes).
    init(
        manifest: Manifest, device: Device, options: Options, load: @escaping GraphResidency.Loader,
        fileBytes: @escaping (String) -> Int64
    ) throws(AetherError) {
        configuration = try Configuration(manifest)
        pipeline = try StatelessPipeline(
            manifest: manifest, device: device, contract: configuration.head.contract, options: options, load: load,
            fileBytes: fileBytes)
        self.manifest = manifest
        self.options = options
        preprocessor = configuration.preprocessor
        postprocess = configuration.postprocess
    }

    /// Roles currently loaded.
    public var residentRoles: [String] { pipeline.residentRoles }

    /// Decodes and prepares `image` for the graph: the CPU half of ``detect(_:_:)-(ImageInput,_)``.
    public nonisolated func prepare(_ image: ImageInput) throws(AetherError) -> PreparedImage {
        try preprocessor.prepare(image)
    }

    /// Detects the objects in `image`. Preparing it runs on the caller's executor, before the request queues.
    public nonisolated func detect(_ image: ImageInput, _ request: Request = Request()) async throws -> [Detection] {
        try await detect(prepare(image), request)
    }

    /// Detects the objects in an image this detector's ``prepare(_:)`` made.
    public func detect(_ prepared: PreparedImage, _ request: Request = Request()) async throws -> [Detection] {
        let c = configuration
        let outputs = try await pipeline.perform(job: request.job) { residency in
            let graph = try await residency.graph("detector")
            try c.head.check(graph.signature)
            let arena = TensorArena()
            var bindings = Bindings(arena: arena)
            try bindings.bindInput(c.image, to: graph.makeBuffer(for: c.image, holding: prepared, in: arena))
            var buffers: [(String, [Int], TensorBuffer)] = []
            for name in c.head.outputs {
                let buffer = try graph.makeBuffer(for: name, role: .output, dimensions: [], in: arena)
                try bindings.bindOutput(name, to: buffer)
                buffers.append((name, graph.signature.output(name)!.shape, buffer))
            }
            try await residency.run(graph, &bindings)
            var outputs: [String: DetectorOutput] = [:]
            for (name, shape, buffer) in buffers { outputs[name] = DetectorOutput(shape: shape, values: try buffer.readFloats()) }
            return outputs
        }
        // After the GPU grant: selection, boxes and masks are host work.
        return c.head.detections(outputs, geometry: prepared.geometry, request: request)
    }

    /// Releases the graph, waiting for a running request. The next request loads it again.
    public func unload() async { await pipeline.unload() }
}
