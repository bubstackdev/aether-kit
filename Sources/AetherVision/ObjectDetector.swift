public import AetherCore
public import AetherHost
public import AetherRuntime
import Foundation

/// `detector-dense@1` (MANIFEST.md §1.3, §2.8): object detection from one static graph with one row per anchor, then
/// host NMS (YOLOX). Loads an ``ObjectDetectorModel``: set its options (a shared coordinator, GPU admission, a
/// ledger), then make the ``ObjectDetector``.
public enum DenseDetectorDriver: ModelDriver {
    public static let id: DriverID = .detectorDense

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> ObjectDetectorModel {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        _ = try ObjectDetector.Configuration(manifest)
        return ObjectDetectorModel(
            bundle: bundle, manifest: manifest, device: options.device,
            options: .init(cachePolicy: options.cachePolicy, cache: options.cache))
    }
}

/// An opened detector bundle. Nothing is loaded yet; ``options`` start from the `LoadOptions` cache settings.
public struct ObjectDetectorModel: Sendable {
    public let bundle: InstalledBundle
    public let manifest: Manifest
    public let device: Device
    public var options: PipelineOptions

    public func detector() throws -> ObjectDetector {
        try ObjectDetector(bundle: bundle, manifest: manifest, device: device, options: options)
    }
}

/// One loaded object detector (DESIGN.md §8): ``ImageInput`` → ``ImagePreprocessor`` (CPU) → the `detector` graph
/// (GPU, on AetherHost's ``StatelessPipeline``) → the manifest's `postprocess` (CPU) → ``Detection``s, highest score
/// first, with boxes normalized in the source image.
///
/// - **`denseNms`** (`detector-dense@1`, YOLOX): per anchor, score = objectness × the largest class score and the class
///   its column (the first on ties), after a sigmoid when `activation` is `sigmoid`; anchors below the threshold are
///   dropped; then greedy NMS in descending score (ties in anchor order), per class or across classes, dropping boxes
///   whose IoU with a kept one exceeds `nms.iou`; then the first `maxDetections` by score. Upstream YOLOX's
///   `postprocess`, in Double from the graph's floats (``DenseNMS``).
/// - **Boxes** are mapped from the graph's input to the source through the prepared image's geometry (the letterbox's
///   content rectangle) and clipped to the image.
/// - **Memory and the GPU:** the graph stays loaded until ``unload()``. Each request is admitted against the measured
///   footprint (``FootprintLedger``) or the file's size, with the shared `ResourceCoordinator` when the options set
///   one, and through the options' GPU admission (by default ``GPUAdmission/shared``), one grant per request, held only
///   for the graph call; the postprocess runs after it.
/// - **One at a time:** requests run in GPU priority order (the request's `job`), in the order made within a priority;
///   ``unload()`` waits for the running one.
/// - **Cancellation** stops after the graph call with `CancellationError`. Any failure releases the graph; the detector
///   stays usable.
/// - **Real time:** ``prepare(_:)`` is the CPU half and ``detect(_:_:)-(PreparedImage,_)`` the GPU half.
public actor ObjectDetector {
    public typealias Options = PipelineOptions
    public typealias Request = DetectionRequest

    /// What the driver reads from a manifest, checked before anything loads.
    struct Configuration: Sendable {
        enum Head: Sendable {
            case dense(Manifest.Postprocess.DenseNMS, preds: String)
        }

        var contract: DriverContract
        var image: String
        var head: Head
        var labels: DetectionLabels
        var preprocessor: ImagePreprocessor

        init(_ manifest: Manifest) throws(AetherError) {
            switch manifest.driver {
            case .detectorDense:
                contract = .detectorDense
                try contract.validate(manifest)
                guard case .denseNms(let dense)? = manifest.postprocess else {
                    throw .manifestInvalid(field: "postprocess", reason: "\(manifest.driver) requires kind denseNms")
                }
                let graph = manifest.graphs["detector"]!
                head = .dense(dense, preds: graph.outputs["preds"]!.name)
            default:
                throw .unsupported(driver: manifest.driver)
            }
            image = manifest.graphs["detector"]!.inputs["image"]!.name
            preprocessor = try ImagePreprocessor(manifest: manifest)
            guard (preprocessor.configuration.layout ?? .NCHW) == .NCHW else {
                throw .manifestInvalid(field: "preprocess.image.layout", reason: "\(manifest.driver) takes NCHW")
            }
            labels = try DetectionLabels(manifest.labels)
        }

        var defaults: Manifest.Postprocess.DetectionDefaults {
            switch head {
            case .dense(let dense, _): dense.defaults
            }
        }
    }

    public nonisolated let manifest: Manifest
    public nonisolated let options: Options
    /// The manifest's `preprocess.image`, as a transform.
    public nonisolated let preprocessor: ImagePreprocessor
    /// The manifest's `labels`.
    public nonisolated let labels: DetectionLabels
    /// The manifest's `postprocess` thresholds, which a request may override.
    public nonisolated let defaults: Manifest.Postprocess.DetectionDefaults
    let configuration: Configuration
    private let pipeline: StatelessPipeline

    /// Opens an installed bundle; the graph loads on the first request. `options.memory` absent means
    /// ``MemoryBudget/platformDefault``.
    public init(bundle: InstalledBundle, manifest: Manifest, device: Device, options: Options = Options()) throws(AetherError) {
        let configuration = try Configuration(manifest)
        try self.init(manifest: manifest, options: options, configuration: configuration, pipeline: StatelessPipeline(
            bundle: bundle, manifest: manifest, device: device, contract: configuration.contract, options: options))
    }

    /// The detector over any graph loader (tests drive it with fakes).
    init(
        manifest: Manifest, device: Device, options: Options, load: @escaping GraphResidency.Loader,
        fileBytes: @escaping (String) -> Int64
    ) throws(AetherError) {
        let configuration = try Configuration(manifest)
        try self.init(manifest: manifest, options: options, configuration: configuration, pipeline: StatelessPipeline(
            manifest: manifest, device: device, contract: configuration.contract, options: options, load: load,
            fileBytes: fileBytes))
    }

    private init(manifest: Manifest, options: Options, configuration: Configuration, pipeline: StatelessPipeline) {
        self.pipeline = pipeline
        self.manifest = manifest
        self.options = options
        self.configuration = configuration
        preprocessor = configuration.preprocessor
        labels = configuration.labels
        defaults = configuration.defaults
    }

    /// Roles currently loaded.
    public var residentRoles: [String] { pipeline.residentRoles }

    /// Decodes and prepares `image` for the graph: the CPU half of ``detect(_:_:)-(ImageInput,_)``.
    public nonisolated func prepare(_ image: ImageInput) throws(AetherError) -> PreparedImage {
        try preprocessor.prepare(image)
    }

    /// Detects the objects in `image`. Preparing it runs on the caller's executor, before the request queues.
    public nonisolated func detect(_ image: ImageInput, _ request: Request = Request()) async throws -> [Detection] {
        _ = try request.resolved(defaults) // a bad request fails before the decode
        return try await detect(prepare(image), request)
    }

    /// Detects the objects in an image this detector's ``prepare(_:)`` made.
    public func detect(_ prepared: PreparedImage, _ request: Request = Request()) async throws -> [Detection] {
        let c = configuration
        let (threshold, limit) = try request.resolved(c.defaults)
        switch c.head {
        case .dense(let dense, let preds):
            let (values, columns) = try await pipeline.perform(job: request.job) { residency in
                let graph = try await residency.graph("detector")
                let columns = try Self.denseColumns(preds, of: graph, labels: c.labels, classIndex: dense.classIndex ?? .contiguous)
                let arena = TensorArena()
                var bindings = Bindings(arena: arena)
                try bindings.bindInput(c.image, to: graph.makeBuffer(for: c.image, holding: prepared, in: arena))
                let out = try graph.makeBuffer(for: preds, role: .output, dimensions: [], in: arena)
                try bindings.bindOutput(preds, to: out)
                try await residency.run(graph, &bindings)
                return (try out.readFloats(), columns)
            }
            // After the GPU grant: NMS and the box mapping are host work.
            let nms = DenseNMS(dense, columns: columns)
            let boxes = DetectionBoxes(format: dense.boxFormat, space: dense.boxSpace, geometry: prepared.geometry)
            let classIndex = dense.classIndex ?? .contiguous
            return nms.select(values, scoreThreshold: threshold, maxDetections: limit).compactMap { kept in
                guard let label = c.labels.label(column: kept.column, classIndex: classIndex) else { return nil }
                return Detection(label: label, score: Float(kept.score),
                                 box: boxes.source(kept.box[0], kept.box[1], kept.box[2], kept.box[3]))
            }
        }
    }

    /// Releases the graph, waiting for a running request. The next request loads it again.
    public func unload() async { await pipeline.unload() }

    /// `preds`' row width, which must be `[1, A, 4 + 1 + C]` with C ≥ 1, and C the label set's size when columns are
    /// contiguous.
    static func denseColumns(
        _ name: String, of graph: any GraphBackend, labels: DetectionLabels, classIndex: Manifest.Postprocess.ClassIndex
    ) throws(AetherError) -> Int {
        let shape = graph.signature.output(name)?.shape
        let classes = classIndex == .contiguous ? labels.labels.count : nil
        guard let shape, shape.count == 3, shape[0] == 1, shape[1] > 0, shape[2] > 5,
              classes.map({ shape[2] == 5 + $0 }) ?? true else {
            let c = classes.map(String.init) ?? "C ≥ 1"
            throw .manifestMismatch(
                field: "graphs.detector.outputs", expected: "'\(name)' of shape [1, A, 5 + \(c)]",
                found: shape.map { "\($0)" } ?? "no such output")
        }
        return shape[2]
    }
}

/// The `denseNms` postprocess on one image's `preds`, row-major `[A, columns]` with each row `[box(4), objectness,
/// classes…]` (``ObjectDetector``). Upstream YOLOX's `postprocess`, as `Forge/yolox/yolox_graphs.py`'s `dense_nms`
/// restates it: in Double from the floats; score = objectness × max class and the class its first column; keep score ≥
/// threshold; per class (or across classes), greedy NMS in descending score, ties in anchor order, suppressing IoU >
/// `iou` on xyxy boxes (no +1, negative extents as zero area); all kept merged by descending score, ties in anchor
/// order, the first `maxDetections`. Non-finite scores are never kept.
struct DenseNMS: Sendable {
    struct Kept: Sendable, Hashable {
        var anchor: Int
        /// The class column, 0-based after the objectness.
        var column: Int
        var score: Double
        /// The four box values as the graph wrote them.
        var box: SIMD4<Double>
    }

    let columns: Int
    let format: Manifest.Postprocess.BoxFormat
    let sigmoid: Bool
    let perClass: Bool
    let iou: Double

    init(_ postprocess: Manifest.Postprocess.DenseNMS, columns: Int) {
        self.columns = columns
        format = postprocess.boxFormat
        sigmoid = postprocess.activation == .sigmoid
        perClass = postprocess.nms.type == .perClass
        iou = postprocess.nms.iou
    }

    func select(_ preds: [Float], scoreThreshold: Double, maxDetections: Int) -> [Kept] {
        precondition(preds.count % columns == 0)
        let anchors = preds.count / columns, classes = columns - 5
        var candidates: [Kept] = []
        preds.withUnsafeBufferPointer { p in
            for a in 0..<anchors {
                let row = a * columns
                var best = 0, high = activated(p[row + 5])
                for k in 1..<classes {
                    let v = activated(p[row + 5 + k])
                    if v > high { (best, high) = (k, v) }
                }
                let score = activated(p[row + 4]) * high
                guard score.isFinite, score >= scoreThreshold else { continue }
                candidates.append(Kept(anchor: a, column: best, score: score,
                                       box: SIMD4(Double(p[row]), Double(p[row + 1]), Double(p[row + 2]), Double(p[row + 3]))))
            }
        }
        let byScore: (Kept, Kept) -> Bool = { $0.score != $1.score ? $0.score > $1.score : $0.anchor < $1.anchor }
        let groups = perClass ? Dictionary(grouping: candidates, by: \.column).values.map { $0 } : [candidates]
        var kept: [Kept] = []
        for group in groups {
            let ordered = group.sorted(by: byScore)
            let boxes = ordered.map { xyxy($0.box) }
            var alive = [Bool](repeating: true, count: ordered.count)
            for i in ordered.indices where alive[i] {
                kept.append(ordered[i])
                for j in (i + 1)..<ordered.count where alive[j] && !(Self.iou(boxes[i], boxes[j]) <= iou) {
                    alive[j] = false
                }
            }
        }
        return Array(kept.sorted(by: byScore).prefix(maxDetections))
    }

    private func activated(_ x: Float) -> Double {
        sigmoid ? 1 / (1 + exp(-Double(x))) : Double(x)
    }

    private func xyxy(_ b: SIMD4<Double>) -> SIMD4<Double> {
        switch format {
        case .cxcywh: SIMD4(b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2)
        case .xyxy: b
        }
    }

    /// Intersection over union of two xyxy boxes (torchvision's convention: no +1); 0 when the union is empty.
    static func iou(_ a: SIMD4<Double>, _ b: SIMD4<Double>) -> Double {
        let iw = max(min(a[2], b[2]) - max(a[0], b[0]), 0), ih = max(min(a[3], b[3]) - max(a[1], b[1]), 0)
        let inter = iw * ih
        let areaA = max(a[2] - a[0], 0) * max(a[3] - a[1], 0), areaB = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
        let union = areaA + areaB - inter
        return union > 0 ? inter / union : 0
    }
}
