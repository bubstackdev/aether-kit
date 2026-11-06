public import AetherCore
public import AetherHost
public import AetherRuntime
import CoreGraphics

/// `depth-estimator@1` (MANIFEST.md §1.3, §2.8): monocular depth from one static graph (Depth Anything 3). Loads a
/// ``DepthEstimatorModel``: set its options (a shared coordinator, GPU admission, a ledger), then make the
/// ``DepthEstimator``.
public enum DepthEstimatorDriver: ModelDriver {
    public static let id: DriverID = .depthEstimator

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> DepthEstimatorModel {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        _ = try DepthEstimator.Configuration(manifest)
        return DepthEstimatorModel(
            bundle: bundle, manifest: manifest, device: options.device,
            options: .init(cachePolicy: options.cachePolicy, cache: options.cache))
    }
}

/// An opened depth bundle. Nothing is loaded yet; ``options`` start from the `LoadOptions` cache settings.
public struct DepthEstimatorModel: Sendable {
    public let bundle: InstalledBundle
    public let manifest: Manifest
    public let device: Device
    public var options: PipelineOptions

    public func estimator() throws -> DepthEstimator {
        try DepthEstimator(bundle: bundle, manifest: manifest, device: device, options: options)
    }
}

extension Registry {
    /// The drivers AetherVision links: `depth-estimator@1`, `detector-dense@1`, `detector-detr@1`, `clip-joint@1`.
    public static let vision = Registry([
        DepthEstimatorDriver.self, DenseDetectorDriver.self, DetrDetectorDriver.self, ImageTextEncoderDriver.self,
    ])
}

/// One depth request's settings.
public struct DepthRequest: Sendable, Hashable {
    /// Whether the map is resampled to the source image's size; absent, the manifest's `postprocess.depth.resizeBack`.
    /// `false` skips the resample (a real-time caller drawing at the graph's resolution) and keeps ``DepthMap/geometry``
    /// to place the map.
    public var resizeBack: Bool?
    /// The request's GPU priority (absent: `.medium`). Keep the job to change it while the request waits.
    public var job: GPUJob?

    public init(resizeBack: Bool? = nil, job: GPUJob? = nil) {
        self.resizeBack = resizeBack
        self.job = job
    }
}

/// A depth map: the graph's output, row-major, either at the source image's size or at the graph's.
public struct DepthMap: Sendable, Hashable {
    public typealias Polarity = Manifest.Postprocess.Depth.Polarity

    public let width: Int
    public let height: Int
    /// `width · height` values, row-major from the top-left. They are the model's own output, unnormalized: for Depth
    /// Anything 3 small, relative depth (affine-invariant, not metric). ``polarity`` says which way is near.
    public let values: [Float]
    /// The model's per-pixel confidence, the same size, when its manifest binds one (Depth Anything 3: ≥ 1, larger is
    /// more confident).
    public let confidence: [Float]?
    /// `.largerIsFarther`: a larger value is farther from the camera; `.largerIsNearer`: nearer (disparity).
    public let polarity: Polarity
    /// Where the source image sits in the graph's input. When ``isSourceSized``, pixel (x, y) of the map is pixel (x, y)
    /// of the decoded, upright source; otherwise the map is the graph's resolution and `geometry.content` is the
    /// source's rectangle on it.
    public let geometry: ImageGeometry

    public init(width: Int, height: Int, values: [Float], confidence: [Float]?, polarity: Polarity, geometry: ImageGeometry) {
        self.width = width
        self.height = height
        self.values = values
        self.confidence = confidence
        self.polarity = polarity
        self.geometry = geometry
    }

    /// Whether the map was resampled to the source image's size.
    public var isSourceSized: Bool { width == geometry.sourceWidth && height == geometry.sourceHeight }

    /// The value at column `x`, row `y`.
    public subscript(x x: Int, y y: Int) -> Float { values[y * width + x] }

    /// The smallest and largest finite values, or nil when there are none.
    public var range: ClosedRange<Float>? {
        var low = Float.infinity, high = -Float.infinity
        for v in values where v.isFinite {
            low = min(low, v)
            high = max(high, v)
        }
        return low <= high ? low...high : nil
    }
}

/// One loaded depth model (DESIGN.md §8): ``ImageInput`` → ``ImagePreprocessor`` (CPU) → the `estimator` graph (GPU,
/// on AetherHost's ``StatelessPipeline``) → the `depth` postprocess (CPU) → ``DepthMap``.
///
/// - **Postprocess:** with `resizeBack` (the manifest's, or the request's), the map and confidence are resampled to the
///   source image's size through the prepared image's geometry: the source's rectangle on the graph's input (the whole
///   canvas when stretched, inset when letterboxed, so padding is never sampled) is mapped onto the source's pixels
///   with bilinear interpolation at pixel centres (half-pixel offsets, as `F.interpolate(align_corners=False)`, no
///   antialiasing), clamped to the content's edge pixels. Bilinear keeps the values inside their neighbours' range, so the
///   ordering ``DepthMap/polarity`` describes holds; a source smaller than the graph's input is point-sampled more
///   coarsely, not averaged.
/// - **Memory and the GPU:** the graph stays loaded until ``unload()``. Each request is admitted against the measured
///   footprint (``FootprintLedger``) or the file's size, with the shared `ResourceCoordinator` when the options set
///   one, and through the options' GPU admission (by default ``GPUAdmission/shared``), one grant per request, held only
///   for the graph call.
/// - **One at a time:** requests run in GPU priority order (the request's `job`), in the order made within a priority;
///   ``unload()`` waits for the running one.
/// - **Cancellation** stops after the graph call with `CancellationError`. Any failure releases the graph; the
///   estimator stays usable.
/// - **Real time:** ``prepare(_:)`` is the CPU half and ``estimate(_:_:)-(PreparedImage,_)`` the GPU half, so a caller
///   can prepare the next frame while the last one runs.
public actor DepthEstimator {
    public typealias Options = PipelineOptions
    public typealias Request = DepthRequest

    /// What the driver reads from a manifest, checked before anything loads.
    struct Configuration: Sendable {
        var image: String, depth: String, confidence: String?
        var postprocess: Manifest.Postprocess.Depth
        var preprocessor: ImagePreprocessor

        init(_ manifest: Manifest) throws(AetherError) {
            try DriverContract.depthEstimator.validate(manifest)
            guard case .depth(let depth)? = manifest.postprocess else {
                throw .manifestInvalid(field: "postprocess", reason: "\(DepthEstimatorDriver.id) requires kind depth")
            }
            preprocessor = try ImagePreprocessor(manifest: manifest)
            guard (preprocessor.configuration.layout ?? .NCHW) == .NCHW else {
                throw .manifestInvalid(field: "preprocess.image.layout", reason: "\(DepthEstimatorDriver.id) takes NCHW")
            }
            let graph = manifest.graphs["estimator"]!
            image = graph.inputs["image"]!.name
            self.depth = graph.outputs["depth"]!.name
            confidence = graph.outputs["confidence"]?.name
            postprocess = depth
        }
    }

    public nonisolated let manifest: Manifest
    public nonisolated let options: Options
    /// The manifest's `preprocess.image`, as a transform.
    public nonisolated let preprocessor: ImagePreprocessor
    /// The manifest's `postprocess.depth`.
    public nonisolated let postprocess: Manifest.Postprocess.Depth
    let configuration: Configuration
    private let pipeline: StatelessPipeline

    /// Opens an installed bundle; the graph loads on the first request. `options.memory` absent means
    /// ``MemoryBudget/platformDefault``.
    public init(bundle: InstalledBundle, manifest: Manifest, device: Device, options: Options = Options()) throws(AetherError) {
        configuration = try Configuration(manifest)
        pipeline = try StatelessPipeline(
            bundle: bundle, manifest: manifest, device: device, contract: .depthEstimator, options: options)
        self.manifest = manifest
        self.options = options
        preprocessor = configuration.preprocessor
        postprocess = configuration.postprocess
    }

    /// The estimator over any graph loader (tests drive it with fakes).
    init(
        manifest: Manifest, device: Device, options: Options, load: @escaping GraphResidency.Loader,
        fileBytes: @escaping (String) -> Int64
    ) throws(AetherError) {
        configuration = try Configuration(manifest)
        pipeline = try StatelessPipeline(
            manifest: manifest, device: device, contract: .depthEstimator, options: options, load: load, fileBytes: fileBytes)
        self.manifest = manifest
        self.options = options
        preprocessor = configuration.preprocessor
        postprocess = configuration.postprocess
    }

    /// Roles currently loaded.
    public var residentRoles: [String] { pipeline.residentRoles }

    /// Decodes and prepares `image` for the graph: the CPU half of ``estimate(_:_:)-(ImageInput,_)``.
    public nonisolated func prepare(_ image: ImageInput) throws(AetherError) -> PreparedImage {
        try preprocessor.prepare(image)
    }

    /// Estimates `image`'s depth. Preparing it runs on the caller's executor, before the request queues.
    public nonisolated func estimate(_ image: ImageInput, _ request: Request = Request()) async throws -> DepthMap {
        try await estimate(prepare(image), request)
    }

    /// Estimates the depth of an image this estimator's ``prepare(_:)`` made.
    public func estimate(_ prepared: PreparedImage, _ request: Request = Request()) async throws -> DepthMap {
        let c = configuration
        let geometry = prepared.geometry
        let (depth, confidence) = try await pipeline.perform(job: request.job) { residency in
            let graph = try await residency.graph("estimator")
            let arena = TensorArena()
            var bindings = Bindings(arena: arena)
            try bindings.bindInput(c.image, to: graph.makeBuffer(for: c.image, holding: prepared, in: arena))
            let depth = try Self.output(c.depth, of: graph, geometry: geometry, in: arena)
            try bindings.bindOutput(c.depth, to: depth)
            var confidence: TensorBuffer?
            if let name = c.confidence {
                confidence = try Self.output(name, of: graph, geometry: geometry, in: arena)
                try bindings.bindOutput(name, to: confidence!)
            }
            try await residency.run(graph, &bindings)
            return (try depth.readFloats(), try confidence?.readFloats())
        }
        // After the GPU grant: the resample is host work.
        guard request.resizeBack ?? c.postprocess.resizeBack ?? false else {
            return DepthMap(width: geometry.width, height: geometry.height, values: depth, confidence: confidence,
                            polarity: c.postprocess.polarity, geometry: geometry)
        }
        let resampler = DepthResampler(geometry)
        return DepthMap(
            width: geometry.sourceWidth, height: geometry.sourceHeight, values: resampler.resample(depth),
            confidence: confidence.map(resampler.resample), polarity: c.postprocess.polarity, geometry: geometry)
    }

    /// Releases the graph, waiting for a running request. The next request loads it again.
    public func unload() async { await pipeline.unload() }

    /// An output buffer for `name`, which must be `[1, H, W]` at the prepared image's H and W.
    static func output(
        _ name: String, of graph: any GraphBackend, geometry: ImageGeometry, in arena: TensorArena
    ) throws(AetherError) -> TensorBuffer {
        let expected = [1, geometry.height, geometry.width]
        guard let spec = graph.signature.output(name), spec.shape == expected else {
            throw .manifestMismatch(
                field: "graphs.estimator.outputs", expected: "'\(name)' of shape \(expected) (the prepared image's)",
                found: graph.signature.output(name).map { "\($0.shape)" } ?? "no such output")
        }
        return try graph.makeBuffer(for: name, role: .output, dimensions: [], in: arena)
    }
}

/// The `depth` postprocess's `resizeBack`: a map at the graph's resolution, resampled onto the source image's pixels
/// through the prepared image's geometry (bilinear at pixel centres, clamped to the content's edges; ``DepthEstimator``).
struct DepthResampler: Sendable {
    let width: Int, height: Int
    let sourceWidth: Int, sourceHeight: Int
    /// Per source column and row: the two map indices and the second one's weight.
    let columns: [(Int, Int, Float)], rows: [(Int, Int, Float)]

    init(_ geometry: ImageGeometry) {
        width = geometry.width
        height = geometry.height
        sourceWidth = geometry.sourceWidth
        sourceHeight = geometry.sourceHeight
        columns = Self.taps(count: geometry.sourceWidth, origin: geometry.content.minX, extent: geometry.content.width,
                            limit: geometry.width)
        rows = Self.taps(count: geometry.sourceHeight, origin: geometry.content.minY, extent: geometry.content.height,
                         limit: geometry.height)
    }

    /// Source pixel `i`'s centre, `i + 0.5` of `count`, lands at `origin + (i + 0.5) · extent / count` on the map,
    /// whose pixel `j` is centred at `j + 0.5`. Positions clamp to the map pixels the content covers, so letterbox
    /// padding is never read.
    static func taps(count: Int, origin: Double, extent: Double, limit: Int) -> [(Int, Int, Float)] {
        let first = max(0, origin), last = max(first, min(Double(limit - 1), origin + extent - 1))
        return (0..<count).map { i in
            let at = origin + (Double(i) + 0.5) * extent / Double(count) - 0.5
            let clamped = min(max(at, first), last)
            let low = Int(clamped.rounded(.down))
            let high = min(low + 1, limit - 1)
            return (low, high, Float(clamped - Double(low)))
        }
    }

    /// `map` (`height · width`, row-major) at the source's size.
    func resample(_ map: [Float]) -> [Float] {
        precondition(map.count == width * height)
        var out = [Float](repeating: 0, count: sourceWidth * sourceHeight)
        map.withUnsafeBufferPointer { m in
            out.withUnsafeMutableBufferPointer { o in
                for (y, (r0, r1, fy)) in rows.enumerated() {
                    let top = r0 * width, bottom = r1 * width, line = y * sourceWidth
                    for (x, (c0, c1, fx)) in columns.enumerated() {
                        let upper = m[top + c0] + (m[top + c1] - m[top + c0]) * fx
                        let lower = m[bottom + c0] + (m[bottom + c1] - m[bottom + c0]) * fx
                        o[line + x] = upper + (lower - upper) * fy
                    }
                }
            }
        }
        return out
    }
}
