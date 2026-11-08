import AetherCore
public import AetherHost
public import AetherRuntime

// What the image drivers share (DESIGN.md §6.4): their options and request, the admission prologue every generation
// runs, and the load-time check of each graph against its driver contract.

/// How an image pipeline loads its graphs and shares memory and the GPU: AetherHost's ``PipelineOptions``, whose absent
/// `memory` means ``MemoryBudget/imageGeneration`` here.
public typealias ImagePipelineOptions = PipelineOptions

extension MemoryBudget {
    /// The image pipelines' default budget: on macOS three quarters of RAM, about Metal's recommended working set on
    /// Apple silicon, since these models' graphs are almost all GPU memory (Krea 2 needs about 34 GiB a generation per
    /// phase, measured: more than ``platformDefault``'s half of a 64 GB Mac, which suits a decoder's own buffers); on iOS
    /// what the process may still allocate, less 512 MB, as ``platformDefault``.
    public static var imageGeneration: MemoryBudget {
        #if os(iOS)
        .platformDefault
        #else
        MemoryBudget(.fractionOfRAM(0.75))
        #endif
    }
}

/// One text-to-image generation.
public struct ImageRequest: Sendable, Hashable {
    public var prompt: String
    public var seed: UInt64
    /// Absent: the manifest's default.
    public var steps: Int?
    /// Absent: the export's default size. One side alone makes a square.
    public var width: Int?
    public var height: Int?
    /// The request's GPU priority (absent: `.medium`). Keep the job to change it while the request waits.
    public var job: GPUJob?

    public init(
        prompt: String, seed: UInt64 = 0, steps: Int? = nil, width: Int? = nil, height: Int? = nil, job: GPUJob? = nil
    ) {
        self.prompt = prompt
        self.seed = seed
        self.steps = steps
        self.width = width
        self.height = height
        self.job = job
    }

    /// The requested size, or `fallback` (the export's default) when neither side is given.
    func size(default fallback: (width: Int, height: Int)?) -> (width: Int, height: Int)? {
        switch (width, height) {
        case (let w?, let h?): (w, h)
        case (let w?, nil): (w, w)
        case (nil, let h?): (h, h)
        case (nil, nil): fallback
        }
    }
}

/// Progress through one image generation, reported in order: the prompt's conditioning, the denoiser graphs ready,
/// each step, the decoded image. The content filter's reviews come before the first and after the last.
///
/// A handler that throws stops the generation with its error between graph calls, as cancellation does, and the
/// pipeline releases its graphs (as for any failure). Use it for the host's own checks at these boundaries (a memory
/// budget, say); keep it fast, since it runs on the generation's path.
public enum ImageProgress: Sendable, Hashable {
    /// The text encoder has run (`rows`: the conditioning rows the denoisers take).
    case conditioningReady(rows: Int)
    /// Every denoiser graph is loaded for this generation's shapes.
    case denoisersReady
    /// Step `index` (0-based) of `total` has updated the latents.
    case step(Int, of: Int, latents: ImageLatents)
    /// The decoder has produced the image; the filter's review of it follows.
    case decoded
}

/// A step's latents, copied for the handler: `rows · columns` packed tokens in raster order, each `channels` values
/// (`values[(row · columns + column) · channels + channel]`). For previews; changing them changes nothing.
public struct ImageLatents: Sendable, Hashable {
    public let rows: Int
    public let columns: Int
    public let channels: Int
    public let values: [Float]

    public init(rows: Int, columns: Int, channels: Int, values: [Float]) {
        self.rows = rows
        self.columns = columns
        self.channels = channels
        self.values = values
    }
}

/// Where one generation's time and memory went. Seconds are wall-clock on the continuous clock.
public struct ImageGenerationMetrics: Sendable, Hashable, Codable {
    /// The whole call, from the request to the reviewed image, including the waits below.
    public var totalSeconds: Double = 0
    /// Waiting for the GPU grant and for this pipeline's earlier requests.
    public var waitSeconds: Double = 0
    /// The content filter's review of the prompt, then of the image.
    public var promptReviewSeconds: Double = 0
    public var imageReviewSeconds: Double = 0
    /// The text encoder's phase, including loading it when it wasn't resident.
    public var conditioningSeconds: Double = 0
    /// Loading the denoiser graphs (zero when they were resident) and preparing the step inputs.
    public var denoiserPreparationSeconds: Double = 0
    /// Every step, then each step's own time. A graph's first call at a new shape is slower than later ones.
    public var samplingSeconds: Double = 0
    public var stepSeconds: [Double] = []
    /// The decoder's phase, including loading it when it wasn't resident.
    public var decodeSeconds: Double = 0
    public var conditioningRows: Int = 0
    public var imageTokens: Int = 0
    /// The graph roles this generation loaded because they weren't resident when it began, sorted.
    public var loadedRoles: [String] = []
    /// The highest Metal allocation the pipeline's probe read during the generation (the whole process's, so it
    /// includes other allocators').
    public var peakAllocatedBytes: Int64 = 0

    public init() {}
}

/// What an image pipeline does with its graphs after a generation.
public enum ImageRetention: Sendable, Hashable {
    /// Keep what the residency plan keeps (everything, when it fits the budget): the next generation starts warm.
    case keepLoaded
    /// Release every graph after each generation; the next one loads them again.
    case releaseAfterEachGeneration
}

/// A generated image with its metrics.
public struct ImageGeneration: Sendable {
    public let image: GeneratedImage
    public let metrics: ImageGenerationMetrics

    public init(image: GeneratedImage, metrics: ImageGenerationMetrics) {
        self.image = image
        self.metrics = metrics
    }
}

/// Times one generation's phases and samples the probe at each boundary (the pipeline's actor owns it).
final class ImageGenerationMeter {
    private let clock = ContinuousClock()
    private let started: ContinuousClock.Instant
    private var mark: ContinuousClock.Instant
    private let probe: GPUMemoryProbe
    private(set) var metrics = ImageGenerationMetrics()

    init(probe: GPUMemoryProbe) {
        self.probe = probe
        started = clock.now
        mark = started
        sample()
    }

    /// The seconds since the last lap, starting the next one.
    func lap() -> Double {
        let now = clock.now
        defer { mark = now }
        sample()
        return Self.seconds(now - mark)
    }

    /// Ends the lap and stores its seconds in `field`.
    func lap(into field: WritableKeyPath<ImageGenerationMetrics, Double>) {
        let seconds = lap()
        metrics[keyPath: field] = seconds
    }

    func sample() { metrics.peakAllocatedBytes = max(metrics.peakAllocatedBytes, probe.allocatedBytes()) }

    func update(_ change: (inout ImageGenerationMetrics) -> Void) { change(&metrics) }

    /// The metrics with the total set.
    func finish() -> ImageGenerationMetrics {
        metrics.totalSeconds = Self.seconds(clock.now - started)
        sample()
        return metrics
    }

    static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}

extension PipelineOptions {
    /// An image pipeline's admission (absent `memory`: ``MemoryBudget/imageGeneration``).
    func admission(
        manifest: Manifest, device: Device, bundle: InstalledBundle?, load: @escaping GraphResidency.Loader,
        fileBytes: @escaping (String) -> Int64, maximumShapes: Int = 4
    ) throws(AetherError) -> ResidentAdmission {
        try admission(manifest: manifest, device: device, defaultMemory: .imageGeneration, maximumShapes: maximumShapes,
                      bundle: bundle, load: load, fileBytes: fileBytes)
    }
}

enum ImageGraphs {
    /// Wraps `loader` so each graph is checked against its manifest role and `contract` before use (MANIFEST.md §2.1.1).
    static func validating(
        _ loader: @escaping GraphResidency.Loader, manifest: Manifest, contract: DriverContract
    ) -> GraphResidency.Loader {
        GraphResidency.validating(loader, manifest: manifest, contract: contract)
    }
}
