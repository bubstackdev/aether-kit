public import AetherCore
public import AetherHost
public import AetherRuntime
import AetherText
import Foundation

/// `krea2@1` (MANIFEST.md §5.5–5.6): Krea 2 Turbo text-to-image, vanilla or with a merged adapter (realism). A Qwen3-VL
/// text encoder whose tapped layers condition a flow-matching transformer split into four segments chained by their
/// carries, then a VAE decoder, on a `flowMatchEuler` schedule with a fixed exponential shift. macOS only in 0.1.0.
///
/// Loading gives a ``Krea2Model``, not a pipeline: the Krea 2 Community License (§4.2) requires a content filter, so a
/// ``Krea2Pipeline`` is only made with the app's ``ContentFilter`` (``Krea2Model/pipeline(filter:)``).
public enum Krea2Driver: ModelDriver {
    public static let id: DriverID = .krea2

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> Krea2Model {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        try Krea2Pipeline.requireMacOS(manifest, options.device)
        try DriverContract.krea2.validate(manifest)
        return Krea2Model(
            bundle: bundle, manifest: manifest, device: options.device,
            options: .init(cachePolicy: options.cachePolicy, cache: options.cache))
    }
}

/// An opened Krea 2 bundle, before a content filter is attached. Nothing is loaded yet.
public struct Krea2Model: Sendable {
    public let bundle: InstalledBundle
    public let manifest: Manifest
    public let device: Device
    public var options: ImagePipelineOptions

    /// The pipeline, reviewing every generation with `filter` (the app's; see ``ContentFilter``).
    public func pipeline(filter: some ContentFilter) async throws -> Krea2Pipeline {
        try await Krea2Pipeline(bundle: bundle, manifest: manifest, device: device, filter: filter, options: options)
    }
}

/// One loaded Krea 2 model (DESIGN.md §6.4): filter the prompt → text encoder → per step the four denoiser segments with
/// the host Euler update → VAE decoder → filter the image → ``GeneratedImage``.
///
/// - **Content filter:** required at construction, no default (``ContentFilter``). The prompt is reviewed before
///   anything runs and the image before it is returned; a refusal is `generation(.contentRefused)` and no image.
/// - **Prompt:** `krea2.prompt.systemPrefix + prompt + suffix`, tokenized without added specials and never padded. A
///   rendering longer than `maxTokens` keeps its first `maxTokens − keepTrailingTokens` ids and its last
///   `keepTrailingTokens` (the suffix's). The text encoder drops the `dropLeadingRows` prefix rows from its output.
/// - **Text length:** an export whose denoisers take `textMask` runs them at a fixed text length, as Diffusers does:
///   the conditioning's real rows first, zero rows after, and the mask (1 real, 0 padding) keeping the padding out of
///   every attention. The length is the segments' static text axis, the same for text-only and reference prompts, so
///   every prompt runs at one shape. The manifest declares it (`krea2.textLength`; Forge's default is 1,116, the most a
///   reference prompt has, and a shorter export such as 256 rows runs faster steps); without the field it is the most
///   rows a prompt of the export may have. A prompt with more conditioning rows is refused before anything loads, by
///   ``prepare(prompt:)``, the encodes and ``generate(_:)`` alike, with
///   `AetherError.capacity(what: Krea2Pipeline.textCapacityResource, needed: rows, available: length)`, so a host can
///   route it to an export with a longer text axis. An export without the mask runs at each prompt's own row count.
/// - **Memory, ordering, cancellation:** as ``LatentDiffusionPipeline``: residency planned from measured footprints
///   before each generation (`capacity` before anything loads), one generation at a time, cancellation between graph
///   calls, and any failure releases every graph.
/// - **Noise:** the seed draws the packed initial latents from Kit's ``SeededGaussianNoise`` (not PyTorch's generator),
///   so a seed gives the same image here and in the Forge reference.
/// - **Refine** (image-to-image) needs the export's optional `vaeEncoder`: ``encode(image:job:)`` gives an image's
///   packed latents, and ``generate(_:refining:reusing:noise:observing:)`` runs the steps from a start step.
/// - **Reference images** need the export's `krea2.references` block and `vision` graph: ``encode(prompt:references:job:)``
///   gives conditioning that ``generate(_:reusing:observing:)`` renders from.
public actor Krea2Pipeline {
    public typealias Options = ImagePipelineOptions
    public typealias Request = ImageRequest
    /// The rendered token ids (`systemPrefix + prompt + suffix`, no added specials) for a prompt.
    typealias PromptEncoder = @Sendable (String) throws -> [Int32]

    public nonisolated let manifest: Manifest
    public nonisolated let schedule: Manifest.Schedule
    public nonisolated let latent: Manifest.Latent
    public nonisolated let krea2: Manifest.Krea2
    public nonisolated let options: Options
    nonisolated let filter: any ContentFilter
    nonisolated let wiring: Wiring
    nonisolated let encodePrompt: PromptEncoder
    let device: Device
    private let load: GraphResidency.Loader
    let admission: ResidentAdmission
    /// One request at a time, in arrival order (after the GPU grant).
    let gate = SerialGate()
    /// Generations running or waiting (for the GPU or their turn).
    var pending = 0
    /// What happens to the graphs after each generation (default ``ImageRetention/keepLoaded``).
    public private(set) var retention: ImageRetention = .keepLoaded
    private var decoderTiling: Krea2VAETiling?
    /// The export a ``Krea2Conditioning`` must come from: the manifest's ``Manifest/contentDigest``.
    nonisolated let export: String
    /// The canvas of the last generation, so an ``encode(prompt:job:)`` plans against the residency it would keep.
    private var lastProfile: String?
    /// What an encode plans against: the last generation's canvas, else a conditioning-only profile.
    var lastProfileForEncode: String { lastProfile ?? "conditioning" }

    /// Opens an installed bundle with the app's content filter; graphs load on the first generation.
    public init(
        bundle: InstalledBundle, manifest: Manifest, device: Device, filter: some ContentFilter, options: Options = Options()
    ) async throws {
        try Self.requireMacOS(manifest, device)
        let tokenizer = try await TokenizerAdapter(bundle: bundle, manifest: manifest, forGeneration: false)
        let rule = manifest.krea2?.prompt
        let addSpecialTokens = manifest.tokenizer?.addSpecialTokens ?? false
        let loader = GraphResidency.bundleLoader(
            bundle: bundle, manifest: manifest, device: device, cachePolicy: options.cachePolicy, cache: options.cache)
        try self.init(
            manifest: manifest, device: device, filter: filter, options: options,
            encodePrompt: { text in
                guard let rule else { throw AetherError.manifestInvalid(field: "krea2", reason: "required by \(Krea2Driver.id)") }
                return tokenizer.encode(rule.systemPrefix + text + rule.suffix, addSpecialTokens: addSpecialTokens)
            },
            bundle: bundle, load: ImageGraphs.validating(loader, manifest: manifest, contract: .krea2),
            fileBytes: { role in GraphResidency.fileBytes(bundle: bundle, manifest: manifest, role: role, device: device) })
    }

    /// The pipeline over any prompt encoder and graph loader (tests drive it with fakes).
    init(
        manifest: Manifest, device: Device, filter: any ContentFilter, options: Options, encodePrompt: @escaping PromptEncoder,
        bundle: InstalledBundle? = nil, load: @escaping GraphResidency.Loader, fileBytes: @escaping @Sendable (String) -> Int64
    ) throws(AetherError) {
        try Self.requireMacOS(manifest, device)
        try DriverContract.krea2.validate(manifest)
        guard let schedule = manifest.schedule, schedule.kind == .flowMatchEuler else {
            throw .manifestInvalid(field: "schedule", reason: "\(Krea2Driver.id) requires a flowMatchEuler schedule")
        }
        guard schedule.guidance.kind == .none else {
            throw .manifestInvalid(field: "schedule.guidance", reason: "Krea 2 Turbo is distilled without guidance")
        }
        guard let latent = manifest.latent, latent.packing == .patch2x2ChannelMajor else {
            throw .manifestInvalid(field: "latent", reason: "\(Krea2Driver.id) requires patch2x2ChannelMajor latents")
        }
        // These exports have no fixed sizes, so the token bound is what keeps a request's canvas (and its unmeasured
        // memory) finite.
        guard latent.maxImageTokens != nil else {
            throw .manifestInvalid(field: "latent.maxImageTokens", reason: "required by \(Krea2Driver.id)")
        }
        guard let krea2 = manifest.krea2 else {
            throw .manifestInvalid(field: "krea2", reason: "required by \(Krea2Driver.id)")
        }
        guard manifest.tokenizer != nil else {
            throw .manifestInvalid(field: "tokenizer", reason: "\(Krea2Driver.id) renders the prompt with it")
        }
        wiring = try Wiring(manifest: manifest, krea2: krea2)
        self.manifest = manifest
        self.schedule = schedule
        self.latent = latent
        self.krea2 = krea2
        self.filter = filter
        self.device = device
        self.options = options
        self.encodePrompt = encodePrompt
        self.load = load
        export = manifest.contentDigest
        admission = try options.admission(manifest: manifest, device: device, bundle: bundle, load: load, fileBytes: fileBytes)
    }

    /// Krea is qualified for macOS only in 0.1.0 (docs/INTEGRATION.md).
    static func requireMacOS(_ manifest: Manifest, _ device: Device) throws(AetherError) {
        guard device.platform == .macOS else {
            throw .unsupportedDevice(catalogID: manifest.id, platform: device.platform.rawValue, architecture: device.architecture)
        }
    }

    /// Generates one image, reviewed by the content filter before and after. Krea exports have no default size: pass
    /// `width` and `height` (positive multiples of `latent.pixelsPerToken`, at most `latent.maxImageTokens` tokens).
    public func generate(_ request: Request) async throws -> GeneratedImage {
        try await generate(request, observing: nil).image
    }

    /// Generates one image as ``generate(_:)`` does, reporting each phase and step to `onProgress` (see
    /// ``ImageProgress``; a handler that throws stops the generation), and returns it with its metrics.
    public func generate(
        _ request: Request, observing onProgress: (@Sendable (ImageProgress) throws -> Void)?
    ) async throws -> ImageGeneration {
        try await generate(request, source: nil, refine: nil, observing: onProgress)
    }

    /// Generates one image from conditioning ``encode(prompt:job:)`` made, without loading or running the text
    /// encoder: the image a ``generate(_:observing:)`` of the same request renders, pixel for pixel. `request.prompt`
    /// must be the conditioning's prompt; the filter reviews it again, then the image, as for any generation. The
    /// conditioning must come from this export (``Krea2Conditioning`` from another bundle or export is refused with
    /// `invalidParameters`). ``ImageProgress/conditioningReady(rows:)`` is still reported, first, and the metrics'
    /// `conditioningSeconds` is the time to bind it.
    public func generate(
        _ request: Request, reusing conditioning: Krea2Conditioning,
        observing onProgress: (@Sendable (ImageProgress) throws -> Void)? = nil
    ) async throws -> ImageGeneration {
        guard conditioning.export == export else {
            throw AetherError.generation(.invalidParameters("the Krea 2 conditioning was encoded by another export"))
        }
        guard conditioning.prompt == request.prompt else {
            throw AetherError.generation(.invalidParameters("the request's prompt is not the conditioning's"))
        }
        return try await generate(request, source: conditioning, refine: nil, observing: onProgress)
    }

    private func generate(
        _ request: Request, source reused: Krea2Conditioning?, refine: Refine?,
        observing onProgress: (@Sendable (ImageProgress) throws -> Void)?
    ) async throws -> ImageGeneration {
        pending += 1
        defer { pending -= 1 }
        let meter = ImageGenerationMeter(probe: options.probe)
        try Task.checkCancellation()
        // Validate the request, then the filter's view of the prompt, before waiting for a turn or the GPU.
        guard let size = request.size(default: latent.defaultSize) else {
            throw AetherError.generation(.invalidParameters("Krea 2 has no default size; pass width and height"))
        }
        let grid = try latent.tokenGrid(width: size.width, height: size.height)
        var table = try FlowMatchSchedule(schedule, steps: request.steps, imageTokens: grid.rows * grid.columns)
        var plan: RefinePlan?
        if let refine {
            plan = try refine.plan(grid: grid, schedule: &table, trainTimesteps: schedule.trainTimesteps)
        }
        let source: ConditioningSource = if let reused {
            .reused(reused)
        } else {
            .encode(try Prompt(rendered: try encodePrompt(request.prompt), rule: krea2.prompt,
                               rows: krea2.conditioning.rowRange ?? 1...1))
        }
        try requireTextCapacity(rows: source.rows)
        _ = meter.lap()
        try await filter.requireAllowed(prompt: request.prompt)
        meter.lap(into: \.promptReviewSeconds)
        try Task.checkCancellation()

        let image = try await render(source, seed: request.seed, schedule: table, refine: plan, grid: grid, size: size,
                                     job: request.job, meter: meter, onProgress: onProgress)
        _ = meter.lap()
        try await filter.requireAllowed(image: image, prompt: request.prompt)
        meter.lap(into: \.imageReviewSeconds)
        return ImageGeneration(image: image, metrics: meter.finish())
    }

    /// How the pipeline would render `prompt`, on the CPU: no graph loads and the content filter isn't consulted. A
    /// prompt this throws for is refused by ``generate(_:)`` too (a negative id, or conditioning rows outside the
    /// export's range). Hosts that must not render a cut prompt check ``Krea2PreparedPrompt/truncated`` first:
    /// ``generate(_:)`` keeps the first `maxTokens − keepTrailingTokens` ids and the last `keepTrailingTokens`.
    public nonisolated func prepare(prompt: String) throws -> Krea2PreparedPrompt {
        let rendered = try encodePrompt(prompt)
        let prepared = try Prompt(rendered: rendered, rule: krea2.prompt, rows: krea2.conditioning.rowRange ?? 1...1)
        try requireTextCapacity(rows: prepared.conditioningRows)
        return Krea2PreparedPrompt(
            tokenIDs: prepared.ids, renderedTokenCount: rendered.count, conditioningRows: prepared.conditioningRows,
            truncated: prepared.truncated, maximumTokens: krea2.prompt.maxTokens)
    }

    /// Runs the text encoder on `prompt` alone and returns its conditioning, for ``generate(_:reusing:observing:)``:
    /// a prompt rendered at several seeds or sizes encodes once. The filter reviews the prompt first, as a generation
    /// does. Waits for the GPU (`job`'s priority) and this pipeline's turn, and plans memory against the canvas of the
    /// last generation (so its kept denoisers stay), or before any generation against a conditioning-only profile, whose
    /// text encoder the first generation's canvas then releases (use ``encode(prompt:width:height:job:)`` to keep it);
    /// the text encoder then stays loaded or is released as for a generation (``setRetention(_:)``). Refusals, pressure
    /// and cancellation are a generation's.
    public func encode(prompt text: String, job: GPUJob? = nil) async throws -> Krea2Conditioning {
        try await encode(prompt: text, profile: lastProfileForEncode, job: job)
    }

    /// ``encode(prompt:job:)`` for the generation that follows at `width` × `height`: the encode plans memory against
    /// that canvas, as the generation will, so with ``ImageRetention/keepLoaded`` the text encoder stays resident beside
    /// that canvas's denoisers and decoder instead of being released by the generation's change of canvas (the first
    /// encode on a new pipeline, or one for another size than the last generation's). The canvas is checked as a
    /// generation's would be (`invalidParameters`); nothing else changes.
    public func encode(prompt text: String, width: Int, height: Int, job: GPUJob? = nil) async throws -> Krea2Conditioning {
        _ = try latent.tokenGrid(width: width, height: height)
        return try await encode(prompt: text, profile: Self.profile(width: width, height: height), job: job)
    }

    /// The admission profile of a canvas: the key generations and canvas-aware encodes plan under.
    static func profile(width: Int, height: Int) -> String { "\(width)x\(height)" }

    private func encode(prompt text: String, profile: String, job: GPUJob?) async throws -> Krea2Conditioning {
        pending += 1
        defer { pending -= 1 }
        try Task.checkCancellation()
        let prompt = try Prompt(rendered: try encodePrompt(text), rule: krea2.prompt, rows: krea2.conditioning.rowRange ?? 1...1)
        try requireTextCapacity(rows: prompt.conditioningRows)
        try await filter.requireAllowed(prompt: text)
        try Task.checkCancellation()
        let grant = try await options.gpu.admit(job: job)
        defer { grant.release() }
        return try await gate.withTurn {
            let residency = try admission.begin(profile: profile, shape: denoiserShape(rows: prompt.conditioningRows))
            if let pressure = options.pressure, !admission.residentRoles.contains(wiring.text.role),
               await pressure.level == .critical {
                admission.release()
                throw AetherError.memoryPressure
            }
            do {
                try await residency.beginPhase(.perPrompt)
                let rows = try await encode(prompt, with: try await residency.graph(wiring.text.role), in: TensorArena())
                residency.sample()
                residency.endPhase(.perPrompt)
                try Task.checkCancellation()
                let conditioning = Krea2Conditioning(
                    prompt: text, rows: prompt.conditioningRows, columns: krea2.conditioning.columns,
                    truncated: prompt.truncated, referenceDigests: [], export: export, scalarType: rows.scalarType,
                    bytes: try rows.readBytes())
                admission.finish()
                if retention == .releaseAfterEachGeneration { admission.release() }
                return conditioning
            } catch {
                settle(after: error)
                throw error
            }
        }
    }

    /// After an encode or a generation threw inside its turn. A cancellation (`CancellationError`, bare or as
    /// ``AetherError/runtime(underlying:)``) keeps the resident graphs under ``ImageRetention/keepLoaded``, so the next
    /// request starts warm (``ResidentAdmission/keepAfterCancellation()``: only graphs already measured stay); any other
    /// failure, or a cancellation under ``ImageRetention/releaseAfterEachGeneration``, releases every graph.
    func settle(after error: any Error) {
        if retention == .keepLoaded, Self.isCancellation(error) {
            admission.keepAfterCancellation()
        } else {
            admission.release()
        }
    }

    /// Whether `error` is a cancellation: `CancellationError`, bare or wrapped as ``AetherError/runtime(underlying:)``.
    nonisolated static func isCancellation(_ error: any Error) -> Bool {
        if error is CancellationError { return true }
        if case AetherError.runtime(let underlying) = error, underlying is CancellationError { return true }
        return false
    }

    /// Whether the export has the VAE encoder that refine needs.
    public nonisolated var supportsRefine: Bool { wiring.encoder != nil }

    /// Encodes `image` (planar RGB on [-1, 1], as ``GeneratedImage`` holds it; each side a multiple of
    /// `latent.pixelsPerToken`, at most `latent.maxImageTokens` tokens) into the packed latents a refine starts from:
    /// `[rows, columns, channels]`, as the decoder takes them. Runs only the export's `vaeEncoder` (absent:
    /// `unsupportedOperation`), in the prompt phase, after the GPU grant and this pipeline's turn, and plans against
    /// the last generation's canvas, as ``encode(prompt:job:)`` does. The content filter isn't consulted: nothing is
    /// generated, and the refine that uses the latents reviews its prompt and its image.
    public func encode(image: GeneratedImage, job: GPUJob? = nil) async throws -> ImageLatents {
        guard let encoder = wiring.encoder else {
            throw AetherError.unsupportedOperation("this Krea 2 export has no vaeEncoder, which refine needs")
        }
        pending += 1
        defer { pending -= 1 }
        try Task.checkCancellation()
        let grid = try latent.tokenGrid(width: image.width, height: image.height)
        let grant = try await options.gpu.admit(job: job)
        defer { grant.release() }
        return try await gate.withTurn {
            let residency = try admission.begin(profile: lastProfile ?? "\(image.width)x\(image.height)")
            if let pressure = options.pressure, !admission.residentRoles.contains(encoder.role),
               await pressure.level == .critical {
                admission.release()
                throw AetherError.memoryPressure
            }
            do {
                let arena = TensorArena()
                try await residency.beginPhase(.perPrompt)
                let graph = try await residency.graph(encoder.role)
                let pixels = try graph.makeBuffer(
                    for: encoder.pixels, role: .input, dimensions: [image.height, image.width], in: arena)
                let packed = try graph.makeBuffer(
                    for: encoder.latents, role: .output, dimensions: [grid.rows, grid.columns], in: arena)
                guard pixels.shape == [3, image.height, image.width],
                      packed.shape == [grid.rows, grid.columns, latent.channels] else {
                    throw AetherError.manifestMismatch(
                        field: "graphs.\(encoder.role)",
                        expected: "[3, \(image.height), \(image.width)] → [\(grid.rows), \(grid.columns), \(latent.channels)]",
                        found: "\(pixels.shape) → \(packed.shape)")
                }
                let planar = image.planarRGB
                switch pixels.scalarType {
                case .float16: try pixels.write(Float16.self, count: planar.count) { Float16(planar[$0]) }
                default: try pixels.write(planar)
                }
                var bindings = Bindings(arena: arena)
                try bindings.bindInput(encoder.pixels, to: pixels)
                try bindings.bindOutput(encoder.latents, to: packed)
                try await graph.run(&bindings)
                residency.sample()
                let values = try packed.readFloats()
                residency.endPhase(.perPrompt)
                guard values.allSatisfy(\.isFinite) else {
                    throw AetherError.tensor(name: encoder.latents, reason: .unsupported("the encoder returned a non-finite latent"))
                }
                try Task.checkCancellation()
                admission.finish()
                if retention == .releaseAfterEachGeneration { admission.release() }
                return ImageLatents(rows: grid.rows, columns: grid.columns, channels: latent.channels, values: values)
            } catch {
                settle(after: error)
                throw error
            }
        }
    }

    /// Refines: generates `request` from `start`'s latents instead of pure noise, running only the steps from its
    /// start (see ``Krea2RefineStart``), with the content filter's reviews as for any generation. `start.latents` must
    /// be the request's canvas (`[height / pixelsPerToken, width / pixelsPerToken, channels]`). With `conditioning`
    /// (from ``encode(prompt:job:)``, for `request.prompt`), the text encoder neither loads nor runs, so tiles of one
    /// picture share one encode. `noise` (`rows · columns · channels` finite values in packed-token order, as
    /// ``ImageLatents/values``) replaces the seed's: tiles cut from one canvas-wide field get matching noise where they
    /// overlap. The start latents are `(1 − σ) · x0 + σ · noise` at the start step's sigma.
    public func generate(
        _ request: Request, refining start: Krea2RefineStart, reusing conditioning: Krea2Conditioning? = nil,
        noise: [Float]? = nil, observing onProgress: (@Sendable (ImageProgress) throws -> Void)? = nil
    ) async throws -> ImageGeneration {
        if let conditioning {
            guard conditioning.export == export else {
                throw AetherError.generation(.invalidParameters("the Krea 2 conditioning was encoded by another export"))
            }
            guard conditioning.prompt == request.prompt else {
                throw AetherError.generation(.invalidParameters("the request's prompt is not the conditioning's"))
            }
        }
        let x0 = start.latents
        guard x0.channels == latent.channels, x0.values.count == x0.rows * x0.columns * x0.channels else {
            throw AetherError.generation(.invalidParameters(
                "refine latents must be [rows, columns, \(latent.channels)], as encode(image:) returns them"))
        }
        if let noise {
            guard noise.count == x0.values.count, noise.allSatisfy(\.isFinite) else {
                throw AetherError.generation(.invalidParameters(
                    "refine noise must be \(x0.values.count) finite values (rows · columns · channels)"))
            }
        }
        return try await generate(request, source: conditioning, refine: Refine(start: start, noise: noise), observing: onProgress)
    }

    /// The text encoder's conditioning rows for `prompt` (`[rows, taps · width]` row-major; the gate's text check).
    /// Loads the text encoder for the call and releases it; nothing is measured and the filter isn't consulted.
    func conditioningRows(for prompt: String) async throws -> [Float] {
        let prompt = try Prompt(rendered: try encodePrompt(prompt), rule: krea2.prompt, rows: krea2.conditioning.rowRange ?? 1...1)
        return try await gate.withTurn {
            let text = try await load(wiring.text.role)
            return try await encode(prompt, with: text, in: TensorArena()).readFloats()
        }
    }

    /// Releases every graph (waiting for a running generation). The next generation loads them again.
    public func unload() async {
        await gate.withTurnIgnoringCancellation {
            admission.unload()
        }
    }

    /// Roles currently loaded (for tests and diagnostics).
    public var residentRoles: [String] { admission.residentRoles }

    /// What the graphs kept between generations hold, measured on the pipeline's probe (Metal allocations, so Core AI's
    /// memory is counted): the last generation's plan of what stays loaded, which is what stays reserved on a
    /// coordinator. 0 when nothing is. Planned, not read at the call.
    public var plannedResidentBytes: Int64 { admission.residentBytes }

    /// What the graphs kept between generations hold now, read on the pipeline's probe at the call (Metal allocations
    /// included, so Core AI's memory is counted): the process's level above where it stood before the graphs loaded.
    /// 0 when nothing is resident. Live where ``plannedResidentBytes`` is the plan: it follows whatever the kept graphs
    /// retained after the last generation, and it counts anything else the process allocated since they loaded, so a
    /// host that loads other models after this pipeline reads them here too. Read during a generation, it includes
    /// that generation's working memory.
    public var liveResidentBytes: Int64 { admission.liveResidentBytes }

    /// Sets what happens to the graphs after each generation. A generation already running applies the new choice
    /// when it ends.
    public func setRetention(_ retention: ImageRetention) { self.retention = retention }

    /// Uses overlapping VAE tiles to reduce decoder working memory. Sizes are output pixels, multiples of 16:
    /// `tileSize` must be at least 32 and `overlap` smaller than it. `nil` restores the default full decode.
    ///
    /// Tiles approximate the decoder's global attention, so pixels differ slightly from a full decode. 512-pixel
    /// tiles with 256-pixel overlap reduced isolated decoder Metal memory from 11.49 to 2.54 GiB at 1024², with
    /// SSIM 0.995 against the full decode on the qualification image. Decoding takes longer.
    /// Waits for this pipeline's turn. Changing the setting releases its graphs so old decoder specializations
    /// cannot keep the full-decode allocation alive; the next generation loads again.
    public func setDecoderTiling(tileSize: Int?, overlap: Int = 256) async throws {
        let tiling: Krea2VAETiling?
        if let tileSize {
            guard latent.pixelsPerToken == 16 else {
                throw AetherError.generation(.invalidParameters("decoder tiling requires 16 pixels per latent token"))
            }
            tiling = try Krea2VAETiling(tileSize: tileSize, overlap: overlap)
        }
        else { tiling = nil }
        await gate.withTurnIgnoringCancellation {
            if decoderTiling != tiling {
                admission.release()
                lastProfile = nil
                decoderTiling = tiling
            }
        }
    }

    /// Releases every graph now, but only while no generation is running or waiting: it never waits and never
    /// interrupts one. Returns `false`, changing nothing, while a generation is in progress (it keeps its graphs; call
    /// again later, or use ``unload()`` to wait for it). The bundle stays open, unlike ``unload()``.
    @discardableResult
    public func releaseIfIdle() -> Bool {
        guard pending == 0 else { return false }
        admission.release()
        return true
    }

    // MARK: - The generation

    private func render(
        _ source: ConditioningSource, seed: UInt64, schedule: FlowMatchSchedule, refine: RefinePlan?, grid: (rows: Int, columns: Int),
        size: (width: Int, height: Int), job: GPUJob?, meter: ImageGenerationMeter,
        onProgress: (@Sendable (ImageProgress) throws -> Void)?
    ) async throws -> GeneratedImage {
        // The GPU first, so a request's priority also orders it among this pipeline's own requests.
        let grant = try await options.gpu.admit(job: job)
        defer { grant.release() }
        return try await gate.withTurn {
            // The prompt's row count shapes the text encoder's output and every segment's text carry, and those segments
            // load without the reshape hint, so each new count keeps its own specialization (ResidentAdmission.Settings).
            meter.lap(into: \.waitSeconds)
            let used = (source.reused ? [] : [wiring.text.role]) + wiring.segments.map(\.role) + [wiring.decoder.role]
            let profile = Self.profile(width: size.width, height: size.height) + (decoderTiling.map { ".\($0.profile)" } ?? "")
            let residency = try admission.begin(profile: profile, shape: denoiserShape(rows: source.rows))
            lastProfile = profile
            // What is resident once admission has planned: it may have released the graphs (a new shape that doesn't
            // fit beside the others, or a profile change), and then they all load again.
            let residentAtStart = Set(admission.residentRoles)
            // Under critical pressure, run only on graphs already resident: loading is what pressure can't afford.
            if let pressure = options.pressure, !Set(used).isSubset(of: residentAtStart), await pressure.level == .critical {
                admission.release()
                throw AetherError.memoryPressure
            }
            meter.update {
                $0.conditioningRows = source.rows
                $0.imageTokens = grid.rows * grid.columns
                $0.loadedRoles = Set(used).subtracting(residentAtStart).sorted()
            }
            do {
                let image = try await run(residency, source: source, seed: seed, schedule: schedule, refine: refine, grid: grid, size: size,
                                          meter: meter, onProgress: onProgress)
                admission.finish()
                if retention == .releaseAfterEachGeneration { admission.release() }
                return image
            } catch {
                settle(after: error)
                throw error
            }
        }
    }

    /// The three phases. The denoisers and the decoder run in their own functions, so their graphs are referenced
    /// only by those functions' locals and the residency: when ``GraphResidency/endPhase(_:)`` releases them (per
    /// phase), nothing else keeps them alive into the next phase.
    private func run(
        _ residency: GraphResidency, source: ConditioningSource, seed: UInt64, schedule: FlowMatchSchedule,
        refine: RefinePlan?, grid: (rows: Int, columns: Int), size: (width: Int, height: Int), meter: ImageGenerationMeter,
        onProgress: (@Sendable (ImageProgress) throws -> Void)?
    ) async throws -> GeneratedImage {
        let arena = TensorArena()

        // Prompt: the conditioning rows outlive the text encoder (it may be released before the denoisers load). Reused
        // conditioning skips the phase: the text encoder neither loads nor runs.
        let conditioning: Conditioning
        switch source {
        case .encode(let prompt):
            try await residency.beginPhase(.perPrompt)
            conditioning = .buffer(try await encode(prompt, with: try await residency.graph(wiring.text.role), in: arena))
            residency.sample()
            residency.endPhase(.perPrompt)
        case .reused(let value):
            conditioning = .reused(value)
        }
        meter.lap(into: \.conditioningSeconds)
        try onProgress?(.conditioningReady(rows: source.rows))
        try Task.checkCancellation()

        // Steps: every segment loads, then each step writes the latents and sigma and runs the segments in order.
        try await residency.beginPhase(.perStep)
        let latents = try await denoise(
            conditioning: conditioning, textRows: source.rows, seed: seed, schedule: schedule, refine: refine, grid: grid,
            residency: residency, arena: arena, meter: meter, onProgress: onProgress)
        residency.endPhase(.perStep)
        if case .buffer(let rows) = conditioning { try arena.release(rows) }
        try Task.checkCancellation()

        // Image: the packed tokens, raster order, are the decoder's [rows, columns, channels].
        try await residency.beginPhase(.perImage)
        let image = try await decode(latents, grid: grid, size: size, residency: residency, arena: arena)
        residency.endPhase(.perImage)
        meter.lap(into: \.decodeSeconds)
        try onProgress?(.decoded)
        // A cancellation during the decoder's call: don't report success for a request the caller has given up on.
        try Task.checkCancellation()
        return image
    }

    private func denoise(
        conditioning source: Conditioning, textRows: Int, seed: UInt64, schedule: FlowMatchSchedule, refine: RefinePlan?,
        grid: (rows: Int, columns: Int), residency: GraphResidency, arena: TensorArena, meter: ImageGenerationMeter,
        onProgress: (@Sendable (ImageProgress) throws -> Void)?
    ) async throws -> [Float] {
        // Handles do not own storage: dropping this function's locals leaves their arrays in the arena. Return
        // every step buffer before the decoder starts, including on cancellation or a failed graph call.
        var scratch: [TensorBuffer] = []
        defer { for buffer in scratch { try? arena.release(buffer) } }
        let tokens = grid.rows * grid.columns, count = tokens * latent.channels
        var loaded: [any GraphBackend] = []
        for segment in wiring.segments { loaded.append(try await residency.graph(segment.role)) }
        // The functions this prompt runs and their text length: the rows themselves, or a masked bucket's fixed length.
        let (graphs, length) = try textLayout(rows: textRows, graphs: loaded)
        let dims = Dimensions(textRows: length, imageTokens: tokens)
        let first = graphs[0]
        let sample = try first.makeBuffer(for: wiring.latents, role: .input, dimensions: [tokens], in: arena)
        scratch.append(sample)
        let sigma = try first.makeBuffer(for: wiring.sigma, role: .input, dimensions: [], in: arena)
        scratch.append(sigma)
        let imageRotary = try first.makeBuffer(for: wiring.imageRotary, role: .input, dimensions: [tokens], in: arena)
        scratch.append(imageRotary)
        guard sample.shape == [tokens, latent.channels] else {
            throw AetherError.manifestMismatch(
                field: "latent.channels", expected: "denoiser latents [\(tokens), \(latent.channels)]", found: "\(sample.shape)")
        }
        try imageRotary.write(Krea2Rotary.image(rows: grid.rows, columns: grid.columns, rotary: krea2.rotary))
        let conditioning = try conditioningBuffer(source, rows: textRows, length: length, graph: first, dims: dims, arena: arena)
        if case .buffer(let original) = source, conditioning.isSameStorage(as: original) {
            // The caller owns the encoder's rows.
        } else {
            scratch.append(conditioning)
        }
        var textMask: TensorBuffer?
        if let name = wiring.textMask {
            let mask = try first.makeBuffer(for: name, role: .input, dimensions: dims.of("textMask", in: first, name: name), in: arena)
            scratch.append(mask)
            guard mask.shape == [length] else {
                throw AetherError.manifestMismatch(
                    field: "graphs.\(wiring.segments[0].role).inputs.textMask", expected: "[\(length)]", found: "\(mask.shape)")
            }
            let values = Krea2TextMask.values(rows: textRows, length: length)
            switch mask.scalarType {
            case .float16: try mask.write(Float16.self, count: length) { Float16(values[$0]) }
            default: try mask.write(values)
            }
            textMask = mask
        }
        var outputs: [String: TensorBuffer] = [:]
        // Reuse only after the last consumer, never within a call. Keep the velocity live through the host update.
        var lastUse: [String: Int] = [wiring.velocity: wiring.segments.count]
        for (index, segment) in wiring.segments.enumerated() {
            for input in segment.inputs {
                if case .output(let ref) = input.source { lastUse[ref] = max(lastUse[ref] ?? index, index) }
            }
        }
        var slots: [(buffer: TensorBuffer, lastUse: Int)] = []
        for (index, pair) in zip(wiring.segments, graphs).enumerated() {
            let (segment, graph) = pair
            for output in segment.outputs {
                let candidate = try graph.makeBuffer(
                    for: output.name, role: .output, dimensions: dims.of(output.role, in: graph, name: output.name, output: true),
                    in: arena)
                // Ask the backend for its actual layout rather than guessing Core AI's preferred strides.
                if let slot = slots.firstIndex(where: {
                    $0.lastUse < index && $0.buffer.shape == candidate.shape
                        && $0.buffer.scalarType == candidate.scalarType && $0.buffer.strides == candidate.strides
                }) {
                    try arena.release(candidate)
                    outputs[output.ref] = slots[slot].buffer
                    slots[slot].lastUse = lastUse[output.ref] ?? index
                } else {
                    scratch.append(candidate)
                    outputs[output.ref] = candidate
                    slots.append((candidate, lastUse[output.ref] ?? index))
                }
            }
        }
        let velocity = outputs[wiring.velocity]!
        guard velocity.shape == [tokens, latent.channels] else {
            throw AetherError.manifestMismatch(
                field: "latent.channels", expected: "velocity [\(tokens), \(latent.channels)]", found: "\(velocity.shape)")
        }
        func buffer(_ source: Wiring.Source) -> TensorBuffer {
            switch source {
            case .latents: sample
            case .conditioning: conditioning
            case .sigma: sigma
            case .imageRotary: imageRotary
            case .textMask: textMask!
            case .output(let ref): outputs[ref]!
            }
        }

        var latents = try refine?.noise ?? SeededGaussianNoise.generate(seed: seed, count: count)
        if let refine { latents = try schedule.refineStart(x0: refine.x0, noise: latents, step: refine.startStep) }
        // Host buffers reused across steps: the velocity is read into `v`, and each update goes into `next` in one
        // validating pass before the two swap.
        var v = [Float](repeating: 0, count: count), next = [Float](repeating: 0, count: count)
        meter.lap(into: \.denoiserPreparationSeconds)
        try onProgress?(.denoisersReady)
        for step in (refine?.startStep ?? 0)..<schedule.steps {
            try latents.withUnsafeBufferPointer { x throws(AetherError) in
                try sample.write(Float16.self, count: count) { Float16(x[$0]) }
            }
            try sigma.write([schedule.timesteps[step]])
            for (segment, graph) in zip(wiring.segments, graphs) {
                try Task.checkCancellation()
                var bindings = Bindings(arena: arena)
                for input in segment.inputs { try bindings.bindInput(input.name, to: buffer(input.source)) }
                for output in segment.outputs { try bindings.bindOutput(output.name, to: outputs[output.ref]!) }
                try await graph.run(&bindings)
                residency.sample()
            }
            try velocity.readFloats(0..<count, into: &v)
            try schedule.advance(step: step, latents: &latents, velocity: v, scratch: &next)
            let seconds = meter.lap()
            meter.update { $0.stepSeconds.append(seconds); $0.samplingSeconds += seconds }
            if let onProgress {
                try onProgress(.step(step, of: schedule.steps, latents: ImageLatents(
                    rows: grid.rows, columns: grid.columns, channels: latent.channels, values: latents)))
            }
        }
        return latents
    }

    /// The text length the segments run `rows` conditioning rows at: the rows themselves without the mask, else the
    /// segments' static text axis (a masked export whose text axis is dynamic runs at the rows, the mask all ones).
    /// A static axis must be the manifest's `krea2.textLength` when it declares one; rows past it are refused
    /// (`capacity`, as ``requireTextCapacity(rows:)``).
    private func textLayout(rows: Int, graphs: [any GraphBackend]) throws(AetherError) -> (graphs: [any GraphBackend], length: Int) {
        guard wiring.textMask != nil else { return (graphs, rows) }
        guard let spec = graphs[0].signature.input(wiring.conditioning), spec.rank == 2 else {
            throw .tensor(name: wiring.conditioning, reason: .unknownName)
        }
        let length = spec.shape[0]
        if length == TensorSpec.dynamic { return (graphs, rows) }
        if let declared = krea2.textLength, declared != length {
            throw .manifestMismatch(
                field: "krea2.textLength", expected: "\(declared) (the denoisers' static text axis)",
                found: "graphs.\(wiring.segments[0].role).inputs.conditioning [\(length), \(spec.shape[1])]")
        }
        guard rows <= length else {
            throw .capacity(what: Self.textCapacityResource, needed: Int64(rows), available: Int64(length))
        }
        return (graphs, length)
    }

    /// The `what` of the `capacity` error a prompt with more conditioning rows than a masked export's text length
    /// (`krea2.textLength`) fails with: `needed` is the prompt's rows, `available` the length.
    public static let textCapacityResource = "krea2.textLength"

    /// The most conditioning rows a prompt may have on this export's denoisers: the masked text length the manifest
    /// declares (`krea2.textLength`), nil when it declares none (an exact-length export, or a masked one whose length is
    /// the most rows its prompts can have).
    public nonisolated var textCapacity: Int? { wiring.textMask == nil ? nil : krea2.textLength }

    /// Refuses `rows` conditioning rows past ``textCapacity``, before anything loads.
    nonisolated func requireTextCapacity(rows: Int) throws(AetherError) {
        guard let length = textCapacity, rows > length else { return }
        throw .capacity(what: Self.textCapacityResource, needed: Int64(rows), available: Int64(length))
    }

    /// The first segment's conditioning input: the text encoder's buffer as it is when it already has the run's
    /// length, else the rows copied into a `[length, columns]` buffer whose rows after them are zero (finite padding,
    /// kept out of attention by the mask).
    private func conditioningBuffer(
        _ source: Conditioning, rows: Int, length: Int, graph: any GraphBackend, dims: Dimensions, arena: TensorArena
    ) throws(AetherError) -> TensorBuffer {
        let columns = krea2.conditioning.columns
        let real: TensorBuffer
        switch source {
        case .buffer(let buffer):
            if buffer.shape == [length, columns] { return buffer }
            real = buffer
        case .reused(let value):
            if length == value.rows {
                let input = try graph.makeBuffer(
                    for: wiring.conditioning, role: .input, dimensions: dims.of("conditioning", in: graph, name: wiring.conditioning),
                    in: arena)
                guard input.scalarType == value.scalarType, input.shape == [value.rows, value.columns],
                      input.storageByteCount == value.bytes.count else {
                    throw .manifestMismatch(
                        field: "graphs.\(wiring.segments[0].role).inputs.conditioning",
                        expected: "\(value.scalarType) [\(value.rows), \(value.columns)] (the text encoder's output)",
                        found: "\(input.scalarType) \(input.shape)")
                }
                try input.writeBytes(value.bytes)
                return input
            }
            real = try arena.allocate(name: "\(wiring.conditioning).rows", shape: [value.rows, value.columns], scalarType: value.scalarType)
            try real.writeBytes(value.bytes)
        }
        // Reused rows are staging storage; only the padded destination is needed by the steps.
        defer { if case .reused = source { try? arena.release(real) } }

        let padded = try graph.makeBuffer(
            for: wiring.conditioning, role: .input, dimensions: dims.of("conditioning", in: graph, name: wiring.conditioning), in: arena)
        guard padded.scalarType == real.scalarType, padded.shape == [length, columns], real.shape == [rows, columns] else {
            throw .manifestMismatch(
                field: "graphs.\(wiring.segments[0].role).inputs.conditioning",
                expected: "\(real.scalarType) [\(length), \(columns)] holding \(rows) rows",
                found: "\(padded.scalarType) \(padded.shape) from \(real.shape)")
        }
        try padded.zero()
        try padded.copyRegion(from: real)
        return padded
    }

    private func decode(
        _ latents: [Float], grid: (rows: Int, columns: Int), size: (width: Int, height: Int), residency: GraphResidency,
        arena: TensorArena
    ) async throws -> GeneratedImage {
        let count = grid.rows * grid.columns * latent.channels
        let vae = try await residency.graph(wiring.decoder.role)
        if let tiling = decoderTiling, size.width > tiling.tileSize || size.height > tiling.tileSize {
            return try await decodeTiles(latents, grid: grid, size: size, tiling: tiling, vae: vae, residency: residency, arena: arena)
        }
        let input = try vae.makeBuffer(for: wiring.decoder.latents, role: .input, dimensions: [grid.rows, grid.columns], in: arena)
        let output = try vae.makeBuffer(for: wiring.decoder.image, role: .output, dimensions: [size.height, size.width], in: arena)
        guard input.shape == [grid.rows, grid.columns, latent.channels], output.shape == [3, size.height, size.width] else {
            throw AetherError.manifestMismatch(
                field: "graphs.\(wiring.decoder.role)",
                expected: "[\(grid.rows), \(grid.columns), \(latent.channels)] → [3, \(size.height), \(size.width)]",
                found: "\(input.shape) → \(output.shape)")
        }
        try input.write(Float16.self, count: count) { Float16(latents[$0]) }
        var bindings = Bindings(arena: arena)
        try bindings.bindInput(wiring.decoder.latents, to: input)
        try bindings.bindOutput(wiring.decoder.image, to: output)
        try await vae.run(&bindings)
        residency.sample()
        return try GeneratedImage(width: size.width, height: size.height, planarRGB: try output.readFloats())
    }

    private func decodeTiles(
        _ latents: [Float], grid: (rows: Int, columns: Int), size: (width: Int, height: Int), tiling: Krea2VAETiling,
        vae: any GraphBackend, residency: GraphResidency, arena: TensorArena
    ) async throws -> GeneratedImage {
        var accumulator = Krea2VAETiling.Accumulator(width: size.width, height: size.height, tiling: tiling)
        let rows = accumulator.y.size / 16, columns = accumulator.x.size / 16, channels = latent.channels
        let input = try vae.makeBuffer(for: wiring.decoder.latents, role: .input, dimensions: [rows, columns], in: arena)
        let output = try vae.makeBuffer(
            for: wiring.decoder.image, role: .output, dimensions: [accumulator.y.size, accumulator.x.size], in: arena)
        guard input.shape == [rows, columns, channels], output.shape == [3, accumulator.y.size, accumulator.x.size] else {
            throw AetherError.manifestMismatch(
                field: "graphs.\(wiring.decoder.role)",
                expected: "[\(rows), \(columns), \(channels)] → [3, \(accumulator.y.size), \(accumulator.x.size)]",
                found: "\(input.shape) → \(output.shape)")
        }
        var bindings = Bindings(arena: arena)
        try bindings.bindInput(wiring.decoder.latents, to: input)
        try bindings.bindOutput(wiring.decoder.image, to: output)
        var pixels: [Float] = []
        for (iy, y) in accumulator.y.starts.enumerated() {
            for (ix, x) in accumulator.x.starts.enumerated() {
                try Task.checkCancellation()
                try input.write(Float16.self, count: rows * columns * channels) { i in
                    let token = i / channels
                    let source = ((y / 16 + token / columns) * grid.columns + x / 16 + token % columns) * channels + i % channels
                    return Float16(latents[source])
                }
                try await vae.run(&bindings)
                residency.sample()
                try Task.checkCancellation()
                try output.readFloats(0..<output.count, into: &pixels)
                try accumulator.add(pixels, row: iy, column: ix)
            }
        }
        return try GeneratedImage(width: size.width, height: size.height, planarRGB: accumulator.finish())
    }

    /// Runs the text encoder on `prompt`: its conditioning rows, `[n − dropLeadingRows, taps · width]`, in `arena`.
    private func encode(_ prompt: Prompt, with text: any GraphBackend, in arena: TensorArena) async throws -> TensorBuffer {
        let n = prompt.ids.count
        let ids = try text.makeBuffer(for: wiring.text.tokens, role: .input, dimensions: [n], in: arena)
        defer { try? arena.release(ids) }
        let positions = try text.makeBuffer(for: wiring.text.positions, role: .input, dimensions: [n], in: arena)
        defer { try? arena.release(positions) }
        let rows = try text.makeBuffer(for: wiring.text.conditioning, role: .output, dimensions: [prompt.conditioningRows], in: arena)
        guard rows.shape == [prompt.conditioningRows, krea2.conditioning.columns] else {
            throw AetherError.manifestMismatch(
                field: "krea2.conditioning", expected: "text rows [\(prompt.conditioningRows), \(krea2.conditioning.columns)]",
                found: "\(rows.shape)")
        }
        try ids.write(prompt.ids)
        try positions.write(Int32.self, count: n) { Int32($0) }
        var bindings = Bindings(arena: arena)
        try bindings.bindInput(wiring.text.tokens, to: ids)
        try bindings.bindInput(wiring.text.positions, to: positions)
        try bindings.bindOutput(wiring.text.conditioning, to: rows)
        try await text.run(&bindings)
        return rows
    }
}

/// A prompt as ``Krea2Pipeline`` renders it (``Krea2Pipeline/prepare(prompt:)``): `systemPrefix + prompt + suffix`,
/// tokenized without added specials, then bounded by the export's `maxTokens`.
public struct Krea2PreparedPrompt: Sendable, Hashable {
    /// The ids the text encoder receives (after any truncation).
    public let tokenIDs: [Int32]
    /// The rendered ids before truncation.
    public let renderedTokenCount: Int
    /// The conditioning rows the denoisers take: the ids after the dropped prefix.
    public let conditioningRows: Int
    /// The rendering was longer than ``maximumTokens`` and was cut (its tail kept).
    public let truncated: Bool
    /// The export's `maxTokens`: the most ids the text encoder receives.
    public let maximumTokens: Int
}

/// A prompt's conditioning from ``Krea2Pipeline/encode(prompt:job:)``: the text encoder's output rows, owned by the
/// host, for ``Krea2Pipeline/generate(_:reusing:observing:)`` at any seed and size without the text encoder. Valid
/// only for the export that encoded it (another bundle's pipeline refuses it) and only with its own prompt.
public struct Krea2Conditioning: Sendable, Hashable {
    /// The prompt it was encoded from, as the host wrote it.
    public let prompt: String
    /// The conditioning rows (the denoisers' text length) and their width.
    public let rows: Int
    public let columns: Int
    /// The rendering was longer than `maxTokens` and was cut (``Krea2PreparedPrompt/truncated``).
    public let truncated: Bool
    /// The reference images' digests, in prompt order (``Krea2ReferenceImage/digest``); empty for a text-only prompt.
    public let referenceDigests: [String]
    /// The export it belongs to (``Manifest/contentDigest``).
    let export: String
    let scalarType: ScalarType
    /// The text encoder's output, byte for byte, so a reuse renders what an encode would.
    let bytes: [UInt8]

    /// The bytes the conditioning holds.
    public var byteCount: Int { bytes.count }
}

/// Where a refine starts (``Krea2Pipeline/generate(_:refining:reusing:noise:observing:)``): an image's packed latents
/// (``Krea2Pipeline/encode(image:job:)``) and the step the run begins at. The start latents are
/// `(1 − σ) · x0 + σ · noise` at that step's sigma, so a later step keeps more of the image (Turbo, 8 steps: σ ≈ 0.31 at
/// step 7, 0.51 at step 6, 0.66 at step 5).
public struct Krea2RefineStart: Sendable, Hashable {
    /// Where on the schedule the refine begins.
    public enum Start: Sendable, Hashable {
        /// Steps `step…steps − 1` of the schedule run; `step` 0 would ignore the image, so it's refused.
        case step(Int)
        /// One step from `sigma` to 0: the schedule's last step with its sigma replaced. Gentler than the last grid
        /// step; it must lie below that step's sigma. Turbo is distilled at its grid sigmas, so qualify an off-grid
        /// start before relying on it.
        case lastStepFrom(sigma: Float)
    }

    /// The image's packed latents (``Krea2Pipeline/encode(image:job:)``), at the canvas the refine renders.
    public let latents: ImageLatents
    /// The step, or the off-grid sigma, the run begins at.
    public let start: Start

    /// Fails unless the latents are non-empty and finite, and the step is at least 1 (or the sigma finite in (0, 1)).
    /// The request's schedule checks the rest when the refine runs.
    public init(latents: ImageLatents, start: Start) throws(AetherError) {
        guard latents.rows > 0, latents.columns > 0, latents.channels > 0,
              latents.values.count == latents.rows * latents.columns * latents.channels,
              latents.values.allSatisfy(\.isFinite) else {
            throw .generation(.invalidParameters("refine latents must be non-empty, finite [rows, columns, channels]"))
        }
        switch start {
        case .step(let step):
            guard step >= 1 else { throw .generation(.invalidParameters("a refine starts at step 1 or later")) }
        case .lastStepFrom(let sigma):
            guard sigma.isFinite, sigma > 0, sigma < 1 else {
                throw .generation(.invalidParameters("an off-grid refine sigma must be in (0, 1)"))
            }
        }
        self.latents = latents
        self.start = start
    }
}

// MARK: - Host pieces

extension Krea2Pipeline {
    /// A refine as requested: its start and any caller noise.
    struct Refine: Sendable {
        var start: Krea2RefineStart
        var noise: [Float]?

        /// Checks the start against the request's canvas and schedule; an off-grid start replaces `schedule`'s last
        /// sigma.
        func plan(grid: (rows: Int, columns: Int), schedule: inout FlowMatchSchedule, trainTimesteps: Int) throws(AetherError) -> RefinePlan {
            guard start.latents.rows == grid.rows, start.latents.columns == grid.columns else {
                throw .generation(.invalidParameters(
                    "refine latents are \(start.latents.rows)x\(start.latents.columns) tokens; the request is \(grid.rows)x\(grid.columns)"))
            }
            let step: Int
            switch start.start {
            case .step(let value):
                guard value < schedule.steps else {
                    throw .generation(.invalidParameters("a refine start step must be below the \(schedule.steps) steps"))
                }
                step = value
            case .lastStepFrom(let sigma):
                guard schedule.steps >= 2, sigma < schedule.sigmas[schedule.steps - 1] else {
                    throw .generation(.invalidParameters(
                        "an off-grid refine sigma must lie below the last grid sigma (\(schedule.sigmas[max(0, schedule.steps - 1)]))"))
                }
                schedule = try schedule.replacingLastSigma(sigma, trainTimesteps: trainTimesteps)
                step = schedule.steps - 1
            }
            return RefinePlan(x0: start.latents.values, startStep: step, noise: noise)
        }
    }

    /// A checked refine: the start latents, the first step to run and any caller noise.
    struct RefinePlan: Sendable {
        var x0: [Float]
        var startStep: Int
        var noise: [Float]?
    }

    /// Where a generation's conditioning comes from: the text encoder now, or an earlier ``encode(prompt:job:)``.
    enum ConditioningSource: Sendable {
        case encode(Prompt)
        case reused(Krea2Conditioning)

        var rows: Int {
            switch self {
            case .encode(let prompt): prompt.conditioningRows
            case .reused(let value): value.rows
            }
        }

        var reused: Bool { if case .reused = self { true } else { false } }
    }

    /// The conditioning the first segment binds: the text encoder's output buffer, or reused bytes to copy in.
    enum Conditioning {
        case buffer(TensorBuffer)
        case reused(Krea2Conditioning)
    }

    /// The text encoder's input: the rendered ids, unpadded. A rendering longer than `maxTokens` keeps its first
    /// `maxTokens − keepTrailingTokens` ids and its last `keepTrailingTokens`.
    struct Prompt: Sendable, Hashable {
        var ids: [Int32]
        var truncated: Bool
        /// Rows the text encoder emits: the ids after the dropped prefix.
        var conditioningRows: Int

        init(rendered: [Int32], rule: Manifest.Krea2.Prompt, rows: ClosedRange<Int>) throws(AetherError) {
            guard rendered.allSatisfy({ $0 >= 0 }) else {
                throw .generation(.invalidParameters("the rendered Krea 2 prompt has a negative token id"))
            }
            truncated = rendered.count > rule.maxTokens
            ids = truncated
                ? Array(rendered.prefix(rule.maxTokens - rule.keepTrailingTokens)) + rendered.suffix(rule.keepTrailingTokens)
                : rendered
            conditioningRows = ids.count - rule.dropLeadingRows
            guard rows.contains(conditioningRows) else {
                throw .generation(.invalidParameters(
                    "the rendered Krea 2 prompt has \(ids.count) tokens; its \(conditioningRows) conditioning rows must be in \(rows)"))
            }
        }
    }

    /// Which dynamic dimension each role's tensors have: text-side tensors the run's text length, image-side ones the
    /// generated token count.
    struct Dimensions {
        var textRows: Int
        var imageTokens: Int

        func of(_ role: String) -> [Int] {
            switch role {
            case "carryText", "conditioning", "textMask": [textRows]
            case "carryImage", "imageRotary", "latents", "velocity": [imageTokens]
            default: []
            }
        }

        /// ``of(_:)`` for tensor `name` of `graph`: only as many values as the tensor has dynamic axes (a masked
        /// export's text axis is static, so its text-side tensors take none).
        func of(_ role: String, in graph: any GraphBackend, name: String, output: Bool = false) -> [Int] {
            let wanted = of(role)
            guard let spec = output ? graph.signature.output(name) : graph.signature.input(name) else { return wanted }
            return Array(wanted.prefix(spec.dynamicAxes.count))
        }
    }

    /// The admission shape of a generation with `rows` conditioning rows: the rows (each count its own denoiser
    /// specialization) or, with the text mask, one shape for every prompt: the manifest's `krea2.textLength`, else the
    /// most rows any prompt of the export may have (the references range's upper bound, else the conditioning range's).
    nonisolated func denoiserShape(rows: Int) -> Int {
        guard wiring.textMask != nil else { return rows }
        return krea2.textLength ?? krea2.references?.rowRange?.upperBound ?? krea2.conditioning.rowRange?.upperBound ?? rows
    }

    /// The manifest's tensor names and how each segment input is fed, checked once at load: an input chained with
    /// `source.from` must name the text encoder's conditioning or an earlier segment's output, and only the last
    /// segment emits `velocity`.
    struct Wiring: Sendable {
        enum Source: Sendable, Hashable { case latents, conditioning, sigma, imageRotary, textMask, output(String) }
        struct Input: Sendable { var role: String; var name: String; var source: Source }
        struct Output: Sendable { var role: String; var name: String; var ref: String }
        struct Segment: Sendable { var role: String; var inputs: [Input]; var outputs: [Output] }

        var text: (role: String, tokens: String, positions: String, conditioning: String)
        var segments: [Segment]
        var decoder: (role: String, latents: String, image: String)
        /// The optional VAE encoder (refine).
        var encoder: (role: String, pixels: String, latents: String)?
        /// The first segment's names for the host-fed inputs, and for its conditioning input.
        var latents: String, sigma: String, imageRotary: String, conditioning: String
        /// `role.name` of the last segment's velocity.
        var velocity: String
        /// The first segment's `textMask` input name, when the segments take the text mask (all four do, or none).
        var textMask: String?

        init(manifest: Manifest, krea2: Manifest.Krea2) throws(AetherError) {
            let textGraph = manifest.graphs["textEncoder"]!, vae = manifest.graphs["vaeDecoder"]!
            let conditioningRef = "textEncoder.\(textGraph.outputs["conditioning"]!.name)"
            text = ("textEncoder", textGraph.inputs["tokens"]!.name, textGraph.inputs["positions"]!.name,
                    textGraph.outputs["conditioning"]!.name)
            decoder = ("vaeDecoder", vae.inputs["latents"]!.name, vae.outputs["image"]!.name)
            encoder = manifest.graphs["vaeEncoder"].map { ("vaeEncoder", $0.inputs["pixels"]!.name, $0.outputs["latents"]!.name) }
            let contracted = Set(DriverContract.krea2.graphs.keys.filter { $0.hasPrefix("denoiser") })
            guard Set(krea2.segments) == contracted, krea2.segments.count == contracted.count,
                  krea2.segments == contracted.sorted() else {
                throw .manifestInvalid(field: "krea2.segments", reason: "must list \(contracted.sorted()) in call order")
            }
            var produced: Set<String> = []
            var segments: [Segment] = []
            for (index, role) in krea2.segments.enumerated() {
                let graph = manifest.graphs[role]!
                var inputs: [Input] = []
                for (inputRole, binding) in graph.inputs.sorted(by: { $0.key < $1.key }) {
                    let field = "graphs.\(role).inputs.\(inputRole)"
                    let source: Source
                    if case .from(let ref)? = binding.source {
                        if ref == conditioningRef, inputRole == "conditioning" {
                            source = .conditioning
                        } else if produced.contains(ref) {
                            source = .output(ref)
                        } else {
                            throw .manifestInvalid(
                                field: "\(field).source", reason: "'\(ref)' is not \(conditioningRef) or an earlier segment's output")
                        }
                    } else {
                        switch inputRole {
                        case "latents": source = .latents
                        case "sigma": source = .sigma
                        case "imageRotary": source = .imageRotary
                        case "textMask": source = .textMask
                        default:
                            throw .manifestInvalid(field: "\(field).source", reason: "must be chained with source.from")
                        }
                    }
                    inputs.append(Input(role: inputRole, name: binding.name, source: source))
                }
                let outputs = graph.outputs.sorted(by: { $0.key < $1.key }).map {
                    Output(role: $0.key, name: $0.value.name, ref: "\(role).\($0.value.name)")
                }
                let emitsVelocity = outputs.contains { $0.role == "velocity" }
                guard emitsVelocity == (index == krea2.segments.count - 1) else {
                    throw .manifestInvalid(field: "krea2.segments", reason: "only the last segment, \(krea2.segments.last!), emits velocity")
                }
                produced.formUnion(outputs.map(\.ref))
                segments.append(Segment(role: role, inputs: inputs, outputs: outputs))
            }
            self.segments = segments
            let first = manifest.graphs[krea2.segments[0]]!
            guard case .from(conditioningRef)? = first.inputs["conditioning"]?.source else {
                throw .manifestInvalid(field: "graphs.\(krea2.segments[0]).inputs.conditioning.source", reason: "must be from \(conditioningRef)")
            }
            latents = first.inputs["latents"]!.name
            sigma = first.inputs["sigma"]!.name
            imageRotary = first.inputs["imageRotary"]!.name
            conditioning = first.inputs["conditioning"]!.name
            velocity = segments.last!.outputs.first { $0.role == "velocity" }!.ref
            // The text mask and the segments' functions: every segment alike.
            let masked = krea2.segments.map { manifest.graphs[$0]!.inputs["textMask"] != nil }
            guard Set(masked).count == 1 else {
                throw .manifestInvalid(field: "graphs.\(krea2.segments[0]).inputs.textMask", reason: "every segment takes the text mask, or none")
            }
            textMask = masked[0] ? first.inputs["textMask"]!.name : nil
        }
    }
}

/// Krea 2's host text mask (`textMask`): 1 for each of the `rows` real conditioning rows, which come first, then 0 for
/// the padding to `length`; Diffusers' `encoder_attention_mask` as the graphs apply it, a key-padding mask.
enum Krea2TextMask {
    static func values(rows: Int, length: Int) -> [Float] {
        [Float](repeating: 1, count: rows) + [Float](repeating: 0, count: max(0, length - rows))
    }
}

/// Krea 2's host image rotary table (MANIFEST.md §5.6 `krea2.rotary`): positions `(0, row, column)` in raster order,
/// `[rows · columns, 2, Σ axes]` float32 (cos rows, then sin rows) in the split-halves layout (pair `i` rotates
/// dimensions `i` and `i + Σ axes / 2`). Per axis of dimension `d`, pair `j` turns at `1 / θ^(2j / d)`; angles, cos and
/// sin are computed in Double and rounded once (0.0.x's `KreaRotary.image`).
enum Krea2Rotary {
    static func image(rows: Int, columns: Int, rotary: Manifest.Krea2.Rotary) -> [Float] {
        let width = rotary.dimension, half = width / 2
        var frequencies: [(axis: Int, value: Double)] = []
        for (axis, dimension) in rotary.axes.enumerated() {
            for j in stride(from: 0, to: dimension, by: 2) {
                frequencies.append((axis, 1 / pow(rotary.theta, Double(j) / Double(dimension))))
            }
        }
        var values = [Float](repeating: 0, count: rows * columns * 2 * width)
        for row in 0..<rows {
            for column in 0..<columns {
                let base = (row * columns + column) * 2 * width
                let position = [0, row, column]
                for (index, frequency) in frequencies.enumerated() {
                    let angle = Double(position[frequency.axis]) * frequency.value
                    let c = Float(cos(angle)), s = Float(sin(angle))
                    values[base + index] = c
                    values[base + index + half] = c
                    values[base + width + index] = s
                    values[base + width + index + half] = s
                }
            }
        }
        return values
    }
}
