import AetherCoreAI
import AetherKit
import AetherVision
import Foundation

/// The encoder a ``VisionLanguageSession`` drives (``LanguageVisionEncoder``; fakes in CPU tests).
@available(macOS 27, iOS 27, *)
protocol VisionLanguageEncoding: Sendable {
    func encode(pixels: [Float16], grid: (t: Int, h: Int, w: Int)) async throws -> LanguageImageFeatures
    func unload() async
}

/// The decoder a ``VisionLanguageSession`` drives (``LanguageGenerationSession``; fakes in CPU tests).
@available(macOS 27, iOS 27, *)
protocol VisionLanguageDecoding: Sendable {
    func generate(prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens, maximumTokens: Int, prefillChunkSize: Int?,
                  onToken: (@Sendable (Int) async throws -> Void)?, observer: LanguageGenerationSession.Observer?) async throws
        -> LanguageGenerationSession.Reply
    func generate(prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens, maximumTokens: Int, sampling: LanguageSampling,
                  constraint: (any LanguageLogitsProcessor)?, prefillChunkSize: Int?, observer: LanguageGenerationSession.Observer?,
                  onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?) async throws
        -> LanguageGenerationResult
    func withdrawPendingAdmission() async
    func unload() async
    /// The decoder's cache and reload statistics; nil for decoders without them.
    func decoderStatistics() async -> LanguageGenerationSession.CacheStatistics?
}

@available(macOS 27, iOS 27, *)
extension VisionLanguageDecoding {
    func decoderStatistics() async -> LanguageGenerationSession.CacheStatistics? { nil }

    /// For decoders that only decode greedily (CPU-test fakes): the default request without delivery, through the
    /// greedy entry point.
    func generate(prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens, maximumTokens: Int, sampling: LanguageSampling,
                  constraint: (any LanguageLogitsProcessor)?, prefillChunkSize: Int?, observer: LanguageGenerationSession.Observer?,
                  onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?) async throws
        -> LanguageGenerationResult {
        guard sampling == .greedy, constraint == nil, onToken == nil else {
            throw LanguageGenerationError.invalid("This decoder only decodes greedily, without a constraint or delivery")
        }
        let reply: LanguageGenerationSession.Reply = try await generate(prompt: prompt, imageTokens: imageTokens,
            maximumTokens: maximumTokens, prefillChunkSize: prefillChunkSize, onToken: nil, observer: observer)
        return LanguageGenerationResult(tokens: reply.tokens, stopReason: LanguageStopReason(rawValue: reply.stopReason) ?? .outputLimit,
                                        seed: nil, timeToFirstTokenSeconds: reply.timeToFirstTokenSeconds, totalSeconds: reply.totalSeconds)
    }
}

@available(macOS 27, iOS 27, *)
extension LanguageVisionEncoder: VisionLanguageEncoding {}
@available(macOS 27, iOS 27, *)
extension LanguageGenerationSession: VisionLanguageDecoding {
    func decoderStatistics() async -> CacheStatistics? { cacheStatistics() }
}

/// Image + text generation over a registered Qwen3-VL bundle (experimental).
///
/// The caller renders the chat template with one `<|vision_start|><|image_pad|><|vision_end|>`
/// placeholder per image (tokenization stays the consumer's) and passes the images in the same
/// order. Each image snaps to the registration's fixed patch grid nearest its aspect ratio, is
/// resized to it exactly (antialiased bicubic, like the reference processor) and normalized,
/// encoded by that grid's static-shape encoder function, and spliced into the bucketed decoder's
/// prefill. Decoding is greedy unless a request asks for sampling or a constraint, which work as for a text
/// request (``LanguageSampling``, ``LanguageLogitsProcessor``).
///
/// One request at a time; overlap fails with `busy`. With `gpu`, the encode phase and the
/// decoder's request are each admitted. `unload()` first withdraws a request still waiting for
/// either admission (it fails with `closed`), so close never waits behind the GPU queue; an
/// admitted request finishes first. Then both models are released.
@available(macOS 27, iOS 27, *)
public actor VisionLanguageSession {
    public typealias Reply = LanguageGenerationSession.Reply

    private let language: any VisionLanguageDecoding
    private let encoder: any VisionLanguageEncoding
    private let registration: LanguageVisionRegistration
    private let preprocessing: Qwen3VLImagePreprocessor.Configuration
    private let gpu: GPUAdmissionOwner?
    private let lifecycle = LanguageSessionLifecycle()
    private var pendingEncodeAdmission: GPUAdmissionAttempt?
    private var closeRequested = false
    public var state: LanguageGenerationSession.State { lifecycle.state }

    /// Opens a registered bundle whose `language-model.json` has a `vision` registration. No
    /// download or fallback; both assets are verified first.
    /// `memoryGrowthBudgetBytes` is the decoder's (``LanguageGenerationSession``'s memory budget).
    public static func open(bundle directory: URL, platform: LanguageModelManifest.Platform,
                            memoryGrowthBudgetBytes: Int? = LanguageGenerationSession.defaultMemoryGrowthBudgetBytes,
                            gpu: GPUAdmissionOwner? = nil) async throws -> VisionLanguageSession {
        let manifest = try LanguageModelManifest.load(from: directory)
        let textModel = try manifest.verify(in: directory, platform: platform)
        guard let registration = manifest.vision, let visionModel = try manifest.verifyVision(in: directory) else {
            throw LanguageGenerationError.invalid("The registration has no image input")
        }
        return try await open(textModel: textModel, visionModel: visionModel, registration: registration,
                              maximumContextLength: manifest.maximumContextLength, stopTokenIDs: Set(manifest.stopTokenIDs),
                              exportBounds: manifest.exportBounds, memoryGrowthBudgetBytes: memoryGrowthBudgetBytes, gpu: gpu)
    }

    /// Raw-asset entry point for qualification before registration. The export bounds must
    /// declare an image chunk covering the registration's image-token budget.
    public static func open(textModel: URL, visionModel: URL, registration: LanguageVisionRegistration, maximumContextLength: Int,
                            stopTokenIDs: Set<Int>, exportBounds: LanguageExportBounds?,
                            memoryGrowthBudgetBytes: Int? = LanguageGenerationSession.defaultMemoryGrowthBudgetBytes,
                            gpu: GPUAdmissionOwner? = nil) async throws -> VisionLanguageSession {
        try registration.validate()
        try Self.requireImageBudget(registration: registration, bounds: exportBounds)
        let encoder = try await LanguageVisionEncoder.open(at: visionModel,
            configuration: .init(mergeSize: registration.mergeSize, positionTableSide: registration.positionTableSide,
                                 imageGrids: Self.grids(of: registration)),
            expectFrequentReshapes: try registration.encoderExpectsFrequentReshapes())
        do {
            let language = try await LanguageGenerationSession.open(at: textModel, maximumContextLength: maximumContextLength,
                                                                    stopTokenIDs: stopTokenIDs, memoryGrowthBudgetBytes: memoryGrowthBudgetBytes,
                                                                    exportBounds: exportBounds, gpu: gpu)
            guard language.acceptsImages else {
                await language.unload()
                throw LanguageGenerationError.invalid("The text model has no image entry point")
            }
            return VisionLanguageSession(language: language, encoder: encoder, registration: registration, gpu: gpu)
        } catch {
            await encoder.unload()
            throw error
        }
    }

    /// The registration's fixed grids in the preprocessor's terms (empty when it lists none, which
    /// `validate()` refuses).
    static func grids(of registration: LanguageVisionRegistration) -> [Qwen3VLImagePreprocessor.Grid] {
        (registration.imageGrids ?? []).map { Qwen3VLImagePreprocessor.Grid(height: $0.height, width: $0.width) }
    }

    /// Planning check: an image's patch grid must be one of the registration's fixed grids, so the
    /// encoder and the decoder's image submissions only ever see those shapes.
    static func requireRegisteredGrid(_ grid: (t: Int, h: Int, w: Int), registration: LanguageVisionRegistration) throws {
        guard grid.t == 1, registration.imageGrids?.contains(LanguageImageGrid(height: grid.h, width: grid.w)) == true else {
            throw LanguageGenerationError.invalid("Image grid \(grid.t)x\(grid.h)x\(grid.w) is not one of the registration's fixed grids")
        }
    }

    /// The decoder's image chunk must cover the largest image the registration admits.
    static func requireImageBudget(registration: LanguageVisionRegistration, bounds: LanguageExportBounds?) throws {
        guard let chunk = bounds?.maximumImageChunk, registration.maximumImageTokens <= chunk else {
            throw LanguageGenerationError.invalid("The export bounds must declare an image chunk covering the image-token budget")
        }
    }

    init(language: any VisionLanguageDecoding, encoder: any VisionLanguageEncoding, registration: LanguageVisionRegistration,
         gpu: GPUAdmissionOwner?) {
        self.language = language
        self.encoder = encoder
        self.registration = registration
        self.gpu = gpu
        self.preprocessing = Self.preprocessing(for: registration)
    }

    /// The registration's preprocessing: its fixed grids (every image snaps to one), with the
    /// smart-resize pixel bounds capped to the image-token budget kept for reference.
    public static func preprocessing(for registration: LanguageVisionRegistration) -> Qwen3VLImagePreprocessor.Configuration {
        var configuration = Qwen3VLImagePreprocessor.Configuration()
        configuration.patchSize = registration.patchSize
        configuration.mergeSize = registration.mergeSize
        configuration.temporalPatchSize = registration.temporalPatchSize
        configuration.minimumPixels = registration.minimumPixels
        // `validate()` refuses an overflowing budget before a session opens.
        configuration.maximumPixels = min(registration.maximumPixels, registration.imageTokenPixelBudget ?? registration.maximumPixels)
        configuration.mean = registration.imageMean
        configuration.standardDeviation = registration.imageStandardDeviation
        configuration.imageGrids = grids(of: registration)
        return configuration
    }

    /// The preprocessing applied to every image (the registration's, with its token budget).
    public var imagePreprocessing: Qwen3VLImagePreprocessor.Configuration { preprocessing }

    /// Greedy generation from a rendered prompt and its images.
    public func generate(renderedTokenIDs: [Int], images: [VisionRGBImage], maximumTokens: Int = 256,
                         prefillChunkSize: Int? = nil, onToken: (@Sendable (Int) async throws -> Void)? = nil) async throws -> Reply {
        try Task.checkCancellation()
        try lifecycle.admit()
        defer { lifecycle.finish() }
        let prompt = try await encode(renderedTokenIDs: renderedTokenIDs, images: images)
        return try await language.generate(prompt: prompt, imageTokens: imageTokens, maximumTokens: maximumTokens,
                                           prefillChunkSize: prefillChunkSize, onToken: onToken, observer: nil)
    }

    /// Generation from a rendered prompt and its images with sampling and a constraint, as a text request has them
    /// (``LanguageGenerationRequest``). `.greedy` without a constraint gives the greedy entry point's tokens. The
    /// sampler and the processor see the expanded prompt (each image's pad token repeated to its token count).
    public func generate(renderedTokenIDs: [Int], images: [VisionRGBImage], maximumTokens: Int = 256,
                         sampling: LanguageSampling, constraint: (any LanguageLogitsProcessor)? = nil, prefillChunkSize: Int? = nil,
                         onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)? = nil) async throws
        -> LanguageGenerationResult {
        try Task.checkCancellation()
        try lifecycle.admit()
        defer { lifecycle.finish() }
        let prompt = try await encode(renderedTokenIDs: renderedTokenIDs, images: images)
        return try await language.generate(prompt: prompt, imageTokens: imageTokens, maximumTokens: maximumTokens, sampling: sampling,
                                           constraint: constraint, prefillChunkSize: prefillChunkSize, observer: nil, onToken: onToken)
    }

    private var imageTokens: LanguageImageTokens {
        LanguageImageTokens(visionStart: registration.visionStartTokenID, imagePad: registration.imageTokenID,
                            visionEnd: registration.visionEndTokenID)
    }

    /// Preprocesses and encodes each image (each encode admitted on its own) for an admitted request.
    private func encode(renderedTokenIDs: [Int], images: [VisionRGBImage]) async throws -> LanguageVisionPrompt {
        var features: [LanguageImageFeatures] = []
        for image in images {
            try Task.checkCancellation()
            // Double-precision resampling runs off the actor.
            let processed = try await Self.process(image, configuration: preprocessing)
            guard !closeRequested else { throw LanguageGenerationError.closed }
            try Self.requireRegisteredGrid(processed.grid, registration: registration)
            let grant = try await admitEncode()
            defer { grant?.release() }
            let encoded = try await encoder.encode(pixels: processed.pixels, grid: processed.grid)
            try Self.requireRegisteredGrid(encoded.grid, registration: registration)
            features.append(encoded)
        }
        guard !closeRequested else { throw LanguageGenerationError.closed }
        return LanguageVisionPrompt(renderedTokenIDs: renderedTokenIDs, images: features)
    }

    /// The decoder's cache statistics, including its model reloads and the memory its requests
    /// retain against the budget (``LanguageGenerationSession/cacheStatistics()``).
    public func decoderStatistics() async -> LanguageGenerationSession.CacheStatistics? {
        await language.decoderStatistics()
    }

    /// Terminal close: withdraws a request waiting for GPU admission (encode or decode), waits for
    /// an admitted request to finish, then releases the decoder and the encoder.
    public func unload() async {
        // Mark the close before any suspension, so a request between phases stops at its next check.
        closeRequested = true
        pendingEncodeAdmission?.withdraw()
        await language.withdrawPendingAdmission()
        guard await lifecycle.beginClose() else { return }
        await language.unload()
        await encoder.unload()
        lifecycle.didClose()
    }

    private func admitEncode() async throws -> GPUAdmissionGrant? {
        guard let gpu else { return nil }
        let attempt = GPUAdmissionAttempt(gpu)
        pendingEncodeAdmission = attempt
        defer { pendingEncodeAdmission = nil }
        return try await attempt.admit(orThrow: { LanguageGenerationError.closed })
    }

    @concurrent private nonisolated static func process(_ image: VisionRGBImage, configuration: Qwen3VLImagePreprocessor.Configuration) async throws
        -> (pixels: [Float16], grid: (t: Int, h: Int, w: Int)) {
        let processed = try Qwen3VLImagePreprocessor.process(rgb: image.rgb, width: image.width, height: image.height,
                                                             configuration: configuration)
        return (processed.pixels.map { Float16($0) }, processed.grid)
    }
}
