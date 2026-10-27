import AetherCoreAI
import AetherKit
import AetherVision
import Foundation

/// The encoder a ``VisionLanguageSession`` drives (``LanguageVisionEncoder``; fakes in CPU tests).
protocol VisionLanguageEncoding: Sendable {
    func encode(pixels: [Float16], grid: (t: Int, h: Int, w: Int)) async throws -> LanguageImageFeatures
    func unload() async
}

/// The decoder a ``VisionLanguageSession`` drives (``LanguageGenerationSession``; fakes in CPU tests).
protocol VisionLanguageDecoding: Sendable {
    func generate(prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens, maximumTokens: Int, prefillChunkSize: Int?,
                  onToken: (@Sendable (Int) async throws -> Void)?, observer: LanguageGenerationSession.Observer?) async throws
        -> LanguageGenerationSession.Reply
    func withdrawPendingAdmission() async
    func unload() async
}

extension LanguageVisionEncoder: VisionLanguageEncoding {}
extension LanguageGenerationSession: VisionLanguageDecoding {}

/// Image + text generation over a registered Qwen3-VL bundle (experimental).
///
/// The caller renders the chat template with one `<|vision_start|><|image_pad|><|vision_end|>`
/// placeholder per image (tokenization stays the consumer's) and passes the images in the same
/// order. Each image is preprocessed like the reference processor (smart resize capped at the
/// registration's image-token budget, antialiased bicubic, normalization), encoded by the vision
/// asset, and spliced into the bucketed decoder's prefill. Decoding is greedy.
///
/// One request at a time; overlap fails with `busy`. With `gpu`, the encode phase and the
/// decoder's request are each admitted. `unload()` first withdraws a request still waiting for
/// either admission (it fails with `closed`), so close never waits behind the GPU queue; an
/// admitted request finishes first. Then both models are released.
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
    public static func open(bundle directory: URL, platform: LanguageModelManifest.Platform,
                            gpu: GPUAdmissionOwner? = nil) async throws -> VisionLanguageSession {
        let manifest = try LanguageModelManifest.load(from: directory)
        let textModel = try manifest.verify(in: directory, platform: platform)
        guard let registration = manifest.vision, let visionModel = try manifest.verifyVision(in: directory) else {
            throw LanguageGenerationError.invalid("The registration has no image input")
        }
        return try await open(textModel: textModel, visionModel: visionModel, registration: registration,
                              maximumContextLength: manifest.maximumContextLength, stopTokenIDs: Set(manifest.stopTokenIDs),
                              exportBounds: manifest.exportBounds, gpu: gpu)
    }

    /// Raw-asset entry point for qualification before registration. The export bounds must
    /// declare an image chunk covering the registration's image-token budget.
    public static func open(textModel: URL, visionModel: URL, registration: LanguageVisionRegistration, maximumContextLength: Int,
                            stopTokenIDs: Set<Int>, exportBounds: LanguageExportBounds?,
                            gpu: GPUAdmissionOwner? = nil) async throws -> VisionLanguageSession {
        try registration.validate()
        try Self.requireImageBudget(registration: registration, bounds: exportBounds)
        let encoder = try await LanguageVisionEncoder.open(at: visionModel,
            configuration: .init(mergeSize: registration.mergeSize, positionTableSide: registration.positionTableSide),
            expectFrequentReshapes: try registration.encoderExpectsFrequentReshapes())
        do {
            let language = try await LanguageGenerationSession.open(at: textModel, maximumContextLength: maximumContextLength,
                                                                    stopTokenIDs: stopTokenIDs, exportBounds: exportBounds, gpu: gpu)
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

    /// The registration's preprocessing, with pixels capped to its image-token budget: one token
    /// per (patch · merge)² pixels.
    static func preprocessing(for registration: LanguageVisionRegistration) -> Qwen3VLImagePreprocessor.Configuration {
        var configuration = Qwen3VLImagePreprocessor.Configuration()
        configuration.patchSize = registration.patchSize
        configuration.mergeSize = registration.mergeSize
        configuration.temporalPatchSize = registration.temporalPatchSize
        configuration.minimumPixels = registration.minimumPixels
        // `validate()` refuses an overflowing budget before a session opens.
        configuration.maximumPixels = min(registration.maximumPixels, registration.imageTokenPixelBudget ?? registration.maximumPixels)
        configuration.mean = registration.imageMean
        configuration.standardDeviation = registration.imageStandardDeviation
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
        var features: [LanguageImageFeatures] = []
        for image in images {
            try Task.checkCancellation()
            // Double-precision resampling runs off the actor.
            let processed = try await Self.process(image, configuration: preprocessing)
            guard !closeRequested else { throw LanguageGenerationError.closed }
            let grant = try await admitEncode()
            defer { grant?.release() }
            features.append(try await encoder.encode(pixels: processed.pixels, grid: processed.grid))
        }
        guard !closeRequested else { throw LanguageGenerationError.closed }
        let tokens = LanguageImageTokens(visionStart: registration.visionStartTokenID, imagePad: registration.imageTokenID,
                                         visionEnd: registration.visionEndTokenID)
        return try await language.generate(prompt: LanguageVisionPrompt(renderedTokenIDs: renderedTokenIDs, images: features),
                                           imageTokens: tokens, maximumTokens: maximumTokens, prefillChunkSize: prefillChunkSize,
                                           onToken: onToken, observer: nil)
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
