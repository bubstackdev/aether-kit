import AetherCoreAI
import AetherKit
import AetherVision
import Foundation

/// Image + text generation over a registered Qwen3-VL bundle (experimental).
///
/// The caller renders the chat template with one `<|vision_start|><|image_pad|><|vision_end|>`
/// placeholder per image (tokenization stays the consumer's) and passes the images in the same
/// order. Each image is preprocessed like the reference processor (smart resize capped at the
/// registration's image-token budget, antialiased bicubic, normalization), encoded by the vision
/// asset, and spliced into the bucketed decoder's prefill. Decoding is greedy.
///
/// One request at a time; overlap fails with `busy`. `unload()` waits for an active request and
/// releases both models. With `gpu`, the encode phase and the decoder's request are each admitted.
public actor VisionLanguageSession {
    public typealias Reply = LanguageGenerationSession.Reply

    private let language: LanguageGenerationSession
    private let encoder: LanguageVisionEncoder
    private let registration: LanguageVisionRegistration
    private let preprocessing: Qwen3VLImagePreprocessor.Configuration
    private let gpu: GPUAdmissionOwner?
    private let lifecycle = LanguageSessionLifecycle()
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

    /// Raw-asset entry point for qualification before registration.
    public static func open(textModel: URL, visionModel: URL, registration: LanguageVisionRegistration, maximumContextLength: Int,
                            stopTokenIDs: Set<Int>, exportBounds: LanguageExportBounds?,
                            gpu: GPUAdmissionOwner? = nil) async throws -> VisionLanguageSession {
        try registration.validate()
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

    private init(language: LanguageGenerationSession, encoder: LanguageVisionEncoder, registration: LanguageVisionRegistration,
                 gpu: GPUAdmissionOwner?) {
        self.language = language
        self.encoder = encoder
        self.registration = registration
        self.gpu = gpu
        var configuration = Qwen3VLImagePreprocessor.Configuration()
        configuration.patchSize = registration.patchSize
        configuration.mergeSize = registration.mergeSize
        configuration.temporalPatchSize = registration.temporalPatchSize
        configuration.minimumPixels = registration.minimumPixels
        // The image-token budget caps the pixels: one token per (patch · merge)² pixels.
        let factor = registration.patchSize * registration.mergeSize
        configuration.maximumPixels = min(registration.maximumPixels, registration.maximumImageTokens * factor * factor)
        configuration.mean = registration.imageMean
        configuration.standardDeviation = registration.imageStandardDeviation
        self.preprocessing = configuration
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
            let processed = try Qwen3VLImagePreprocessor.process(rgb: image.rgb, width: image.width, height: image.height,
                                                                 configuration: preprocessing)
            let pixels = processed.pixels.map { Float16($0) }
            let grant = try await gpu?.admit()
            defer { grant?.release() }
            features.append(try await encoder.encode(pixels: pixels, grid: processed.grid))
        }
        let tokens = LanguageImageTokens(visionStart: registration.visionStartTokenID, imagePad: registration.imageTokenID,
                                         visionEnd: registration.visionEndTokenID)
        return try await language.generate(prompt: LanguageVisionPrompt(renderedTokenIDs: renderedTokenIDs, images: features),
                                           imageTokens: tokens, maximumTokens: maximumTokens, prefillChunkSize: prefillChunkSize,
                                           onToken: onToken)
    }

    /// Terminal close: waits for an active request, then releases the decoder and the encoder.
    public func unload() async {
        guard await lifecycle.beginClose() else { return }
        await language.unload()
        await encoder.unload()
        lifecycle.didClose()
    }
}
