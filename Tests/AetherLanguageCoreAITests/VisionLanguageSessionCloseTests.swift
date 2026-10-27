import AetherKit
@testable import AetherLanguageCoreAI
import AetherVision
import Foundation
import Testing

/// Records what the session asked of its encoder.
private actor FakeEncoder: VisionLanguageEncoding {
    private(set) var encodes = 0
    private(set) var unloads = 0
    private let onEncode: @Sendable () async -> Void
    init(onEncode: @escaping @Sendable () async -> Void = {}) { self.onEncode = onEncode }

    func encode(pixels: [Float16], grid: (t: Int, h: Int, w: Int)) async throws -> LanguageImageFeatures {
        encodes += 1
        await onEncode()
        let tokens = grid.t * (grid.h / 2) * (grid.w / 2)
        return try LanguageImageFeatures(grid: grid, mergeSize: 2, hiddenSize: 1, deepstackLevels: 1,
                                         embeddings: Array(repeating: 0, count: tokens), deepstack: Array(repeating: 0, count: tokens))
    }
    func unload() async { unloads += 1 }
}

/// Emulates the language session's own GPU admission (queue, withdrawal and its latch).
private actor FakeDecoder: VisionLanguageDecoding {
    private let gpu: GPUAdmissionOwner
    private var pending: GPUAdmissionAttempt?
    private var withdrawn = false
    private(set) var waiting = false
    private(set) var unloads = 0
    init(gpu: GPUAdmissionOwner) { self.gpu = gpu }

    func generate(prompt: LanguageVisionPrompt, imageTokens: LanguageImageTokens, maximumTokens: Int, prefillChunkSize: Int?,
                  onToken: (@Sendable (Int) async throws -> Void)?, observer: LanguageGenerationSession.Observer?) async throws
        -> LanguageGenerationSession.Reply {
        guard !withdrawn else { throw LanguageGenerationError.closed }
        let attempt = GPUAdmissionAttempt(gpu)
        pending = attempt
        waiting = true
        defer { pending = nil; waiting = false }
        let grant = try await attempt.admit(orThrow: { LanguageGenerationError.closed })
        grant.release()
        throw LanguageGenerationError.invalid("The fake decoder only exercises admission")
    }
    func withdrawPendingAdmission() async { withdrawn = true; pending?.withdraw() }
    func unload() async { unloads += 1 }
}

private actor GrantBox {
    var grant: GPUAdmissionGrant?
    func set(_ g: GPUAdmissionGrant) { grant = g }
    func release() { grant?.release(); grant = nil }
}

private func registration(maximumImageTokens: Int = 64) -> LanguageVisionRegistration {
    LanguageVisionRegistration(modelDirectory: "vision.aimodel", modelSHA256: String(repeating: "a", count: 64), imageTokenID: 151655,
                               visionStartTokenID: 151652, visionEndTokenID: 151653, maximumImageTokens: maximumImageTokens)
}

private let prompt = [151644, 151652, 151655, 151653, 77091]
private func image() throws -> VisionRGBImage { try VisionRGBImage(rgb: Array(repeating: 128, count: 64 * 64 * 3), width: 64, height: 64) }

private func waitUntil(_ condition: @Sendable () async -> Bool) async -> Bool {
    for _ in 0..<1000 { if await condition() { return true }; try? await Task.sleep(for: .milliseconds(5)) }
    return false
}

private func closedWithin(_ limit: Duration, _ session: VisionLanguageSession) async -> Bool {
    let deadline = ContinuousClock.now + limit
    while ContinuousClock.now < deadline {
        if await session.state == .closed { return true }
        try? await Task.sleep(for: .milliseconds(5))
    }
    return false
}

struct VisionLanguageSessionCloseTests {
    /// Another owner holds the GPU while an image prompt waits for encode admission: unload withdraws
    /// it and returns promptly, the prompt fails with `closed`, and the encoder never runs.
    @Test(.timeLimit(.minutes(1))) func unloadWithdrawsAPromptWaitingForEncodeAdmission() async throws {
        let admission = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())       // private, in-process: no lease
        let blocker = try await admission.admit(owner: "other")
        let encoder = FakeEncoder()
        let decoder = FakeDecoder(gpu: admission.owner("decoder"))
        let session = VisionLanguageSession(language: decoder, encoder: encoder, registration: registration(), gpu: admission.owner("vl"))
        let request = Task { try await session.generate(renderedTokenIDs: prompt, images: [try image()], maximumTokens: 4) }
        #expect(await waitUntil { admission.snapshot.queued == 1 })
        let closer = Task { await session.unload() }
        // Fail fast rather than hang on a regression: if close is stuck behind the GPU queue, free it.
        let closedPromptly = await closedWithin(.seconds(2), session)
        #expect(closedPromptly)
        if !closedPromptly { blocker.release() }
        await closer.value
        await #expect(throws: LanguageGenerationError.closed) { try await request.value }
        #expect(await encoder.encodes == 0)
        #expect(await encoder.unloads == 1)
        #expect(await decoder.unloads == 1)
        #expect(admission.snapshot.queued == 0)
        if closedPromptly {
            #expect(admission.snapshot.running == 1)                  // only the other owner remains
            blocker.release()
        }
    }

    /// The prompt is past encoding and waits at the decoder's own admission behind another owner:
    /// unload withdraws that one too before waiting for the request.
    @Test(.timeLimit(.minutes(1))) func unloadWithdrawsAPromptWaitingForDecoderAdmission() async throws {
        let admission = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        let other = GrantBox()
        // While the encode grant is held, another owner queues; it is granted when the encode grant is
        // released, ahead of the decoder's request.
        let encoder = FakeEncoder(onEncode: {
            Task { if let g = try? await admission.admit(owner: "other") { await other.set(g) } }
            _ = await waitUntil { admission.snapshot.queued == 1 }
        })
        let decoder = FakeDecoder(gpu: admission.owner("decoder"))
        let session = VisionLanguageSession(language: decoder, encoder: encoder, registration: registration(), gpu: admission.owner("vl"))
        let request = Task { try await session.generate(renderedTokenIDs: prompt, images: [try image()], maximumTokens: 4) }
        #expect(await waitUntil { await decoder.waiting && admission.snapshot.queued == 1 && admission.snapshot.running == 1 })
        let closer = Task { await session.unload() }
        let closedPromptly = await closedWithin(.seconds(2), session)
        #expect(closedPromptly)
        if !closedPromptly { await other.release() }                 // free a stuck close so the test ends
        await closer.value
        await #expect(throws: LanguageGenerationError.closed) { try await request.value }
        #expect(await encoder.encodes == 1)
        #expect(await decoder.unloads == 1)
        #expect(admission.snapshot.queued == 0)
        await other.release()
        #expect(admission.snapshot.running == 0)
    }

    @Test func imageInputRequiresAnImageChunkAndRefusesOversizedImages() throws {
        let bounds = LanguageExportBounds(maximumPrefillChunk: 512, maximumCacheCapacity: 4096, maximumImageChunk: 64)
        #expect(try LanguageStepContract.imageChunkLimit(acceptsImages: true, bounds: bounds) == 64)
        #expect(try LanguageStepContract.imageChunkLimit(acceptsImages: false, bounds: nil) == nil)
        #expect(throws: LanguageGenerationError.self) {
            try LanguageStepContract.imageChunkLimit(acceptsImages: true,
                                                     bounds: LanguageExportBounds(maximumPrefillChunk: 512, maximumCacheCapacity: 4096))
        }
        // An image of 16x16 merged tokens (256) exceeds a 64-token chunk: refused while planning.
        let features = try LanguageImageFeatures(grid: (1, 32, 32), mergeSize: 2, hiddenSize: 1, deepstackLevels: 0,
                                                 embeddings: Array(repeating: 0, count: 256), deepstack: [])
        let plan = try LanguagePromptPlan.expand(LanguageVisionPrompt(renderedTokenIDs: prompt, images: [features]),
                                                 tokens: LanguageImageTokens(visionStart: 151652, imagePad: 151655, visionEnd: 151653))
        #expect(throws: LanguageGenerationError.self) { try plan.requireImageSpans(within: 64) }
        try plan.requireImageSpans(within: 256)
        // The raw open path needs an image chunk covering the registration's budget.
        try VisionLanguageSession.requireImageBudget(registration: registration(maximumImageTokens: 64), bounds: bounds)
        #expect(throws: LanguageGenerationError.self) {
            try VisionLanguageSession.requireImageBudget(registration: registration(maximumImageTokens: 128), bounds: bounds)
        }
        #expect(throws: LanguageGenerationError.self) {
            try VisionLanguageSession.requireImageBudget(registration: registration(),
                                                         bounds: LanguageExportBounds(maximumPrefillChunk: 512, maximumCacheCapacity: 4096))
        }
    }

    /// The token budget must cover the minimum pixel bound, so smart resize never upscales past the cap.
    @Test func registrationBudgetCoversTheMinimumPixels() throws {
        try registration(maximumImageTokens: 64).validate()                    // 64 · 32² = 65,536 = the minimum
        #expect(throws: AetherError.self) { try registration(maximumImageTokens: 63).validate() }
        let configuration = VisionLanguageSession.preprocessing(for: registration(maximumImageTokens: 64))
        let smallest = try Qwen3VLImagePreprocessor.targetSize(width: 16, height: 16, configuration: configuration)
        #expect(smallest.width * smallest.height <= configuration.maximumPixels)
    }
}
