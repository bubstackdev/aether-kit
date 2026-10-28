import AetherKit

/// A caller-qualified maximum estimate for one language model asset, keyed by the SHA-256 its registration declares
/// (`modelSHA256`, or the vision registration's for a Qwen3-VL encoder).
@available(macOS 27, iOS 27, *)
public struct LanguageModelResourceEstimate: Sendable {
    public let modelSHA256: String
    /// Held while the model is loaded: its load peak, weights, KV caches at the registered context and, for a decoder,
    /// the memory budget's growth allowance (the budget is what bounds retained memory).
    public let residentBytes: UInt64
    /// Held around each request's native work (an encoder's, around each image's encode).
    public let runScratchBytes: UInt64

    public init(modelSHA256: String, residentBytes: UInt64, runScratchBytes: UInt64) throws {
        guard residentBytes > 0, runScratchBytes > 0 else { throw ResourceAdmissionError.invalidByteCount }
        guard modelSHA256.utf8.count == 64, modelSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw LanguageGenerationError.invalid("A resource estimate needs a lowercase hex SHA-256")
        }
        self.modelSHA256 = modelSHA256
        self.residentBytes = residentBytes
        self.runScratchBytes = runScratchBytes
    }
}

/// Opt-in reservations of caller estimates against a host-selected shared `ResourceCoordinator`, as Krea and FLUX.2
/// have them: each loaded model's residency, reserved before it loads (after its GPU grant, when the session has one)
/// and held until unload, across memory-budget reloads (the old model is released before the new one loads); and each
/// request's scratch, reserved after its grant and held until its native work joins. A model whose bytes have no
/// estimate, a registered context above `maximumContextLength` and a request with more image tokens than
/// `maximumImageTokens` are refused before any native work.
///
/// This is admission of estimates, not a measured footprint or allocator enforcement, and the ledger is per process:
/// it cannot see another process's models (see docs/qwen3-vl-resource-admission.md). It does not cover the page cache
/// a model's compiled file is read into when it opens (clean, evictable pages).
@available(macOS 27, iOS 27, *)
public struct LanguageResourceAdmission: Sendable {
    public let coordinator: ResourceCoordinator
    public let maximumContextLength: Int
    /// The image tokens one request may carry, summed over its images; nil for text-only profiles.
    public let maximumImageTokens: Int?
    private let estimates: [String: LanguageModelResourceEstimate]

    public init(coordinator: ResourceCoordinator, models: [LanguageModelResourceEstimate], maximumContextLength: Int,
                maximumImageTokens: Int? = nil) throws {
        guard !models.isEmpty, (1...16384).contains(maximumContextLength), (maximumImageTokens ?? 1) > 0 else {
            throw LanguageGenerationError.invalid("Invalid language resource profile")
        }
        var keyed: [String: LanguageModelResourceEstimate] = [:]
        for model in models {
            guard keyed.updateValue(model, forKey: model.modelSHA256) == nil else {
                throw LanguageGenerationError.invalid("Duplicate language model resource estimate")
            }
        }
        self.coordinator = coordinator
        self.maximumContextLength = maximumContextLength
        self.maximumImageTokens = maximumImageTokens
        estimates = keyed
    }

    /// The model's budget, or a refusal before any native work.
    func budget(modelSHA256: String, contextLength: Int? = nil) throws -> LanguageModelBudget {
        guard let estimate = estimates[modelSHA256] else {
            throw LanguageGenerationError.invalid("No resource estimate for the selected model bytes")
        }
        if let contextLength, contextLength > maximumContextLength {
            throw LanguageGenerationError.invalid("The registered context exceeds the resource profile's")
        }
        return LanguageModelBudget(coordinator: coordinator, estimate: estimate, maximumImageTokens: maximumImageTokens)
    }
}

/// One model's reservations under a ``LanguageResourceAdmission``. Shared by the sessions and CPU tests.
@available(macOS 27, iOS 27, *)
struct LanguageModelBudget: Sendable {
    let coordinator: ResourceCoordinator
    let estimate: LanguageModelResourceEstimate
    let maximumImageTokens: Int?

    func reserveResidency() throws -> ResourceReservation { try coordinator.reserve(bytes: estimate.residentBytes) }
    func reserveScratch() throws -> ResourceReservation { try coordinator.reserve(bytes: estimate.runScratchBytes) }

    func check(imageTokens: Int) throws {
        guard imageTokens > 0 else { return }
        guard let maximumImageTokens, imageTokens <= maximumImageTokens else {
            throw LanguageGenerationError.invalid("The request's image tokens exceed the resource profile's")
        }
    }
}

/// Opening admitted to the GPU: a session's loads and first specializations wait for a grant, as its requests and
/// reloads do, so a model never specializes beside another owner's GPU work (with a lease-backed admission, another
/// process's render). Cancelling the opening task leaves the queue.
@available(macOS 27, iOS 27, *)
func withOpenAdmission<T>(_ gpu: GPUAdmissionOwner?, isolation: isolated (any Actor)? = #isolation,
                          _ body: () async throws -> T) async throws -> T {
    let grant = try await gpu?.admit()
    defer { grant?.release() }
    try Task.checkCancellation()
    return try await body()
}
