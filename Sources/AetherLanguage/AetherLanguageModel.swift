import AetherKit
import AetherLanguageRuntime
import AetherLanguageTokenizers
import Foundation

/// An installed language model bundle (a directory with `language-model.json`, registered by Kit's Forge tools), ready to
/// back ``AetherLanguageSession``s, or, through `AetherLanguageFoundationModels`, a FoundationModels
/// `LanguageModelSession`.
///
/// Creating one is cheap: it reads the registration and checks the OS, the platform and the bundle's paths, and never
/// hashes or loads the model. The backend and tokenizer load, fully verified, on ``prewarm()`` or the first request, and
/// stay loaded until ``unload()``. Copies share one loaded model.
public struct AetherLanguageModel: Sendable {
    public enum Availability: Sendable, Equatable {
        case available
        case unavailable(UnavailableReason)
    }

    public enum UnavailableReason: Sendable, Equatable {
        /// No backend runs on this OS: the Core AI backend needs macOS 27 or iOS 27.
        case unsupportedOS
        /// The registration is for the other platform (for example a macOS export on an iPhone).
        case unsupportedPlatform
        /// No `language-model.json`, or a path it lists is absent.
        case bundleMissing
        /// The registration doesn't parse or fails its checks, or (when loading) an asset fails verification.
        case bundleInvalid(String)
        /// The tokenizer has no chat template, so transcripts can't be rendered.
        case unsupportedTemplate
    }

    public let bundle: URL
    /// Whether this model can serve requests here, from the registration alone (no hashing, no loading).
    public let availability: Availability
    /// Whether the registration declares a thinking mode (``AetherGenerationOptions/Reasoning``).
    public let supportsReasoning: Bool
    /// The registered context: prompt plus output, in tokens. Zero when unavailable.
    public let contextLength: Int
    let resources: LanguageModelResources
    /// Equal for copies of one model, which share its loaded backend.
    package var identity: ObjectIdentifier { ObjectIdentifier(resources) }

    /// The platform registrations are checked against.
    public static var currentPlatform: LanguageModelManifest.Platform {
        #if os(iOS)
        .ios
        #else
        .macos
        #endif
    }

    public init(bundle: URL, platform: LanguageModelManifest.Platform = Self.currentPlatform) {
        let readiness = Self.readiness(bundle: bundle, platform: platform)
        self.init(bundle: bundle, availability: readiness.availability, supportsReasoning: readiness.manifest?.supportsReasoning ?? false,
                  contextLength: readiness.manifest?.maximumContextLength ?? 0) {
            try await Self.loadCoreAI(bundle: bundle, platform: platform)
        }
    }

    /// A model over any backend (CPU tests drive the engine with a fake one).
    init(bundle: URL, availability: Availability, supportsReasoning: Bool, contextLength: Int,
         loader: @escaping LanguageModelResources.Loader) {
        self.bundle = bundle
        self.availability = availability
        self.supportsReasoning = supportsReasoning
        self.contextLength = contextLength
        resources = LanguageModelResources(availability: availability, loader: loader)
    }

    /// Verifies and loads the model now rather than on the first request.
    public func prewarm() async throws { _ = try await resources.loaded() }

    /// Releases the backend and tokenizer. Waits for a running request; later requests load the model again.
    public func unload() async { await resources.unload() }

    /// The registration's structural checks, the OS and the template, with no hashing.
    static func readiness(bundle: URL, platform: LanguageModelManifest.Platform)
        -> (availability: Availability, manifest: LanguageModelManifest?) {
        guard FileManager.default.fileExists(atPath: bundle.appendingPathComponent("language-model.json").path) else {
            return (.unavailable(.bundleMissing), nil)
        }
        let manifest: LanguageModelManifest
        do { manifest = try LanguageModelManifest.load(from: bundle) }
        catch { return (.unavailable(.bundleInvalid("language-model.json does not parse: \(error)")), nil) }
        guard manifest.platform == platform else { return (.unavailable(.unsupportedPlatform), manifest) }
        let tokenizer: URL
        do { tokenizer = try manifest.locate(in: bundle, platform: platform).tokenizer }
        catch AetherError.invalidModel(let reason) where reason.contains("missing") { return (.unavailable(.bundleMissing), manifest) }
        catch { return (.unavailable(.bundleInvalid(String(describing: error))), manifest) }
        guard (try? LanguageTokenizer.template(in: tokenizer)) != nil else { return (.unavailable(.unsupportedTemplate), manifest) }
        guard Self.coreAIAvailable else { return (.unavailable(.unsupportedOS), manifest) }
        return (.available, manifest)
    }
}
