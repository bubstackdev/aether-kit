import AetherAssets
import AetherKit
import AetherLanguageRuntime
import AetherLanguageTokenizers
import Foundation

/// An installed language model bundle (a directory with `language-model.json`, registered by Kit's Forge tools), ready to
/// back ``AetherLanguageSession``s, or, through `AetherLanguageFoundationModels`, a FoundationModels
/// `LanguageModelSession`.
///
/// Creating one is cheap: it reads the registration and checks the OS, the platform and the bundle's paths, and never
/// hashes or loads the model. The backend and tokenizer load, fully verified, on ``prewarm()`` or the first request.
/// They stay loaded until ``unload()``, or until the lifecycle policy unloads them: after an idle period, or under memory
/// pressure (docs/model-lifecycle.md). The next request loads them again. Copies share one loaded model and its
/// lifecycle.
public struct AetherLanguageModel: Sendable {
    public enum Availability: Sendable, Equatable {
        case available
        case unavailable(UnavailableReason)
    }

    @nonexhaustive public enum UnavailableReason: Sendable, Equatable {
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
    /// The store installation this model was opened from (``init(installation:platform:)``). Its read lease lives as long
    /// as this model, its copies and the sessions using it, so the store can't remove or replace the files meanwhile.
    public let installation: AssetReadLease?
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

    public init(bundle: URL, platform: LanguageModelManifest.Platform = Self.currentPlatform,
                lifecycle: ModelLifecyclePolicy = .platformDefault) {
        self.init(bundle: bundle, platform: platform, coreAIAvailable: Self.coreAIAvailable, lifecyclePolicy: lifecycle)
    }

    /// A model in a `LocalAssetStore` installation, typically one `ModelPreparation` produced
    /// (docs/model-preparation.md). Acquiring the lease verified every installed byte; the model holds it for its
    /// lifetime. An installation of anything other than a language bundle is unavailable (`bundleInvalid`).
    public init(installation: AssetReadLease, platform: LanguageModelManifest.Platform = Self.currentPlatform,
                lifecycle: ModelLifecyclePolicy = .platformDefault) {
        self.init(installation: installation, platform: platform, coreAIAvailable: Self.coreAIAvailable, lifecyclePolicy: lifecycle)
    }

    init(installation: AssetReadLease, platform: LanguageModelManifest.Platform, coreAIAvailable: Bool,
         lifecyclePolicy: ModelLifecyclePolicy = .platformDefault) {
        self.init(bundle: installation.directory, platform: platform, coreAIAvailable: coreAIAvailable, installation: installation,
                  lifecyclePolicy: lifecyclePolicy)
    }

    /// The backend choice with the OS check injected, so CPU tests on OS 27 reach the unsupported-OS branch.
    init(bundle: URL, platform: LanguageModelManifest.Platform, coreAIAvailable: Bool, installation: AssetReadLease? = nil,
         lifecyclePolicy: ModelLifecyclePolicy = .platformDefault) {
        var readiness = Self.readiness(bundle: bundle, platform: platform, coreAIAvailable: coreAIAvailable)
        if let installation, installation.manifest.executionContract != Self.installationContract,
           readiness.availability != .unavailable(.unsupportedOS) {
            readiness.availability = .unavailable(.bundleInvalid(
                "installation \(installation.installationID) is \(installation.manifest.executionContract), not a language bundle"))
        }
        self.init(bundle: bundle, availability: readiness.availability, supportsReasoning: readiness.manifest?.supportsReasoning ?? false,
                  contextLength: readiness.manifest?.maximumContextLength ?? 0, installation: installation,
                  lifecycle: ModelLifecycle(policy: lifecyclePolicy)) {
            try await Self.loadCoreAI(bundle: bundle, platform: platform)
        }
    }

    /// The execution contract of ``LanguageModelManifest/installationManifest(in:)``.
    static let installationContract = "aether.language-bundle/1"

    /// A model over any backend (CPU tests drive the engine with a fake one).
    init(bundle: URL, availability: Availability, supportsReasoning: Bool, contextLength: Int,
         installation: AssetReadLease? = nil, lifecycle: ModelLifecycle = ModelLifecycle(policy: .platformDefault),
         loader: @escaping LanguageModelResources.Loader) {
        self.bundle = bundle
        self.installation = installation
        self.availability = availability
        self.supportsReasoning = supportsReasoning
        self.contextLength = contextLength
        resources = LanguageModelResources(availability: availability, lifecycle: lifecycle, loader: loader)
    }

    /// Verifies and loads the model now rather than on the first request.
    public func prewarm() async throws { try await resources.prewarm() }

    /// Releases the backend and tokenizer. Waits for a running request; later requests load the model again.
    public func unload() async { await resources.unload() }

    /// When this model unloads by itself.
    public var lifecyclePolicy: ModelLifecyclePolicy { resources.lifecycle.policy }

    /// Whether the model is loaded, loading, serving a request, or unloaded (and why).
    public var lifecycleState: ModelLifecycleState { resources.lifecycle.current }

    /// The lifecycle state, then each change. A slow reader sees the latest state.
    public var lifecycleStates: AsyncStream<ModelLifecycleState> { resources.lifecycle.states }

    /// The OS, then the registration's structural checks and the template, with no hashing. The OS comes first: below
    /// macOS 27 or iOS 27 no backend runs, whatever the bundle holds.
    static func readiness(bundle: URL, platform: LanguageModelManifest.Platform, coreAIAvailable: Bool)
        -> (availability: Availability, manifest: LanguageModelManifest?) {
        guard coreAIAvailable else { return (.unavailable(.unsupportedOS), nil) }
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
        return (.available, manifest)
    }
}
