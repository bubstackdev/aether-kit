import AetherAssets
import AetherKit

extension ModelRuntimeCheck {
    /// Language bundles (docs/model-readiness.md). Before downloading: Core AI's OS (macOS 27 or iOS 27) and the language
    /// installation contract. Once verified: ``AetherLanguageModel/availability`` for the installed registration, so a
    /// model readiness calls ready is one `AetherLanguageModel(installation:)` reports available.
    public static var language: ModelRuntimeCheck {
        language(platform: AetherLanguageModel.currentPlatform, coreAIAvailable: AetherLanguageModel.coreAIAvailable)
    }

    /// The language check with the platform and the OS check injected, for CPU tests.
    static func language(platform: LanguageModelManifest.Platform, coreAIAvailable: Bool) -> ModelRuntimeCheck {
        ModelRuntimeCheck(entry: { entry in
            guard coreAIAvailable else { return .unsupported(.operatingSystem) }
            guard entry.manifest.executionContract == AetherLanguageModel.installationContract else {
                return .invalid("the entry is \(entry.manifest.executionContract), not a language bundle")
            }
            return .runnable
        }, installation: { lease in
            switch AetherLanguageModel(installation: lease, platform: platform, coreAIAvailable: coreAIAvailable).availability {
            case .available: .runnable
            case .unavailable(.unsupportedOS): .unsupported(.operatingSystem)
            case .unavailable(.unsupportedPlatform): .unsupported(.platform)
            case let .unavailable(reason): .invalid(String(describing: reason))
            }
        })
    }
}
