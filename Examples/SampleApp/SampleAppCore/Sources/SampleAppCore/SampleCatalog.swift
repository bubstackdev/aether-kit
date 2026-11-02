import AetherAssets
import Foundation

/// The catalog entries the app ships: Qwen3-0.6B from the public `aether-models/aether-kit-qwen3-0.6b` repository at a
/// pinned revision, one bundle per platform (written by `make-catalog.sh`).
public enum SampleCatalog {
    /// The platform this build runs on.
    public static var currentPlatform: ModelRequirements.Platform {
        #if os(iOS)
        .ios
        #else
        .macos
        #endif
    }

    public static func resourceName(for platform: ModelRequirements.Platform) -> String {
        platform == .ios ? "qwen3-0.6b-ios" : "qwen3-0.6b-macos"
    }

    /// The shipped entry for `platform`, decoded and validated.
    public static func entry(for platform: ModelRequirements.Platform = currentPlatform) throws -> ModelCatalogEntry {
        let name = resourceName(for: platform)
        guard let url = Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Catalog") else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: "Catalog/\(name).json"])
        }
        let entry = try JSONDecoder().decode(ModelCatalogEntry.self, from: Data(contentsOf: url))
        try entry.validate()
        return entry
    }
}
