import AetherAssets
import AetherKit
import Foundation

extension LanguageModelManifest {
    /// The portable installation contract for the language bundle at `directory`, for `LocalAssetStore` and catalog
    /// entries (docs/model-preparation.md):
    /// - `language-model.json`;
    /// - the model asset (and the vision asset for image input), each as `main.mlirb`, `main.hash` and `metadata.json`;
    /// - the tokenizer files: exactly the registered `tokenizerFiles` when present, or every regular file in the
    ///   tokenizer directory otherwise.
    ///
    /// Model streams are not reread: their hashes come from the registration, which must agree with each asset's
    /// `main.hash`. Installation verifies every copied byte. Export reports are provenance and are not installed.
    public static func installationManifest(in directory: URL) throws -> AssetManifest {
        let root = directory.standardizedFileURL.resolvingSymlinksInPath()
        let registrationData = try Data(contentsOf: BundleFiles.confined("language-model.json", in: root))
        let manifest = try JSONDecoder().decode(Self.self, from: registrationData)
        _ = try manifest.locate(in: root, platform: manifest.platform)
        do {
            var files = [try BundleFiles.portable("language-model.json", in: root)]
            files += try modelFiles(manifest.modelDirectory, sha256: manifest.modelSHA256, in: root)
            if let vision = manifest.vision {
                files += try modelFiles(vision.modelDirectory, sha256: vision.modelSHA256, in: root)
            }
            files += try tokenizerFiles(manifest, in: root)
            let registrationHash = files[0].sha256
            return try AssetManifest(assetIdentity: "aether-language/1:" + registrationHash,
                                     executionContract: AetherLanguageModel.installationContract, files: files)
        } catch let error as AssetStoreError {
            throw AetherError.invalidModel("\(error)")
        }
    }

    private static func modelFiles(_ relative: String, sha256: String, in root: URL) throws -> [AssetFile] {
        let asset = try DeclaredModelAsset.open(relative, in: root)
        guard asset.declaredSHA256 == sha256 else {
            throw AetherError.invalidModel("\(relative)/main.hash disagrees with the registration")
        }
        return try asset.portableFiles(relativeTo: root)
    }

    private static func tokenizerFiles(_ manifest: Self, in root: URL) throws -> [AssetFile] {
        let directory = try BundleFiles.confined(manifest.tokenizerDirectory, in: root, directory: true)
        let prefix = manifest.tokenizerDirectory + "/"
        if let registered = manifest.tokenizerFiles {
            guard registered["tokenizer.json"] == manifest.tokenizerSHA256 else {
                throw AetherError.invalidModel("tokenizer file list disagrees with tokenizerSHA256")
            }
            return try registered.keys.sorted().map {
                try BundleFiles.portable(prefix + $0, in: root, sha256: registered[$0])
            }
        }
        // Without a registered list, install every regular file the loader could read, hashing each one.
        guard let walk = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey]) else {
            throw AetherError.invalidModel("tokenizer directory unreadable")
        }
        let base = directory.path + "/"
        var paths: [String] = []
        for case let url as URL in walk {
            guard try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else { continue }
            let path = url.standardizedFileURL.resolvingSymlinksInPath().path
            guard path.hasPrefix(base) else { throw AetherError.invalidModel("tokenizer path escapes its directory") }
            let relative = String(path.dropFirst(base.count))
            if relative.split(separator: "/").contains(where: { $0.hasPrefix(".") }) { continue }
            paths.append(relative)
        }
        return try paths.sorted().map { path in
            try BundleFiles.portable(prefix + path, in: root,
                                     sha256: path == "tokenizer.json" ? manifest.tokenizerSHA256 : nil)
        }
    }
}
