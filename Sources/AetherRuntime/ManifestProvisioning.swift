public import Foundation

/// Gets an installed bundle's manifest, synthesizing it once for zoo bundles (DESIGN.md §4).
///
/// If `.aether/aether.json` exists it is the contract. Otherwise the decoder graph's signatures are read, the
/// ``ZooBundleAdapter`` runs with the catalog entry and the override table, the result is validated against the
/// signatures, and the manifest is written to `.aether/`, so inference never runs a heuristic.
public enum BundleManifest {
    public typealias SignatureProvider = (_ graph: URL, _ function: String) async throws -> GraphSignature

    public static func loadOrSynthesize(
        _ bundle: InstalledBundle, catalog: Catalog = .builtin, overrides: ZooOverrides = .builtin,
        signature provider: SignatureProvider
    ) async throws -> Manifest {
        let fm = FileManager.default
        if fm.fileExists(atPath: Manifest.url(inBundle: bundle.url).path) {
            return try Manifest.load(fromBundle: bundle.url)
        }
        var entry = try catalog.entry(bundle.catalogID)
        guard entry.revision == bundle.ref.revision else {
            throw AetherError.manifestInvalid(
                field: Manifest.fileName,
                reason: "a stale install (\(bundle.ref.revision)) without its manifest cannot be re-synthesized at \(entry.revision)")
        }
        entry.variants = entry.variants.filter { $0.key == bundle.ref.variant }
        guard let main = bundle.url(ofAsset: "main") else {
            throw AetherError.manifestInvalid(field: "assets.main", reason: "the install has no main asset")
        }
        let metadataURL = main.appending(path: "metadata.json")
        guard fm.fileExists(atPath: metadataURL.path) else {
            throw AetherError.manifestInvalid(
                field: Manifest.fileName, reason: "the install has neither .aether/aether.json nor a zoo metadata.json")
        }
        let metadata = try Data(contentsOf: metadataURL)
        let request = try ZooBundleAdapter.graphRequest(
            entry: entry, variant: bundle.ref.variant, metadata: metadata, overrides: overrides)
        guard let graphURL = bundle.url(of: request.file) else {
            throw AetherError.manifestInvalid(field: "graphs.decoder.file", reason: "'\(request.file)' is not installed")
        }
        var signatures: [String: GraphSignature] = [:]
        for function in request.functions {
            signatures[function] = try await provider(graphURL, function)
        }
        let manifest = try ZooBundleAdapter.manifest(from: ZooBundleAdapter.Input(
            entry: entry, variant: bundle.ref.variant, metadata: metadata, signatures: signatures,
            tokenizerConfig: try? Data(contentsOf: main.appending(path: "tokenizer/tokenizer_config.json")),
            chatTemplate: try? String(contentsOf: main.appending(path: "tokenizer/chat_template.jinja"), encoding: .utf8),
            overrides: overrides))
        try SignatureValidator.validate(manifest, signatures: ["decoder": signatures])
        try manifest.write(toBundle: bundle.url)
        return manifest
    }

    /// Reads signatures by loading the graph with `options`, which the caller states explicitly (the manifest that
    /// would normally supply them does not exist yet).
    public static func loadOrSynthesize(_ bundle: InstalledBundle, inspectingWith options: GraphOptions) async throws -> Manifest {
        try await loadOrSynthesize(bundle) { url, function in
            try await Graph(url: url, function: function, options: options).signature
        }
    }
}
