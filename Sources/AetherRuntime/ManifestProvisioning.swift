public import Foundation

/// Gets an installed bundle's manifest, synthesizing it once for zoo bundles (DESIGN.md §4).
///
/// If `aether.json` exists it is the contract. Otherwise, for a zoo bundle (`metadata.json`), the graph's
/// signature is read, the ``ZooBundleAdapter`` runs, the result is validated against the signature, and the
/// manifest is written next to the bundle, so the heuristics run once per install and never at inference time.
public enum BundleManifest {
    public typealias SignatureProvider = (_ graph: URL, _ function: String) async throws -> GraphSignature

    public static func loadOrSynthesize(
        _ bundle: InstalledBundle, signature provider: SignatureProvider
    ) async throws -> Manifest {
        let root = bundle.url
        let fm = FileManager.default
        if fm.fileExists(atPath: root.appending(path: Manifest.fileName).path) {
            return try Manifest.load(fromBundle: root)
        }
        let metadataURL = root.appending(path: "metadata.json")
        guard fm.fileExists(atPath: metadataURL.path) else {
            throw AetherError.manifestInvalid(field: Manifest.fileName, reason: "the bundle has neither aether.json nor metadata.json")
        }
        let metadata = try Data(contentsOf: metadataURL)
        let request = try ZooBundleAdapter.graphRequests(metadata: metadata)
        var signatures: [String: GraphSignature] = [:]
        for function in request.functions {
            signatures[function] = try await provider(root.appending(path: request.file), function)
        }
        let input = ZooBundleAdapter.Input(
            metadata: metadata,
            signatures: signatures,
            files: try relativeFiles(in: root),
            tokenizerConfig: try? Data(contentsOf: root.appending(path: "tokenizer/tokenizer_config.json")),
            chatTemplate: try? String(contentsOf: root.appending(path: "tokenizer/chat_template.jinja"), encoding: .utf8),
            catalogID: bundle.catalogID,
            repo: bundle.ref.repo,
            revision: bundle.ref.revision)
        let manifest = try ZooBundleAdapter.manifest(from: input)
        let byRole = Dictionary(uniqueKeysWithValues: manifest.graphs.compactMap { role, graph in
            signatures[graph.function].map { (role, $0) }
        })
        try SignatureValidator.validate(manifest, signatures: byRole)
        try manifest.write(toBundle: root)
        return manifest
    }

    /// Reads signatures by loading the graph with `options`, which the caller states explicitly (the manifest that
    /// would normally supply them does not exist yet).
    public static func loadOrSynthesize(_ bundle: InstalledBundle, inspectingWith options: GraphOptions) async throws -> Manifest {
        try await loadOrSynthesize(bundle) { url, function in
            try await Graph(url: url, function: function, options: options).signature
        }
    }

    /// Every regular file under `root`, relative to it, excluding Aether's own bookkeeping files.
    static func relativeFiles(in root: URL) throws -> [String] {
        let base = root.standardizedFileURL.resolvingSymlinksInPath().path + "/"
        guard let walker = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.isRegularFileKey], options: [])
        else { return [] }
        var files: [String] = []
        for case let url as URL in walker {
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true else { continue }
            let path = url.standardizedFileURL.resolvingSymlinksInPath().path
            guard path.hasPrefix(base) else { continue }
            let relative = String(path.dropFirst(base.count))
            if relative == Manifest.fileName || relative.hasPrefix(".aether") { continue }
            files.append(relative)
        }
        return files.sorted()
    }
}
