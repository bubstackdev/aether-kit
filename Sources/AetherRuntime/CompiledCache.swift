public import AetherCore
import CoreAI
import Foundation

/// Whether Core AI already holds a compiled copy of each graph of a bundle for this device, and how many source bytes
/// the graphs still to compile come to: what a host needs before a first load, which writes the compiled copies to
/// disk (on the order of the source graphs' size; measure the ratio for a given model).
public struct CompiledCacheStatus: Sendable, Hashable {
    /// One graph file's lookup.
    public struct Entry: Sendable, Hashable {
        /// The manifest graph role, e.g. `textEncoder`.
        public var role: String
        /// The graph file as this device resolves it (a bucket's own file is a separate entry).
        public var file: AssetPath
        /// Core AI's compile cache already holds a compiled copy for this device and these options.
        public var compiled: Bool
        /// The declared bytes of the graph's source files (the reference's files under `file`).
        public var sourceBytes: Int64

        public init(role: String, file: AssetPath, compiled: Bool, sourceBytes: Int64) {
            self.role = role
            self.file = file
            self.compiled = compiled
            self.sourceBytes = sourceBytes
        }
    }

    /// Sorted by role, then file.
    public var entries: [Entry]

    public init(entries: [Entry]) { self.entries = entries }

    /// Every graph has a compiled copy (a first load writes nothing new to the compile cache).
    public var isComplete: Bool { entries.allSatisfy(\.compiled) }

    /// The declared source bytes of the graphs without a compiled copy.
    public var uncompiledSourceBytes: Int64 {
        entries.filter { !$0.compiled }.reduce(0) { $0 + $1.sourceBytes }
    }
}

/// Lookups in Core AI's compile cache (``status(of:manifest:device:roles:cache:)``). Nothing here compiles or deletes.
public enum CompiledCache {
    /// Looks each graph of `manifest` (only `roles`, when given) up in Core AI's compile cache, with the specialization
    /// options a load from this manifest uses (`compute`, `expectFrequentReshapes`) and `cache`. Nothing is compiled.
    /// Core AI's lookup (`AIModelCache.model(for:options:)`) returns the cached model, which is released at once. On the
    /// Krea 2 realism export the whole lookup took 5 ms and allocated no Metal; still call it off the main actor and
    /// before taking the GPU, not on a generation's path. A graph whose reshape hint is absent and depends on its signature throws
    /// ``AetherError/manifestInvalid(field:reason:)``, as loading it without the signature would.
    public static func status(
        of bundle: InstalledBundle, manifest: Manifest, device: Device, roles: Set<String>? = nil,
        cache: GraphOptions.Cache = .default
    ) throws(AetherError) -> CompiledCacheStatus {
        let aiCache = try GraphOptions(computeUnit: .gpu, expectFrequentReshapes: false, cachePolicy: .default, cache: cache)
            .aiCache()
        return try status(of: bundle, manifest: manifest, device: device, roles: roles, cache: cache) { url, options in
            (try? aiCache.model(for: url, options: options.specializationOptions)) != nil
        }
    }

    /// `status` with the cache lookup injected (tests).
    static func status(
        of bundle: InstalledBundle, manifest: Manifest, device: Device, roles: Set<String>?, cache: GraphOptions.Cache,
        compiled: (URL, GraphOptions) -> Bool
    ) throws(AetherError) -> CompiledCacheStatus {
        var entries: [CompiledCacheStatus.Entry] = []
        for (role, graph) in manifest.graphs where roles?.contains(role) ?? true {
            let field = "graphs.\(role)"
            let options = try GraphOptions(manifest: graph, at: field, cachePolicy: .default, cache: cache)
            var files: [AssetPath] = []
            guard let main = graph.file.resolve(architecture: device.architecture) else {
                throw .manifestInvalid(field: "\(field).file", reason: "no file for architecture '\(device.architecture)'")
            }
            files.append(main)
            for entry in graph.buckets?.entries ?? [] where entry.asset == nil {
                if let file = entry.file, !files.contains(file) { files.append(file) }
            }
            for file in files {
                guard let url = bundle.url(of: file) else {
                    throw .manifestInvalid(field: "\(field).file", reason: "the bundle has no asset '\(file.asset)'")
                }
                entries.append(.init(
                    role: role, file: file, compiled: compiled(url, options),
                    sourceBytes: sourceBytes(of: file, in: bundle.ref)))
            }
        }
        entries.sort { ($0.role, $0.file.description) < ($1.role, $1.file.description) }
        return CompiledCacheStatus(entries: entries)
    }

    /// The declared sizes of the reference's files at or under `file`'s path in its asset.
    static func sourceBytes(of file: AssetPath, in ref: ModelRef) -> Int64 {
        guard let asset = ref.asset(file.asset) else { return 0 }
        let path = file.path
        return asset.files
            .filter { path == "." || $0.path == path || $0.path.hasPrefix(path + "/") }
            .reduce(0) { $0 + $1.size }
    }
}
