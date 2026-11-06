public import Foundation

// The `aether.json` schema, version 1: docs/MANIFEST.md §2 is normative. The schema types live in this folder:
// - Manifest.swift        top level (§2.2), coding, structural validation
// - ManifestGraph.swift   graphs (§2.3): file, functions, bindings, outputs, states, limits, compute
// - ManifestBlocks.swift  variant, assets, tables, tokenizer (§2.4) and the typed-but-loose driver blocks (§2.5–§2.8)
// - ManifestGenerative.swift  the generative drivers' typed blocks (§5.2–§5.3): schedule, latent, residency
// - ManifestFlux2.swift   the `flux2` block of `flux2-klein@1` (§5.7)
// - ManifestKokoro.swift  the `kokoro` block of `kokoro@1` (§5.9)
//
// Rules: unknown fields and `x-` fields are ignored; a `$ref` anywhere is rejected (the adapter inlines refs);
// optional fields are omitted when encoding; every validation error names the dotted field path.

/// A bundle's self-description for one catalog variant: its assets, graphs and their roles, tokenizer, and the
/// blocks its driver owns.
public struct Manifest: Sendable, Hashable, Codable {
    public static let currentSchema = 1
    /// The manifest's file name.
    public static let fileName = "aether.json"
    /// Where an installed bundle keeps Aether's own files (manifest, receipt), relative to the revision root.
    public static let directory = ".aether"

    public var schema: Int
    public var id: String
    public var task: String
    public var driver: DriverID
    public var variant: Variant
    /// Asset name → subtree. Must contain `main`.
    public var assets: [String: Asset]
    public var graphs: [String: Graph]
    public var tables: [String: Table]?
    public var tokenizer: Tokenizer?
    public var preprocess: Preprocess?
    public var image: ImageBlock?
    public var decision: Decision?
    public var postprocess: Postprocess?
    public var labels: Labels?
    public var generation: Generation?
    public var calibration: Calibration?
    /// §5.2: the generative drivers' denoising schedule.
    public var schedule: Schedule?
    /// §5.2: the generative drivers' image-latent contract.
    public var latent: Latent?
    /// §5.7: the block `flux2-klein@1` owns.
    public var flux2: Flux2?
    /// §5.9: the block `kokoro@1` owns.
    public var kokoro: Kokoro?
    public var precision: [String: JSONValue]?
    public var provenance: Provenance?
    public var license: String?

    public init(
        schema: Int = Manifest.currentSchema, id: String, task: String, driver: DriverID, variant: Variant,
        assets: [String: Asset], graphs: [String: Graph], tables: [String: Table]? = nil,
        tokenizer: Tokenizer? = nil, preprocess: Preprocess? = nil, image: ImageBlock? = nil,
        decision: Decision? = nil, postprocess: Postprocess? = nil, labels: Labels? = nil,
        generation: Generation? = nil, calibration: Calibration? = nil, precision: [String: JSONValue]? = nil,
        provenance: Provenance? = nil, license: String? = nil, schedule: Schedule? = nil, latent: Latent? = nil,
        flux2: Flux2? = nil, kokoro: Kokoro? = nil
    ) {
        self.schema = schema
        self.id = id
        self.task = task
        self.driver = driver
        self.variant = variant
        self.assets = assets
        self.graphs = graphs
        self.tables = tables
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.image = image
        self.decision = decision
        self.postprocess = postprocess
        self.labels = labels
        self.generation = generation
        self.calibration = calibration
        self.schedule = schedule
        self.latent = latent
        self.flux2 = flux2
        self.kokoro = kokoro
        self.precision = precision
        self.provenance = provenance
        self.license = license
    }
}

// MARK: - Coding

extension Manifest {
    /// Decodes and structurally validates a manifest. Errors name the offending field.
    public static func decode(_ data: Data) throws(AetherError) -> Manifest {
        let manifest: Manifest = try decodeSanitized(Manifest.self, from: data)
        try manifest.validate()
        return manifest
    }

    /// Reads `.aether/aether.json` under an installed revision root.
    public static func load(fromBundle root: URL) throws(AetherError) -> Manifest {
        let data: Data
        do { data = try Data(contentsOf: url(inBundle: root)) } catch { throw .runtime(underlying: error) }
        return try decode(data)
    }

    public static func url(inBundle root: URL) -> URL {
        root.appending(path: directory, directoryHint: .isDirectory).appending(path: fileName)
    }

    /// Canonical JSON: pretty-printed, sorted keys, unescaped slashes, trailing newline. Goldens use this form.
    public func encoded() throws(AetherError) -> Data {
        try canonicalJSON(self)
    }

    /// Writes `.aether/aether.json` under `root` atomically.
    public func write(toBundle root: URL) throws(AetherError) {
        do {
            let url = Self.url(inBundle: root)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try encoded().write(to: url, options: .atomic)
        } catch let error as AetherError {
            throw error
        } catch {
            throw .runtime(underlying: error)
        }
    }
}

/// Decodes `type` after dropping `x-` fields and rejecting `$ref`s. Decoding errors name their field.
func decodeSanitized<T: Decodable>(_ type: T.Type, from data: Data) throws(AetherError) -> T {
    do {
        let raw = try JSONDecoder().decode(JSONValue.self, from: data)
        let clean = try raw.sanitized()
        return try JSONDecoder().decode(T.self, from: JSONEncoder().encode(clean))
    } catch let error as AetherError {
        throw error
    } catch let error as DecodingError {
        throw AetherError(decoding: error)
    } catch {
        throw .manifestInvalid(field: "", reason: "\(error)")
    }
}

func canonicalJSON<T: Encodable>(_ value: T) throws(AetherError) -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    do {
        var data = try encoder.encode(value)
        data.append(0x0A)
        return data
    } catch {
        throw .runtime(underlying: error)
    }
}

extension AetherError {
    /// Maps a decoding failure onto the field it happened at.
    init(decoding error: DecodingError) {
        func path(_ codingPath: [any CodingKey], _ key: (any CodingKey)? = nil) -> String {
            (codingPath + (key.map { [$0] } ?? [])).map { $0.intValue.map(String.init) ?? $0.stringValue }
                .joined(separator: ".")
        }
        switch error {
        case .keyNotFound(let key, let context):
            self = .manifestInvalid(field: path(context.codingPath, key), reason: "required field is missing")
        case .typeMismatch(let type, let context):
            self = .manifestInvalid(field: path(context.codingPath), reason: "expected \(type)")
        case .valueNotFound(let type, let context):
            self = .manifestInvalid(field: path(context.codingPath), reason: "expected \(type), found null")
        case .dataCorrupted(let context):
            self = .manifestInvalid(field: path(context.codingPath), reason: context.debugDescription)
        @unknown default:
            self = .manifestInvalid(field: "", reason: "\(error)")
        }
    }
}

// MARK: - Structural validation

extension Manifest {
    /// Checks everything that needs neither a driver nor a graph. Errors name the offending field.
    public func validate() throws(AetherError) {
        guard schema == Self.currentSchema else {
            throw .manifestInvalid(
                field: "schema", reason: "unsupported schema \(schema); this build reads \(Self.currentSchema)")
        }
        try require(!id.isEmpty, "id", "must not be empty")
        try require(!task.isEmpty, "task", "must not be empty")
        if DriverCatalog.graphKinds.contains(driver) {
            throw .manifestInvalid(field: "driver", reason: "\(driver) is a graph kind, never a top-level driver")
        }
        try variant.validate()
        try require(assets["main"] != nil, "assets.main", "every manifest has a main asset")
        for name in assets.keys.sorted() { try assets[name]!.validate(at: "assets.\(name)") }
        try require(!graphs.isEmpty, "graphs", "at least one graph is required")
        for name in (tables ?? [:]).keys.sorted() {
            let table = tables![name]!
            try table.validate(at: "tables.\(name)", assets: assets)
        }
        for role in graphs.keys.sorted() {
            try graphs[role]!.validate(at: "graphs.\(role)", manifest: self)
        }
        try tokenizer?.validate(assets: assets)
        try preprocess?.validate()
        try image?.validate()
        try decision?.validate()
        try postprocess?.validate()
        try labels?.validate()
        try calibration?.validate()
        try schedule?.validate()
        try latent?.validate()
        try flux2?.validate(graphs: graphs)
        try kokoro?.validate(tables: tables ?? [:])
    }

    /// Resolves `"<asset>:<path>"` against this manifest's assets. Errors name `field`.
    func checkAssetPath(_ path: AssetPath, _ field: String) throws(AetherError) {
        try require(assets[path.asset] != nil, field, "names asset '\(path.asset)', which is not in assets")
        try require(isSafeRelativePath(path.path, allowDot: true), field, "'\(path.path)' is not a safe relative path")
    }
}

func require(_ condition: Bool, _ field: String, _ reason: String) throws(AetherError) {
    if !condition { throw .manifestInvalid(field: field, reason: reason) }
}

/// Relative, no `..`, no empty components. `"."` (the asset itself) is allowed when `allowDot` is set.
func isSafeRelativePath(_ path: String, allowDot: Bool = false) -> Bool {
    if allowDot && path == "." { return true }
    if path.isEmpty || path.hasPrefix("/") { return false }
    return !path.split(separator: "/", omittingEmptySubsequences: false).contains { $0.isEmpty || $0 == ".." || $0 == "." }
}
