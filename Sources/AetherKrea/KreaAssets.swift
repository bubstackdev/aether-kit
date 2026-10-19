import AetherKit
import CryptoKit
import Foundation

/// An external model stream with its declared content identity.
/// A declaration is not verification of the stream bytes; call `verify()` explicitly.
public struct KreaModelAsset: Sendable {
    public let url: URL
    public let declaredSHA256: String
    public let metadataSHA256: String

    public func verify() throws {
        let declaration = try Data(contentsOf: confinedFile("main.hash", root: url))
        guard declaration.hex == declaredSHA256,
              try hashFile(confinedFile("main.mlirb", root: url)) == declaredSHA256,
              try hashFile(confinedFile("metadata.json", root: url)) == metadataSHA256 else {
            throw AetherError.invalidModel("model hash mismatch: \(url.lastPathComponent)")
        }
    }
}

/// The prototype's dynamic, text-only Krea 2 Turbo realism bundle.
/// Opening reads headers, hash declarations and tokenizer files, not model streams.
/// Keep external files immutable for the lifetime of any model using this bundle.
public struct KreaAssets: Sendable {
    public let root: URL
    public let textEncoder: KreaModelAsset
    public let decoder: KreaModelAsset
    public let denoisers: [KreaModelAsset]
    public let tokenizerDirectory: URL
    public let fingerprint: String
    /// The `visionTokens` reference encoder, present only when the manifest declares Krea's reference contract and its
    /// text asset declares `encodeWithImages`. Text-only use never opens, verifies or fingerprints it.
    public let vision: KreaModelAsset?
    /// Identity for reference-image execution: the text-only fingerprint plus the vision asset's declarations and the
    /// reference contract. A text-only identity and a reference identity can therefore never be confused.
    public let referenceFingerprint: String?
    public var supportsReferences: Bool { vision != nil }
    /// Denoiser text rows: 512 for text-only bundles; the reference export bounds its segments for 1,116.
    public var maximumTextRows: Int { supportsReferences ? KreaReferencePrompt.maximumConditioningRows : 512 }
    private let tokenizerFiles: [URL: String]
    private var specializations: [KreaDenoiserSelection] = []

    public static func open(at root: URL) throws -> Self {
        guard root.isFileURL else { throw AetherError.invalidModel("bundle must be local") }
        let root = root.standardizedFileURL.resolvingSymlinksInPath()
        let metadataData = try Data(contentsOf: confinedFile("metadata.json", root: root))
        let manifestData = try Data(contentsOf: confinedFile("aether.json", root: root))
        let decoder = JSONDecoder()
        let metadata = try decoder.decode(Metadata.self, from: metadataData)
        let manifest = try decoder.decode(Manifest.self, from: manifestData)
        try manifest.validate(name: metadata.name)

        func asset(_ name: String) throws -> KreaModelAsset {
            guard let relative = metadata.assets[name] else {
                throw AetherError.invalidModel("missing asset '\(name)'")
            }
            let directory = try confinedFile(relative, root: root, directory: true)
            let stream = try confinedFile(relative + "/main.mlirb", root: root)
            let hashURL = try confinedFile(relative + "/main.hash", root: root)
            let metadataURL = try confinedFile(relative + "/metadata.json", root: root)
            let metadataHash = try hashFile(metadataURL)
            let declaration = try Data(contentsOf: hashURL)
            guard declaration.count == 32 else {
                throw AetherError.invalidModel("main.hash must contain 32 bytes: \(name)")
            }
            if let pinned = manifest.provenance?.files[relative + "/main.mlirb"], pinned != "sha256:" + declaration.hex {
                throw AetherError.invalidModel("model declaration differs from provenance: \(stream.lastPathComponent)")
            }
            if let pinned = manifest.provenance?.files[relative + "/metadata.json"], pinned != "sha256:" + metadataHash {
                throw AetherError.invalidModel("model metadata differs from provenance: \(name)")
            }
            return KreaModelAsset(url: directory, declaredSHA256: declaration.hex, metadataSHA256: metadataHash)
        }

        let encoder = try asset("textEncoder")
        let vae = try asset("vaeDecoder")
        let denoisers = try (0..<4).map { try asset("denoiser\($0)") }
        guard Set(([encoder, vae] + denoisers).map(\.url)).count == 6 else {
            throw AetherError.invalidModel("Krea requires six distinct model assets")
        }
        var support: [URL: String] = [:]
        for name in ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"] {
            let url = try confinedFile(name, root: root)
            let hash = try hashFile(url)
            if let pinned = manifest.provenance?.files[name], pinned != "sha256:" + hash {
                throw AetherError.invalidModel("tokenizer differs from provenance: \(name)")
            }
            support[url] = hash
        }
        var identity = SHA256()
        identity.update(data: metadataData)
        identity.update(data: manifestData)
        for value in support.sorted(by: { $0.key.path < $1.key.path }) {
            identity.update(data: Data(value.value.utf8))
        }
        // Include model declarations even when older manifests omit provenance.
        for model in [encoder, vae] + denoisers {
            identity.update(data: Data(model.declaredSHA256.utf8))
            identity.update(data: Data(model.metadataSHA256.utf8))
        }
        let fingerprint = Data(identity.finalize()).hex
        var vision: KreaModelAsset?
        var referenceFingerprint: String?
        if try manifest.validateReferences() {
            let asset = try asset("vision")
            guard ![encoder, vae].contains(where: { $0.url == asset.url }), !denoisers.contains(where: { $0.url == asset.url }) else {
                throw AetherError.invalidModel("the Krea vision asset must be distinct")
            }
            var reference = SHA256()
            for part in [fingerprint, asset.declaredSHA256, asset.metadataSHA256, KreaReferenceContract.identity] {
                reference.update(data: Data(part.utf8)); reference.update(data: Data([0]))
            }
            vision = asset
            referenceFingerprint = Data(reference.finalize()).hex
        }
        return Self(
            root: root, textEncoder: encoder, decoder: vae, denoisers: denoisers,
            tokenizerDirectory: root.appendingPathComponent("tokenizer", isDirectory: true),
            fingerprint: fingerprint, vision: vision, referenceFingerprint: referenceFingerprint, tokenizerFiles: support
        )
    }

    /// Streams all six model assets through SHA-256. This can read tens of GB.
    /// No native model is loaded and no specialization cache is changed.
    public func verify() throws {
        for model in [textEncoder, decoder] + denoisers { try model.verify() }
        try verifyTokenizer()
    }

    /// Streams all seven assets (the six text-only assets and the vision asset) before reference execution.
    public func verifyIncludingReferences() throws {
        guard let vision else { throw AetherError.unsupported("this Krea bundle declares no reference images") }
        try verify()
        try vision.verify()
    }

    public func verifyTokenizer() throws {
        for (url, expected) in tokenizerFiles {
            guard try hashFile(url) == expected else {
                throw AetherError.invalidModel("tokenizer changed after bundle opening: \(url.lastPathComponent)")
            }
        }
    }

    /// Registers a trusted offline export against these exact four source streams.
    /// Hashes both source and specialized streams before returning a new asset set.
    public func registeringSpecialization(at registrationURL: URL) throws -> Self {
        guard registrationURL.isFileURL else { throw AetherError.invalidModel("registration must be local") }
        let registration = try JSONDecoder().decode(ShapeRegistration.self, from: Data(contentsOf: registrationURL))
        guard specializations.count < 8, (1...4096).contains(registration.image_tokens),
              (1...maximumTextRows).contains(registration.text_rows), registration.segments.count == 4,
              registration.source_sha256 == denoisers.map(\.declaredSHA256),
              registration.specialized_sha256.count == 4,
              !specializations.contains(where: { $0.imageTokens == registration.image_tokens && $0.textRows == registration.text_rows }) else {
            throw AetherError.invalidModel("invalid, duplicate or incompatible Krea shape registration")
        }
        var models: [KreaModelAsset] = []
        for index in 0..<4 {
            try denoisers[index].verify()
            let path = registration.segments[index]
            guard !path.isEmpty else { throw AetherError.invalidModel("empty specialized asset path") }
            let url = (path.hasPrefix("/") ? URL(fileURLWithPath: path)
                : registrationURL.deletingLastPathComponent().appendingPathComponent(path))
                .standardizedFileURL.resolvingSymlinksInPath()
            let metadataHash = try hashFile(confinedFile("metadata.json", root: url))
            let model = KreaModelAsset(url: url, declaredSHA256: registration.specialized_sha256[index], metadataSHA256: metadataHash)
            try model.verify()
            models.append(model)
        }
        guard Set(models.map(\.url)).count == 4 else {
            throw AetherError.invalidModel("specialization requires four distinct assets")
        }
        var result = self
        result.specializations.append(KreaDenoiserSelection(
            assets: models, imageTokens: registration.image_tokens, textRows: registration.text_rows, specialized: true
        ))
        return result
    }

    public func denoiserSelection(imageTokens: Int, textRows: Int) throws -> KreaDenoiserSelection {
        guard (1...4096).contains(imageTokens), (1...maximumTextRows).contains(textRows) else {
            throw AetherError.invalidRequest("invalid Krea denoiser shape")
        }
        return specializations.first { $0.imageTokens == imageTokens && $0.textRows == textRows }
            ?? KreaDenoiserSelection(assets: denoisers, imageTokens: imageTokens, textRows: textRows, specialized: false)
    }
}

/// A selected set of four denoisers, keyed by stream/metadata identity and both shapes.
public struct KreaDenoiserSelection: Sendable {
    public let assets: [KreaModelAsset]
    public let imageTokens: Int
    public let textRows: Int
    public let specialized: Bool
    public var identity: String {
        let parts = assets.flatMap { [$0.declaredSHA256, $0.metadataSHA256] }
            + [String(imageTokens), String(textRows), String(specialized)]
        return Data(SHA256.hash(data: Data(parts.joined(separator: ":").utf8))).hex
    }
}

private struct ShapeRegistration: Decodable {
    let image_tokens: Int
    let text_rows: Int
    let source_sha256: [String]
    let specialized_sha256: [String]
    let segments: [String]

    init(from decoder: any Decoder) throws {
        let raw = try decoder.container(keyedBy: AnyKey.self)
        let expected = Set(["image_tokens", "text_rows", "source_sha256", "specialized_sha256", "segments"])
        guard Set(raw.allKeys.map(\.stringValue)) == expected else {
            throw AetherError.invalidModel("unknown or missing shape registration fields")
        }
        image_tokens = try raw.decode(Int.self, forKey: AnyKey("image_tokens"))
        text_rows = try raw.decode(Int.self, forKey: AnyKey("text_rows"))
        source_sha256 = try raw.decode([String].self, forKey: AnyKey("source_sha256"))
        specialized_sha256 = try raw.decode([String].self, forKey: AnyKey("specialized_sha256"))
        segments = try raw.decode([String].self, forKey: AnyKey("segments"))
    }
    struct AnyKey: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init(_ value: String) { stringValue = value }
        init?(stringValue: String) { self.init(stringValue) }
        init?(intValue: Int) { return nil }
    }
}

private struct Metadata: Decodable {
    let name: String
    let assets: [String: String]
}

private struct Manifest: Decodable {
    let schema: Int
    let driver: String
    let diffusion: Diffusion
    let text: Text
    let target: Target
    let functions: [String: Role]
    let inputs: [String: Binding]
    let outputs: [String: Binding]
    let provenance: Provenance?
    let vision: Vision?

    struct Text: Decodable { let tokenizer: String }
    struct Vision: Decodable {
        let deepstackLevels: Int
        let hiddenSize: Int
        let imageMean: [Double]
        let imageStd: [Double]
        let maxPixels: Int
        let minPixels: Int
        let mergeSize: Int
        let patchSize: Int
        let positionTableSide: Int
        let positions: String
        let resample: String
        let temporalPatchSize: Int
        let tokens: Tokens
        struct Tokens: Decodable { let image: Int; let visionStart: Int; let visionEnd: Int }
    }
    struct ReferenceImages: Decodable {
        let kind: String
        let encoderAsset: String
        let maxImages: Int
        let maxSide: Int
        let maxTotalTokens: Int
        let label: String
    }
    struct Target: Decodable {
        let backend: String
        let platform: String
        let arch: String
        let compiled: Bool
        let minOS: String
    }
    struct Role: Decodable { let role: String }
    struct Binding: Decodable { let dtype: String; let bind: String?; let mode: String? }
    struct Provenance: Decodable { let files: [String: String] }
    struct Diffusion: Decodable {
        let latentChannels: Int
        let pixelsPerToken: Int
        let steps: Int
        let maxSteps: Int?
        let trainTimesteps: Int?
        let guidance: Bool
        let denoiserSegments: [String]
        let textEncoderAsset: String
        let decoderAsset: String
        let sigmaShift: Shift
        let rotary: Rotary
        let prompt: Prompt
        let referenceImages: ReferenceImages?
        struct Shift: Decodable { let rule: String; let mu: Double }
        struct Rotary: Decodable { let axes: [Int]; let theta: Int; let layout: String; let textRotated: Bool }
        struct Prompt: Decodable {
            let systemMessage: String
            let template: String
            let dropLeadingRows: Int
            let maxLength: Int
            let keepTrailingTokens: Int
            let padding: String
            let thinking: String?
        }
    }

    func validate(name: String) throws {
        let d = diffusion
        let p = d.prompt
        guard schema == 1, driver == "diffusion", name == "krea_2_turbo_realism",
              target.backend == "coreai", target.platform == "macos", target.arch == "any",
              !target.compiled, target.minOS == "27.0",
              d.latentChannels == 64, d.pixelsPerToken == 16, d.steps == 8,
              d.maxSteps == nil || d.maxSteps == 50,
              d.trainTimesteps == nil || d.trainTimesteps == 1000,
              !d.guidance, d.sigmaShift.rule == "constant", d.sigmaShift.mu == 1.15,
              d.rotary.axes == [32, 48, 48], d.rotary.theta == 1000,
              d.rotary.layout == "tyx", !d.rotary.textRotated,
              d.denoiserSegments == ["denoiser0", "denoiser1", "denoiser2", "denoiser3"],
              d.textEncoderAsset == "textEncoder", d.decoderAsset == "vaeDecoder",
              p.systemMessage == KreaPrompt.systemMessage, p.template == "chat",
              p.dropLeadingRows == 34, p.maxLength == 546, p.keepTrailingTokens == 5,
              p.padding == "none", p.thinking == nil, text.tokenizer == "tokenizer/" else {
            throw AetherError.unsupported("expected the dynamic macOS Krea 2 Turbo realism recipe")
        }
        let roles = ["textEncoder/encode": "encodeText", "vaeDecoder/decode": "decodeLatents"]
            .merging(Dictionary(uniqueKeysWithValues: (0..<4).map { ("denoiser\($0)/segment", "denoiseSegment") })) { $1 }
        guard roles.allSatisfy({ functions[$0.key]?.role == $0.value }) else {
            throw AetherError.unsupported("Krea function roles differ from the supported recipe")
        }
        let bindings = [
            "input_ids": ("int32", "tokens"), "attention_mask": ("int32", "tokenMask"),
            "position_ids": ("int32", "positions"), "text_embeds": ("float16", "textEmbeddings"),
            "latents": ("float16", "latents"), "sigma": ("float32", "timestep"),
            "rope_image": ("float32", "imageRotary"),
        ]
        guard bindings.allSatisfy({ inputs[$0.key]?.dtype == $0.value.0 && inputs[$0.key]?.bind == $0.value.1 }),
              inputs["position_ids"]?.mode == "running",
              outputs["text_embeds"]?.dtype == "float16", outputs["velocity"]?.dtype == "float16",
              outputs["image"]?.dtype == "float32" else {
            throw AetherError.unsupported("Krea tensor declarations differ from the supported recipe")
        }
    }

    /// True when the bundle declares exactly Krea's `visionTokens` reference contract (any other declaration is
    /// refused rather than ignored); false when it declares none.
    func validateReferences() throws -> Bool {
        // Without `referenceImages` the bundle is text-only; stray vision declarations stay unused.
        guard let r = diffusion.referenceImages else { return false }
        let c = KreaReferenceContract.self
        guard r.kind == "visionTokens", r.encoderAsset == "vision", r.maxImages == c.maximumImages,
              r.maxSide == c.maximumSide, r.maxTotalTokens == c.maximumImageTokens, r.label == c.label,
              let v = vision, v.deepstackLevels == c.deepstackLevels, v.hiddenSize == c.hiddenSize,
              v.imageMean == [0.5, 0.5, 0.5], v.imageStd == [0.5, 0.5, 0.5],
              v.maxPixels == c.maximumPixels, v.minPixels == c.minimumPixels, v.mergeSize == c.mergeSize,
              v.patchSize == c.patchSize, v.positionTableSide == c.positionTableSide, v.positions == "mrope",
              v.resample == "bicubic-antialias", v.temporalPatchSize == c.temporalPatchSize,
              v.tokens.image == Int(c.imageToken), v.tokens.visionStart == Int(c.visionStartToken),
              v.tokens.visionEnd == Int(c.visionEndToken) else {
            throw AetherError.unsupported("expected Krea's visionTokens reference contract")
        }
        guard functions["textEncoder/encodeWithImages"]?.role == "encodeTextWithImages",
              functions["vision/encode"]?.role == "encodeImage" else {
            throw AetherError.unsupported("Krea reference function roles differ from the supported recipe")
        }
        let bindings = [
            "rope_positions": ("int32", "positions"), "image_embeds": ("float16", "imageEmbeds"),
            "deepstack": ("float16", "deepstack"), "pixels": ("float16", "image"),
            "pos_index": ("int32", "patchPositionIndex"), "pos_weight": ("float32", "patchPositionWeight"),
            "patch_rc": ("int32", "patchCoordinates"),
        ]
        guard bindings.allSatisfy({ inputs[$0.key]?.dtype == $0.value.0 && inputs[$0.key]?.bind == $0.value.1 }),
              inputs["rope_positions"]?.mode == "mrope",
              outputs["image_embeds"]?.dtype == "float16", outputs["deepstack"]?.dtype == "float16" else {
            throw AetherError.unsupported("Krea reference tensor declarations differ from the supported recipe")
        }
        return true
    }
}

private func confinedFile(_ relative: String, root: URL, directory: Bool = false) throws -> URL {
    guard !relative.isEmpty, !relative.hasPrefix("/"),
          !relative.split(separator: "/").contains("..") else {
        throw AetherError.invalidModel("asset path must stay inside its bundle")
    }
    let url = root.appendingPathComponent(relative).standardizedFileURL.resolvingSymlinksInPath()
    guard url.pathComponents.starts(with: root.pathComponents), url.pathComponents.count > root.pathComponents.count else {
        throw AetherError.invalidModel("asset symlink escapes its bundle")
    }
    let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey])
    guard directory ? values.isDirectory == true : values.isRegularFile == true else {
        throw AetherError.invalidModel("missing \(directory ? "directory" : "file"): \(relative)")
    }
    return url
}

func hashFile(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var digest = SHA256()
    // `read(upToCount:)` returns autoreleased buffers. Without a pool per chunk, a synchronous loop keeps
    // every chunk alive until the caller's pool drains: `verify()` then peaks at the sum of all streams.
    while try autoreleasepool(invoking: {
        guard let data = try file.read(upToCount: 4 << 20), !data.isEmpty else { return false }
        try Task.checkCancellation()
        digest.update(data: data)
        return true
    }) {}
    return Data(digest.finalize()).hex
}

extension Data {
    var hex: String { map { String(format: "%02x", $0) }.joined() }
}
