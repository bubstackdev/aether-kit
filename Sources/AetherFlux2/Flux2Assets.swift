import AetherAssets
import AetherKit
import CryptoKit
import Foundation

/// A FLUX.2 [klein] 4B Core AI bundle: Qwen3 text encoder, three sequential denoiser segments and
/// the VAE decoder, plus the optional VAE encoder that conditions generation on reference images.
/// Opening reads the bundle declarations, small metadata and tokenizer files, not model streams.
/// Keep external files immutable for the lifetime of any model using this bundle.
public struct Flux2Assets: Sendable {
    public static let segmentCount = 3
    /// The supported Apache-2.0 checkpoint revision.
    public static let sourceRevision = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    public let root: URL
    public let textEncoder: DeclaredModelAsset
    public let denoisers: [DeclaredModelAsset]
    public let decoder: DeclaredModelAsset
    /// Present only when the manifest declares the `packedLatents` reference-image contract.
    public let referenceEncoder: DeclaredModelAsset?
    public let references: Flux2ReferenceLimits?
    public let tokenizerDirectory: URL
    /// Identity of the text-to-image execution contract: manifest, metadata, tokenizer and model declarations.
    public let fingerprint: String
    /// Text-to-image identity plus the reference encoder's declarations and the reference contract.
    public let referenceFingerprint: String?
    public let sourceRepository: String
    public let sourceRevision: String
    public var supportsReferences: Bool { referenceEncoder != nil }
    private let tokenizerFiles: [String: String]

    public static func open(at root: URL) throws -> Self {
        guard root.isFileURL else { throw AetherError.invalidModel("bundle must be local") }
        let root = root.standardizedFileURL.resolvingSymlinksInPath()
        let metadataData = try Data(contentsOf: BundleFiles.confined("metadata.json", in: root))
        let manifestData = try Data(contentsOf: BundleFiles.confined("aether.json", in: root))
        let metadata = try JSONDecoder().decode(Flux2Metadata.self, from: metadataData)
        let manifest = try JSONDecoder().decode(Flux2Manifest.self, from: manifestData)
        let limits = try manifest.validate(name: metadata.name)
        let pinned = manifest.provenance.files

        func asset(_ role: String) throws -> DeclaredModelAsset {
            guard let relative = metadata.assets[role] else { throw AetherError.invalidModel("missing asset '\(role)'") }
            do { return try DeclaredModelAsset.open(relative, in: root, pinned: pinned) } catch let error as AssetStoreError {
                throw AetherError.invalidModel("\(role): \(error)")
            }
        }
        let encoder = try asset("textEncoder")
        let denoisers = try (0..<segmentCount).map { try asset("denoiser\($0)") }
        let decoder = try asset("vaeDecoder")
        let referenceEncoder = try limits == nil ? nil : asset("vaeEncoder")
        let models = [encoder, decoder] + denoisers + (referenceEncoder.map { [$0] } ?? [])
        guard Set(models.map(\.url)).count == models.count else {
            throw AetherError.invalidModel("FLUX.2 requires distinct model assets")
        }
        guard Set(metadata.assets.keys) == Set(["textEncoder", "vaeDecoder"] + (0..<segmentCount).map { "denoiser\($0)" }
            + (limits == nil ? [] : ["vaeEncoder"])) else {
            throw AetherError.invalidModel("FLUX.2 metadata declares unexpected assets")
        }
        var tokenizer: [String: String] = [:]
        for name in Self.tokenizerNames {
            let hash = try BundleFiles.sha256(of: BundleFiles.confined(name, in: root))
            if let value = pinned[name], value != "sha256:" + hash {
                throw AetherError.invalidModel("tokenizer differs from provenance: \(name)")
            }
            tokenizer[name] = hash
        }
        var identity = SHA256()
        for part in [metadataData, manifestData] { identity.update(data: part) }
        for (name, hash) in tokenizer.sorted(by: { $0.key < $1.key }) { identity.update(data: Data((name + ":" + hash).utf8)) }
        for model in [encoder, decoder] + denoisers { identity.update(data: Data(model.identity.utf8)) }
        let fingerprint = Data(identity.finalize()).hexString
        let referenceFingerprint = try referenceEncoder.map { asset -> String in
            var reference = SHA256()
            for part in [fingerprint, asset.identity, try limits!.identity] {
                reference.update(data: Data(part.utf8)); reference.update(data: Data([0]))
            }
            return Data(reference.finalize()).hexString
        }
        return Self(root: root, textEncoder: encoder, denoisers: denoisers, decoder: decoder,
                    referenceEncoder: referenceEncoder, references: limits,
                    tokenizerDirectory: root.appendingPathComponent("tokenizer", isDirectory: true),
                    fingerprint: fingerprint, referenceFingerprint: referenceFingerprint,
                    sourceRepository: manifest.provenance.source.hfRepo, sourceRevision: manifest.provenance.source.revision,
                    tokenizerFiles: tokenizer)
    }

    static let tokenizerNames = ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]

    /// Streams every declared model through SHA-256 (about 8 GB for the int8 export) and rechecks
    /// the tokenizer. No native model is loaded.
    public func verify() throws {
        for model in [textEncoder, decoder] + denoisers + (referenceEncoder.map { [$0] } ?? []) {
            do { try model.verify() } catch let error as AssetStoreError { throw AetherError.invalidModel("\(error)") }
        }
        try verifyTokenizer()
    }

    public func verifyTokenizer() throws {
        for (name, expected) in tokenizerFiles {
            guard try BundleFiles.sha256(of: BundleFiles.confined(name, in: root)) == expected else {
                throw AetherError.invalidModel("tokenizer changed after bundle opening: \(name)")
            }
        }
    }

    /// Describes the portable bundle for `LocalAssetStore` without reading model streams;
    /// installation verifies every copied byte.
    public func installationManifest() throws -> AssetManifest {
        let current = try Self.open(at: root)
        guard current.fingerprint == fingerprint, current.referenceFingerprint == referenceFingerprint else {
            throw AetherError.invalidModel("FLUX.2 declarations changed after bundle opening")
        }
        do {
            var files = try ["metadata.json", "aether.json"].map { try BundleFiles.portable($0, in: root) }
            files += try tokenizerFiles.map { try BundleFiles.portable($0.key, in: root, sha256: $0.value) }
            for name in ["LICENSE.md", "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/added_tokens.json",
                         "tokenizer/special_tokens_map.json", "tokenizer/chat_template.jinja"]
                where FileManager.default.fileExists(atPath: root.appendingPathComponent(name).path) {
                files.append(try BundleFiles.portable(name, in: root))
            }
            for model in [textEncoder, decoder] + denoisers + (referenceEncoder.map { [$0] } ?? []) {
                files += try model.portableFiles(relativeTo: root)
            }
            return try AssetManifest(assetIdentity: "flux2-klein-bundle/1:" + (referenceFingerprint ?? fingerprint),
                                     executionContract: "flux2.portable-bundle/1", files: files)
        } catch let error as AssetStoreError {
            throw AetherError.invalidModel("\(error)")
        }
    }

    /// Reopens a verified installation, checking that its portable contract exactly covers this bundle.
    public static func open(in installation: AssetReadLease) throws -> Self {
        let assets = try Self.open(at: installation.directory)
        guard try assets.installationManifest().installationID == installation.installationID else {
            throw AetherError.invalidModel("installation is not the exact portable FLUX.2 bundle")
        }
        return assets
    }
}

/// Bounds of the bundle's `packedLatents` reference-image contract.
public struct Flux2ReferenceLimits: Sendable, Equatable, Codable {
    public let maxImages: Int
    public let maxSide: Int
    /// Generated plus reference image tokens in one denoiser call.
    public let maxTotalTokens: Int
    public let positionStride: Int
    var identity: String {
        get throws {
            let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
            return "packedLatents:" + String(decoding: try encoder.encode(self), as: UTF8.self)
        }
    }
}

private struct Flux2Metadata: Decodable {
    let name: String
    let kind: String
    let assets: [String: String]
}

struct Flux2Manifest: Decodable {
    let schema: Int
    let driver: String
    let functions: [String: Role]
    let inputs: [String: Binding]
    let outputs: [String: Binding]
    let diffusion: Diffusion
    let target: Target
    let text: Text
    let provenance: Provenance

    struct Role: Decodable { let role: String }
    struct Binding: Decodable { let dtype: String; let bind: String?; let role: String?; let mode: String? }
    struct Text: Decodable { let tokenizer: String }
    struct Target: Decodable {
        let backend: String; let platform: String; let arch: String; let compiled: Bool; let minOS: String
    }
    struct Provenance: Decodable {
        struct Source: Decodable { let hfRepo: String; let revision: String }
        let source: Source
        let files: [String: String]
    }
    struct Diffusion: Decodable {
        struct Shift: Decodable {
            let rule: String
            let lowSteps: Int; let lowSlope: Double; let lowIntercept: Double
            let highSteps: Int; let highSlope: Double; let highIntercept: Double
            let tokenThreshold: Int
        }
        struct Rotary: Decodable { let axes: [Int]; let theta: Double }
        struct Prompt: Decodable {
            let template: String; let thinking: String; let maxLength: Int; let padding: String; let padToken: Int
        }
        struct References: Decodable {
            let kind: String; let encoderAsset: String; let positionStride: Int
            let maxImages: Int; let maxSide: Int; let maxTotalTokens: Int
        }
        let latentChannels: Int
        let pixelsPerToken: Int
        let steps: Int
        let maxSteps: Int
        let trainTimesteps: Int
        let guidance: Bool
        let textEncoderAsset: String
        let decoderAsset: String
        let denoiserSegments: [String]
        let sigmaShift: Shift
        let rotary: Rotary
        let prompt: Prompt
        let referenceImages: References?
    }

    func validate(name: String) throws -> Flux2ReferenceLimits? {
        let d = diffusion, s = d.sigmaShift, p = d.prompt
        let shift = Flux2Schedule.EmpiricalShift.klein
        guard schema == 1, driver == "diffusion", ["flux2_klein_4b", "flux2_klein_4b_reference"].contains(name),
              target.backend == "coreai", target.platform == "macos", target.arch == "any", !target.compiled,
              target.minOS == "27.0", text.tokenizer == "tokenizer/",
              d.latentChannels == Flux2Configuration.latentChannels, d.pixelsPerToken == 16, d.steps == 4,
              d.maxSteps == 50, d.trainTimesteps == 1000, !d.guidance,
              d.textEncoderAsset == "textEncoder", d.decoderAsset == "vaeDecoder",
              d.denoiserSegments == (0..<Flux2Assets.segmentCount).map({ "denoiser\($0)" }),
              s.rule == "empirical", s.lowSteps == shift.lowSteps, s.lowSlope == shift.lowSlope,
              s.lowIntercept == shift.lowIntercept, s.highSteps == shift.highSteps, s.highSlope == shift.highSlope,
              s.highIntercept == shift.highIntercept, s.tokenThreshold == shift.tokenThreshold,
              d.rotary.axes == Flux2Rotary.axes, d.rotary.theta == Flux2Rotary.theta,
              p.template == "chat", p.thinking == "off", p.maxLength == Flux2Prompt.length, p.padding == "right",
              p.padToken == Int(Flux2Prompt.padToken),
              provenance.source.hfRepo == "black-forest-labs/FLUX.2-klein-4B",
              provenance.source.revision == Flux2Assets.sourceRevision else {
            throw AetherError.unsupported("expected the FLUX.2 [klein] 4B macOS recipe")
        }
        var roles = ["textEncoder/encode": "encodeText", "vaeDecoder/decode": "decodeLatents"]
        for k in 0..<Flux2Assets.segmentCount { roles["denoiser\(k)/segment"] = "denoiseSegment" }
        if d.referenceImages != nil { roles["vaeEncoder/encode"] = "encodeLatents" }
        guard functions.mapValues(\.role) == roles else {
            throw AetherError.unsupported("FLUX.2 function roles differ from the supported recipe")
        }
        let bindings: [String: (String, String)] = [
            "input_ids": ("int32", "tokens"), "attention_mask": ("int32", "tokenMask"), "position_ids": ("int32", "positions"),
            "latents": ("float16", "latents"), "text_embeds": ("float16", "textEmbeddings"), "sigma": ("float32", "timestep"),
            "rope_text": ("float32", "textRotary"), "rope_image": ("float32", "imageRotary"),
        ]
        guard bindings.allSatisfy({ inputs[$0.key]?.dtype == $0.value.0 && inputs[$0.key]?.bind == $0.value.1 }),
              inputs["position_ids"]?.mode == "running",
              outputs["text_embeds"]?.dtype == "float16", outputs["velocity"]?.dtype == "float16",
              outputs["image"]?.dtype == "float32" else {
            throw AetherError.unsupported("FLUX.2 tensor declarations differ from the supported recipe")
        }
        guard let r = d.referenceImages else { return nil }
        guard r.kind == "packedLatents", r.encoderAsset == "vaeEncoder", r.positionStride == 10,
              (1...4).contains(r.maxImages), r.maxSide >= 32, r.maxSide <= 1024, r.maxSide.isMultiple(of: 16),
              (2...Flux2Configuration.maximumImageTokens).contains(r.maxTotalTokens),
              inputs["reference_pixels"]?.dtype == "float32", inputs["reference_pixels"]?.bind == "image",
              outputs["reference_latents"]?.dtype == "float32" else {
            throw AetherError.unsupported("FLUX.2 reference-image declarations differ from the supported recipe")
        }
        return Flux2ReferenceLimits(maxImages: r.maxImages, maxSide: r.maxSide, maxTotalTokens: r.maxTotalTokens,
                                    positionStride: r.positionStride)
    }
}

extension Data {
    var hexString: String { map { String(format: "%02x", $0) }.joined() }
}
