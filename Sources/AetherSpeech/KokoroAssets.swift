import AetherAssets
import AetherKit
import CryptoKit
import Foundation

/// One exported Core AI model directory (`main.mlirb`, `main.hash`, `metadata.json`).
/// Opening checks the declaration and metadata; call `verify()` to stream the model bytes.
public struct KokoroModelAsset: Sendable, Equatable {
    public let url: URL
    public let relativePath: String
    public let function: String
    public let declaredSHA256: String
    public let metadataSHA256: String
    let declarationSHA256: String

    /// Streams `main.mlirb` and rehashes `metadata.json` and `main.hash`, each resolved inside the
    /// model directory (symlinks may not escape it).
    public func verify() throws {
        do {
            try declared.verify()
            guard try BundleFiles.sha256(of: BundleFiles.confined("main.hash", in: url)) == declarationSHA256 else {
                throw AetherError.invalidModel("Kokoro model hash mismatch: \(relativePath)")
            }
        } catch is AssetStoreError {
            throw AetherError.invalidModel("Kokoro model hash mismatch: \(relativePath)")
        }
    }

    var declared: DeclaredModelAsset {
        DeclaredModelAsset(url: url, declaredSHA256: declaredSHA256, metadataSHA256: metadataSHA256)
    }
}

/// A bundled voice style table: 510 rows of 256 float32 values.
public struct KokoroVoiceAsset: Sendable, Equatable {
    public let voice: KokoroVoice
    public let url: URL
    public let relativePath: String
    public let sha256: String
}

/// Owned style rows for one voice, hash-verified when loaded.
public struct KokoroVoiceStyles: Sendable, Equatable {
    public static let rows = 510
    public static let dimension = 256
    public let voice: KokoroVoice
    public let values: [Float]

    public init(voice: KokoroVoice, values: [Float]) throws {
        guard values.count == Self.rows * Self.dimension, values.allSatisfy(\.isFinite) else {
            throw AetherError.invalidModel("Kokoro voice styles must be 510 x 256 finite values")
        }
        self.voice = voice
        self.values = values
    }

    /// Upstream's `pack[len(phonemes) - 1]`: [1,256], decoder half first, prosody half second.
    public func style(for phonemes: KokoroPhonemes) throws -> FloatTensor {
        let start = phonemes.styleRow * Self.dimension
        return try FloatTensor(shape: [1, Self.dimension], values: Array(values[start..<start + Self.dimension]))
    }
}

/// A Kokoro-82M Core AI speech bundle exported by `Forge/kokoro/export_kokoro.py`.
///
/// `kokoro.json` is the execution contract: source provenance, sample format, vocabulary,
/// model directories and every file's SHA-256. Opening validates the contract, confines
/// every path, checks model declarations and metadata, and sizes voice files; it does not
/// stream model bytes. Keep external files immutable while any session uses them.
public struct KokoroAssets: Sendable {
    public static let sampleRate = 24_000
    public static let samplesPerFrame = 600
    public static let harmonics = 9

    public let root: URL
    public let durations: KokoroModelAsset
    public let decoder: KokoroModelAsset
    public let vocabulary: KokoroVocabulary
    public let voices: [KokoroVoice: KokoroVoiceAsset]
    public let maximumFrames: Int
    /// Fixed decoder frame counts, ascending and ending at `maximumFrames`. When present the decoder
    /// takes a `frames` input and each utterance is zero-padded to the smallest bucket that fits, so
    /// the GPU specializes at most once per bucket. Empty for bundles exported without buckets.
    public let decoderBuckets: [Int]
    /// Decoder weights: `float32`, `float16` (float16 conv/linear compute), `float16-storage` (float16 weights,
    /// float32 compute) or weight-only `int8`.
    /// Graph inputs and outputs are float32 for every precision.
    public let decoderPrecision: String
    public let source: Source
    /// SHA-256 of `kokoro.json`, which pins every other file.
    public let fingerprint: String
    let contractByteCount: UInt64

    public struct Source: Sendable, Equatable, Codable {
        public let repository: String
        public let revision: String
        public let checkpointSHA256: String
        public let license: String
        enum CodingKeys: String, CodingKey {
            case repository, revision, license
            case checkpointSHA256 = "checkpoint_sha256"
        }
    }

    /// The decoder shape used for an utterance of `frames` frames.
    public func decoderFrames(for frames: Int) -> Int {
        decoderBuckets.first { $0 >= frames } ?? frames
    }

    public var availableVoices: [KokoroVoice] { voices.keys.sorted { $0.identifier < $1.identifier } }

    public static func open(at root: URL) throws -> Self {
        guard root.isFileURL else { throw AetherError.invalidModel("Kokoro bundle must be local") }
        let root = root.standardizedFileURL.resolvingSymlinksInPath()
        let contractURL = try confinedFile("kokoro.json", root: root)
        let data = try Data(contentsOf: contractURL)
        guard data.count <= 1 << 20 else { throw AetherError.invalidModel("kokoro.json is too large") }
        let contract = try JSONDecoder().decode(Contract.self, from: data)
        try contract.validate()

        func model(_ entry: Contract.Model) throws -> KokoroModelAsset {
            let directory = try confinedFile(entry.path, root: root, directory: true)
            guard Set(entry.sha256.keys) == ["main.mlirb", "main.hash", "metadata.json"],
                  let stream = entry.sha256["main.mlirb"], let metadata = entry.sha256["metadata.json"],
                  let declaration = entry.sha256["main.hash"] else {
                throw AetherError.invalidModel("Kokoro model entries must pin main.mlirb, main.hash and metadata.json")
            }
            let hashData = try Data(contentsOf: confinedFile(entry.path + "/main.hash", root: root))
            guard hashData.count == 32, hashData.hex == stream, digest(hashData) == declaration,
                  try hashFile(confinedFile(entry.path + "/metadata.json", root: root)) == metadata else {
                throw AetherError.invalidModel("Kokoro model declaration differs from kokoro.json: \(entry.path)")
            }
            _ = try confinedFile(entry.path + "/main.mlirb", root: root)
            return KokoroModelAsset(url: directory, relativePath: entry.path, function: entry.function,
                                    declaredSHA256: stream, metadataSHA256: metadata, declarationSHA256: declaration)
        }
        let durations = try model(contract.models.durations)
        let decoder = try model(contract.models.decoder)
        guard durations.url != decoder.url, durations.function == "durations", decoder.function == "decode" else {
            throw AetherError.invalidModel("Kokoro requires distinct durations and decode models")
        }
        var voices: [KokoroVoice: KokoroVoiceAsset] = [:]
        for entry in contract.voices {
            let voice = try KokoroVoice(entry.id)
            guard voice.language.bcp47 == entry.language, voice.gender.rawValue == String(entry.gender.prefix(1)) else {
                throw AetherError.invalidModel("Kokoro voice metadata differs from its identifier: \(entry.id)")
            }
            let url = try confinedFile(entry.file, root: root)
            let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize
            guard size == KokoroVoiceStyles.rows * KokoroVoiceStyles.dimension * 4, isSHA256(entry.sha256),
                  voices.updateValue(KokoroVoiceAsset(voice: voice, url: url, relativePath: entry.file, sha256: entry.sha256),
                                     forKey: voice) == nil else {
                throw AetherError.invalidModel("invalid or duplicate Kokoro voice: \(entry.id)")
            }
        }
        guard !voices.isEmpty, Set(voices.values.map(\.url)).count == voices.count else {
            throw AetherError.invalidModel("Kokoro bundle declares no distinct voices")
        }
        return Self(root: root, durations: durations, decoder: decoder, vocabulary: try KokoroVocabulary(contract.vocabulary),
                    voices: voices, maximumFrames: contract.maximumFrames, decoderBuckets: contract.decoderBuckets ?? [],
                    decoderPrecision: contract.decoderPrecision ?? "float32",
                    source: contract.source,
                    fingerprint: digest(data), contractByteCount: UInt64(data.count))
    }

    /// Reopens a verified installation, checking that its contract exactly covers this bundle.
    public static func open(in installation: AssetReadLease) throws -> Self {
        let assets = try open(at: installation.directory)
        guard try assets.installationManifest().installationID == installation.installationID else {
            throw AetherError.invalidModel("installation is not the exact Kokoro bundle")
        }
        return assets
    }

    /// Streams both model directories through SHA-256 (about 0.3 GB) and every voice file.
    public func verify() throws {
        do {
            // Both streams hash concurrently; the first failure in durations, decoder order is reported.
            try DeclaredModelAsset.verifyConcurrently([durations.declared, decoder.declared])
        } catch let error as AssetStoreError {
            throw AetherError.invalidModel("Kokoro model verification failed: \(error)")
        }
        for model in [durations, decoder] {
            guard (try? BundleFiles.sha256(of: BundleFiles.confined("main.hash", in: model.url))) == model.declarationSHA256 else {
                throw AetherError.invalidModel("Kokoro model hash mismatch: \(model.relativePath)")
            }
        }
        for asset in voices.values.sorted(by: { $0.relativePath < $1.relativePath }) {
            guard try hashFile(asset.url) == asset.sha256 else {
                throw AetherError.invalidModel("Kokoro voice hash mismatch: \(asset.voice)")
            }
        }
    }

    /// Reads and hash-checks one voice's style table.
    public func styles(for voice: KokoroVoice) throws -> KokoroVoiceStyles {
        guard let asset = voices[voice] else {
            throw AetherError.invalidRequest("voice \(voice) is not in this Kokoro bundle")
        }
        let data = try Data(contentsOf: asset.url)
        guard digest(data) == asset.sha256 else { throw AetherError.invalidModel("Kokoro voice hash mismatch: \(voice)") }
        let values = data.withUnsafeBytes { raw in
            (0..<(data.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
        return try KokoroVoiceStyles(voice: voice, values: values)
    }

    /// The portable installation contract: kokoro.json, both model directories and every voice.
    /// Hashes come from kokoro.json; installation verifies the copied bytes.
    public func installationManifest() throws -> AssetManifest {
        guard try Self.open(at: root).fingerprint == fingerprint else {
            throw AetherError.invalidModel("kokoro.json changed after bundle opening")
        }
        func file(_ path: String, _ sha256: String) throws -> AssetFile {
            let url = try confinedFile(path, root: root)
            guard url.path == root.appendingPathComponent(path).standardizedFileURL.path else {
                throw AetherError.unsupported("portable Kokoro installation requires direct file paths")
            }
            let size = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber
            guard let size else { throw AetherError.invalidModel("missing Kokoro file size: \(path)") }
            return AssetFile(path: path, byteCount: size.uint64Value, sha256: sha256)
        }
        var files = [try file("kokoro.json", fingerprint)]
        for model in [durations, decoder] {
            files.append(try file(model.relativePath + "/main.mlirb", model.declaredSHA256))
            files.append(try file(model.relativePath + "/main.hash", model.declarationSHA256))
            files.append(try file(model.relativePath + "/metadata.json", model.metadataSHA256))
        }
        for asset in voices.values { files.append(try file(asset.relativePath, asset.sha256)) }
        return try AssetManifest(assetIdentity: "kokoro-82m/1:" + fingerprint,
                                 executionContract: "kokoro.speech-bundle/1", files: files)
    }
}

private struct Contract: Decodable {
    let schema: Int
    let family: String
    let precision: String
    let source: KokoroAssets.Source
    let sampleRate: Int
    let samplesPerFrame: Int
    let harmonics: Int
    let maximumPhonemes: Int
    let maximumFrames: Int
    let decoderBuckets: [Int]?
    let decoderPrecision: String?
    let styleDimension: Int
    let boundaryToken: Int
    let vocabulary: [String: Int]
    let models: Models
    let voices: [Voice]

    struct Models: Decodable { let durations: Model; let decoder: Model }
    struct Model: Decodable { let path: String; let function: String; let sha256: [String: String] }
    struct Voice: Decodable {
        let id: String
        let language: String
        let gender: String
        let file: String
        let sha256: String
    }

    func validate() throws {
        guard schema == 1, family == "kokoro-82m", precision == "float32",
              sampleRate == KokoroAssets.sampleRate, samplesPerFrame == KokoroAssets.samplesPerFrame,
              harmonics == KokoroAssets.harmonics, maximumPhonemes == KokoroPhonemes.maximumCount,
              (2...65_536).contains(maximumFrames), styleDimension == KokoroVoiceStyles.dimension,
              ["float32", "float16", "float16-storage", "int8"].contains(decoderPrecision ?? "float32"),
              decoderBuckets.map({ b in !b.isEmpty && b.count <= 32 && b.first! >= 2 && b.last == maximumFrames
                  && zip(b, b.dropFirst()).allSatisfy { $0 < $1 } }) ?? true,
              boundaryToken == Int(KokoroVocabulary.boundaryToken), source.repository == "hexgrad/Kokoro-82M",
              source.revision.utf8.count == 40, isSHA256(source.checkpointSHA256), source.license == "Apache-2.0",
              !voices.isEmpty, voices.count <= 256,
              [models.durations, models.decoder].allSatisfy({ $0.sha256.values.allSatisfy(isSHA256) }) else {
            throw AetherError.unsupported("expected the float32 Kokoro-82M speech bundle contract")
        }
    }
}

func isSHA256(_ value: String) -> Bool {
    value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
}

func confinedFile(_ relative: String, root: URL, directory: Bool = false) throws -> URL {
    guard !relative.isEmpty, !relative.hasPrefix("/"), !relative.contains("\\"),
          !relative.split(separator: "/", omittingEmptySubsequences: false).contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }) else {
        throw AetherError.invalidModel("Kokoro asset path must stay inside its bundle")
    }
    let url = root.appendingPathComponent(relative).standardizedFileURL.resolvingSymlinksInPath()
    guard url.pathComponents.starts(with: root.pathComponents), url.pathComponents.count > root.pathComponents.count else {
        throw AetherError.invalidModel("Kokoro asset symlink escapes its bundle")
    }
    let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey])
    guard directory ? values.isDirectory == true : values.isRegularFile == true else {
        throw AetherError.invalidModel("missing Kokoro \(directory ? "directory" : "file"): \(relative)")
    }
    return url
}

func hashFile(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var hash = SHA256()
    while try autoreleasepool(invoking: { () throws -> Bool in
        guard let data = try file.read(upToCount: 4 << 20), !data.isEmpty else { return false }
        try Task.checkCancellation()
        hash.update(data: data)
        return true
    }) {}
    return Data(hash.finalize()).hex
}

func digest(_ data: Data) -> String { Data(SHA256.hash(data: data)).hex }

extension Data {
    var hex: String { map { String(format: "%02x", $0) }.joined() }
}
