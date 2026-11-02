import CryptoKit
import Foundation

@nonexhaustive public enum AssetStoreError: Error, Sendable, Equatable {
    case invalidManifest(String)
    case integrityMismatch(String)
    case unsafePath(String)
    case busy
    case notInstalled
    case fileSystem(Int32)
}

/// A complete portable file. Hashes and byte counts describe the bytes, not their source location.
public struct AssetFile: Codable, Sendable, Equatable {
    public let path: String
    public let byteCount: UInt64
    public let sha256: String
    public init(path: String, byteCount: UInt64, sha256: String) {
        self.path = path; self.byteCount = byteCount; self.sha256 = sha256
    }
}

/// A versioned local installation contract, independent of a model family's execution manifest.
/// Different execution contracts may refer to identical weights without sharing an installation identity.
public struct AssetManifest: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let assetIdentity: String
    public let executionContract: String
    public let requiredFeatures: [String]
    public let files: [AssetFile]

    public init(assetIdentity: String, executionContract: String, files: [AssetFile]) throws {
        schemaVersion = 1
        self.assetIdentity = assetIdentity; self.executionContract = executionContract
        requiredFeatures = []
        self.files = files.sorted { $0.path < $1.path }
        try validate()
    }

    private enum CodingKeys: String, CodingKey { case schemaVersion, assetIdentity, executionContract, requiredFeatures, files }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try values.decode(Int.self, forKey: .schemaVersion)
        assetIdentity = try values.decode(String.self, forKey: .assetIdentity)
        executionContract = try values.decode(String.self, forKey: .executionContract)
        requiredFeatures = try values.decodeIfPresent([String].self, forKey: .requiredFeatures) ?? []
        files = try values.decode([AssetFile].self, forKey: .files).sorted { $0.path < $1.path }
        try validate()
    }

    public func validate() throws {
        guard schemaVersion == 1, requiredFeatures.isEmpty, !assetIdentity.isEmpty, assetIdentity.utf8.count <= 4096,
              !executionContract.isEmpty, executionContract.utf8.count <= 256,
              !files.isEmpty, files.count <= 100_000 else {
            throw AssetStoreError.invalidManifest("unsupported schema or invalid identity/file count")
        }
        var paths = Set<String>()
        for file in files {
            guard Self.validPath(file.path), paths.insert(file.path).inserted,
                  file.sha256.utf8.count == 64,
                  file.sha256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
                throw AssetStoreError.invalidManifest("invalid or duplicate file: \(file.path)")
            }
        }
        // Refuse a file that is also another file's parent before touching the filesystem.
        for path in paths {
            var components = path.split(separator: "/"); components.removeLast()
            while !components.isEmpty {
                guard !paths.contains(components.joined(separator: "/")) else {
                    throw AssetStoreError.invalidManifest("file/directory collision")
                }
                components.removeLast()
            }
        }
    }

    public var installationID: String {
        get throws { try digest(canonicalData()) }
    }

    func canonicalData() throws -> Data {
        try validate()
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        // Decode through the public initializer to normalize file ordering.
        let normalized = try AssetManifest(assetIdentity: assetIdentity, executionContract: executionContract, files: files)
        return try encoder.encode(normalized)
    }

    static func validPath(_ path: String) -> Bool {
        guard !path.isEmpty, path.utf8.count <= 4096, !path.hasPrefix("/"),
              !path.utf8.contains(0), !path.contains("\\") else { return false }
        return path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
}

func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
