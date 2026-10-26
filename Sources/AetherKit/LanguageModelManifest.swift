import CryptoKit
import Foundation

/// Versioned registration for a local dense language model. Registration is an
/// explicit caller-owned contract; it does not establish numerical qualification.
public struct LanguageModelManifest: Codable, Sendable {
    public enum Platform: String, Codable, Sendable { case macos, ios }
    public let schemaVersion: Int
    public let modelDirectory: String
    public let tokenizerDirectory: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let modelSHA256: String
    public let tokenizerSHA256: String
    public let maximumContextLength: Int
    public let stopTokenIDs: [Int]
    public let platform: Platform
    public let quantization: String
    public let supportsReasoning: Bool?

    public static func load(from directory: URL) throws -> Self {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: directory.appendingPathComponent("language-model.json")))
    }
    public func verify(in directory: URL, platform expectedPlatform: Platform) throws -> URL {
        guard schemaVersion == 1, platform == expectedPlatform,
              maximumContextLength > 0, maximumContextLength <= 16384, !stopTokenIDs.isEmpty,
              stopTokenIDs.allSatisfy({ $0 >= 0 }), Self.hex(sourceRevision, count: 40),
              Self.hex(modelSHA256, count: 64), Self.hex(tokenizerSHA256, count: 64),
              !sourceRepository.isEmpty else { throw AetherError.invalidModel("invalid language model registration or platform") }
        let model = try Self.child(modelDirectory, of: directory)
        let tokenizer = try Self.child(tokenizerDirectory, of: directory)
        guard try Self.fileHash(model.appendingPathComponent("main.mlirb")) == modelSHA256,
              try Self.fileHash(tokenizer.appendingPathComponent("tokenizer.json")) == tokenizerSHA256 else {
            throw AetherError.invalidModel("language model or tokenizer checksum mismatch")
        }
        return model
    }
    public static func fileHash(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hash = SHA256()
        // Drain each autoreleased chunk; otherwise a multi-GB model stays resident
        // until the synchronous verification returns.
        while try autoreleasepool(invoking: { () throws -> Bool in
            guard let data = try file.read(upToCount: 4 * 1024 * 1024), !data.isEmpty else { return false }
            hash.update(data: data)
            return true
        }) {}
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    private static func hex(_ string: String, count: Int) -> Bool {
        string.utf8.count == count && string.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
    private static func child(_ path: String, of directory: URL) throws -> URL {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.split(separator: "/").contains("..") else {
            throw AetherError.invalidModel("language model path must be relative to its bundle")
        }
        let base = directory.standardizedFileURL.resolvingSymlinksInPath()
        let url = directory.appendingPathComponent(path).standardizedFileURL.resolvingSymlinksInPath()
        guard url.path.hasPrefix(base.path + "/") else { throw AetherError.invalidModel("language model path escapes bundle") }
        return url
    }
}
