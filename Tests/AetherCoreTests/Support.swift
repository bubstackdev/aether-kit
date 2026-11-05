import Foundation
import Testing
@testable import AetherCore

/// Fixtures copied into the test bundle (`Fixtures/`).
enum Fixtures {
    static var root: URL { Bundle.module.resourceURL!.appending(path: "Fixtures") }

    static func data(_ path: String) throws -> Data {
        try Data(contentsOf: root.appending(path: path))
    }

    static func zoo(_ id: String, _ file: String) throws -> Data {
        try data("zoo/\(id)/\(file)")
    }

    static func signature(_ id: String) throws -> GraphSignature {
        try JSONDecoder().decode(GraphSignature.self, from: zoo(id, "signature.json"))
    }

    /// The adapter input for a catalog entry's pinned zoo bundle: its small files plus the builtin catalog listing.
    static func adapterInput(_ id: String) throws -> ZooBundleAdapter.Input {
        let entry = try Catalog.builtin.entry(id)
        let variant = try #require(entry.variants["macos"])
        return ZooBundleAdapter.Input(
            metadata: try zoo(id, "metadata.json"),
            signatures: ["main": try signature(id)],
            files: variant.files.map(\.path),
            tokenizerConfig: try zoo(id, "tokenizer_config.json"),
            chatTemplate: String(decoding: try zoo(id, "chat_template.jinja"), as: UTF8.self),
            catalogID: id, repo: entry.repo, revision: entry.revision)
    }

    /// `aether-kit/` (for catalog/catalog.json).
    static var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
}

/// A fresh temporary directory, removed when `body` returns.
func withTemporaryDirectory<R>(_ body: (URL) async throws -> R) async throws -> R {
    let url = FileManager.default.temporaryDirectory.appending(path: "aether-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: url) }
    return try await body(url)
}

/// The field named by a manifest error, or nil for other errors.
func field(of error: any Error) -> String? {
    switch error as? AetherError {
    case .manifestInvalid(let field, _)?, .manifestMismatch(let field, _, _)?: field
    default: nil
    }
}

/// Runs `body`, expecting a manifest error naming `expected`.
func expectField(_ expected: String, sourceLocation: SourceLocation = #_sourceLocation, _ body: () throws -> Void) {
    do {
        try body()
        Issue.record("expected an error naming '\(expected)', got success", sourceLocation: sourceLocation)
    } catch {
        #expect(field(of: error) == expected, "\(error)", sourceLocation: sourceLocation)
    }
}
