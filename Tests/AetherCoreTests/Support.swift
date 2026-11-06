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

    static let goldenIDs = [
        "qwen3-0.6b", "gemma-3-4b-it", "minicpm5-1b", "youtu-llm-2b", "nanbeige4.1-3b", "qwen3.5-0.8b",
        "lfm2.5-1.2b", "granite-4.0-h-1b", "nemotron-3-nano-4b", "gemma-4-e2b",
    ]

    static func signatures(_ id: String) throws -> [String: GraphSignature] {
        try JSONDecoder().decode([String: GraphSignature].self, from: zoo(id, "signatures.json"))
    }

    static func optional(_ id: String, _ file: String) -> Data? { try? zoo(id, file) }

    /// The adapter input for a catalog entry's macOS variant: its fixture files plus the builtin catalog.
    static func adapterInput(_ id: String) throws -> ZooBundleAdapter.Input {
        ZooBundleAdapter.Input(
            entry: try Catalog.builtin.entry(id), variant: "macos",
            metadata: try zoo(id, "metadata.json"),
            signatures: try signatures(id),
            tokenizerConfig: optional(id, "tokenizer_config.json"),
            chatTemplate: optional(id, "chat_template.jinja").map { String(decoding: $0, as: UTF8.self) })
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
