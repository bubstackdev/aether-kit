import Foundation
import Testing
@testable import AetherRuntime

/// The zoo fixtures live with AetherCoreTests; read them by path so they have one home.
enum Fixtures {
    static var zoo: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "AetherCoreTests/Fixtures/zoo")
    }

    static func data(_ id: String, _ file: String) throws -> Data {
        try Data(contentsOf: zoo.appending(path: "\(id)/\(file)"))
    }

    /// Every catalog entry with a golden (the chat entries downloaded in lab/models).
    static let goldenIDs = [
        "qwen3-0.6b", "gemma-3-4b-it", "minicpm5-1b", "youtu-llm-2b", "nanbeige4.1-3b", "qwen3.5-0.8b",
        "lfm2.5-1.2b", "granite-4.0-h-1b", "nemotron-3-nano-4b", "gemma-4-e2b",
    ]

    /// The recorded decoder signatures, keyed by function name.
    static func signatures(_ id: String) throws -> [String: GraphSignature] {
        try JSONDecoder().decode([String: GraphSignature].self, from: data(id, "signatures.json"))
    }

    static func manifest(_ id: String) throws -> Manifest {
        try Manifest.decode(data(id, "aether.json"))
    }
}

func withTemporaryDirectory<R>(_ body: (URL) async throws -> R) async throws -> R {
    let url = FileManager.default.temporaryDirectory.appending(path: "aether-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: url) }
    return try await body(url)
}

func field(of error: any Error) -> String? {
    switch error as? AetherError {
    case .manifestInvalid(let field, _)?, .manifestMismatch(let field, _, _)?: field
    default: nil
    }
}

func tensorFailure(of error: any Error) -> (name: String, reason: AetherError.TensorFailure)? {
    if case .tensor(let name, let reason)? = error as? AetherError { return (name, reason) }
    return nil
}

func expectField(_ expected: String, sourceLocation: SourceLocation = #_sourceLocation, _ body: () throws -> Void) {
    do {
        try body()
        Issue.record("expected an error naming '\(expected)', got success", sourceLocation: sourceLocation)
    } catch {
        #expect(field(of: error) == expected, "\(error)", sourceLocation: sourceLocation)
    }
}
