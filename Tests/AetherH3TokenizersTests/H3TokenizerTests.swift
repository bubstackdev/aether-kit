import AetherKit
import AetherH3Tokenizers
import Foundation
import Testing

/// Opt-in: set `AETHER_H3_TOKENIZER` to the checkpoint's `FL2VA/tokenizer` folder (local files
/// only). Token IDs must equal the Hugging Face Transformers oracle for the same files.
@Test(.enabled(if: ProcessInfo.processInfo.environment["AETHER_H3_TOKENIZER"] != nil))
func h3TokenizerMatchesTransformersIncludingDialogueTokens() async throws {
    struct Case: Decodable { let prompt: String; let ids: [Int32] }
    struct Oracle: Decodable { let files: [String: String]; let special: [String: Int32]; let cases: [Case] }
    let url = try #require(Bundle.module.url(forResource: "token-oracle", withExtension: "json", subdirectory: "Fixtures"))
    let oracle = try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: url))
    let directory = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["AETHER_H3_TOKENIZER"]))
    let tokenizer = try await H3Tokenizer.open(directory: directory)
    #expect(tokenizer.fileHashes == oracle.files)
    for (token, id) in oracle.special { #expect(tokenizer.addedTokens[token] == id, "\(token)") }
    for item in oracle.cases { #expect(try tokenizer.tokenIDs(item.prompt) == item.ids, "\(item.prompt)") }
    // The prototype's production profile was exported for this prompt at 49 text rows.
    #expect(try oracle.cases.last.map { try tokenizer.tokenIDs($0.prompt).count } == 49)
    #expect(throws: AetherError.self) { try tokenizer.tokenIDs("") }
}

@Test func h3TokenizerRefusesAFolderWithoutItsFiles() async throws {
    let empty = FileManager.default.temporaryDirectory.appending(path: "h3-tokenizer-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: empty) }
    await #expect(throws: AetherError.self) { try await H3Tokenizer.open(directory: empty) }
}
