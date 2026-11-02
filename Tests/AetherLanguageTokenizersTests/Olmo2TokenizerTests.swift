import AetherLanguageTokenizers
import CryptoKit
import Foundation
import Testing

/// Pinned Transformers oracle; reads tokenizer files only when explicitly opted in.
@Test(.enabled(if: ProcessInfo.processInfo.environment["AETHER_OLMO2_CHECKPOINT"] != nil))
func olmo2ChatTemplateAndByteLevelTokenizerMatchUpstream() async throws {
    let directory = URL(filePath: try #require(ProcessInfo.processInfo.environment["AETHER_OLMO2_CHECKPOINT"]))
    let fixture = try #require(Bundle.module.url(forResource: "olmo2-chat-oracle", withExtension: "json", subdirectory: "Fixtures"))
    let oracle = try JSONDecoder().decode(Olmo2TokenOracle.self, from: Data(contentsOf: fixture))
    let data = try Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))
    #expect(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == oracle.tokenizerSHA256)
    let tokenizer = try await LanguageTokenizer.open(directory: directory)
    for entry in oracle.cases {
        let messages = try entry.messages.map {
            try LanguageChatMessage(role: #require(LanguageChatMessage.Role(rawValue: $0.role)),
                                    parts: $0.parts.map { .text($0.text) })
        }
        #expect(try tokenizer.render(messages, thinking: entry.thinking).tokenIDs == entry.tokenIDs, "\(entry.name)")
        #expect(tokenizer.encode(entry.text) == entry.tokenIDs, "\(entry.name)")
        #expect(tokenizer.decode(entry.tokenIDs, skipSpecialTokens: false) == entry.text, "\(entry.name)")
    }
    #expect(tokenizer.decode(oracle.decode.tokenIDs) == oracle.decode.text)
}

private struct Olmo2TokenOracle: Decodable {
    let tokenizerSHA256: String
    let cases: [Case]
    let decode: Decode
    struct Case: Decodable {
        let name: String, text: String
        let thinking: Bool
        let tokenIDs: [Int]
        let messages: [Message]
    }
    struct Message: Decodable {
        let role: String
        let parts: [Part]
    }
    struct Part: Decodable { let text: String }
    struct Decode: Decodable { let tokenIDs: [Int]; let text: String }
}
