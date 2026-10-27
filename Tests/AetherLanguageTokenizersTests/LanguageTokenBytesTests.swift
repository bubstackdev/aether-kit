import AetherKit
@testable import AetherLanguageTokenizers
import Foundation
import Testing

private let environment = ProcessInfo.processInfo.environment

/// GPT-2's byte-to-unicode map, written out independently of the implementation's inverse: a vocabulary token's
/// spelling of raw bytes.
private func spelled(_ bytes: some Sequence<UInt8>) -> String {
    let printable = Set(Array(33...126) + Array(161...172) + Array(174...255))
    var shifted: [Int: Int] = [:]
    var next = 256
    for byte in 0...255 where !printable.contains(byte) { shifted[byte] = next; next += 1 }
    return String(String.UnicodeScalarView(bytes.map { Unicode.Scalar(UInt32(shifted[Int($0)] ?? Int($0)))! }))
}

private func tokenizerJSON(vocab: [String: Int], added: [(Int, String, Bool)], decoder: [String: Any]) throws -> Data {
    try JSONSerialization.data(withJSONObject: [
        "model": ["type": "BPE", "vocab": vocab],
        "added_tokens": added.map { ["id": $0.0, "content": $0.1, "special": $0.2] },
        "decoder": decoder,
    ] as [String: Any])
}

struct LanguageTokenBytesTests {
    @Test func byteLevelTokensDecodeToTheirBytes() throws {
        let moon = Array("🌙".utf8)  // F0 9F 8C 99
        let pieces: [[UInt8]] = [Array("Hello".utf8), Array(" world".utf8), [0x0A], Array("é".utf8), Array(" 🌙".utf8),
                                 Array(moon.prefix(2)), Array(moon.suffix(2))]  // a scalar split across two tokens
        var vocab: [String: Int] = [:]
        for (id, piece) in pieces.enumerated() { vocab[spelled(piece)] = id }
        let data = try tokenizerJSON(vocab: vocab, added: [(8, "<|im_end|>", true), (9, "<think>", false)],
                                     decoder: ["type": "ByteLevel"])
        let table = try LanguageTokenBytes(tokenizerJSON: data)
        #expect(table.count == 10)
        for (id, piece) in pieces.enumerated() { #expect(table.bytes(of: id).map(Array.init) == piece, "token \(id)") }
        #expect(table.bytes(of: 7) == nil && table.bytes(of: 10) == nil && table.bytes(of: -1) == nil)
        #expect(table.bytes(of: 8).map(Array.init) == Array("<|im_end|>".utf8) && table.isSpecial(8))
        #expect(table.bytes(of: 9).map(Array.init) == Array("<think>".utf8) && !table.isSpecial(9))
        #expect(String(decoding: table.bytes(of: [0, 1, 8, 2]), as: UTF8.self) == "Hello world\n")
        #expect(String(decoding: table.bytes(of: [5, 6]), as: UTF8.self) == "🌙")
        #expect(String(decoding: table.bytes(of: [9, 0, 8], skipSpecialTokens: false), as: UTF8.self) == "<think>Hello<|im_end|>")
    }

    @Test func everyByteRoundTrips() throws {
        var vocab: [String: Int] = [:]
        for byte in 0...255 { vocab[spelled([UInt8(byte)])] = byte }
        let table = try LanguageTokenBytes(tokenizerJSON: try tokenizerJSON(vocab: vocab, added: [], decoder: [
            "type": "Sequence", "decoders": [["type": "Replace"], ["type": "ByteLevel"]],
        ]))
        #expect((0...255).allSatisfy { table.bytes(of: $0).map(Array.init) == [UInt8($0)] })
        #expect(LanguageTokenBytes.unicodeToByte.count == 256)
    }

    @Test func nonByteLevelTokenizersAreRefused() throws {
        let data = try tokenizerJSON(vocab: ["▁Hello": 0], added: [], decoder: ["type": "Metaspace"])
        #expect(throws: AetherError.self) { try LanguageTokenBytes(tokenizerJSON: data) }
    }

    /// Opt-in: on every model in the chat oracle (pinned hub snapshots under `AETHER_LANGUAGE_TOKENIZER_HUB`), the bytes
    /// of each rendered prompt are exactly upstream's rendered text, and of the decode case its decoded text.
    @Test(.enabled(if: environment["AETHER_LANGUAGE_TOKENIZER_HUB"] != nil))
    func bytesMatchUpstreamTextForEveryOracleCase() async throws {
        struct Oracle: Decodable {
            struct Model: Decodable { let repository: String; let revision: String; let cases: [Case]; let decode: Case }
            struct Case: Decodable { let tokenIDs: [Int]; let text: String }
            let models: [Model]
        }
        let hub = URL(fileURLWithPath: try #require(environment["AETHER_LANGUAGE_TOKENIZER_HUB"]))
        let fixture = try #require(Bundle.module.url(forResource: "chat-oracle", withExtension: "json", subdirectory: "Fixtures"))
        let oracle = try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: fixture))
        var checked = 0
        for model in oracle.models {
            let snapshot = hub.appendingPathComponent("models--" + model.repository.replacingOccurrences(of: "/", with: "--"))
                .appendingPathComponent("snapshots/\(model.revision)")
            let table = try LanguageTokenizer.tokenBytes(directory: snapshot)
            let tokenizer = try await LanguageTokenizer.open(directory: snapshot)
            for entry in model.cases {
                #expect(String(decoding: table.bytes(of: entry.tokenIDs, skipSpecialTokens: false), as: UTF8.self) == entry.text,
                        "\(model.repository)")
                checked += 1
            }
            #expect(String(decoding: table.bytes(of: model.decode.tokenIDs), as: UTF8.self) == model.decode.text)
            #expect(String(decoding: table.bytes(of: model.decode.tokenIDs), as: UTF8.self) == tokenizer.decode(model.decode.tokenIDs))
            #expect(table.count == 151_669 && table.isSpecial(151_645) && !table.isSpecial(151_667), "\(model.repository)")
        }
        #expect(checked == 24)
    }
}
