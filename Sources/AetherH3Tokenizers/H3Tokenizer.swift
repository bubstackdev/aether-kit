import AetherKit
import CryptoKit
import Foundation
import Tokenizers

/// Optional local tokenizer adapter for H3's Qwen3-VL tokenizer. Never downloads files.
/// Tokenization is the verbatim prompt with no chat template and no added special tokens, as
/// the H3 conditioner expects. H3's own tokens (`<d>`, `</d>`, `<|cutoff|>`, ...) are listed only
/// in `tokenizer_config.json`; Transformers appends those absent from `tokenizer.json` after its
/// highest added id, in config order. This adapter reproduces that table, splits the prompt on
/// every added token (exact, unnormalized matches, longest first) and encodes the text between
/// them, as Transformers does.
public struct H3Tokenizer: Sendable {
    /// Files read from the tokenizer directory, with their SHA-256 at open.
    public let fileHashes: [String: String]
    /// Every added token and its id, including H3's appended ones.
    public let addedTokens: [String: Int32]
    private let tokenizer: any Tokenizer

    public static let files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]

    /// `directory` is the checkpoint's `FL2VA/tokenizer` folder. Files must stay immutable
    /// while the tokenizer is in use; they are hashed before and after loading.
    public static func open(directory: URL) async throws -> Self {
        let before = try hashes(directory)
        let added = try addedTokenTable(directory)
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        guard try hashes(directory) == before else { throw AetherError.invalidModel("H3 tokenizer files changed while loading") }
        return Self(fileHashes: before, addedTokens: added, tokenizer: tokenizer)
    }

    public func tokenIDs(_ prompt: String) throws -> [Int32] {
        guard !prompt.isEmpty else { throw AetherError.invalidRequest("H3 prompt must not be empty") }
        let ordered = addedTokens.keys.sorted { $0.utf8.count > $1.utf8.count }
        var ids: [Int32] = [], text = Substring(prompt), pending = ""
        func flush() throws {
            guard !pending.isEmpty else { return }
            let encoded = tokenizer.encode(text: pending, addSpecialTokens: false)
            guard encoded.allSatisfy({ $0 >= 0 && $0 <= Int(Int32.max) }) else { throw AetherError.invalidRequest("H3 prompt produced invalid tokens") }
            ids += encoded.map(Int32.init); pending = ""
        }
        while !text.isEmpty {
            if let token = ordered.first(where: { text.utf8.starts(with: $0.utf8) }), let id = addedTokens[token] {
                try flush(); ids.append(id); text = text[text.utf8.index(text.startIndex, offsetBy: token.utf8.count)...]
            } else {
                pending.append(text.removeFirst())
            }
        }
        try flush()
        guard !ids.isEmpty else { throw AetherError.invalidRequest("H3 prompt produced no tokens") }
        return ids
    }

    static func addedTokenTable(_ directory: URL) throws -> [String: Int32] {
        struct Added: Decodable { let id: Int32; let content: String }
        struct Model: Decodable { let added_tokens: [Added] }
        struct Config: Decodable { let additional_special_tokens: [String]? }
        let model = try JSONDecoder().decode(Model.self, from: Data(contentsOf: directory.appending(path: "tokenizer.json")))
        let config = try JSONDecoder().decode(Config.self, from: Data(contentsOf: directory.appending(path: "tokenizer_config.json")))
        var table = Dictionary(model.added_tokens.map { ($0.content, $0.id) }, uniquingKeysWith: { first, _ in first })
        guard table.count == model.added_tokens.count, var next = model.added_tokens.map(\.id).max() else {
            throw AetherError.invalidModel("H3 tokenizer added tokens are missing or duplicated")
        }
        for token in config.additional_special_tokens ?? [] where table[token] == nil {
            guard !token.isEmpty, next < Int32.max else { throw AetherError.invalidModel("invalid H3 additional token") }
            next += 1; table[token] = next
        }
        return table
    }

    private static func hashes(_ directory: URL) throws -> [String: String] {
        var result: [String: String] = [:]
        for name in files {
            let url = directory.appending(path: name)
            guard FileManager.default.fileExists(atPath: url.path) else { throw AetherError.invalidModel("H3 tokenizer is missing \(name)") }
            result[name] = SHA256.hash(data: try Data(contentsOf: url, options: .mappedIfSafe)).map { String(format: "%02x", $0) }.joined()
        }
        return result
    }
}
