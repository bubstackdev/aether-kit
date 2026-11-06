public import Foundation
public import AetherCore

/// CLIP's byte-level BPE as `CLIPTokenizerFast` applies it (MANIFEST.md §5, `tokenizer.renderer: clip-bpe`): NFC and
/// lowercase normalization, whitespace dropped, end-of-word `</w>` on each word's last piece, BOS and EOS, truncation to
/// the context length and padding with the pad id. swift-transformers' BPE lacks the `</w>` suffix, so this is its
/// own small implementation, checked id-for-id against the pinned tokenizers (the SD-Turbo and CLIP gates' cases).
///
/// The added tokens, which split the text before BPE and map straight to their ids, come from the tokenizer's own
/// files: `tokenizer.json`'s `added_tokens`, `tokenizer_config.json`'s `added_tokens_decoder` and the special tokens
/// (`bos`, `eos`, `unk`, `pad`) it and `special_tokens_map.json` name, besides BOS and EOS. So `!` is the pad token 0 in
/// SD 2.x's tokenizer, which adds it, and an ordinary piece in OpenAI CLIP's (`hello!` → 3306, 256).
///
/// Shared by AetherImage (SD-Turbo's prompt) and AetherVision (CLIP's text tower), so it lives here, below both.
public struct CLIPTokenizer: Sendable {
    public static let bos: Int32 = 49406
    public static let eos: Int32 = 49407

    /// Ids per prompt, padded (77 for SD-Turbo and CLIP ViT-B/32).
    public let length: Int
    /// The padding id: the given one, else the tokenizer files' `pad_token` (`!`, 0, for SD 2.x checkpoints; EOS for
    /// OpenAI CLIP), else EOS (transformers' default).
    public let pad: Int32
    /// The added tokens' strings, matched before BPE.
    public let addedTokens: Set<String>
    private let vocabulary: [String: Int32]
    private let ranks: [Pair: Int]
    private let byteSymbols: [String]
    private let words: NSRegularExpression
    private let specials: NSRegularExpression

    private struct Pair: Hashable, Sendable {
        let first: String
        let second: String
    }

    /// Reads `vocab.json` and `merges.txt` from `directory`, and the added tokens and pad token from whichever of
    /// `tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json` it holds. `pad` nil takes the files'.
    public init(directory: URL, length: Int = 77, pad: Int32? = nil) throws(AetherError) {
        let vocabulary: [String: Int32]
        let merges: String
        let added: AddedTokens
        do {
            vocabulary = try JSONDecoder().decode([String: Int32].self, from: Data(contentsOf: directory.appending(path: "vocab.json")))
            merges = try String(contentsOf: directory.appending(path: "merges.txt"), encoding: .utf8)
            added = try AddedTokens(directory: directory)
        } catch {
            throw .runtime(underlying: error)
        }
        guard vocabulary["<|startoftext|>"] == Self.bos, vocabulary["<|endoftext|>"] == Self.eos, length >= 2 else {
            throw .manifestInvalid(field: "tokenizer.source", reason: "not a CLIP vocabulary (BOS 49406, EOS 49407)")
        }
        let addedTokens = added.tokens.union(["<|startoftext|>", "<|endoftext|>"])
        for token in addedTokens.sorted() where vocabulary[token] == nil {
            throw .manifestInvalid(field: "tokenizer.source", reason: "added token '\(token)' is not in vocab.json")
        }
        let resolved: Int32? = if let pad { pad } else if let token = added.pad { vocabulary[token] } else { Self.eos }
        guard let pad = resolved, vocabulary.values.contains(pad) else {
            throw .manifestInvalid(field: "tokenizer.source", reason: "the pad token '\(added.pad ?? "")' is not in vocab.json")
        }
        var ranks: [Pair: Int] = [:]
        for line in merges.split(separator: "\n") where !line.hasPrefix("#") {
            let pair = line.split(separator: " ")
            guard pair.count == 2 else {
                throw .manifestInvalid(field: "tokenizer.source", reason: "merges.txt line '\(line)' is not a pair")
            }
            ranks[Pair(first: String(pair[0]), second: String(pair[1]))] = ranks.count
        }
        // GPT-2's byte-to-unicode table: printable bytes map to themselves, the rest to 256 and up.
        let printable = Set(Array(33...126) + Array(161...172) + Array(174...255))
        var symbols: [String] = [], extra = 0
        for byte in 0..<256 {
            let scalar = printable.contains(byte) ? byte : 256 + { defer { extra += 1 }; return extra }()
            symbols.append(String(Character(Unicode.Scalar(scalar)!)))
        }
        self.vocabulary = vocabulary
        self.ranks = ranks
        self.byteSymbols = symbols
        self.length = length
        self.pad = pad
        self.addedTokens = addedTokens
        do {
            words = try NSRegularExpression(pattern: #"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#)
            // Longest first, so a token containing another wins.
            let alternatives = addedTokens.sorted { ($0.count, $0) > ($1.count, $1) }.map(NSRegularExpression.escapedPattern(for:))
            specials = try NSRegularExpression(pattern: alternatives.joined(separator: "|"))
        } catch {
            throw .runtime(underlying: error)
        }
    }

    /// The prompt's ids: BOS, at most `length − 2` pieces, EOS, then `pad` up to `length`.
    public func encode(_ text: String) throws(AetherError) -> [Int32] {
        let normalized = text.precomposedStringWithCanonicalMapping.lowercased()
        let source = normalized as NSString
        // The added tokens (`<|startoftext|>`, `<|endoftext|>`, SD 2.x's `!`) split the text first and map to their ids.
        var pieces: [(text: String, special: Bool)] = []
        var offset = 0
        func words(upTo end: Int) {
            for match in self.words.matches(in: normalized, range: NSRange(location: offset, length: end - offset)) {
                pieces.append((source.substring(with: match.range), false))
            }
        }
        for match in specials.matches(in: normalized, range: NSRange(location: 0, length: source.length)) {
            words(upTo: match.range.location)
            pieces.append((source.substring(with: match.range), true))
            offset = NSMaxRange(match.range)
        }
        words(upTo: source.length)

        var ids: [Int32] = [Self.bos]
        let limit = length - 1
        for piece in pieces where ids.count < limit {
            if piece.special {
                ids.append(vocabulary[piece.text]!)
                continue
            }
            for symbol in try bpe(piece.text) {
                guard let id = vocabulary[symbol] else {
                    throw .tensor(name: "input_ids", reason: .unsupported("the CLIP vocabulary has no piece '\(symbol)'"))
                }
                ids.append(id)
            }
        }
        ids = Array(ids.prefix(limit)) + [Self.eos]
        return ids + Array(repeating: pad, count: length - ids.count)
    }

    /// One word's pieces: its bytes as symbols, `</w>` on the last, merged by rank.
    private func bpe(_ word: String) throws(AetherError) -> [String] {
        var symbols = word.utf8.map { byteSymbols[Int($0)] }
        guard !symbols.isEmpty else { return [] }
        symbols[symbols.count - 1] += "</w>"
        while symbols.count > 1 {
            var best: (rank: Int, index: Int)?
            for i in 0..<(symbols.count - 1) {
                if let rank = ranks[Pair(first: symbols[i], second: symbols[i + 1])], rank < (best?.rank ?? .max) {
                    best = (rank, i)
                }
            }
            guard let best else { break }
            let first = symbols[best.index], second = symbols[best.index + 1]
            var merged: [String] = []
            merged.reserveCapacity(symbols.count)
            var i = 0
            while i < symbols.count {
                if i + 1 < symbols.count, symbols[i] == first, symbols[i + 1] == second {
                    merged.append(first + second)
                    i += 2
                } else {
                    merged.append(symbols[i])
                    i += 1
                }
            }
            symbols = merged
        }
        return symbols
    }
}

/// The added tokens and pad token a Hugging Face tokenizer directory declares, from whichever files it has.
private struct AddedTokens {
    var tokens: Set<String> = []
    var pad: String?

    init(directory: URL) throws {
        func object(_ name: String) throws -> [String: Any]? {
            let url = directory.appending(path: name)
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            return try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        }
        /// A special-token entry: a bare string or `{content: …}`.
        func content(_ value: Any?) -> String? {
            (value as? String) ?? ((value as? [String: Any])?["content"] as? String)
        }
        if let json = try object("tokenizer.json"), let added = json["added_tokens"] as? [[String: Any]] {
            tokens.formUnion(added.compactMap { $0["content"] as? String })
        }
        for name in ["special_tokens_map.json", "tokenizer_config.json"] {
            guard let config = try object(name) else { continue }
            if let decoder = config["added_tokens_decoder"] as? [String: Any] {
                tokens.formUnion(decoder.values.compactMap(content))
            }
            for key in ["bos_token", "eos_token", "unk_token", "pad_token"] {
                if let token = content(config[key]) { tokens.insert(token) }
            }
            if let token = content(config["pad_token"]) { pad = token }
        }
    }
}
