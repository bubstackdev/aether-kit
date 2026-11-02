import AetherKit
import Foundation

/// CLIP byte BPE with end-of-word suffixes, NFC/lowercase normalization, BOS/EOS,
/// truncation and SD2 zero-padding. Matches the pinned CLIPTokenizerFast export.
/// The generic swift-transformers BPE does not currently implement CLIP's </w> suffix.
public struct SDTurboTokenizer: Sendable {
    private struct Pair: Hashable, Sendable { let a: String; let b: String }
    private let vocabulary: [String: Int]
    private let ranks: [Pair: Int]
    private let bytes: [String]

    public static func open(bundle: SDTurboBundle) throws -> Self {
        try Self(directory: bundle.root.appendingPathComponent("tokenizer"))
    }

    init(directory: URL) throws {
        vocabulary = try JSONDecoder().decode([String: Int].self, from: Data(contentsOf: directory.appendingPathComponent("vocab.json")))
        guard vocabulary["<|startoftext|>"] == 49406, vocabulary["<|endoftext|>"] == 49407,
              vocabulary["!"] == 0, vocabulary.count == 49408 else { throw AetherError.invalidModel("invalid SD-Turbo vocabulary") }
        let lines = try String(contentsOf: directory.appendingPathComponent("merges.txt"), encoding: .utf8).split(separator: "\n")
        var ranks: [Pair: Int] = [:]
        for line in lines where !line.hasPrefix("#") {
            let pair = line.split(separator: " ")
            guard pair.count == 2 else { throw AetherError.invalidModel("invalid CLIP merge") }
            ranks[Pair(a: String(pair[0]), b: String(pair[1]))] = ranks.count
        }
        self.ranks = ranks
        let printable = Array(33...126) + Array(161...172) + Array(174...255)
        var map = Array(repeating: "", count: 256), extra = 0
        for byte in 0..<256 {
            let scalar: Int
            if printable.contains(byte) { scalar = byte } else { scalar = 256 + extra; extra += 1 }
            map[byte] = String(UnicodeScalar(scalar)!)
        }
        bytes = map
    }

    public func prepare(_ text: String) throws -> Int32Tensor {
        let normalized = text.precomposedStringWithCanonicalMapping.lowercased()
        // Splitting on whitespace and discarding it implements CLIP fast's whitespace normalization.
        let pattern = #"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#
        let regex = try NSRegularExpression(pattern: pattern)
        let source = normalized as NSString
        var ids = [49406]
        let special = try NSRegularExpression(pattern: #"<\|startoftext\|>|<\|endoftext\|>|!"#)
        var tokens: [String] = [], offset = 0
        for match in special.matches(in: normalized, range: NSRange(location: 0, length: source.length)) {
            let range = NSRange(location: offset, length: match.range.location - offset)
            tokens += regex.matches(in: normalized, range: range).map { source.substring(with: $0.range) }
            tokens.append(source.substring(with: match.range)); offset = NSMaxRange(match.range)
        }
        tokens += regex.matches(in: normalized, range: NSRange(location: offset, length: source.length - offset)).map { source.substring(with: $0.range) }
        for token in tokens {
            if token == "<|startoftext|>" || token == "<|endoftext|>" || token == "!" { ids.append(vocabulary[token]!); continue }
            var word = token.utf8.map { bytes[Int($0)] }
            guard !word.isEmpty else { continue }
            word[word.count - 1] += "</w>"
            while word.count > 1 {
                let candidates = (0..<(word.count - 1)).compactMap { i -> (Int, Pair)? in
                    let pair = Pair(a: word[i], b: word[i + 1]); return ranks[pair].map { ($0, pair) }
                }
                guard let best = candidates.min(by: { $0.0 < $1.0 })?.1 else { break }
                var merged: [String] = [], i = 0
                while i < word.count {
                    if i + 1 < word.count, word[i] == best.a, word[i + 1] == best.b {
                        merged.append(word[i] + word[i + 1]); i += 2
                    } else { merged.append(word[i]); i += 1 }
                }
                word = merged
            }
            for piece in word {
                guard let id = vocabulary[piece] else { throw AetherError.invalidModel("CLIP vocabulary omits BPE piece") }
                ids.append(id)
            }
            if ids.count >= 76 { break }
        }
        ids = Array(ids.prefix(76)) + [49407]
        ids += Array(repeating: 0, count: 77 - ids.count)
        return try Int32Tensor(shape: [1, 77], values: ids.map(Int32.init))
    }
}
