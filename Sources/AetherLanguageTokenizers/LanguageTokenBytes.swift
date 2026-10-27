import AetherKit
import Foundation

/// The raw UTF-8 bytes each token ID stands for, read from a byte-level BPE `tokenizer.json` (the GPT-2 and Qwen
/// family): vocabulary entries spell bytes with GPT-2's byte-to-unicode map, and added tokens are their own text, as the
/// `ByteLevel` decoder and Transformers' added-token handling decode them. Streaming text from bytes never splits a
/// UTF-8 scalar, and a constrained decoder walks the same bytes.
public struct LanguageTokenBytes: Sendable {
    /// One past the largest token ID the tokenizer defines. A model's head can have more rows (Qwen3: 151,936 rows,
    /// 151,669 tokens); those rows have no bytes.
    public let count: Int
    private let storage: [UInt8]
    /// `storage[offsets[id]..<offsets[id + 1]]` are token `id`'s bytes.
    private let offsets: [Int32]
    private let defined: [Bool]
    private let special: [Bool]

    /// Reads `tokenizer.json`. Throws for a tokenizer whose decoder isn't byte-level.
    public init(tokenizerJSON data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any], let vocab = model["vocab"] as? [String: Int]
        else { throw AetherError.invalidModel("tokenizer.json has no vocabulary") }
        guard Self.isByteLevel(root["decoder"]) else {
            throw AetherError.invalidModel("token bytes need a byte-level decoder")
        }
        let added = (root["added_tokens"] as? [[String: Any]]) ?? []
        var entries: [(id: Int, bytes: [UInt8], special: Bool)] = []
        entries.reserveCapacity(vocab.count + added.count)
        let inverse = Self.unicodeToByte
        for (token, id) in vocab {
            var bytes: [UInt8] = []
            bytes.reserveCapacity(token.unicodeScalars.count)
            for scalar in token.unicodeScalars {
                guard let byte = inverse[scalar] else {
                    throw AetherError.invalidModel("vocabulary token \(id) is not byte-level")
                }
                bytes.append(byte)
            }
            entries.append((id, bytes, false))
        }
        for token in added {
            guard let id = token["id"] as? Int, let content = token["content"] as? String
            else { throw AetherError.invalidModel("malformed added token") }
            entries.append((id, Array(content.utf8), token["special"] as? Bool ?? false))
        }
        guard entries.allSatisfy({ $0.id >= 0 }) else { throw AetherError.invalidModel("negative token ID") }
        let count = (entries.map(\.id).max() ?? -1) + 1
        var perID = [[UInt8]?](repeating: nil, count: count)
        var special = [Bool](repeating: false, count: count)
        // Added tokens come last, so an added token that reuses a vocabulary ID wins, as it does when decoding.
        for entry in entries {
            perID[entry.id] = entry.bytes
            special[entry.id] = entry.special
        }
        var storage: [UInt8] = []
        var offsets: [Int32] = [0]
        offsets.reserveCapacity(count + 1)
        for bytes in perID {
            storage += bytes ?? []
            offsets.append(Int32(storage.count))
        }
        self.count = count
        self.storage = storage
        self.offsets = offsets
        self.defined = perID.map { $0 != nil }
        self.special = special
    }

    /// The bytes of `id`, or nil for an ID the tokenizer doesn't define.
    public func bytes(of id: Int) -> ArraySlice<UInt8>? {
        guard id >= 0, id < count, defined[id] else { return nil }
        return storage[Int(offsets[id])..<Int(offsets[id + 1])]
    }

    /// Special tokens (for example `<|im_end|>`) contribute no text when decoding with special tokens skipped.
    public func isSpecial(_ id: Int) -> Bool { id >= 0 && id < count && special[id] }

    /// The bytes `decode(tokenIDs, skipSpecialTokens:)` would produce, before UTF-8 decoding.
    public func bytes(of tokenIDs: some Sequence<Int>, skipSpecialTokens: Bool = true) -> [UInt8] {
        var out: [UInt8] = []
        for id in tokenIDs where !(skipSpecialTokens && isSpecial(id)) { out += bytes(of: id) ?? [] }
        return out
    }

    private static func isByteLevel(_ decoder: Any?) -> Bool {
        guard let decoder = decoder as? [String: Any] else { return false }
        if decoder["type"] as? String == "ByteLevel" { return true }
        if decoder["type"] as? String == "Sequence", let steps = decoder["decoders"] as? [Any] {
            return steps.contains { isByteLevel($0) }
        }
        return false
    }

    /// GPT-2's `bytes_to_unicode`, inverted: printable Latin-1 bytes stand for themselves, and the other 68 bytes take
    /// the code points from U+0100 up, in byte order.
    static let unicodeToByte: [Unicode.Scalar: UInt8] = {
        let printable = Array(33...126) + Array(161...172) + Array(174...255)
        var map: [Unicode.Scalar: UInt8] = [:]
        for byte in printable { map[Unicode.Scalar(UInt32(byte))!] = UInt8(byte) }
        var next: UInt32 = 256
        for byte in 0...255 where !printable.contains(byte) {
            map[Unicode.Scalar(next)!] = UInt8(byte)
            next += 1
        }
        return map
    }()
}

extension LanguageTokenizer {
    /// The token byte table of a local tokenizer folder (its `tokenizer.json`).
    public static func tokenBytes(directory: URL) throws -> LanguageTokenBytes {
        try LanguageTokenBytes(tokenizerJSON: Data(contentsOf: directory.appendingPathComponent("tokenizer.json")))
    }

    /// The token byte table of a registered bundle, after verifying its tokenizer files as `open(bundle:)` does.
    public static func tokenBytes(bundle: URL) throws -> LanguageTokenBytes {
        let manifest = try LanguageModelManifest.load(from: bundle)
        return try tokenBytes(directory: try manifest.verifyTokenizer(in: bundle))
    }
}
