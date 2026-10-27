import AetherKit

/// Kokoro's phoneme alphabet: single Unicode scalars mapped to token ids (178 slots in v1.0).
public struct KokoroVocabulary: Sendable, Equatable {
    public static let boundaryToken: Int32 = 0
    public let tokens: [Unicode.Scalar: Int32]

    public init(_ entries: [String: Int]) throws {
        var tokens: [Unicode.Scalar: Int32] = [:]
        var ids = Set<Int>()
        for (key, id) in entries {
            let scalars = Array(key.unicodeScalars)
            guard scalars.count == 1, (1...4095).contains(id), ids.insert(id).inserted else {
                throw AetherError.invalidModel("Kokoro vocabulary entries must be unique single scalars with ids 1...4095")
            }
            tokens[scalars[0]] = Int32(id)
        }
        guard !tokens.isEmpty else { throw AetherError.invalidModel("empty Kokoro vocabulary") }
        self.tokens = tokens
    }
}

/// A validated Kokoro phoneme string for one utterance.
///
/// Kokoro consumes phonemes, not text. Each Unicode scalar must be in the bundle's vocabulary:
/// upstream silently drops unknown symbols, whereas this type refuses them so that a G2P
/// mismatch is visible. Produce phonemes with a `SpeechPhonemizer` or supply them directly.
public struct KokoroPhonemes: Sendable, Equatable, CustomStringConvertible {
    public static let maximumCount = 510

    public let string: String
    /// Boundary token, one id per phoneme scalar, boundary token.
    public let tokenIDs: [Int32]
    public var count: Int { tokenIDs.count - 2 }
    /// Upstream selects the voice style row by phoneme count; row `count - 1` of 510.
    public var styleRow: Int { count - 1 }
    public var description: String { string }

    public init(_ string: String, vocabulary: KokoroVocabulary) throws {
        var ids: [Int32] = [KokoroVocabulary.boundaryToken]
        for scalar in string.unicodeScalars {
            guard let id = vocabulary.tokens[scalar] else {
                let code = String(scalar.value, radix: 16, uppercase: true)
                throw AetherError.invalidRequest("phoneme '\(scalar)' (U+\(code)) is not in the Kokoro vocabulary")
            }
            ids.append(id)
            guard ids.count <= Self.maximumCount + 1 else {
                throw AetherError.invalidRequest("Kokoro accepts at most 510 phonemes per utterance; split longer input")
            }
        }
        guard ids.count > 1 else { throw AetherError.invalidRequest("phoneme string must not be empty") }
        ids.append(KokoroVocabulary.boundaryToken)
        self.string = string
        tokenIDs = ids
    }
}

/// Grapheme-to-phoneme conversion supplied by the app.
///
/// AetherSpeech deliberately ships no G2P. The common upstream route (misaki) can fall back to
/// espeak-ng, which is GPL-3.0 and incompatible with this Apache-2.0 SDK. Implementations must
/// return Kokoro phoneme strings (see `docs/kokoro-speech.md` for license-clean routes), split
/// into utterances of at most 510 scalars, ideally at sentence or clause boundaries.
public protocol SpeechPhonemizer: Sendable {
    func phonemize(_ text: String, language: KokoroVoice.Language) async throws -> [String]
}
