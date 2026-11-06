public import AetherCore

/// Kokoro's phoneme alphabet: single Unicode scalars mapped to token ids (`kokoro.phonemes`, MANIFEST.md §5.9).
public struct KokoroVocabulary: Sendable, Hashable {
    public let tokens: [Unicode.Scalar: Int32]
    public let boundaryToken: Int32
    public let maximumCount: Int

    /// A manifest's vocabulary: one Unicode scalar per key, ids that fit `Int32`. (A decoded manifest has been
    /// validated already; a hand-built `Phonemes` value may not have been.)
    public init(_ phonemes: Manifest.Kokoro.Phonemes) throws(AetherError) {
        var tokens: [Unicode.Scalar: Int32] = [:]
        for (key, id) in phonemes.vocabulary {
            let scalars = Array(key.unicodeScalars)
            guard scalars.count == 1, let token = Int32(exactly: id) else {
                throw .manifestInvalid(field: "kokoro.phonemes.vocabulary.\(key)", reason: "must be one Unicode scalar with an Int32 id")
            }
            tokens[scalars[0]] = token
        }
        guard let boundary = Int32(exactly: phonemes.boundaryToken) else {
            throw .manifestInvalid(field: "kokoro.phonemes.boundaryToken", reason: "must fit Int32")
        }
        self.tokens = tokens
        boundaryToken = boundary
        maximumCount = phonemes.maxPhonemes
    }
}

/// A validated Kokoro phoneme string for one utterance.
///
/// Kokoro reads phonemes, not text. Each Unicode scalar must be in the bundle's vocabulary: upstream silently drops
/// unknown symbols, this type refuses them so a G2P mismatch is visible. Scalars are taken as given, never
/// normalized: Kokoro's nasal vowels are `a` + U+0303, and precomposed U+00E3 is not in its alphabet (Swift `String`
/// comparison would treat the two as equal, so tokenizing walks `unicodeScalars`).
public struct KokoroPhonemes: Sendable, Hashable, CustomStringConvertible {
    public let string: String
    /// The boundary token, one id per phoneme scalar, the boundary token.
    public let tokenIDs: [Int32]
    /// Phoneme scalars, without the boundaries.
    public var count: Int { tokenIDs.count - 2 }
    /// The voice-style row: upstream selects it by phoneme count.
    public var styleRow: Int { count - 1 }
    public var description: String { string }

    /// Refuses (`generation(.invalidParameters)`) an empty string, a scalar outside `vocabulary`, or more than
    /// `vocabulary.maximumCount` scalars.
    public init(_ string: String, vocabulary: KokoroVocabulary) throws(AetherError) {
        var ids: [Int32] = [vocabulary.boundaryToken]
        for scalar in string.unicodeScalars {
            guard let id = vocabulary.tokens[scalar] else {
                let code = String(scalar.value, radix: 16, uppercase: true)
                throw .generation(.invalidParameters("phoneme '\(scalar)' (U+\(code)) is not in the Kokoro vocabulary"))
            }
            ids.append(id)
            guard ids.count <= vocabulary.maximumCount + 1 else {
                throw .generation(.invalidParameters(
                    "Kokoro takes at most \(vocabulary.maximumCount) phonemes per utterance; split longer input"))
            }
        }
        guard ids.count > 1 else { throw .generation(.invalidParameters("the phoneme string is empty")) }
        ids.append(vocabulary.boundaryToken)
        self.string = string
        tokenIDs = ids
    }
}

/// Text to Kokoro phonemes, supplied by the app.
///
/// AetherSpeech ships no grapheme-to-phoneme conversion: Kokoro's usual route (misaki) falls back to espeak-ng, which
/// is GPL-3.0. License-clean routes include a CMUdict (BSD) lexicon with a permissive fallback, precomputed phonemes,
/// or hand-authored ones. Return Kokoro phoneme strings split into utterances of at most 510 scalars, ideally at
/// sentence or clause boundaries.
public protocol SpeechPhonemizer: Sendable {
    func phonemize(_ text: String, language: KokoroVoice.Language) async throws -> [String]
}
