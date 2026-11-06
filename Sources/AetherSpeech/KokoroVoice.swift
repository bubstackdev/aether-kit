public import AetherCore

/// A Kokoro-82M voice. The identifier's first letter is its language, the second its gender, as published in the
/// model's VOICES.md. Voices are style tables, not separate models: any voice reads any phonemes, but quality is best
/// in its own language. Which voices a bundle carries is its `kokoro.voices`.
public struct KokoroVoice: Hashable, Sendable, Codable, CustomStringConvertible {
    public enum Language: String, Sendable, Hashable, Codable, CaseIterable {
        case americanEnglish = "a", britishEnglish = "b", spanish = "e", french = "f", hindi = "h"
        case italian = "i", japanese = "j", brazilianPortuguese = "p", mandarinChinese = "z"

        /// The BCP 47 tag the manifest records.
        public var bcp47: String {
            switch self {
            case .americanEnglish: "en-US"
            case .britishEnglish: "en-GB"
            case .spanish: "es"
            case .french: "fr"
            case .hindi: "hi"
            case .italian: "it"
            case .japanese: "ja"
            case .brazilianPortuguese: "pt-BR"
            case .mandarinChinese: "zh"
            }
        }
    }

    public enum Gender: String, Sendable, Hashable, Codable { case female = "f", male = "m" }

    public let identifier: String
    public let language: Language
    public let gender: Gender
    public var description: String { identifier }

    /// `<language><gender>_<name>`, for example `af_heart`.
    public init(_ identifier: String) throws(AetherError) {
        let scalars = Array(identifier.unicodeScalars)
        guard Manifest.Kokoro.isVoiceIdentifier(identifier), let language = Language(rawValue: String(scalars[0])),
              let gender = Gender(rawValue: String(scalars[1])) else {
            throw .generation(.invalidParameters("'\(identifier)' is not a Kokoro voice identifier"))
        }
        self.identifier = identifier
        self.language = language
        self.gender = gender
    }

    public init(from decoder: any Decoder) throws {
        let text = try decoder.singleValueContainer().decode(String.self)
        do {
            try self.init(text)
        } catch {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: error.description))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(identifier)
    }

    /// The literal voices below (the tests parse each one).
    private init(known identifier: String) {
        let scalars = Array(identifier.unicodeScalars)
        self.identifier = identifier
        language = Language(rawValue: String(scalars[0]))!
        gender = Gender(rawValue: String(scalars[1]))!
    }

    public static let afHeart = KokoroVoice(known: "af_heart")
    public static let afBella = KokoroVoice(known: "af_bella")
    public static let amMichael = KokoroVoice(known: "am_michael")
    public static let bfEmma = KokoroVoice(known: "bf_emma")
    public static let bmGeorge = KokoroVoice(known: "bm_george")
}
