public import AetherCore

/// A Kokoro-82M voice. The identifier's first letter is its language, the second its gender, as published in the
/// model's VOICES.md. Voices are style tables, not separate models: any voice reads any phonemes, but quality is best
/// in its own language. Which voices a bundle carries is its `kokoro.voices`.
public struct KokoroVoice: Hashable, Sendable, Codable, CustomStringConvertible {
    /// A voice's language: its identifier's first letter. Open, so a bundle with a language this release doesn't
    /// know still loads and its voices still speak; the known ones are constants.
    public struct Language: RawRepresentable, Hashable, Sendable, Codable, CustomStringConvertible {
        public let rawValue: String

        /// Any lowercase letter `a`–`z`, as the manifest allows.
        public init?(rawValue: String) {
            let scalars = Array(rawValue.unicodeScalars)
            guard scalars.count == 1, ("a"..."z").contains(scalars[0]) else { return nil }
            self.rawValue = rawValue
        }

        fileprivate init(known: String) { rawValue = known }

        public static let americanEnglish = Language(known: "a"), britishEnglish = Language(known: "b")
        public static let spanish = Language(known: "e"), french = Language(known: "f"), hindi = Language(known: "h")
        public static let italian = Language(known: "i"), japanese = Language(known: "j")
        public static let brazilianPortuguese = Language(known: "p"), mandarinChinese = Language(known: "z")

        /// The languages Kokoro-82M v1.0 publishes.
        public static let known: [Language] = [
            .americanEnglish, .britishEnglish, .spanish, .french, .hindi, .italian, .japanese, .brazilianPortuguese,
            .mandarinChinese,
        ]

        /// The BCP 47 tag for a known language (the manifest records each voice's own); nil otherwise.
        public var bcp47: String? {
            switch rawValue {
            case "a": "en-US"
            case "b": "en-GB"
            case "e": "es"
            case "f": "fr"
            case "h": "hi"
            case "i": "it"
            case "j": "ja"
            case "p": "pt-BR"
            case "z": "zh"
            default: nil
            }
        }

        public var description: String { rawValue }
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
        language = Language(known: String(scalars[0]))
        gender = Gender(rawValue: String(scalars[1]))!
    }

    public static let afHeart = KokoroVoice(known: "af_heart")
    public static let afBella = KokoroVoice(known: "af_bella")
    public static let amMichael = KokoroVoice(known: "am_michael")
    public static let bfEmma = KokoroVoice(known: "bf_emma")
    public static let bmGeorge = KokoroVoice(known: "bm_george")
}
