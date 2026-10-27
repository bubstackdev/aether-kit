import AetherKit

/// A Kokoro-82M v1.0 voice. The identifier's first letter is the voice's language and the
/// second its gender, as published in the model's VOICES.md. Voices are style vectors, not
/// separate models; any voice accepts any phoneme string, but quality is best in its language.
public struct KokoroVoice: Hashable, Sendable, Codable, CustomStringConvertible {
    public enum Language: String, Sendable, Codable, CaseIterable {
        case americanEnglish = "a", britishEnglish = "b", spanish = "e", french = "f", hindi = "h"
        case italian = "i", japanese = "j", brazilianPortuguese = "p", mandarinChinese = "z"

        /// BCP 47 tag recorded in the bundle contract.
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

    public enum Gender: String, Sendable, Codable { case female = "f", male = "m" }

    public let identifier: String
    public let language: Language
    public let gender: Gender
    public var description: String { identifier }

    /// Accepts identifiers of the form `<language><gender>_<name>`, for example `af_heart`.
    public init(_ identifier: String) throws {
        let scalars = Array(identifier.unicodeScalars)
        guard scalars.count >= 4, scalars.count <= 64, scalars[2] == "_",
              let language = Language(rawValue: String(scalars[0])), let gender = Gender(rawValue: String(scalars[1])),
              scalars[3...].allSatisfy({ ("a"..."z").contains($0) || ("0"..."9").contains($0) || $0 == "_" }) else {
            throw AetherError.invalidRequest("invalid Kokoro voice identifier '\(identifier)'")
        }
        self.identifier = identifier
        self.language = language
        self.gender = gender
    }

    public init(from decoder: any Decoder) throws {
        try self.init(try decoder.singleValueContainer().decode(String.self))
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(identifier)
    }

    private init(known identifier: String) {
        // Only used for the literal catalog below, which the tests check against `init(_:)`.
        let scalars = Array(identifier.unicodeScalars)
        self.identifier = identifier
        language = Language(rawValue: String(scalars[0]))!
        gender = Gender(rawValue: String(scalars[1]))!
    }

    // American English
    public static let afHeart = KokoroVoice(known: "af_heart")
    public static let afAlloy = KokoroVoice(known: "af_alloy")
    public static let afAoede = KokoroVoice(known: "af_aoede")
    public static let afBella = KokoroVoice(known: "af_bella")
    public static let afJessica = KokoroVoice(known: "af_jessica")
    public static let afKore = KokoroVoice(known: "af_kore")
    public static let afNicole = KokoroVoice(known: "af_nicole")
    public static let afNova = KokoroVoice(known: "af_nova")
    public static let afRiver = KokoroVoice(known: "af_river")
    public static let afSarah = KokoroVoice(known: "af_sarah")
    public static let afSky = KokoroVoice(known: "af_sky")
    public static let amAdam = KokoroVoice(known: "am_adam")
    public static let amEcho = KokoroVoice(known: "am_echo")
    public static let amEric = KokoroVoice(known: "am_eric")
    public static let amFenrir = KokoroVoice(known: "am_fenrir")
    public static let amLiam = KokoroVoice(known: "am_liam")
    public static let amMichael = KokoroVoice(known: "am_michael")
    public static let amOnyx = KokoroVoice(known: "am_onyx")
    public static let amPuck = KokoroVoice(known: "am_puck")
    public static let amSanta = KokoroVoice(known: "am_santa")
    // British English
    public static let bfAlice = KokoroVoice(known: "bf_alice")
    public static let bfEmma = KokoroVoice(known: "bf_emma")
    public static let bfIsabella = KokoroVoice(known: "bf_isabella")
    public static let bfLily = KokoroVoice(known: "bf_lily")
    public static let bmDaniel = KokoroVoice(known: "bm_daniel")
    public static let bmFable = KokoroVoice(known: "bm_fable")
    public static let bmGeorge = KokoroVoice(known: "bm_george")
    public static let bmLewis = KokoroVoice(known: "bm_lewis")
    // Spanish, French, Hindi, Italian
    public static let efDora = KokoroVoice(known: "ef_dora")
    public static let emAlex = KokoroVoice(known: "em_alex")
    public static let emSanta = KokoroVoice(known: "em_santa")
    public static let ffSiwis = KokoroVoice(known: "ff_siwis")
    public static let hfAlpha = KokoroVoice(known: "hf_alpha")
    public static let hfBeta = KokoroVoice(known: "hf_beta")
    public static let hmOmega = KokoroVoice(known: "hm_omega")
    public static let hmPsi = KokoroVoice(known: "hm_psi")
    public static let ifSara = KokoroVoice(known: "if_sara")
    public static let imNicola = KokoroVoice(known: "im_nicola")
    // Japanese
    public static let jfAlpha = KokoroVoice(known: "jf_alpha")
    public static let jfGongitsune = KokoroVoice(known: "jf_gongitsune")
    public static let jfNezumi = KokoroVoice(known: "jf_nezumi")
    public static let jfTebukuro = KokoroVoice(known: "jf_tebukuro")
    public static let jmKumo = KokoroVoice(known: "jm_kumo")
    // Brazilian Portuguese
    public static let pfDora = KokoroVoice(known: "pf_dora")
    public static let pmAlex = KokoroVoice(known: "pm_alex")
    public static let pmSanta = KokoroVoice(known: "pm_santa")
    // Mandarin Chinese
    public static let zfXiaobei = KokoroVoice(known: "zf_xiaobei")
    public static let zfXiaoni = KokoroVoice(known: "zf_xiaoni")
    public static let zfXiaoxiao = KokoroVoice(known: "zf_xiaoxiao")
    public static let zfXiaoyi = KokoroVoice(known: "zf_xiaoyi")
    public static let zmYunjian = KokoroVoice(known: "zm_yunjian")
    public static let zmYunxi = KokoroVoice(known: "zm_yunxi")
    public static let zmYunxia = KokoroVoice(known: "zm_yunxia")
    public static let zmYunyang = KokoroVoice(known: "zm_yunyang")

    /// The 54 voices published with Kokoro-82M v1.0 (revision f3ff357). A bundle may carry fewer.
    public static let v1_0: [KokoroVoice] = [
        afHeart, afAlloy, afAoede, afBella, afJessica, afKore, afNicole, afNova, afRiver, afSarah, afSky,
        amAdam, amEcho, amEric, amFenrir, amLiam, amMichael, amOnyx, amPuck, amSanta,
        bfAlice, bfEmma, bfIsabella, bfLily, bmDaniel, bmFable, bmGeorge, bmLewis,
        efDora, emAlex, emSanta, ffSiwis, hfAlpha, hfBeta, hmOmega, hmPsi, ifSara, imNicola,
        jfAlpha, jfGongitsune, jfNezumi, jfTebukuro, jmKumo, pfDora, pmAlex, pmSanta,
        zfXiaobei, zfXiaoni, zfXiaoxiao, zfXiaoyi, zmYunjian, zmYunxi, zmYunxia, zmYunyang,
    ]
}
