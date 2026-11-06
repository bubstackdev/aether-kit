/// A versioned driver key such as `decoder-lm@1` (DESIGN.md §4, §6.3).
///
/// Matching is exact: `decoder-lm@2` is a different driver from `decoder-lm@1`, and a bundle asking for a version
/// nobody registered is unsupported rather than loaded by a near match.
public struct DriverID: Hashable, Sendable, Codable, CustomStringConvertible, ExpressibleByStringLiteral {
    public let name: String
    public let version: Int

    public init(name: String, version: Int) throws(AetherError) {
        guard Self.isValidName(name), version >= 0 else {
            throw AetherError.manifestInvalid(field: "driver", reason: "'\(name)@\(version)' is not a valid driver id")
        }
        self.name = name
        self.version = version
    }

    /// Parses `name@version`.
    public init(parsing text: String) throws(AetherError) {
        let parts = text.split(separator: "@", omittingEmptySubsequences: false)
        guard parts.count == 2, let version = Int(parts[1]), Self.isValidName(String(parts[0])), version >= 0 else {
            throw AetherError.manifestInvalid(
                field: "driver", reason: "'\(text)' is not of the form name@version (for example decoder-lm@1)")
        }
        self.name = String(parts[0])
        self.version = version
    }

    /// For compiled-in literals that `scripts/gen-catalog` already validated. Traps on a malformed literal.
    public init(validated text: String) {
        do { try self.init(parsing: text) } catch { preconditionFailure("\(error)") }
    }

    /// A literal that must be well formed (traps otherwise), for compiled-in ids.
    public init(stringLiteral value: String) { self.init(validated: value) }

    public var description: String { "\(name)@\(version)" }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        let text = try container.decode(String.self)
        do {
            try self.init(parsing: text)
        } catch {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: error.description)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(description)
    }

    private static func isValidName(_ name: String) -> Bool {
        guard let first = name.unicodeScalars.first, LowerAlnum.contains(first) else { return false }
        return name.unicodeScalars.allSatisfy { LowerAlnum.contains($0) || $0 == "-" || $0 == "." }
    }
}

extension DriverID {
    /// Causal decoders: chat, VLM decoders, decision readout (DESIGN.md §6.2).
    public static let decoderLM = DriverID(validated: "decoder-lm@1")
    /// Text-to-image latent diffusion with a U-Net and an epsilon Euler schedule (SD-Turbo; MANIFEST.md §5).
    public static let latentDiffusion = DriverID(validated: "latent-diffusion@1")
    /// FLUX.2 [klein]: a Qwen3 text encoder, a segmented flow-matching transformer and a VAE (MANIFEST.md §5.7).
    public static let flux2Klein = DriverID(validated: "flux2-klein@1")
    /// Krea 2 Turbo: a Qwen3-VL text encoder, a four-segment flow-matching transformer and a VAE (MANIFEST.md §5.5).
    public static let krea2 = DriverID(validated: "krea2@1")
    /// Kokoro-82M speech: a durations graph and a bucketed vocoder over app-supplied phonemes (MANIFEST.md §5.9).
    public static let kokoro = DriverID(validated: "kokoro@1")
    /// MiniMax H3 text to video with audio: a Qwen3-VL conditioner and a 50-block transformer, both weights-as-inputs
    /// graphs chained on the GPU, then window-tiled video and audio decoders (MANIFEST.md §5.10).
    public static let minimaxH3 = DriverID(validated: "minimax-h3@1")
    /// Monocular depth: one static graph from an image to a depth map and an optional confidence map (MANIFEST.md
    /// §1.3, §2.8).
    public static let depthEstimator = DriverID(validated: "depth-estimator@1")
    /// DETR-style detection: one static graph from an image to query boxes and class logits, scored by a sigmoid with
    /// no NMS, and optional per-query instance masks (RF-DETR; MANIFEST.md §1.3, §2.8).
    public static let detectorDetr = DriverID(validated: "detector-detr@1")
}

private enum LowerAlnum {
    static func contains(_ scalar: Unicode.Scalar) -> Bool {
        ("a"..."z").contains(scalar) || ("0"..."9").contains(scalar)
    }
}
