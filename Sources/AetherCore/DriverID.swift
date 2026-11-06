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
}

private enum LowerAlnum {
    static func contains(_ scalar: Unicode.Scalar) -> Bool {
        ("a"..."z").contains(scalar) || ("0"..."9").contains(scalar)
    }
}
