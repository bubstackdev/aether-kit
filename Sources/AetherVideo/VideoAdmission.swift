public import AetherCore
public import AetherRuntime
import Foundation

/// How ``MiniMaxH3Pipeline`` admits a generation (MANIFEST.md §5.10).
public enum VideoAdmissionPolicy: Sendable, Hashable {
    /// The default, and the only policy a host uses: a generation runs only under a measured envelope
    /// (``VideoMemoryEnvelope``) for its admission key (``VideoAdmissionKey``) and recipe, on this export and Core AI
    /// architecture, and only on a device with the profile's minimum physical memory. Anything else is refused before
    /// anything loads (``VideoRefusal``). The pipeline's ledger is seeded from the envelopes.
    case measuredOnly
    /// The memory gate's (`MiniMaxH3ProfileGateTests`, `scripts/h3-memory-gate`): every profile is admitted, planned
    /// from its graphs' files (or what the ledger has measured so far), with nothing seeded and no physical-memory
    /// minimum, so its renders measure it. SPI: the kit's gates only.
    @_spi(Testing) case measuring
}

/// What a measured envelope is recorded under and admission looks up: a profile (text to video), or the profile with
/// one keyframe (`<profile>/fl2va1`) or first and last (`<profile>/fl2va2`). ``description`` is the manifest's
/// spelling (`h3.envelopes[].key`).
public struct VideoAdmissionKey: Sendable, Hashable, Comparable, CustomStringConvertible {
    /// The profile's key (`960x544x124`).
    public let profile: String
    /// The keyframes the clip is anchored to: 0, 1 or 2.
    public let keyframeCount: Int

    public init(profile: String, keyframeCount: Int = 0) {
        self.profile = profile
        self.keyframeCount = keyframeCount
    }

    /// Reads the manifest's spelling: `<profile>`, `<profile>/fl2va1` or `<profile>/fl2va2`. Nil for anything else.
    public init?(_ description: String) {
        let parts = description.split(separator: "/", omittingEmptySubsequences: false)
        guard let profile = parts.first, !profile.isEmpty, parts.count <= 2 else { return nil }
        switch parts.count == 2 ? String(parts[1]) : nil {
        case nil: keyframeCount = 0
        case "fl2va1": keyframeCount = 1
        case "fl2va2": keyframeCount = 2
        default: return nil
        }
        self.profile = String(profile)
    }

    public var description: String { keyframeCount == 0 ? profile : "\(profile)/fl2va\(keyframeCount)" }

    public static func < (a: Self, b: Self) -> Bool { a.description < b.description }
}

/// What admission holds one key to: the measured envelope (each phase's peak and held bytes, and the sampling margin)
/// and its profile's memory class (the bar and the physical memory it needs).
public struct VideoMemoryEnvelope: Sendable, Hashable {
    /// The admission key the envelope was measured for.
    public let key: VideoAdmissionKey
    /// The recipes measured.
    public let recipes: [String]
    /// Each phase's measured peak and held bytes.
    public let phases: [Manifest.Residency: PhaseFootprint]
    /// The sampling margin added to each peak against the bar.
    public let marginBytes: Int64
    /// The profile's bar: every phase's peak plus the margin stays within it.
    public let barBytes: Int64
    /// The physical memory a device needs to run the profile (0: no minimum beyond the budget).
    public let minimumPhysicalMemoryBytes: Int64
    /// The largest phase peak.
    public var stagePeakBytes: Int64 { phases.values.map(\.peakBytes).max() ?? 0 }
}

/// Why ``MiniMaxH3Pipeline`` refuses a request before anything waits or loads. Generation throws ``error``, an
/// ``AetherError`` of the usual kind (`unsupportedOperation` or `capacity`); ``init(_:)`` reads one back. Act on
/// this type, never on an error's text: the text is for people and may change.
public enum VideoRefusal: Sendable, Hashable {
    /// No measured memory envelope covers the admission key with this recipe on this export and chip, so the default
    /// measured-only admission refuses it. The memory gate (`scripts/h3-memory-gate`) measures it; until then the clip
    /// isn't available on this Mac.
    case unmeasured(admissionKey: VideoAdmissionKey, recipe: String)
    /// The profile doesn't offer the recipe: its `recipes` lack it, or its shapes don't take the recipe's modulation.
    /// `offered`: the recipes it does run (for the request's task), sorted.
    case recipeNotOffered(profile: String, recipe: String, offered: [String])
    /// The device has less physical memory than the profile's measured envelope needs: a Mac with at least
    /// `neededBytes` might run it.
    case insufficientPhysicalMemory(admissionKey: VideoAdmissionKey, neededBytes: Int64, availableBytes: Int64)

    /// The `what` of an ``insufficientPhysicalMemory(admissionKey:neededBytes:availableBytes:)`` capacity error, before
    /// the key.
    static let physicalMemoryResource = "physical memory for "
    static let unmeasuredSuffix = " has no measured memory envelope for this export on this chip, so it is refused (unmeasured); the memory gate measures it"

    /// The error generation throws: `unsupportedOperation` for ``unmeasured(admissionKey:recipe:)`` and
    /// ``recipeNotOffered(profile:recipe:offered:)``, `capacity` (needed and available bytes) for
    /// ``insufficientPhysicalMemory(admissionKey:neededBytes:availableBytes:)``.
    public var error: AetherError {
        switch self {
        case .unmeasured(let key, let recipe):
            .unsupportedOperation("\(key) with '\(recipe)'" + Self.unmeasuredSuffix)
        case .recipeNotOffered(let profile, let recipe, let offered):
            .unsupportedOperation("profile \(profile) offers \(offered.isEmpty ? "no recipe" : offered.joined(separator: ", ")), not '\(recipe)'")
        case .insufficientPhysicalMemory(let key, let needed, let available):
            .capacity(what: Self.physicalMemoryResource + key.description, needed: needed, available: available)
        }
    }

    /// The refusal `error` carries, or nil when it isn't one.
    public init?(_ error: AetherError) {
        switch error {
        case .capacity(let what, let needed, let available):
            guard what.hasPrefix(Self.physicalMemoryResource),
                  let key = VideoAdmissionKey(String(what.dropFirst(Self.physicalMemoryResource.count))) else { return nil }
            self = .insufficientPhysicalMemory(admissionKey: key, neededBytes: needed, availableBytes: available)
        case .unsupportedOperation(let message):
            if message.hasSuffix(Self.unmeasuredSuffix) {
                let head = message.dropLast(Self.unmeasuredSuffix.count)
                guard let split = head.range(of: " with '"), head.hasSuffix("'"),
                      let key = VideoAdmissionKey(String(head[..<split.lowerBound])) else { return nil }
                let recipe = head[split.upperBound...].dropLast()
                guard !recipe.isEmpty, !recipe.contains("'") else { return nil }
                self = .unmeasured(admissionKey: key, recipe: String(recipe))
            } else if message.hasPrefix("profile "), message.hasSuffix("'") {
                let body = message.dropFirst("profile ".count).dropLast()
                guard let offers = body.range(of: " offers "), let not = body.range(of: ", not '", options: .backwards),
                      offers.upperBound <= not.lowerBound else { return nil }
                let profile = String(body[..<offers.lowerBound]), recipe = String(body[not.upperBound...])
                let list = String(body[offers.upperBound..<not.lowerBound])
                guard !profile.isEmpty, !profile.contains(" "), !recipe.isEmpty, !recipe.contains("'") else { return nil }
                let offered = list == "no recipe" ? [] : list.components(separatedBy: ", ")
                self = .recipeNotOffered(profile: profile, recipe: recipe, offered: offered)
            } else {
                return nil
            }
        default:
            return nil
        }
    }
}
