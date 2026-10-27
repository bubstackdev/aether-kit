import AetherKit
import Foundation

/// Measured memory peaks for one exact execution identity (profile manifest and recipe) on one device architecture,
/// recorded by the memory gate (`Scripts/qualify-h3-memory.py`) from 20 ms Metal-allocation timelines of complete
/// renders. H3 is held to an 18 GiB bar: an entry whose largest stage peak plus its operation overlap and margin
/// exceeds the bar is refused. `admission(coordinator:)` turns an accepted entry into stage reservations that cover
/// every measured stage peak, so a coordinator without that capacity refuses the run before any stage loads.
public struct H3MeasuredResources: Sendable, Codable, Equatable {
    /// The H3 memory bar: the Metal allocation peak over a complete render.
    public static let memoryBarBytes: UInt64 = 18 << 30
    /// Added to every measured stage peak (sampling can miss short peaks between 20 ms samples).
    public static let marginBytes: UInt64 = 256 << 20

    public let executionIdentity: String
    public let deviceArchitecture: String
    /// Canvases measured, "WIDTHxHEIGHT"; admission allows at most their largest pixel count.
    public let canvases: [String]
    public let maximumFrames: Int
    /// Per `H3VideoStage` raw value, the largest sample of max(Metal allocation, physical footprint) over the stage,
    /// including memory the previous stage has not yet returned.
    public let stagePeakBytes: [String: UInt64]
    /// The session's CPU state carried across stages (latents, conditioning, one decoded window, PCM), sized from the
    /// measured clip's tensors.
    public let operationBytes: UInt64
    /// The gate record the numbers come from.
    public let evidence: String

    private enum CodingKeys: String, CodingKey {
        case executionIdentity = "execution_identity", deviceArchitecture = "device_architecture", canvases
        case maximumFrames = "maximum_frames", stagePeakBytes = "stage_peak_bytes", operationBytes = "operation_bytes", evidence
    }

    public init(executionIdentity: String, deviceArchitecture: String, canvases: [String], maximumFrames: Int,
                stagePeakBytes: [String: UInt64], operationBytes: UInt64, evidence: String) throws {
        self.executionIdentity = executionIdentity; self.deviceArchitecture = deviceArchitecture; self.canvases = canvases
        self.maximumFrames = maximumFrames; self.stagePeakBytes = stagePeakBytes; self.operationBytes = operationBytes; self.evidence = evidence
        try validate()
    }

    /// Valid stages and canvases, the stages every generation needs, and the 18 GiB bar.
    public func validate() throws {
        guard !executionIdentity.isEmpty, !deviceArchitecture.isEmpty, (1...360).contains(maximumFrames), operationBytes > 0,
              !canvases.isEmpty, canvases.allSatisfy({ $0.wholeMatch(of: /[1-9][0-9]*x[1-9][0-9]*/) != nil }),
              stagePeakBytes.keys.allSatisfy({ H3VideoStage(rawValue: $0) != nil }), stagePeakBytes.values.allSatisfy({ $0 > 0 }),
              Set(["encoding", "denoising", "decodingVideo"]).isSubset(of: Set(stagePeakBytes.keys)) else {
            throw AetherError.invalidModel("invalid H3 measured resource entry")
        }
        let worst = stagePeakBytes.values.max()! + Self.marginBytes + operationBytes
        guard worst <= Self.memoryBarBytes else {
            throw AetherError.unsupported("H3 measured peak \(worst) bytes exceeds the \(Self.memoryBarBytes)-byte memory bar")
        }
    }

    public var maximumPixels: Int {
        canvases.map { c in c.split(separator: "x").map { Int($0)! }.reduce(1, *) }.max()!
    }

    /// Reservations covering every measured stage peak plus margin, and the operation overlap.
    public func admission(coordinator: ResourceCoordinator, maximumPromptUTF8Bytes: Int = 16_384) throws -> H3ResourceAdmission {
        try validate()
        var stages: [H3VideoStage: UInt64] = [:]
        for (key, bytes) in stagePeakBytes { stages[H3VideoStage(rawValue: key)!] = bytes + Self.marginBytes }
        return try H3ResourceAdmission(coordinator: coordinator, executionIdentity: executionIdentity, operationBytes: operationBytes,
                                       stageBytes: stages, maximumPixels: maximumPixels, maximumFrames: maximumFrames,
                                       maximumPromptUTF8Bytes: maximumPromptUTF8Bytes)
    }

    /// The entries recorded with this release (`Resources/h3-measured-resources.json`).
    public static func bundled() throws -> [H3MeasuredResources] {
        guard let url = Bundle.module.url(forResource: "h3-measured-resources", withExtension: "json", subdirectory: "Resources") else {
            throw AetherError.invalidModel("missing H3 measured resources")
        }
        struct File: Decodable { let schema: Int; let entries: [H3MeasuredResources] }
        let file = try JSONDecoder().decode(File.self, from: Data(contentsOf: url))
        guard file.schema == 1 else { throw AetherError.invalidModel("unknown H3 measured resources schema") }
        for entry in file.entries { try entry.validate() }
        return file.entries
    }

    /// The recorded entry for this identity and device; there is no estimate for an unmeasured combination.
    public static func bundled(executionIdentity: String, deviceArchitecture: String) throws -> H3MeasuredResources {
        guard let entry = try bundled().first(where: { $0.executionIdentity == executionIdentity && $0.deviceArchitecture == deviceArchitecture }) else {
            throw AetherError.unsupported("no measured H3 memory for \(executionIdentity) on \(deviceArchitecture)")
        }
        return entry
    }
}
