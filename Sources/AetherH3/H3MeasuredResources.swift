import AetherKit
import Foundation

/// One measured entry: memory peaks for one exact execution identity (profile manifest and recipe) on one device
/// architecture, over one region of canvases, lengths and tasks, recorded by the memory gate
/// (`Scripts/qualify-h3-memory.py`) from 20 ms Metal-allocation timelines of complete renders. An identity may have
/// several entries; each becomes one `H3ResourceAdmission.Envelope`, and each request is held to one of them.
///
/// Each entry is held to its own memory bar: its largest stage peak plus margin and operation overlap must fit it.
/// The 18 GiB bar is the default and covers today's region (at most 960x544's pixels and 124 frames). The 24 GiB bar
/// (decided 2026-10-29) is allowed only for an entry beyond that region (a larger canvas or a longer clip), and such an
/// entry is used only on a device with at least 64 GiB of physical memory. Before anything is reserved the admission
/// checks that the operation plus the largest stage the request needs fits the coordinator's capacity (and, once the
/// operation is reserved, that the largest stage fits what remains), so a run that cannot fit is refused before any
/// stage loads. Stages are still reserved one at a time: capacity other holders take mid-run can refuse a later stage.
/// The bars apply only to sessions created with this admission (`H3VideoSession(stages:resources:)`); a session
/// without resources reserves nothing.
public struct H3MeasuredResources: Sendable, Codable, Equatable {
    /// The H3 memory bar for today's region: the Metal allocation peak over a complete render.
    public static let memoryBarBytes: UInt64 = 18 << 30
    /// The bar for entries beyond today's region (larger canvases or longer clips).
    public static let extendedMemoryBarBytes: UInt64 = 24 << 30
    /// Physical memory a device needs to use an entry under the extended bar.
    public static let extendedPhysicalMemoryBytes: UInt64 = 64 << 30
    /// Today's region, held to `memoryBarBytes`: at most these pixels (960x544) and frames.
    public static let baselinePixels = 960 * 544
    public static let baselineFrames = 124
    /// Added to every measured stage peak (sampling can miss short peaks between 20 ms samples).
    public static let marginBytes: UInt64 = 256 << 20

    public let executionIdentity: String
    public let deviceArchitecture: String
    /// Canvases measured, "WIDTHxHEIGHT". A request is covered by a listed canvas of the same orientation that it fits
    /// inside (see `H3ResourceAdmission.Envelope.covers`).
    public let canvases: [String]
    public let maximumFrames: Int
    /// The tasks measured; nil (absent from the JSON) means both.
    public let tasks: Set<H3Profile.Task>?
    /// Per `H3VideoStage` raw value, the largest sample of max(Metal allocation, physical footprint) over the stage,
    /// including memory the previous stage has not yet returned.
    public let stagePeakBytes: [String: UInt64]
    /// The session's CPU state carried across stages (latents, conditioning, one decoded window, PCM), sized from the
    /// measured clip's tensors.
    public let operationBytes: UInt64
    /// The bar this entry is held to: `memoryBarBytes` (the default when absent from the JSON) or
    /// `extendedMemoryBarBytes`.
    public let memoryBarBytes: UInt64
    /// The gate record the numbers come from (a path in this repository).
    public let evidence: String

    private enum CodingKeys: String, CodingKey {
        case executionIdentity = "execution_identity", deviceArchitecture = "device_architecture", canvases, tasks
        case maximumFrames = "maximum_frames", stagePeakBytes = "stage_peak_bytes", operationBytes = "operation_bytes"
        case memoryBarBytes = "memory_bar_bytes", evidence
    }

    public init(executionIdentity: String, deviceArchitecture: String, canvases: [String], maximumFrames: Int,
                tasks: Set<H3Profile.Task>? = nil, stagePeakBytes: [String: UInt64], operationBytes: UInt64,
                memoryBarBytes: UInt64 = H3MeasuredResources.memoryBarBytes, evidence: String) throws {
        self.executionIdentity = executionIdentity; self.deviceArchitecture = deviceArchitecture; self.canvases = canvases
        self.maximumFrames = maximumFrames; self.tasks = tasks; self.stagePeakBytes = stagePeakBytes; self.operationBytes = operationBytes
        self.memoryBarBytes = memoryBarBytes; self.evidence = evidence
        try validate()
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        executionIdentity = try c.decode(String.self, forKey: .executionIdentity)
        deviceArchitecture = try c.decode(String.self, forKey: .deviceArchitecture)
        canvases = try c.decode([String].self, forKey: .canvases)
        maximumFrames = try c.decode(Int.self, forKey: .maximumFrames)
        tasks = try c.decodeIfPresent([String].self, forKey: .tasks).map { names in
            Set(try names.map { name in
                guard let task = H3Profile.Task(rawValue: name) else { throw AetherError.invalidModel("unknown H3 measured task \(name)") }
                return task
            })
        }
        stagePeakBytes = try c.decode([String: UInt64].self, forKey: .stagePeakBytes)
        operationBytes = try c.decode(UInt64.self, forKey: .operationBytes)
        memoryBarBytes = try c.decodeIfPresent(UInt64.self, forKey: .memoryBarBytes) ?? Self.memoryBarBytes
        evidence = try c.decode(String.self, forKey: .evidence)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(executionIdentity, forKey: .executionIdentity); try c.encode(deviceArchitecture, forKey: .deviceArchitecture)
        try c.encode(canvases, forKey: .canvases); try c.encode(maximumFrames, forKey: .maximumFrames)
        try c.encodeIfPresent(tasks.map { $0.map(\.rawValue).sorted() }, forKey: .tasks)
        try c.encode(stagePeakBytes, forKey: .stagePeakBytes); try c.encode(operationBytes, forKey: .operationBytes)
        if memoryBarBytes != Self.memoryBarBytes { try c.encode(memoryBarBytes, forKey: .memoryBarBytes) }
        try c.encode(evidence, forKey: .evidence)
    }

    /// Whether the entry reaches beyond today's region: a canvas over 960x544's pixels or more than 124 frames.
    public var isBeyondBaseline: Bool { maximumPixels > Self.baselinePixels || maximumFrames > Self.baselineFrames }

    /// Physical memory a device needs to use this entry: `extendedPhysicalMemoryBytes` under the extended bar, else 0.
    public var minimumPhysicalMemoryBytes: UInt64 { memoryBarBytes > Self.memoryBarBytes ? Self.extendedPhysicalMemoryBytes : 0 }

    /// Valid stages, canvases and tasks, the stages every generation needs, a known bar that the entry's region
    /// allows, and its worst case (largest stage peak + margin + operation) within that bar.
    public func validate() throws {
        guard !executionIdentity.isEmpty, !deviceArchitecture.isEmpty, (1...360).contains(maximumFrames), operationBytes > 0,
              !canvases.isEmpty, canvases.allSatisfy({ $0.wholeMatch(of: /[1-9][0-9]*x[1-9][0-9]*/) != nil }), tasks?.isEmpty != true,
              stagePeakBytes.keys.allSatisfy({ H3VideoStage(rawValue: $0) != nil }), stagePeakBytes.values.allSatisfy({ $0 > 0 }),
              Set(["encoding", "denoising", "decodingVideo"]).isSubset(of: Set(stagePeakBytes.keys)),
              !evidence.isEmpty, !evidence.hasPrefix("/"), !evidence.hasPrefix("~") else {
            throw AetherError.invalidModel("invalid H3 measured resource entry")
        }
        guard [Self.memoryBarBytes, Self.extendedMemoryBarBytes].contains(memoryBarBytes) else {
            throw AetherError.invalidModel("H3 measured entry names an unknown memory bar \(memoryBarBytes)")
        }
        guard memoryBarBytes == Self.memoryBarBytes || isBeyondBaseline else {
            throw AetherError.invalidModel("the 24 GiB H3 bar applies only beyond 960x544's pixels or 124 frames")
        }
        let worst = stagePeakBytes.values.max()! + Self.marginBytes + operationBytes
        guard worst <= memoryBarBytes else {
            throw AetherError.unsupported("H3 measured peak \(worst) bytes exceeds the \(memoryBarBytes)-byte memory bar")
        }
    }

    public var maximumPixels: Int {
        canvases.map { c in c.split(separator: "x").map { Int($0)! }.reduce(1, *) }.max()!
    }

    /// This entry's admission envelope: reservations covering every measured stage peak plus margin, and the operation.
    public func envelope() throws -> H3ResourceAdmission.Envelope {
        try validate()
        var stages: [H3VideoStage: UInt64] = [:]
        for (key, bytes) in stagePeakBytes { stages[H3VideoStage(rawValue: key)!] = bytes + Self.marginBytes }
        return try H3ResourceAdmission.Envelope(canvases: canvases, maximumPixels: maximumPixels, maximumFrames: maximumFrames,
                                                tasks: tasks ?? Set(H3Profile.Task.allCases), operationBytes: operationBytes,
                                                stageBytes: stages, memoryBarBytes: memoryBarBytes,
                                                minimumPhysicalMemoryBytes: minimumPhysicalMemoryBytes)
    }

    /// An admission holding requests to this one entry.
    public func admission(coordinator: ResourceCoordinator, maximumPromptUTF8Bytes: Int = 16_384,
                          physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) throws -> H3ResourceAdmission {
        try Self.admission(entries: [self], coordinator: coordinator, maximumPromptUTF8Bytes: maximumPromptUTF8Bytes,
                           physicalMemoryBytes: physicalMemoryBytes)
    }

    /// An admission over several entries of one identity and device, one envelope per entry. Entries this device's
    /// physical memory does not allow stay listed, so a request only they cover is refused naming the requirement;
    /// at least one entry must be allowed.
    public static func admission(entries: [H3MeasuredResources], coordinator: ResourceCoordinator, maximumPromptUTF8Bytes: Int = 16_384,
                                 physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) throws -> H3ResourceAdmission {
        guard let first = entries.first,
              entries.allSatisfy({ $0.executionIdentity == first.executionIdentity && $0.deviceArchitecture == first.deviceArchitecture }) else {
            throw AetherError.invalidModel("H3 measured entries must share one identity and device")
        }
        guard let needed = entries.map(\.minimumPhysicalMemoryBytes).min(), physicalMemoryBytes >= needed else {
            throw AetherError.unsupported("measured H3 memory for \(first.executionIdentity) needs at least "
                                          + "\(entries.map(\.minimumPhysicalMemoryBytes).min()! >> 30) GiB of physical memory")
        }
        return try H3ResourceAdmission(coordinator: coordinator, executionIdentity: first.executionIdentity,
                                       maximumPromptUTF8Bytes: maximumPromptUTF8Bytes, envelopes: entries.map { try $0.envelope() },
                                       physicalMemoryBytes: physicalMemoryBytes)
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

    /// The entries of `entries` for this identity and device that this device's physical memory allows. Throws when
    /// none is recorded, or when every recorded one needs more physical memory (naming the requirement).
    public static func entries(_ entries: [H3MeasuredResources], executionIdentity: String, deviceArchitecture: String,
                               physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) throws -> [H3MeasuredResources] {
        let recorded = entries.filter { $0.executionIdentity == executionIdentity && $0.deviceArchitecture == deviceArchitecture }
        guard !recorded.isEmpty else { throw AetherError.unsupported("no measured H3 memory for \(executionIdentity) on \(deviceArchitecture)") }
        let allowed = recorded.filter { physicalMemoryBytes >= $0.minimumPhysicalMemoryBytes }
        guard !allowed.isEmpty else {
            throw AetherError.unsupported("measured H3 memory for \(executionIdentity) needs at least "
                                          + "\(recorded.map(\.minimumPhysicalMemoryBytes).min()! >> 30) GiB of physical memory")
        }
        return allowed
    }

    /// The bundled entries this device may use for this identity (see `entries(_:executionIdentity:deviceArchitecture:physicalMemoryBytes:)`).
    public static func bundledEntries(executionIdentity: String, deviceArchitecture: String,
                                      physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) throws -> [H3MeasuredResources] {
        try entries(bundled(), executionIdentity: executionIdentity, deviceArchitecture: deviceArchitecture, physicalMemoryBytes: physicalMemoryBytes)
    }

    /// The one bundled entry this device may use for this identity. Throws when there is none, or when several
    /// match: an identity with several entries is admitted through `bundledAdmission`.
    public static func bundled(executionIdentity: String, deviceArchitecture: String,
                               physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) throws -> H3MeasuredResources {
        let matching = try bundledEntries(executionIdentity: executionIdentity, deviceArchitecture: deviceArchitecture,
                                          physicalMemoryBytes: physicalMemoryBytes)
        guard matching.count == 1 else {
            throw AetherError.invalidRequest("\(matching.count) measured H3 entries for \(executionIdentity); use bundledAdmission")
        }
        return matching[0]
    }

    /// The admission over every bundled entry for this identity and device. Entries this device's physical memory does
    /// not allow are kept as envelopes it refuses with the requirement named; at least one entry must be allowed.
    public static func bundledAdmission(executionIdentity: String, deviceArchitecture: String, coordinator: ResourceCoordinator,
                                        maximumPromptUTF8Bytes: Int = 16_384,
                                        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) throws -> H3ResourceAdmission {
        let recorded = try entries(bundled(), executionIdentity: executionIdentity, deviceArchitecture: deviceArchitecture, physicalMemoryBytes: .max)
        return try admission(entries: recorded, coordinator: coordinator, maximumPromptUTF8Bytes: maximumPromptUTF8Bytes,
                             physicalMemoryBytes: physicalMemoryBytes)
    }
}
