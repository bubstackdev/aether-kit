import AetherKit
import Foundation

/// Admission limits and stage reservations for one exact execution identity, as one or more envelopes. Stage bytes
/// must include model loading, native scratch, reusable buffers and both streamed weight banks. Operation bytes cover
/// overlapping CPU intermediates across stages. Caller-qualified envelopes are estimates; measured ones come from
/// `H3MeasuredResources` entries (one envelope per entry).
///
/// Each request is held to exactly one envelope: among those that cover it (`Envelope.covers`) and that this
/// device's physical memory allows, the 18 GiB bar class first, then the cheapest (operation plus the largest stage
/// the request needs). A canvas and length that no single envelope covers is refused, even when separate envelopes
/// cover the canvas and the length; the refusal names the nearest envelope.
public struct H3ResourceAdmission: Sendable {
    /// One admitted region: the canvases, length and tasks it covers, its reservations, and the memory bar its
    /// numbers were held to.
    public struct Envelope: Sendable, Equatable {
        /// "WIDTHxHEIGHT" canvases measured. Empty for a caller-qualified envelope bounded by `maximumPixels` alone.
        public let canvases: [String]
        public let maximumPixels: Int
        public let maximumFrames: Int
        public let tasks: Set<H3Profile.Task>
        public let operationBytes: UInt64
        public let stageBytes: [H3VideoStage: UInt64]
        public let memoryBarBytes: UInt64
        /// Physical memory the device must have for this envelope to be used; 0 for none.
        public let minimumPhysicalMemoryBytes: UInt64

        public init(canvases: [String] = [], maximumPixels: Int, maximumFrames: Int, tasks: Set<H3Profile.Task> = Set(H3Profile.Task.allCases),
                    operationBytes: UInt64, stageBytes: [H3VideoStage: UInt64], memoryBarBytes: UInt64 = H3MeasuredResources.memoryBarBytes,
                    minimumPhysicalMemoryBytes: UInt64 = 0) throws {
            let sides = canvases.map { $0.split(separator: "x").compactMap { Int($0) } }
            guard (1...1_032_192).contains(maximumPixels), (1...360).contains(maximumFrames), !tasks.isEmpty, operationBytes > 0,
                  canvases.allSatisfy({ $0.wholeMatch(of: /[1-9][0-9]*x[1-9][0-9]*/) != nil }),
                  sides.allSatisfy({ $0.count == 2 && $0[0] * $0[1] <= maximumPixels }), memoryBarBytes > 0,
                  stageBytes.values.allSatisfy({ $0 > 0 }),
                  Set([H3VideoStage.encoding, .denoising, .decodingVideo]).isSubset(of: Set(stageBytes.keys)) else {
                throw AetherError.invalidRequest("invalid H3 resource envelope")
            }
            self.canvases = canvases; self.maximumPixels = maximumPixels; self.maximumFrames = maximumFrames; self.tasks = tasks
            self.operationBytes = operationBytes; self.stageBytes = stageBytes; self.memoryBarBytes = memoryBarBytes
            self.minimumPhysicalMemoryBytes = minimumPhysicalMemoryBytes
        }

        /// The request's task (FL2VA with a first frame, else T2VA) is listed, every stage it needs has a reservation,
        /// its length is at most `maximumFrames`, and its canvas fits inside a listed canvas of the same orientation
        /// (width and height each at most the listed ones; a square canvas on either side matches any orientation).
        /// An envelope without listed canvases covers any canvas up to `maximumPixels`. An envelope above the 18 GiB bar
        /// never covers a request inside today's region (at most 960x544's pixels and 124 frames), which stays under
        /// 18 GiB entries.
        public func covers(_ request: H3VideoRequest) -> Bool {
            let w = request.geometry.width, h = request.geometry.height
            let task: H3Profile.Task = request.firstFrame == nil ? .t2va : .fl2va
            let insideToday = w * h <= H3MeasuredResources.baselinePixels && request.geometry.frames <= H3MeasuredResources.baselineFrames
            guard memoryBarBytes <= H3MeasuredResources.memoryBarBytes || !insideToday else { return false }
            guard tasks.contains(task), request.geometry.frames <= maximumFrames,
                  !request.audio || stageBytes[.decodingAudio] != nil,
                  request.firstFrame == nil || stageBytes[.encodingKeyframe] != nil else { return false }
            guard !canvases.isEmpty else { return w * h <= maximumPixels }
            return canvases.contains { canvas in
                let s = canvas.split(separator: "x").map { Int($0)! }, width = s[0], height = s[1]
                return w <= width && h <= height && (w == h || width == height || (w > h) == (width > height))
            }
        }

        /// The largest reservation among the stages `request` uses.
        public func peakStageBytes(_ request: H3VideoRequest) -> UInt64 {
            var used: [H3VideoStage] = [.encoding, .denoising, .decodingVideo]
            if request.firstFrame != nil { used.append(.encodingKeyframe) }
            if request.audio { used.append(.decodingAudio) }
            return used.compactMap { stageBytes[$0] }.max() ?? 0
        }

        var summary: String {
            let region = canvases.isEmpty ? "up to \(maximumPixels) px" : canvases.joined(separator: ", ")
            return "\(region) x \(maximumFrames) frames (\(tasks.map(\.rawValue).sorted().joined(separator: "/")))"
        }
    }

    public let coordinator: ResourceCoordinator
    public let executionIdentity: String
    public let maximumPromptUTF8Bytes: Int
    /// The envelopes a request is held to, one of them per request.
    public let envelopes: [Envelope]
    /// The device's physical memory, against which each envelope's `minimumPhysicalMemoryBytes` is checked.
    public let physicalMemoryBytes: UInt64
    /// Union bounds over every envelope, for diagnostics only: their combination need not be admitted (a large canvas
    /// at the longest length, say). A UI offers only what `admits(_:)` or a covering envelope allows.
    public var maximumPixels: Int { envelopes.map(\.maximumPixels).max()! }
    public var maximumFrames: Int { envelopes.map(\.maximumFrames).max()! }
    public var operationBytes: UInt64 { envelopes.map(\.operationBytes).max()! }

    /// One caller-qualified envelope covering any canvas up to `maximumPixels`, both tasks, under the 18 GiB bar.
    public init(coordinator: ResourceCoordinator, executionIdentity: String, operationBytes: UInt64,
                stageBytes: [H3VideoStage: UInt64], maximumPixels: Int, maximumFrames: Int,
                maximumPromptUTF8Bytes: Int) throws {
        try self.init(coordinator: coordinator, executionIdentity: executionIdentity, maximumPromptUTF8Bytes: maximumPromptUTF8Bytes,
                      envelopes: [Envelope(maximumPixels: maximumPixels, maximumFrames: maximumFrames, operationBytes: operationBytes,
                                           stageBytes: stageBytes)])
    }

    public init(coordinator: ResourceCoordinator, executionIdentity: String, maximumPromptUTF8Bytes: Int, envelopes: [Envelope],
                physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) throws {
        guard !executionIdentity.isEmpty, executionIdentity.utf8.count <= 512, maximumPromptUTF8Bytes > 0, !envelopes.isEmpty else {
            throw AetherError.invalidRequest("invalid H3 resource profile")
        }
        self.coordinator = coordinator; self.executionIdentity = executionIdentity; self.maximumPromptUTF8Bytes = maximumPromptUTF8Bytes
        self.envelopes = envelopes; self.physicalMemoryBytes = physicalMemoryBytes
    }

    /// Whether some envelope admits `request` on this device (capacity is checked only when the run starts).
    public func admits(_ request: H3VideoRequest) -> Bool { (try? envelope(for: request)) != nil }

    /// The envelope `request` is held to: covering, allowed by physical memory, the 18 GiB class first, then the
    /// cheapest. Refusals name the physical-memory requirement or the nearest envelope.
    public func envelope(for request: H3VideoRequest) throws -> Envelope {
        guard request.prompt.utf8.count <= maximumPromptUTF8Bytes else {
            throw AetherError.invalidRequest("H3 prompt exceeds \(maximumPromptUTF8Bytes) UTF-8 bytes")
        }
        let covering = envelopes.filter { $0.covers(request) }
        let allowed = covering.filter { physicalMemoryBytes >= $0.minimumPhysicalMemoryBytes }
        if let chosen = allowed.min(by: { a, b in
            (a.memoryBarBytes, a.operationBytes + a.peakStageBytes(request)) < (b.memoryBarBytes, b.operationBytes + b.peakStageBytes(request))
        }) { return chosen }
        let g = request.geometry, task = request.firstFrame == nil ? "t2va" : "fl2va"
        let asked = "\(g.width)x\(g.height) x \(g.frames) frames (\(task)\(request.audio ? ", audio" : ""))"
        if let needed = covering.map(\.minimumPhysicalMemoryBytes).min() {
            throw AetherError.unsupported("H3 \(asked) needs at least \(needed >> 30) GiB of physical memory; this device has \(physicalMemoryBytes >> 30) GiB")
        }
        throw AetherError.invalidRequest("H3 \(asked) is outside every measured envelope; nearest: \(nearest(to: request).summary)")
    }

    /// The envelope needing the smallest relative stretch (width, height, frames; a missing task or stage counts as 1).
    private func nearest(to request: H3VideoRequest) -> Envelope {
        let g = request.geometry
        func stretch(_ e: Envelope) -> Double {
            let task: H3Profile.Task = request.firstFrame == nil ? .t2va : .fl2va
            let frames = max(0, Double(g.frames) / Double(e.maximumFrames) - 1)
            let missing = (e.tasks.contains(task) ? 0 : 1) + (!request.audio || e.stageBytes[.decodingAudio] != nil ? 0 : 1)
            let sides = e.canvases.isEmpty ? [max(0, Double(g.width * g.height) / Double(e.maximumPixels) - 1)]
                : e.canvases.map { c -> Double in
                    let s = c.split(separator: "x").map { Double($0)! }
                    return max(0, Double(g.width) / s[0] - 1) + max(0, Double(g.height) / s[1] - 1)
                }
            return frames + Double(missing) + sides.min()!
        }
        return envelopes.min { stretch($0) < stretch($1) }!
    }

    /// The admission restricted to the one envelope `request` is held to; a session checks and reserves through it.
    func resolved(for request: H3VideoRequest) throws -> H3ResourceAdmission {
        try H3ResourceAdmission(coordinator: coordinator, executionIdentity: executionIdentity, maximumPromptUTF8Bytes: maximumPromptUTF8Bytes,
                                envelopes: [envelope(for: request)], physicalMemoryBytes: physicalMemoryBytes)
    }

    /// Before anything is reserved: some envelope admits the request (every stage it needs has a reservation: the
    /// first-frame encoder for FL2VA, the audio decoder when audio is produced), and the operation plus the largest
    /// of those stages fits the coordinator's capacity, so a run that can never fit is refused before any stage loads.
    func check(_ request: H3VideoRequest) throws {
        let envelope = try envelope(for: request)
        let needed = envelope.operationBytes + envelope.peakStageBytes(request)
        guard needed <= coordinator.capacityBytes else {
            throw ResourceAdmissionError.insufficientCapacity(requested: needed, available: coordinator.capacityBytes)
        }
    }

    /// Once the operation is reserved (after any GPU grant): the largest stage the request needs must fit what is
    /// still available, so a run refused for capacity has loaded no stage. Stages are reserved one at a time, so other
    /// holders taking capacity mid-run can still refuse a later stage.
    func checkStageHeadroom(_ request: H3VideoRequest) throws {
        let peak = try envelope(for: request).peakStageBytes(request), available = coordinator.snapshot.availableBytes
        guard peak <= available else { throw ResourceAdmissionError.insufficientCapacity(requested: peak, available: available) }
    }

    /// Reserves `stage` from a resolved (single-envelope) admission.
    func reserve(_ stage: H3VideoStage) throws -> ResourceReservation {
        guard envelopes.count == 1, let bytes = envelopes[0].stageBytes[stage] else {
            throw AetherError.invalidRequest("missing H3 stage resource estimate")
        }
        return try coordinator.reserve(bytes: bytes)
    }
}
