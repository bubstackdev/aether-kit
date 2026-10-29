import AetherKit

/// Caller-qualified maxima for one exact execution identity. Stage bytes must include model loading,
/// native scratch, reusable buffers and both streamed weight banks. Operation bytes cover overlapping
/// CPU intermediates across stages. These are estimates, not measured footprint or allocator limits.
///
/// One identity can carry several envelopes (`init(envelopes:)`), each with its own limits and estimates, such as a
/// small canvas measured at a long clip and a large canvas at a short one. A request is held to the cheapest envelope
/// that covers it, so a combination no envelope covers (the large canvas at the long clip) is refused.
public struct H3ResourceAdmission: Sendable {
    public let coordinator: ResourceCoordinator
    public let executionIdentity: String
    public let operationBytes: UInt64
    public let maximumPixels: Int
    public let maximumFrames: Int
    public let maximumPromptUTF8Bytes: Int
    private let stages: [H3VideoStage: UInt64]
    /// The envelopes a combined admission chooses from; empty for a single envelope.
    private let envelopes: [H3ResourceAdmission]

    public init(coordinator: ResourceCoordinator, executionIdentity: String, operationBytes: UInt64,
                stageBytes: [H3VideoStage: UInt64], maximumPixels: Int, maximumFrames: Int,
                maximumPromptUTF8Bytes: Int) throws {
        guard !executionIdentity.isEmpty, executionIdentity.utf8.count <= 512, operationBytes > 0,
              (1...1_032_192).contains(maximumPixels), (1...360).contains(maximumFrames), maximumPromptUTF8Bytes > 0,
              stageBytes.values.allSatisfy({ $0 > 0 }),
              Set([H3VideoStage.encoding, .denoising, .decodingVideo]).isSubset(of: Set(stageBytes.keys)) else {
            throw AetherError.invalidRequest("invalid H3 resource profile")
        }
        self.coordinator = coordinator; self.executionIdentity = executionIdentity; self.operationBytes = operationBytes
        self.maximumPixels = maximumPixels; self.maximumFrames = maximumFrames; self.maximumPromptUTF8Bytes = maximumPromptUTF8Bytes
        stages = stageBytes; envelopes = []
    }

    /// Several single envelopes for one identity and coordinator. The combined maxima are the largest of any envelope;
    /// each request is checked and reserved against the cheapest envelope that covers it.
    public init(envelopes: [H3ResourceAdmission]) throws {
        guard let first = envelopes.first,
              envelopes.allSatisfy({ $0.envelopes.isEmpty && $0.executionIdentity == first.executionIdentity && $0.coordinator === first.coordinator }) else {
            throw AetherError.invalidRequest("H3 resource envelopes must be single envelopes of one identity and coordinator")
        }
        coordinator = first.coordinator; executionIdentity = first.executionIdentity
        operationBytes = envelopes.map(\.operationBytes).max()!
        maximumPixels = envelopes.map(\.maximumPixels).max()!; maximumFrames = envelopes.map(\.maximumFrames).max()!
        maximumPromptUTF8Bytes = envelopes.map(\.maximumPromptUTF8Bytes).max()!
        stages = envelopes.reduce(into: [:]) { all, e in all.merge(e.stages, uniquingKeysWith: max) }
        self.envelopes = envelopes.count == 1 ? [] : envelopes
    }

    /// The single envelope `request` is held to: the cheapest (operation plus largest needed stage) that covers it.
    func envelope(for request: H3VideoRequest) throws -> H3ResourceAdmission {
        guard !envelopes.isEmpty else { return self }
        guard let chosen = envelopes.filter({ $0.covers(request) })
            .min(by: { $0.operationBytes + $0.peakStageBytes(request) < $1.operationBytes + $1.peakStageBytes(request) }) else {
            throw AetherError.invalidRequest("H3 request exceeds or lacks a stage in its resource profile")
        }
        return chosen
    }

    private func covers(_ request: H3VideoRequest) -> Bool {
        request.geometry.width * request.geometry.height <= maximumPixels
            && request.geometry.frames <= maximumFrames && request.prompt.utf8.count <= maximumPromptUTF8Bytes
            && (!request.audio || stages[.decodingAudio] != nil) && (request.firstFrame == nil || stages[.encodingKeyframe] != nil)
    }

    /// Before anything is reserved: the request is within the profile's limits, every stage it needs has an estimate
    /// (the first-frame encoder for FL2VA, the audio decoder when audio is produced), and the operation plus the
    /// largest of those stages fits the coordinator's capacity, so a run that can never fit is refused before any
    /// stage loads.
    func check(_ request: H3VideoRequest) throws {
        guard envelopes.isEmpty else { return try envelope(for: request).check(request) }
        guard covers(request) else {
            throw AetherError.invalidRequest("H3 request exceeds or lacks a stage in its resource profile")
        }
        let needed = operationBytes + peakStageBytes(request)
        guard needed <= coordinator.capacityBytes else {
            throw ResourceAdmissionError.insufficientCapacity(requested: needed, available: coordinator.capacityBytes)
        }
    }

    /// Once the operation is reserved (after any GPU grant): the largest stage the request needs must fit what is
    /// still available, so a run refused for capacity has loaded no stage. Stages are reserved one at a time, so other
    /// holders taking capacity mid-run can still refuse a later stage.
    func checkStageHeadroom(_ request: H3VideoRequest) throws {
        let peak = peakStageBytes(request), available = coordinator.snapshot.availableBytes
        guard peak <= available else { throw ResourceAdmissionError.insufficientCapacity(requested: peak, available: available) }
    }

    /// The largest estimate among the stages `request` uses.
    func peakStageBytes(_ request: H3VideoRequest) -> UInt64 {
        var used: [H3VideoStage] = [.encoding, .denoising, .decodingVideo]
        if request.firstFrame != nil { used.append(.encodingKeyframe) }
        if request.audio { used.append(.decodingAudio) }
        return used.compactMap { stages[$0] }.max() ?? 0
    }

    func reserve(_ stage: H3VideoStage) throws -> ResourceReservation {
        guard let bytes = stages[stage] else { throw AetherError.invalidRequest("missing H3 stage resource estimate") }
        return try coordinator.reserve(bytes: bytes)
    }
}
