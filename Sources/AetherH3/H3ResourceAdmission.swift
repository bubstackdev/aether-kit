import AetherKit

/// Caller-qualified maxima for one exact execution identity. Stage bytes must include model loading,
/// native scratch, reusable buffers and both streamed weight banks. Operation bytes cover overlapping
/// CPU intermediates across stages. These are estimates, not measured footprint or allocator limits.
public struct H3ResourceAdmission: Sendable {
    public let coordinator: ResourceCoordinator
    public let executionIdentity: String
    public let operationBytes: UInt64
    public let maximumPixels: Int
    public let maximumFrames: Int
    public let maximumPromptUTF8Bytes: Int
    private let stages: [H3VideoStage: UInt64]

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
        stages = stageBytes
    }

    /// Before anything is reserved: the request is within the profile's limits, every stage it needs has an estimate
    /// (the first-frame encoder for FL2VA, the audio decoder when audio is produced), and the operation plus the
    /// largest of those stages fits the coordinator's capacity, so a run that can never fit is refused before any
    /// stage loads.
    func check(_ request: H3VideoRequest) throws {
        guard request.geometry.width * request.geometry.height <= maximumPixels,
              request.geometry.frames <= maximumFrames, request.prompt.utf8.count <= maximumPromptUTF8Bytes,
              !request.audio || stages[.decodingAudio] != nil,
              request.firstFrame == nil || stages[.encodingKeyframe] != nil else {
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
