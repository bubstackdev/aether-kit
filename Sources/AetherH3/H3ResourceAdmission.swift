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

    func check(_ request: H3VideoRequest) throws {
        guard request.geometry.width * request.geometry.height <= maximumPixels,
              request.geometry.frames <= maximumFrames, request.prompt.utf8.count <= maximumPromptUTF8Bytes,
              !request.audio || stages[.decodingAudio] != nil,
              request.firstFrame == nil || stages[.encodingKeyframe] != nil else {
            throw AetherError.invalidRequest("H3 request exceeds or lacks a stage in its resource profile")
        }
    }

    func reserve(_ stage: H3VideoStage) throws -> ResourceReservation {
        guard let bytes = stages[stage] else { throw AetherError.invalidRequest("missing H3 stage resource estimate") }
        return try coordinator.reserve(bytes: bytes)
    }
}
