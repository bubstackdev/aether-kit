import AetherCoreAI
import AetherKit
import AetherKrea

/// Caller-qualified maximum estimates for one exact model stream and descriptor pair.
public struct KreaModelResourceEstimate: Sendable {
    let identity: String
    let residentBytes: UInt64
    let runScratchBytes: UInt64
    public init(asset: KreaModelAsset, residentBytes: UInt64, runScratchBytes: UInt64) throws {
        guard residentBytes > 0, runScratchBytes > 0 else { throw ResourceAdmissionError.invalidByteCount }
        identity = asset.declaredSHA256 + ":" + asset.metadataSHA256
        self.residentBytes = residentBytes; self.runScratchBytes = runScratchBytes
    }
}

/// Opt-in reservations for host operation storage and every loaded model/native call. Estimates must
/// include load peaks, overlapping host carries/copies and all shapes up to the stated limits.
/// This is admission of caller estimates, not allocator enforcement or a measured memory guarantee.
public struct KreaResourceAdmission: Sendable {
    public let coordinator: ResourceCoordinator
    public let operationBytes: UInt64
    public let maximumImageTokens: Int
    public let maximumConditioningRows: Int
    private let estimates: [String: KreaModelResourceEstimate]

    public init(coordinator: ResourceCoordinator, operationBytes: UInt64,
                maximumImageTokens: Int, maximumConditioningRows: Int,
                models: [KreaModelResourceEstimate]) throws {
        guard operationBytes > 0, (1...4096).contains(maximumImageTokens),
              (1...KreaReferencePrompt.maximumConditioningRows).contains(maximumConditioningRows), !models.isEmpty else {
            throw AetherError.invalidRequest("invalid Krea resource profile")
        }
        var keyed: [String: KreaModelResourceEstimate] = [:]
        for model in models {
            guard keyed.updateValue(model, forKey: model.identity) == nil else {
                throw AetherError.invalidRequest("duplicate Krea model resource estimate")
            }
        }
        self.coordinator = coordinator; self.operationBytes = operationBytes
        self.maximumImageTokens = maximumImageTokens; self.maximumConditioningRows = maximumConditioningRows
        estimates = keyed
    }

    func estimate(for asset: KreaModelAsset) throws -> KreaModelResourceEstimate {
        guard let value = estimates[asset.declaredSHA256 + ":" + asset.metadataSHA256] else {
            throw AetherError.invalidRequest("no resource estimate for the selected Krea model bytes")
        }
        return value
    }

    func check(imageTokens: Int = 0, conditioningRows: Int = 0) throws {
        guard imageTokens <= maximumImageTokens, conditioningRows <= maximumConditioningRows else {
            throw AetherError.invalidRequest("request exceeds the Krea resource profile's shape limits")
        }
    }
}

/// The decorator is also used by controlled pipeline tests, so native and fake factories share admission ordering.
/// Residency and per-call scratch are reserved by the package's shared `openBudgeted` decorator.
struct BudgetedKreaFactory: KreaModelFactory {
    let base: any KreaModelFactory
    let resources: KreaResourceAdmission
    func open(_ asset: KreaModelAsset, frequentReshapes: Bool) async throws -> any KreaExecutable {
        // Cancellation is checked before the estimate lookup, as it always was.
        try Task.checkCancellation()
        let estimate = try resources.estimate(for: asset)
        return try await openBudgeted(coordinator: resources.coordinator, residentBytes: estimate.residentBytes,
                                      runScratchBytes: estimate.runScratchBytes) {
            try await base.open(asset, frequentReshapes: frequentReshapes)
        }
    }
    func finish() async { await base.finish() }
}
