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
struct BudgetedKreaFactory: KreaModelFactory {
    let base: any KreaModelFactory
    let resources: KreaResourceAdmission
    func open(_ asset: KreaModelAsset, frequentReshapes: Bool) async throws -> any KreaExecutable {
        try Task.checkCancellation()
        let estimate = try resources.estimate(for: asset)
        let residency = try resources.coordinator.reserve(bytes: estimate.residentBytes)
        do {
            let model = try await base.open(asset, frequentReshapes: frequentReshapes)
            if Task.isCancelled {
                await model.unload()
                throw CancellationError()
            }
            return BudgetedKreaModel(model: model, residency: residency, resources: resources, estimate: estimate)
        } catch {
            residency.release()
            throw error
        }
    }
    func finish() async { await base.finish() }
}

private actor BudgetedKreaModel: KreaExecutable {
    nonisolated let functions: [ModelFunctionDescription]
    private var model: (any KreaExecutable)?
    private let residency: ResourceReservation
    private let resources: KreaResourceAdmission
    private let estimate: KreaModelResourceEstimate
    private var active: Task<[String: TensorValue], any Error>?
    private var closing = false
    private var closingTask: Task<Void, Never>?

    init(model: any KreaExecutable, residency: ResourceReservation, resources: KreaResourceAdmission,
         estimate: KreaModelResourceEstimate) {
        functions = model.functions
        self.model = model; self.residency = residency; self.resources = resources; self.estimate = estimate
    }

    func runTensors(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue] {
        try Task.checkCancellation()
        guard !closing else { throw AetherError.sessionClosed }
        guard active == nil else { throw AetherError.sessionBusy }
        let task = try startRun(function: function, inputs: inputs)
        active = task
        defer { active = nil }
        return try await withTaskCancellationHandler {
            let result = try await task.value
            try Task.checkCancellation()
            return result
        } onCancel: { task.cancel() }
    }

    private func startRun(function: String, inputs: [String: TensorValue]) throws -> Task<[String: TensorValue], any Error> {
        guard let model else { throw AetherError.sessionClosed }
        let scratch = try resources.coordinator.reserve(bytes: estimate.runScratchBytes)
        return Task {
            defer { scratch.release() }
            try Task.checkCancellation()
            return try await model.runTensors(function: function, inputs: inputs)
        }
    }

    func unload() async {
        if let closingTask { await closingTask.value; return }
        closing = true
        let task = Task { [self] in
            if let active { _ = await active.result }
            await model?.unload()
            model = nil
            residency.release()
        }
        closingTask = task
        await task.value
    }

    deinit { model = nil; residency.release() }
}
