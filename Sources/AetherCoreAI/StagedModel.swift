import AetherKit

/// A loaded stateless model as seen by staged generation pipelines. `ModelSession` conforms; tests
/// substitute scripted models. Package-internal so family modules share one admission decorator.
@available(macOS 27, iOS 27, *)
package protocol StagedModel: Sendable {
    var functions: [ModelFunctionDescription] { get }
    func runTensors(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue]
    func unload() async
}

@available(macOS 27, iOS 27, *)
extension ModelSession: StagedModel {}

@available(macOS 27, iOS 27, *)
package func signature(_ model: any StagedModel, named name: String) throws -> ModelFunctionDescription {
    guard let value = model.functions.first(where: { $0.name == name }) else {
        throw AetherError.invalidModel("missing function '\(name)'")
    }
    return value
}

/// Reserves a model's residency before `open` and its per-call scratch before each native
/// submission, holding scratch until the submitted call joins. Residency is released only after
/// joined unload. These are reservations of caller estimates, not measured memory.
@available(macOS 27, iOS 27, *)
package func openBudgeted(
    coordinator: ResourceCoordinator, residentBytes: UInt64, runScratchBytes: UInt64,
    open: () async throws -> any StagedModel
) async throws -> any StagedModel {
    try Task.checkCancellation()
    let residency = try coordinator.reserve(bytes: residentBytes)
    do {
        let model = try await open()
        if Task.isCancelled {
            await model.unload()
            throw CancellationError()
        }
        return BudgetedStagedModel(model: model, residency: residency, coordinator: coordinator, scratch: runScratchBytes)
    } catch {
        residency.release()
        throw error
    }
}

@available(macOS 27, iOS 27, *)
private actor BudgetedStagedModel: StagedModel {
    nonisolated let functions: [ModelFunctionDescription]
    private var model: (any StagedModel)?
    private let residency: ResourceReservation
    private let coordinator: ResourceCoordinator
    private let scratch: UInt64
    private var active: Task<[String: TensorValue], any Error>?
    private var closing = false
    private var closingTask: Task<Void, Never>?

    init(model: any StagedModel, residency: ResourceReservation, coordinator: ResourceCoordinator, scratch: UInt64) {
        functions = model.functions
        self.model = model; self.residency = residency; self.coordinator = coordinator; self.scratch = scratch
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
        let reservation = try coordinator.reserve(bytes: scratch)
        return Task {
            defer { reservation.release() }
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
