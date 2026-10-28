import AetherKit
@testable import AetherLanguageCoreAI
import Foundation
import Testing

private let decoderHash = String(repeating: "d", count: 64)
private let encoderHash = String(repeating: "e", count: 64)

private final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    func set() { lock.withLock { flag = true } }
    var value: Bool { lock.withLock { flag } }
}

struct LanguageResourceAdmissionTests {
    private func profile(capacity: UInt64 = 100, maximumImageTokens: Int? = 512) throws -> LanguageResourceAdmission {
        try LanguageResourceAdmission(coordinator: try ResourceCoordinator(capacityBytes: capacity),
                                      models: [try LanguageModelResourceEstimate(modelSHA256: decoderHash, residentBytes: 60, runScratchBytes: 10),
                                               try LanguageModelResourceEstimate(modelSHA256: encoderHash, residentBytes: 20, runScratchBytes: 5)],
                                      maximumContextLength: 4096, maximumImageTokens: maximumImageTokens)
    }

    @Test func estimatesAndProfilesAreValidated() throws {
        #expect(throws: ResourceAdmissionError.invalidByteCount) {
            try LanguageModelResourceEstimate(modelSHA256: decoderHash, residentBytes: 0, runScratchBytes: 1)
        }
        #expect(throws: ResourceAdmissionError.invalidByteCount) {
            try LanguageModelResourceEstimate(modelSHA256: decoderHash, residentBytes: 1, runScratchBytes: 0)
        }
        for bad in ["", String(repeating: "D", count: 64), String(repeating: "d", count: 63), String(repeating: "g", count: 64)] {
            #expect(throws: LanguageGenerationError.self) { try LanguageModelResourceEstimate(modelSHA256: bad, residentBytes: 1, runScratchBytes: 1) }
        }
        let coordinator = try ResourceCoordinator(capacityBytes: 10)
        let estimate = try LanguageModelResourceEstimate(modelSHA256: decoderHash, residentBytes: 1, runScratchBytes: 1)
        #expect(throws: LanguageGenerationError.self) {
            try LanguageResourceAdmission(coordinator: coordinator, models: [], maximumContextLength: 4096)
        }
        #expect(throws: LanguageGenerationError.self) {
            try LanguageResourceAdmission(coordinator: coordinator, models: [estimate, estimate], maximumContextLength: 4096)
        }
        for context in [0, 16385] {
            #expect(throws: LanguageGenerationError.self) {
                try LanguageResourceAdmission(coordinator: coordinator, models: [estimate], maximumContextLength: context)
            }
        }
        #expect(throws: LanguageGenerationError.self) {
            try LanguageResourceAdmission(coordinator: coordinator, models: [estimate], maximumContextLength: 4096, maximumImageTokens: 0)
        }
    }

    /// Other model bytes, and a registered context above the profile's, are refused before anything is reserved.
    @Test func unknownBytesAndLargerContextsAreRefusedBeforeReserving() throws {
        let resources = try profile()
        #expect(throws: LanguageGenerationError.self) { try resources.budget(modelSHA256: String(repeating: "0", count: 64)) }
        #expect(throws: LanguageGenerationError.self) { try resources.budget(modelSHA256: decoderHash, contextLength: 4097) }
        _ = try resources.budget(modelSHA256: decoderHash, contextLength: 4096)
        #expect(resources.coordinator.snapshot.reservedBytes == 0)
    }

    /// Residency and scratch reserve their estimates, refuse past the capacity, and release.
    @Test func residencyAndScratchReserveTheirEstimates() throws {
        let resources = try profile()
        let decoder = try resources.budget(modelSHA256: decoderHash), encoder = try resources.budget(modelSHA256: encoderHash)
        let decoderResidency = try decoder.reserveResidency()
        let encoderResidency = try encoder.reserveResidency()
        #expect(resources.coordinator.snapshot.reservedBytes == 80)
        let scratch = try decoder.reserveScratch()
        #expect(resources.coordinator.snapshot.reservedBytes == 90)
        #expect(throws: ResourceAdmissionError.insufficientCapacity(requested: 20, available: 10)) { try encoder.reserveResidency() }
        scratch.release()
        encoderResidency.release()
        decoderResidency.release()
        #expect(resources.coordinator.snapshot.reservedBytes == 0)
    }

    @Test func imageTokensAreCheckedAgainstTheProfile() throws {
        let budget = try profile().budget(modelSHA256: encoderHash)
        try budget.check(imageTokens: 0)
        try budget.check(imageTokens: 512)
        #expect(throws: LanguageGenerationError.self) { try budget.check(imageTokens: 513) }
        let textOnly = try profile(maximumImageTokens: nil).budget(modelSHA256: decoderHash)
        try textOnly.check(imageTokens: 0)
        #expect(throws: LanguageGenerationError.self) { try textOnly.check(imageTokens: 1) }
    }

    /// Opening waits for a grant while another owner holds the GPU, holds it for the load and releases it after,
    /// also when the load fails.
    @Test(.timeLimit(.minutes(1))) func openingWaitsForAGrantAndReleasesIt() async throws {
        let admission = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        let blocker = try await admission.admit(owner: "render")
        let loaded = Flag()
        let open = Task {
            try await withOpenAdmission(admission.owner("describer")) {
                #expect(admission.snapshot.running == 1)
                loaded.set()
            }
        }
        for _ in 0..<200 where admission.snapshot.queued == 0 { try await Task.sleep(for: .milliseconds(5)) }
        #expect(admission.snapshot.queued == 1 && !loaded.value)
        blocker.release()
        try await open.value
        #expect(loaded.value && admission.snapshot.running == 0)
        struct LoadFailure: Error {}
        await #expect(throws: LoadFailure.self) {
            try await withOpenAdmission(admission.owner("describer")) { throw LoadFailure() }
        }
        #expect(admission.snapshot.running == 0 && admission.snapshot.queued == 0)
    }

    /// Cancelling an opening that waits for the GPU leaves the queue without loading.
    @Test(.timeLimit(.minutes(1))) func cancellingAWaitingOpenLeavesTheQueue() async throws {
        let admission = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        let blocker = try await admission.admit(owner: "render")
        let loaded = Flag()
        let open = Task { try await withOpenAdmission(admission.owner("describer")) { loaded.set() } }
        for _ in 0..<200 where admission.snapshot.queued == 0 { try await Task.sleep(for: .milliseconds(5)) }
        open.cancel()
        await #expect(throws: (any Error).self) { try await open.value }
        #expect(!loaded.value && admission.snapshot.queued == 0)
        blocker.release()
        #expect(admission.snapshot.running == 0)
    }

    /// Without a GPU owner, opening runs at once.
    @Test func openingWithoutAdmissionRunsAtOnce() async throws {
        #expect(try await withOpenAdmission(nil) { 7 } == 7)
    }
}
