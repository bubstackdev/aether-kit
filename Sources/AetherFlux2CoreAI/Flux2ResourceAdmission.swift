import AetherAssets
import AetherCoreAI
import AetherFlux2
import AetherKit

/// A caller-qualified maximum estimate for one exact model stream and metadata pair.
@available(macOS 27, iOS 27, *)
public struct Flux2ModelResourceEstimate: Sendable {
    let identity: String
    let residentBytes: UInt64
    let runScratchBytes: UInt64
    public init(asset: DeclaredModelAsset, residentBytes: UInt64, runScratchBytes: UInt64) throws {
        guard residentBytes > 0, runScratchBytes > 0 else { throw ResourceAdmissionError.invalidByteCount }
        identity = asset.identity
        self.residentBytes = residentBytes; self.runScratchBytes = runScratchBytes
    }
}

/// Opt-in reservations of caller estimates against a host-selected shared `ResourceCoordinator`:
/// operation storage for each prepare/generate, each loaded model's residency (including its load
/// peak) and each native call's scratch. Estimates must cover every shape up to `maximumImageTokens`
/// (generated plus reference tokens). Unknown model bytes and larger shapes refuse before loading.
/// This is admission of estimates, not a measured footprint or allocator enforcement.
@available(macOS 27, iOS 27, *)
public struct Flux2ResourceAdmission: Sendable {
    public let coordinator: ResourceCoordinator
    public let operationBytes: UInt64
    public let maximumImageTokens: Int
    private let estimates: [String: Flux2ModelResourceEstimate]

    public init(coordinator: ResourceCoordinator, operationBytes: UInt64, maximumImageTokens: Int,
                models: [Flux2ModelResourceEstimate]) throws {
        guard operationBytes > 0, (1...Flux2Configuration.maximumImageTokens).contains(maximumImageTokens), !models.isEmpty else {
            throw AetherError.invalidRequest("invalid FLUX.2 resource profile")
        }
        var keyed: [String: Flux2ModelResourceEstimate] = [:]
        for model in models {
            guard keyed.updateValue(model, forKey: model.identity) == nil else {
                throw AetherError.invalidRequest("duplicate FLUX.2 model resource estimate")
            }
        }
        self.coordinator = coordinator; self.operationBytes = operationBytes
        self.maximumImageTokens = maximumImageTokens; estimates = keyed
    }

    func estimate(for asset: DeclaredModelAsset) throws -> Flux2ModelResourceEstimate {
        guard let value = estimates[asset.identity] else {
            throw AetherError.invalidRequest("no resource estimate for the selected FLUX.2 model bytes")
        }
        return value
    }

    func check(imageTokens: Int) throws {
        guard imageTokens <= maximumImageTokens else {
            throw AetherError.invalidRequest("request exceeds the FLUX.2 resource profile's token limit")
        }
    }
}

@available(macOS 27, iOS 27, *)
struct BudgetedFlux2Factory: Flux2ModelFactory {
    let base: any Flux2ModelFactory
    let resources: Flux2ResourceAdmission
    func open(_ asset: DeclaredModelAsset, frequentReshapes: Bool) async throws -> any StagedModel {
        let estimate = try resources.estimate(for: asset)
        return try await openBudgeted(coordinator: resources.coordinator, residentBytes: estimate.residentBytes,
                                      runScratchBytes: estimate.runScratchBytes) {
            try await base.open(asset, frequentReshapes: frequentReshapes)
        }
    }
    func finish() async { await base.finish() }
}
