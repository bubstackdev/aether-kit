import AetherAssets
import AetherLanguage
import Foundation

struct ReadinessResult: Codable {
    struct Transition: Codable { let seconds: Double; let state: String }

    var entryInstallationID = ""
    var requirements = ""
    var deviceMemoryLimitBytes: UInt64?
    var transitions: [Transition] = []
    var final = ""
    var reply: String?
    var replySeconds: Double?
}

/// `probe.json` mode "readiness": a `ModelReadiness` over `Assets/entry.json` with `BackgroundTransport` and the language
/// check (docs/model-readiness.md). The states it passes through are recorded; a model that isn't installed is prepared.
/// Once ready, the model opens from the ready state's lease and answers a short greedy prompt. `memoryBytes` replaces the
/// entry's memory requirement, to see the device check against the app's real memory limit.
func runReadiness(assets: URL, store storeName: String, memoryBytes: UInt64?, report: inout DeviceReport) async throws {
    var entry = try JSONDecoder().decode(ModelCatalogEntry.self, from: Data(contentsOf: assets.appendingPathComponent("entry.json")))
    if let memoryBytes {
        entry = try ModelCatalogEntry(manifest: entry.manifest, source: entry.source, displayName: entry.displayName,
                                      requirements: ModelRequirements(platforms: entry.requirements?.platforms, memoryBytes: memoryBytes))
    }
    let support = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    var headers: @Sendable (URL) -> [String: String] = { _ in [:] }
    if let token = ProcessInfo.processInfo.environment["HF_TOKEN"], !token.isEmpty {
        headers = ModelPreparation.huggingFaceAuthorization(token: token)
    }
    let preparation = ModelPreparation(store: try LocalAssetStore(root: support.appendingPathComponent(storeName)),
                                       transport: BackgroundTransport(identifier: preparationSessionIdentifier), headers: headers)
    var result = ReadinessResult(entryInstallationID: try entry.installationID, requirements: String(describing: entry.requirements),
                                 deviceMemoryLimitBytes: ModelReadiness.deviceMemoryLimit)
    report.bundleModelSHA256 = entry.manifest.files.first { $0.path.hasSuffix("main.mlirb") }?.sha256 ?? ""

    let started = ContinuousClock.now
    let readiness = ModelReadiness(entry: entry, preparation: preparation, check: .language)
    var prepared = false
    var lastKind = ""
    var lease: AssetReadLease?
    for await state in readiness.states {
        let kind: String
        switch state {
        case let .notInstalled(present, total): kind = "notInstalled \(present)/\(total)"
        case let .installing(progress): kind = "installing \(progress.phase)"
        case .verifying: kind = "verifying"
        case let .ready(held): kind = "ready \(held.installationID)"
        case let .unsupported(reason): kind = "unsupported \(reason)"
        case let .failed(failure): kind = "failed \(failure) retryable=\(failure.isRetryable)"
        @unknown default: kind = "\(state)"
        }
        if kind != lastKind {
            lastKind = kind
            result.transitions.append(.init(seconds: (ContinuousClock.now - started).seconds, state: kind))
            print("READINESS_STATE " + kind)
            fflush(stdout)
        }
        if case .installing = state { continue }
        if case .verifying = state { continue }
        if case .notInstalled = state, !prepared { prepared = true; readiness.prepare(); continue }
        if case let .ready(held) = state { lease = held }
        result.final = kind
        break
    }

    if let lease {
        let model = AetherLanguageModel(installation: lease, platform: .ios)
        guard model.availability == .available else { throw PreparationProbeError.unavailable("\(model.availability)") }
        let replyStarted = ContinuousClock.now
        let response = try await AetherLanguageSession(model: model).respond(
            to: "In one sentence, what is a hash function?",
            options: AetherGenerationOptions(samplingMode: .greedy, maximumResponseTokens: 48))
        result.reply = response.content
        result.replySeconds = (ContinuousClock.now - replyStarted).seconds
        await model.unload()
    }
    report.readiness = result
}
