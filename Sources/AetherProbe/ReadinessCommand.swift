import AetherAssets
import AetherKit
import AetherLanguage
import Foundation

/// `readiness ENTRY.json STORE [--prepare] [--memory-bytes BYTES]`: qualifies typed readiness (docs/model-readiness.md).
/// Prints the states a `ModelReadiness` passes through until it settles: ready, unsupported, failed, or not installed
/// without `--prepare`. With `--prepare` a model that isn't installed is prepared over HTTPS (`HF_TOKEN` for a gated
/// repository). `--memory-bytes` replaces the entry's memory requirement, to see the device check refuse or admit it.
/// Language entries use the language check. CPU and network only.
@available(macOS 27, iOS 27, *)
enum ReadinessCommand {
    static let usage = """
        aether-probe readiness ENTRY.json STORE_DIRECTORY [--prepare] [--memory-bytes BYTES]
        """

    private struct Transition: Encodable {
        let seconds: Double
        let state: String
    }

    private struct Report: Encodable {
        let installationID: String
        let requirements: String
        let deviceMemoryLimitBytes: UInt64?
        var transitions: [Transition] = []
        var progressStates = 0
        var final = ""
        var seconds = 0.0
    }

    static func run(_ arguments: [String]) async throws -> Bool {
        guard arguments.first == "readiness" else { return false }
        guard arguments.count >= 3 else { throw AetherError.invalidRequest("usage:\n" + usage) }
        var prepare = false
        var memoryBytes: UInt64?
        var options = arguments.dropFirst(3)
        while let flag = options.popFirst() {
            switch flag {
            case "--prepare" where !prepare: prepare = true
            case "--memory-bytes" where memoryBytes == nil:
                guard let value = options.popFirst().flatMap(UInt64.init) else {
                    throw AetherError.invalidRequest("--memory-bytes takes a byte count")
                }
                memoryBytes = value
            default: throw AetherError.invalidRequest("usage:\n" + usage)
            }
        }
        var entry = try JSONDecoder().decode(ModelCatalogEntry.self, from: Data(contentsOf: URL(fileURLWithPath: arguments[1])))
        if let memoryBytes {
            entry = try ModelCatalogEntry(manifest: entry.manifest, source: entry.source, displayName: entry.displayName,
                                          requirements: ModelRequirements(platforms: entry.requirements?.platforms, memoryBytes: memoryBytes))
        }
        var headers: @Sendable (URL) -> [String: String] = { _ in [:] }
        if let token = ProcessInfo.processInfo.environment["HF_TOKEN"], !token.isEmpty {
            headers = ModelPreparation.huggingFaceAuthorization(token: token)
        }
        let preparation = ModelPreparation(store: try LocalAssetStore(root: URL(fileURLWithPath: arguments[2])),
                                           transport: URLSessionTransport(), headers: headers)
        let check: ModelRuntimeCheck = entry.manifest.executionContract == "aether.language-bundle/1" ? .language : .none
        let start = ContinuousClock.now
        let readiness = ModelReadiness(entry: entry, preparation: preparation, check: check)
        var report = Report(installationID: try entry.installationID, requirements: String(describing: entry.requirements),
                            deviceMemoryLimitBytes: ModelReadiness.deviceMemoryLimit)
        var prepared = false
        var lastKind = ""
        for await state in readiness.states {
            let seconds = (ContinuousClock.now - start) / .seconds(1)
            let kind = describe(state, progress: false)
            if case .installing = state { report.progressStates += 1 }
            // Every change of kind, and each progress phase's first state; the last progress state is in `final`'s lead-up.
            if kind != lastKind { report.transitions.append(Transition(seconds: seconds, state: describe(state, progress: true))) }
            lastKind = kind
            switch state {
            case .ready, .unsupported: break
            case .failed where prepared || !prepare: break
            case .notInstalled, .failed:
                if prepare && !prepared { prepared = true; readiness.prepare(); continue }
            case .verifying, .installing: continue
            }
            report.final = describe(state, progress: true)
            break
        }
        report.seconds = (ContinuousClock.now - start) / .seconds(1)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        FileHandle.standardOutput.write(try encoder.encode(report) + Data("\n".utf8))
        return true
    }

    private static func describe(_ state: ModelReadiness.State, progress: Bool) -> String {
        switch state {
        case let .notInstalled(present, total): "notInstalled \(present)/\(total)"
        case let .installing(step): progress ? "installing \(step.phase) \(step.bytesDone)/\(step.bytesTotal)" : "installing \(step.phase)"
        case .verifying: "verifying"
        case let .ready(lease): "ready \(lease.installationID)"
        case let .unsupported(reason): "unsupported \(reason)"
        case let .failed(failure): "failed \(failure) retryable=\(failure.isRetryable)"
        }
    }
}
