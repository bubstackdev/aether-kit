import AetherAssets
import AetherKit
import AetherLanguage
import Foundation

/// Real-network qualification of model preparation (docs/model-preparation.md#qualification):
/// - `prepare-model ENTRY.json STORE [--cut-after BYTES]` prepares a catalog entry over HTTPS into a store. With
///   `--cut-after`, the first run is cancelled once BYTES have downloaded, the kept partial status is recorded, and a
///   second run resumes and installs. A Hugging Face token, when needed, comes from `HF_TOKEN`. CPU and network only.
/// - `reply-installed STORE INSTALLATION_ID PROMPT` opens the installed language model through the facade
///   (`AetherLanguageModel(installation:)`) and prints a greedy reply. Takes the GPU lease.
@available(macOS 27, iOS 27, *)
enum PrepareModelCommand {
    static let usage = """
        aether-probe prepare-model ENTRY.json STORE_DIRECTORY [--cut-after BYTES]
        aether-probe reply-installed STORE_DIRECTORY INSTALLATION_ID PROMPT
        """

    private struct RunReport: Encodable {
        var terminal = ""
        var downloadEvents = 0
        var installingEvents = 0
        var firstDownloadedBytes: UInt64?
        var lastDownloadedBytes: UInt64?
        var seconds = 0.0
    }

    private struct Report: Encodable {
        let entryInstallationID: String
        let bytesTotal: UInt64
        let files: Int
        let source: String
        var runs: [RunReport] = []
        var statusBetweenRuns: String?
        var finalStatus = ""
        var installationID: String?
    }

    static func run(_ arguments: [String]) async throws -> Bool {
        switch arguments.first {
        case "prepare-model":
            guard arguments.count == 3 || (arguments.count == 5 && arguments[3] == "--cut-after" && UInt64(arguments[4]) != nil) else {
                throw AetherError.invalidRequest("usage:\n" + usage)
            }
            try await prepare(entryPath: arguments[1], store: arguments[2], cutAfter: arguments.count == 5 ? UInt64(arguments[4]) : nil)
            return true
        case "reply-installed":
            guard arguments.count == 4 else { throw AetherError.invalidRequest("usage:\n" + usage) }
            try await reply(store: arguments[1], installationID: arguments[2], prompt: arguments[3])
            return true
        default:
            return false
        }
    }

    private static func prepare(entryPath: String, store storePath: String, cutAfter: UInt64?) async throws {
        let entry = try JSONDecoder().decode(ModelCatalogEntry.self, from: Data(contentsOf: URL(fileURLWithPath: entryPath)))
        let store = try LocalAssetStore(root: URL(fileURLWithPath: storePath))
        let token = ProcessInfo.processInfo.environment["HF_TOKEN"].flatMap { $0.isEmpty ? nil : $0 }
        var headers: @Sendable (URL) -> [String: String] = { _ in [:] }
        if let token { headers = ModelPreparation.huggingFaceAuthorization(token: token) }
        let preparation = ModelPreparation(store: store, transport: URLSessionTransport(), headers: headers)
        let source: String
        switch entry.source {
        case let .https(base): source = "https " + (base.host ?? "")
        case let .huggingFace(repository, revision, _): source = "huggingFace \(repository)@\(revision)"
        }
        var report = Report(entryInstallationID: try entry.installationID, bytesTotal: entry.byteCount,
                            files: entry.manifest.files.count, source: source)
        if let cutAfter {
            report.runs.append(await observe(preparation.prepare(entry), stopAfter: cutAfter))
            report.statusBetweenRuns = describe(try preparation.status(entry))
        }
        report.runs.append(await observe(preparation.prepare(entry), stopAfter: nil))
        let status = try preparation.status(entry)
        report.finalStatus = describe(status)
        if case let .installed(id) = status { report.installationID = id }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        FileHandle.standardOutput.write(try encoder.encode(report) + Data("\n".utf8))
    }

    /// Consumes a run; with `stopAfter`, stops iterating (which cancels the run) once that many bytes have downloaded.
    private static func observe(_ events: AsyncStream<PreparationEvent>, stopAfter: UInt64?) async -> RunReport {
        var run = RunReport()
        let start = ContinuousClock.now
        for await event in events {
            switch event {
            case let .downloading(_, _, _, done, _):
                run.downloadEvents += 1
                if run.firstDownloadedBytes == nil { run.firstDownloadedBytes = done }
                run.lastDownloadedBytes = done
                if let stopAfter, done >= stopAfter { run.terminal = "stopped by the qualification at \(done) bytes" }
            case .installing: run.installingEvents += 1
            case let .installed(id): run.terminal = "installed \(id)"
            case let .failed(error): run.terminal = "failed \(error)"
            case .cancelled: run.terminal = "cancelled"
            case .checking: break
            }
            if !run.terminal.isEmpty { break }
        }
        run.seconds = (ContinuousClock.now - start) / .seconds(1)
        return run
    }

    private static func describe(_ status: PreparationStatus) -> String {
        switch status {
        case let .installed(id): "installed \(id)"
        case let .partial(present, total): "partial \(present)/\(total)"
        case let .notInstalled(total): "notInstalled \(total)"
        }
    }

    private static func reply(store storePath: String, installationID: String, prompt: String) async throws {
        let store = try LocalAssetStore(root: URL(fileURLWithPath: storePath))
        let model = AetherLanguageModel(installation: try await store.acquire(installationID))
        guard model.availability == .available else { throw AetherError.invalidModel("unavailable: \(model.availability)") }
        let start = ContinuousClock.now
        let response = try await AetherLanguageSession(model: model).respond(
            to: prompt, options: AetherGenerationOptions(samplingMode: .greedy, maximumResponseTokens: 48))
        let seconds = (ContinuousClock.now - start) / .seconds(1)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        FileHandle.standardOutput.write(try encoder.encode([
            "installationID": installationID, "prompt": prompt, "reply": response.content, "seconds": String(format: "%.2f", seconds),
        ]) + Data("\n".utf8))
        await model.unload()
    }
}
