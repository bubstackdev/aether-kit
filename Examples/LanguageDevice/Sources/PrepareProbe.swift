import AetherAssets
import AetherLanguage
import Foundation

/// The background-session identifier the app registers for model preparation.
let preparationSessionIdentifier = "dev.aether.language-device-probe.models"

struct PreparationResult: Codable {
    var entryInstallationID = ""
    var bytesTotal: UInt64 = 0
    var statusAtStart = ""
    var terminal = ""
    var downloadEvents = 0
    var firstDownloadedBytes: UInt64?
    var lastDownloadedBytes: UInt64?
    var preparationSeconds = 0.0
    var reply: String?
    var replySeconds: Double?
}

/// `probe.json` mode "prepare": prepares `Assets/entry.json` with `BackgroundTransport` into Application Support, then
/// opens the installation through `AetherLanguageModel(installation:)` and asks for a short greedy reply. Progress goes
/// to stdout and to `Documents/prepare-log.jsonl`, which survives the app being killed and relaunched mid-transfer. A
/// token for a private repository comes from the launch environment (`HF_TOKEN`); it's only needed when tasks are created.
func runPreparation(assets: URL, report: inout DeviceReport) async throws {
    let entry = try JSONDecoder().decode(ModelCatalogEntry.self, from: Data(contentsOf: assets.appendingPathComponent("entry.json")))
    let support = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    let store = try LocalAssetStore(root: support.appendingPathComponent("Models"))
    var headers: @Sendable (URL) -> [String: String] = { _ in [:] }
    if let token = ProcessInfo.processInfo.environment["HF_TOKEN"], !token.isEmpty {
        headers = ModelPreparation.huggingFaceAuthorization(token: token)
    }
    let preparation = ModelPreparation(store: store, transport: BackgroundTransport(identifier: preparationSessionIdentifier),
                                       headers: headers)
    var result = PreparationResult(entryInstallationID: try entry.installationID, bytesTotal: entry.byteCount)
    result.statusAtStart = "\(try preparation.status(entry))"
    log(["event": "start", "status": result.statusAtStart])

    let started = ContinuousClock.now
    var lastLogged = ContinuousClock.now - .seconds(10)
    for await event in preparation.prepare(entry) {
        switch event {
        case let .downloading(path, fileBytes, _, done, total):
            result.downloadEvents += 1
            if result.firstDownloadedBytes == nil { result.firstDownloadedBytes = done }
            result.lastDownloadedBytes = done
            if ContinuousClock.now - lastLogged >= .seconds(1) {
                lastLogged = .now
                log(["event": "downloading", "path": path, "fileBytes": "\(fileBytes)", "done": "\(done)", "total": "\(total)"])
            }
        case let .installing(done, total):
            if ContinuousClock.now - lastLogged >= .seconds(1) || done == 0 {
                lastLogged = .now
                log(["event": "installing", "done": "\(done)", "total": "\(total)"])
            }
        case let .installed(id): result.terminal = "installed \(id)"
        case let .failed(error): result.terminal = "failed \(error)"
        case .cancelled: result.terminal = "cancelled"
        case .checking: break
        }
    }
    result.preparationSeconds = (ContinuousClock.now - started).seconds
    log(["event": "terminal", "terminal": result.terminal])
    report.bundleModelSHA256 = entry.manifest.files.first { $0.path.hasSuffix("main.mlirb") }?.sha256 ?? ""

    if result.terminal.hasPrefix("installed") {
        let model = AetherLanguageModel(installation: try await store.acquire(try entry.installationID), platform: .ios)
        guard model.availability == .available else { throw PreparationProbeError.unavailable("\(model.availability)") }
        let replyStarted = ContinuousClock.now
        let response = try await AetherLanguageSession(model: model).respond(
            to: "In one sentence, what is a hash function?",
            options: AetherGenerationOptions(samplingMode: .greedy, maximumResponseTokens: 48))
        result.reply = response.content
        result.replySeconds = (ContinuousClock.now - replyStarted).seconds
        await model.unload()
        log(["event": "reply", "reply": response.content])
    } else {
        report.failures.append("preparation ended: \(result.terminal)")
    }
    report.preparation = result
}

enum PreparationProbeError: Error { case unavailable(String) }

/// One JSON line to stdout and to `Documents/prepare-log.jsonl`, with a wall-clock timestamp.
private func log(_ fields: [String: String]) {
    var line = fields
    line["time"] = ISO8601DateFormatter().string(from: Date())
    guard let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) else { return }
    print("PREPARE_EVENT " + String(decoding: data, as: UTF8.self))
    fflush(stdout)
    guard let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
    let url = documents.appendingPathComponent("prepare-log.jsonl")
    if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
    if let handle = try? FileHandle(forWritingTo: url) {
        try? handle.seekToEnd()
        try? handle.write(contentsOf: data + Data("\n".utf8))
        try? handle.close()
    }
}
