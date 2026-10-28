@testable import AetherLanguage
import AetherKit
import AetherLanguageRuntime
import Foundation
import Testing

/// A temporary bundle with a registration, a placeholder model file and a tokenizer folder.
private struct TemporaryBundle {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("language-bundle-\(UUID().uuidString)")

    init(platform: String = "macos", template: Bool = true, model: Bool = true, schemaVersion: Int = 1,
         modelDirectory: String = "model.aimodel") throws {
        let files = FileManager.default
        try files.createDirectory(at: url.appendingPathComponent("tokenizer"), withIntermediateDirectories: true)
        if model {
            try files.createDirectory(at: url.appendingPathComponent("model.aimodel"), withIntermediateDirectories: true)
            try Data("graph".utf8).write(to: url.appendingPathComponent("model.aimodel/main.mlirb"))
        }
        try Data("{}".utf8).write(to: url.appendingPathComponent("tokenizer/tokenizer.json"))
        if template { try Data("{{ messages }}".utf8).write(to: url.appendingPathComponent("tokenizer/chat_template.jinja")) }
        let manifest: [String: Any] = [
            "schemaVersion": schemaVersion, "modelDirectory": modelDirectory, "tokenizerDirectory": "tokenizer",
            "sourceRepository": "example/model", "sourceRevision": String(repeating: "a", count: 40),
            "modelSHA256": String(repeating: "b", count: 64), "tokenizerSHA256": String(repeating: "c", count: 64),
            "maximumContextLength": 2048, "stopTokenIDs": [2], "platform": platform, "quantization": "fp16",
            "supportsReasoning": true,
        ]
        try JSONSerialization.data(withJSONObject: manifest).write(to: url.appendingPathComponent("language-model.json"))
    }

    func remove() { try? FileManager.default.removeItem(at: url) }
}

private actor LoadCounter {
    var loads = 0
    var failNext = false
    func load() throws -> Int {
        loads += 1
        if failNext { failNext = false; throw AetherError.invalidModel("checksum mismatch") }
        return loads
    }
    func setFailNext() { failNext = true }
}

struct AetherLanguageModelTests {
    @Test func readinessReasonsComeFromTheRegistrationAlone() throws {
        let good = try TemporaryBundle()
        defer { good.remove() }
        let model = AetherLanguageModel(bundle: good.url, platform: .macos)
        #expect(model.availability == .available && model.supportsReasoning && model.contextLength == 2048)
        #expect(AetherLanguageModel(bundle: good.url, platform: .ios).availability == .unavailable(.unsupportedPlatform))

        let empty = FileManager.default.temporaryDirectory.appendingPathComponent("no-bundle-\(UUID().uuidString)")
        #expect(AetherLanguageModel(bundle: empty).availability == .unavailable(.bundleMissing))
        let noModel = try TemporaryBundle(model: false)
        defer { noModel.remove() }
        #expect(AetherLanguageModel(bundle: noModel.url, platform: .macos).availability == .unavailable(.bundleMissing))
        let noTemplate = try TemporaryBundle(template: false)
        defer { noTemplate.remove() }
        #expect(AetherLanguageModel(bundle: noTemplate.url, platform: .macos).availability == .unavailable(.unsupportedTemplate))
        for invalid in [try TemporaryBundle(schemaVersion: 2), try TemporaryBundle(modelDirectory: "../escape")] {
            defer { invalid.remove() }
            guard case .unavailable(.bundleInvalid) = AetherLanguageModel(bundle: invalid.url, platform: .macos).availability else {
                Issue.record("expected bundleInvalid for \(invalid.url.lastPathComponent)"); continue
            }
        }
        let unparsable = try TemporaryBundle()
        defer { unparsable.remove() }
        try Data("{".utf8).write(to: unparsable.url.appendingPathComponent("language-model.json"))
        guard case .unavailable(.bundleInvalid) = AetherLanguageModel(bundle: unparsable.url).availability else {
            Issue.record("expected bundleInvalid for a manifest that doesn't parse"); return
        }
    }

    @Test func belowTheCoreAIFloorEveryBundleIsUnsupportedOS() async throws {
        let good = try TemporaryBundle()
        defer { good.remove() }
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent("no-bundle-\(UUID().uuidString)")
        for bundle in [good.url, empty] {
            let model = AetherLanguageModel(bundle: bundle, platform: .macos, coreAIAvailable: false)
            #expect(model.availability == .unavailable(.unsupportedOS) && model.contextLength == 0)
            do { try await model.prewarm(); Issue.record("loaded below the Core AI floor") }
            catch AetherLanguageError.unavailable(let reason) { #expect(reason == .unsupportedOS) }
        }
        #expect(AetherLanguageModel(bundle: good.url, platform: .macos, coreAIAvailable: true).availability == .available)
    }

    @Test func unavailableModelsRefuseRequestsWithTheirReason() async throws {
        let model = AetherLanguageModel(bundle: URL(filePath: "/nonexistent"), availability: .unavailable(.unsupportedOS),
                                        supportsReasoning: false, contextLength: 0) { Issue.record("loaded"); throw CancellationError() }
        await #expect(throws: AetherLanguageError.self) { try await model.prewarm() }
        do { try await model.prewarm() } catch AetherLanguageError.unavailable(let reason) { #expect(reason == .unsupportedOS) }
    }

    @Test func loadsOnceLazilyAndRetriesAFailedLoad() async throws {
        let counter = LoadCounter()
        await counter.setFailNext()
        let model = FakeModel.make { _ = try await counter.load() }
        await #expect(throws: AetherLanguageError.self) { try await model.prewarm() }
        try await model.prewarm()
        try await model.prewarm()
        let copy = model
        try await copy.prewarm()
        #expect(await counter.loads == 2, "one failed load, then one load shared by every copy")
        await model.unload()
        try await model.prewarm()
        #expect(await counter.loads == 3, "unloading releases; the next request loads again")
    }

    @Test(.timeLimit(.minutes(1))) func requestsRunOneAtATimeInOrderAndCancelWhileWaiting() async throws {
        let model = FakeModel.make {}
        let trace = Trace()
        let gate = Gate()
        let first = Task { try await model.resources.withModel { _ in await trace.add("first:start"); await gate.wait(); await trace.add("first:end") } }
        while await !trace.events.contains("first:start") { await Task.yield() }
        let second = Task { try await model.resources.withModel { _ in await trace.add("second") } }
        let cancelled = Task { try await model.resources.withModel { _ in await trace.add("cancelled ran") } }
        try await Task.sleep(for: .milliseconds(50))
        cancelled.cancel()
        await #expect(throws: CancellationError.self) { try await cancelled.value }
        await gate.open()
        try await first.value
        try await second.value
        #expect(await trace.events == ["first:start", "first:end", "second"])
    }
}

actor Trace {
    private(set) var events: [String] = []
    func add(_ event: String) { events.append(event) }
}

actor Gate {
    private var opened = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        guard !opened else { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func open() {
        opened = true
        waiters.forEach { $0.resume() }
        waiters.removeAll()
    }
}
