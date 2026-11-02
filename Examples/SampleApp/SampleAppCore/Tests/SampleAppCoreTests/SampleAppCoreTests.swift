import AetherAssets
import AetherKit
import AetherLanguage
import CryptoKit
import Foundation
import Testing
@testable import SampleAppCore

@Suite struct SampleCatalogTests {
    @Test(arguments: [ModelRequirements.Platform.macos, .ios])
    func theShippedEntriesArePinnedToThePublicBundle(_ platform: ModelRequirements.Platform) throws {
        let entry = try SampleCatalog.entry(for: platform)
        guard case let .huggingFace(repository, revision, subdirectory) = entry.source else {
            Issue.record("expected a Hugging Face source, got \(entry.source)")
            return
        }
        #expect(repository == "aether-models/aether-kit-qwen3-0.6b")
        #expect(revision == "7757d22bff5df7e106f7492a12d7278347aa3fb0")
        #expect(subdirectory == (platform == .ios ? "ios" : "mac"))
        #expect(entry.manifest.executionContract == "aether.language-bundle/1")
        #expect(entry.manifest.files.count == 11)
        #expect(entry.manifest.files.contains { $0.path == "language-model.json" })
        // The bundle sizes the model-preparation qualification recorded (docs/model-preparation.md).
        #expect(entry.byteCount == (platform == .ios ? 1_208_370_448 : 1_208_370_450))
        #expect(entry.requirements?.platforms == [platform])
        // iOS: 220 MiB, the qualified peak footprint (Examples/SampleApp/make-catalog.sh cites the record). Mac: none.
        #expect(entry.requirements?.memoryBytes == (platform == .ios ? 230_686_720 : nil))
        #expect(throws: Never.self) { try entry.installationID }
    }
}

@Suite struct ReadinessDisplayTests {
    @Test func notInstalledOffersADownloadAndAResumeAfterAPause() {
        let fresh = ReadinessDisplay(.notInstalled(bytesPresent: 0, bytesTotal: 1_000_000))
        #expect(fresh.title == "Not installed" && fresh.actions == [.download] && fresh.progress == nil)
        let paused = ReadinessDisplay(.notInstalled(bytesPresent: 250_000, bytesTotal: 1_000_000))
        #expect(paused.title == "Paused" && paused.actions == [.download, .delete] && paused.progress == 0.25)
    }

    @Test func installingShowsItsPhaseAndOffersCancel() {
        let checking = ReadinessDisplay(.installing(.init(phase: .checking, bytesDone: 0, bytesTotal: 0)))
        #expect(checking.progress == nil && checking.actions == [.cancel] && checking.tone == .busy)
        let downloading = ReadinessDisplay(.installing(.init(phase: .downloading, bytesDone: 50, bytesTotal: 200)))
        #expect(downloading.progress == 0.25 && downloading.detail.hasPrefix("Downloading"))
        let installing = ReadinessDisplay(.installing(.init(phase: .installing, bytesDone: 200, bytesTotal: 200)))
        #expect(installing.progress == 1 && installing.detail.hasPrefix("Installing"))
    }

    @Test func verifyingOffersNothing() {
        #expect(ReadinessDisplay(.verifying).actions.isEmpty)
    }

    @Test func unsupportedSaysWhy() {
        let memory = ReadinessDisplay(.unsupported(.memory(requiredBytes: 8_000_000_000, availableBytes: 4_000_000_000)))
        #expect(memory.tone == .problem && memory.actions == [.refresh] && memory.detail.contains("memory"))
        #expect(ReadinessDisplay(.unsupported(.operatingSystem)).detail.contains("27"))
        #expect(ReadinessDisplay(.unsupported(.platform)).detail.contains("platform"))
    }

    @Test func failuresOfferRetryOnlyWhenItCanHelp() {
        let network = ReadinessDisplay(.failed(.preparation(.network("offline"))))
        #expect(network.actions == [.retry, .delete] && network.detail.contains("offline"))
        let space = ReadinessDisplay(.failed(.preparation(.insufficientSpace(needed: 2_000_000_000, available: 1_000))))
        #expect(space.actions == [.delete])
        let invalid = ReadinessDisplay(.failed(.invalidModel("no chat template")))
        #expect(invalid.actions == [.delete] && invalid.detail.contains("no chat template"))
        #expect(ReadinessDisplay(.failed(.verification("hash"))).actions == [.retry, .delete])
    }

    @Test func lifecycleStripNamesEveryState() {
        #expect(LifecycleDisplay(state: nil, pressure: .normal).model == "No model open")
        #expect(LifecycleDisplay(state: .unloaded(nil), pressure: .normal).model == "Not loaded yet")
        #expect(LifecycleDisplay(state: .unloaded(.idle), pressure: .normal).model == "Unloaded after idling")
        #expect(LifecycleDisplay(state: .unloaded(.memoryPressure), pressure: .warning).model == "Unloaded under memory pressure")
        #expect(LifecycleDisplay(state: .unloaded(.requested), pressure: .normal).model == "Unloaded")
        #expect(LifecycleDisplay(state: .loading, pressure: .normal).isLoaded == false)
        let running = LifecycleDisplay(state: .running, pressure: .critical)
        #expect(running.isLoaded && running.model == "Running" && running.pressure.contains("critical"))
        #expect(LifecycleDisplay(state: .loaded, pressure: .normal).model == "Loaded, idle")
    }

    @Test func theQualificationRecordNamesLifecycleStatesPlainly() {
        #expect(QualificationRun.name(.unloaded(nil)) == "unloaded")
        #expect(QualificationRun.name(.unloaded(.idle)) == "unloaded(idle)")
        #expect(QualificationRun.name(.unloaded(.requested)) == "unloaded(requested)")
        #expect(QualificationRun.name(.unloaded(.memoryPressure)) == "unloaded(memoryPressure)")
        #expect(QualificationRun.name(.loading) == "loading")
        #expect(QualificationRun.name(.loaded) == "loaded")
        #expect(QualificationRun.name(.running) == "running")
    }
}

/// A responder that replays scripted updates, optionally pausing between them.
struct ScriptedResponder: ChatResponder {
    var updates: [ChatUpdate]
    var delay: Duration = .zero
    var failure: (any Error & Sendable)?

    func reply(to prompt: String) -> AsyncThrowingStream<ChatUpdate, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                for update in updates {
                    if delay > .zero { try? await Task.sleep(for: delay) }
                    if Task.isCancelled { break }
                    continuation.yield(update)
                }
                if let failure { continuation.finish(throwing: failure) } else { continuation.finish() }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

@MainActor
@Suite struct ChatControllerTests {
    @Test func aReplyStreamsThenCarriesItsStopReasonAndUsage() async {
        let chat = ChatController()
        #expect(!chat.isAvailable)
        chat.send("ignored without a model")
        #expect(chat.messages.isEmpty)
        await chat.attach { ScriptedResponder(updates: [.text("Hel"), .text("Hello"), .finished(stopReason: "end of turn", promptTokens: 12, outputTokens: 2)]) }
        #expect(chat.isAvailable)
        chat.send("  hi  ")
        #expect(chat.isResponding)
        await chat.waitForReply()
        #expect(chat.messages.map(\.role) == [.user, .assistant])
        #expect(chat.messages[0].text == "hi")
        #expect(chat.messages[1].text == "Hello" && !chat.messages[1].isStreaming)
        #expect(chat.messages[1].footnote == "end of turn · 12 prompt + 2 reply tokens")
        #expect(!chat.isResponding)
    }

    @Test func stopKeepsTheTextSoFar() async throws {
        let chat = ChatController()
        await chat.attach { ScriptedResponder(updates: [.text("one"), .text("one two"), .text("one two three")], delay: .milliseconds(200)) }
        chat.send("count")
        while chat.messages.last?.text.isEmpty ?? true { try await Task.sleep(for: .milliseconds(5)) }
        chat.stop()
        await chat.waitForReply()
        #expect(chat.messages.last?.footnote == "stopped")
        #expect(chat.messages.last?.text == "one")
        #expect(!chat.isResponding)
    }

    @Test func aFailedReplySaysSo() async {
        struct Boom: Error {}
        let chat = ChatController()
        await chat.attach { ScriptedResponder(updates: [.text("partial")], failure: Boom()) }
        chat.send("q")
        await chat.waitForReply()
        #expect(chat.messages.last?.footnote?.hasPrefix("failed") == true)
    }

    @Test func newChatAndDetachClearTheConversation() async {
        let chat = ChatController()
        var made = 0
        await chat.attach { made += 1; return ScriptedResponder(updates: [.text("a")]) }
        chat.send("1"); await chat.waitForReply()
        chat.send("2"); await chat.waitForReply()
        #expect(made == 1, "one responder per conversation")
        await chat.newChat()
        #expect(chat.messages.isEmpty)
        chat.send("3"); await chat.waitForReply()
        #expect(made == 2)
        await chat.attach(nil)
        #expect(!chat.isAvailable && chat.messages.isEmpty)
    }

    @Test(.timeLimit(.minutes(1))) func sendIsRefusedWhileTheConversationIsReplacedEvenWithNoReplyRunning() async throws {
        let chat = ChatController()
        await chat.attach { ScriptedResponder(updates: [.text("old")]) }
        chat.send("1"); await chat.waitForReply()
        let gate = Gate()
        chat.drained = { await gate.wait() }
        // New chat and attach each hold after the reply has ended, so only the switching guard refuses.
        let clearing = Task { await chat.newChat() }
        while gate.waiting < 1 { try await Task.sleep(for: .milliseconds(1)) }
        #expect(!chat.isResponding && chat.isAvailable)
        #expect(!chat.send("during new chat"))
        gate.release(0)
        await clearing.value
        #expect(chat.messages.isEmpty)
        let replacing = Task { await chat.attach { ScriptedResponder(updates: [.text("new")]) } }
        while gate.waiting < 2 { try await Task.sleep(for: .milliseconds(1)) }
        #expect(!chat.send("during attach"))
        gate.release(1)
        await replacing.value
        #expect(chat.messages.isEmpty && chat.send("2"))
        await chat.waitForReply()
        #expect(chat.messages.last?.text == "new")
    }

    @Test(.timeLimit(.minutes(1))) func anEarlierAttachThatFinishesLaterDoesNotOverwriteALaterOne() async throws {
        let chat = ChatController()
        let gate = Gate()
        chat.drained = { await gate.wait() }
        let first = Task { await chat.attach { ScriptedResponder(updates: [.text("first")]) } }
        while gate.waiting < 1 { try await Task.sleep(for: .milliseconds(1)) }
        let second = Task { await chat.attach { ScriptedResponder(updates: [.text("second")]) } }
        while gate.waiting < 2 { try await Task.sleep(for: .milliseconds(1)) }
        gate.release(1)
        await second.value
        gate.release(0)
        await first.value
        #expect(chat.isAvailable && chat.send("q"))
        await chat.waitForReply()
        #expect(chat.messages.last?.text == "second")
    }
}

@Suite struct TodoItemTests {
    @Test func theSchemaDescribesEveryField() throws {
        let schema = try #require(JSONSerialization.jsonObject(with: Data(TodoItem.generationSchema.json.utf8)) as? [String: Any])
        let properties = try #require(schema["properties"] as? [String: Any])
        #expect(Set(properties.keys) == ["title", "priority", "dueInDays"])
        #expect(TodoItem.generationSchema.json.contains("\"high\""))
        #expect((schema["required"] as? [String]).map(Set.init) == ["title", "priority"])
    }

    @Test func aConstrainedReplyDecodes() throws {
        let item = try JSONDecoder().decode(TodoItem.self, from: Data(#"{"title":"Renew passport","priority":"high","dueInDays":7}"#.utf8))
        #expect(item == TodoItem(title: "Renew passport", priority: .high, dueInDays: 7))
        let undated = try JSONDecoder().decode(TodoItem.self, from: Data(#"{"title":"Call mum","priority":"low"}"#.utf8))
        #expect(undated.dueInDays == nil)
        #expect(TodoController.encode(item).contains("\"priority\" : \"high\""))
    }

    @MainActor @Test func theControllerReportsTheItemOrTheError() async {
        let todo = TodoController()
        await todo.attach { _ in TodoItem(title: "Renew passport", priority: .high, dueInDays: 7) }
        todo.run(TodoController.example)
        await todo.waitForResult()
        #expect(todo.item?.priority == .high && todo.json != nil && todo.seconds != nil && todo.error == nil)
        struct Nope: Error {}
        await todo.attach { _ in throw Nope() }
        #expect(todo.item == nil)
        #expect(todo.run("x"))
        await todo.waitForResult()
        #expect(todo.item == nil && todo.error != nil)
    }
}

@MainActor
@Suite struct ReplacingTheModelTests {
    /// An extraction that finishes 300 ms after it starts, whether or not it's cancelled.
    static func slowExtraction(_ sentence: String) async throws -> TodoItem {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.3) { continuation.resume() }
        }
        return TodoItem(title: sentence, priority: .low, dueInDays: nil)
    }

    @Test func runIsRefusedWhileTheModelIsReplaced() async throws {
        let todo = TodoController()
        await todo.attach(Self.slowExtraction)
        #expect(todo.run("first"))
        #expect(!todo.run("overlapping"), "one extraction at a time")
        let replacing = Task { await todo.attach { _ in TodoItem(title: "new", priority: .high, dueInDays: 1) } }
        while todo.isAvailable { try await Task.sleep(for: .milliseconds(1)) }
        #expect(!todo.run("during"), "refused while the old extraction drains")
        await replacing.value
        #expect(todo.isAvailable && todo.item == nil)
        #expect(todo.run("after"))
        await todo.waitForResult()
        #expect(todo.item?.title == "new")
    }

    @Test(.timeLimit(.minutes(1))) func runIsRefusedAfterTheOldExtractionEndsUntilTheReplacementTakesOver() async throws {
        let todo = TodoController()
        await todo.attach { _ in TodoItem(title: "old", priority: .low, dueInDays: nil) }
        #expect(todo.run("first"))
        let gate = Gate()
        todo.drained = { await gate.wait() }
        let replacing = Task { await todo.attach { _ in TodoItem(title: "new", priority: .high, dueInDays: 1) } }
        while gate.waiting < 1 { try await Task.sleep(for: .milliseconds(1)) }
        // The old extraction has ended and its closure is still attached: only the switching guard refuses.
        #expect(!todo.isWorking && todo.item?.title == "old")
        #expect(!todo.run("during"))
        gate.release(0)
        await replacing.value
        #expect(todo.run("after"))
        await todo.waitForResult()
        #expect(todo.item?.title == "new")
    }

    @Test(.timeLimit(.minutes(1))) func anEarlierAttachThatFinishesLaterDoesNotOverwriteALaterOne() async throws {
        let todo = TodoController()
        let gate = Gate()
        todo.drained = { await gate.wait() }
        let first = Task { await todo.attach { _ in TodoItem(title: "first", priority: .low, dueInDays: nil) } }
        while gate.waiting < 1 { try await Task.sleep(for: .milliseconds(1)) }
        let second = Task { await todo.attach { _ in TodoItem(title: "second", priority: .low, dueInDays: nil) } }
        while gate.waiting < 2 { try await Task.sleep(for: .milliseconds(1)) }
        gate.release(1)
        await second.value
        gate.release(0)
        await first.value
        #expect(todo.isAvailable && todo.run("q"))
        await todo.waitForResult()
        #expect(todo.item?.title == "second")
    }
}

/// Holds each caller of `wait()` until the test releases it, in any order.
@MainActor
final class Gate {
    private var waiters: [CheckedContinuation<Void, Never>?] = []
    /// Callers that have arrived, released or not.
    var waiting: Int { waiters.count }

    func wait() async { await withCheckedContinuation { waiters.append($0) } }

    /// Releases the caller that arrived `index`-th (from 0).
    func release(_ index: Int) {
        waiters[index]?.resume()
        waiters[index] = nil
    }
}

/// Copies fixture files from memory: a transport for the controller test, with no network.
struct MemoryTransport: ModelTransport {
    let files: [String: Data]
    func fetch(_ request: TransportRequest, to destination: URL, resume: Bool, progress: @Sendable (UInt64) -> Void) async throws {
        guard let data = files[request.file.path] else { throw URLError(.fileDoesNotExist) }
        try data.write(to: destination)
        progress(UInt64(data.count))
    }
}

@MainActor
@Suite struct ModelControllerTests {
    /// A one-file entry that isn't a language bundle: readiness gets it ready, and the controller refuses to open it as a
    /// model, then deletes it.
    @Test(.timeLimit(.minutes(1))) func readinessDrivesTheScreenFromDownloadToDelete() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("sample-app-core-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let payload = Data("not a model".utf8)
        let sha = SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
        let manifest = try AssetManifest(assetIdentity: "sample-test/1:\(sha)", executionContract: "sample.test/1",
                                         files: [AssetFile(path: "payload.bin", byteCount: UInt64(payload.count), sha256: sha)])
        let entry = try ModelCatalogEntry(manifest: manifest, source: .https(base: URL(string: "https://example.invalid/m/")!))
        let preparation = ModelPreparation(store: try LocalAssetStore(root: root),
                                           transport: MemoryTransport(files: ["payload.bin": payload]))
        let controller = ModelController(entry: entry, preparation: preparation, check: .none, lifecycle: .manual)
        let app = SampleAppState(model: controller)
        var seen: [String] = []
        controller.stateDidChange = { seen.append($0) }
        controller.start()
        defer { controller.stop() }

        try await until { controller.stateName.hasPrefix("notInstalled") }
        #expect(controller.display.actions == [.download])
        await controller.perform(.download)
        try await until { controller.display.title == "Can't open the model" }
        #expect(seen.contains("ready"))
        #expect(controller.model == nil && !app.chat.isAvailable && !app.todo.isAvailable)

        // A second delete while the first runs is ignored rather than failing busy.
        async let first: Void = controller.perform(.delete)
        async let second: Void = controller.perform(.delete)
        _ = await (first, second)
        #expect(controller.actionError == nil && !controller.isDeleting)
        try await until { controller.stateName == "notInstalled(0/\(payload.count))" }
        #expect(controller.display.actions == [.download])
    }

    private func until(_ condition: @MainActor () -> Bool) async throws {
        while !condition() { try await Task.sleep(for: .milliseconds(10)) }
    }
}
