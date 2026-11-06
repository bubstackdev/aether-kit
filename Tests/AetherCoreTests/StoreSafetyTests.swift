import Foundation
import Testing
@testable import AetherCore

/// Second pre-release review, items 1, 8 and 9: references and manifests can't reach outside the store or a bundle,
/// and an install is reused only for the same installation.
@Suite("Store and manifest safety")
struct StoreSafetyTests {
    static func ref(repo: String = "owner/repo", revision: String = "rev", variant: String = "macos") -> ModelRef {
        ModelRef(catalogID: "test", repo: repo, revision: revision, variant: variant, driver: .decoderLM, assets: [])
    }

    /// Item 1: `remove` with repo `../outside` deleted a sibling of the store root.
    @Test func referencesThatEscapeTheStoreAreRefusedBeforeAnyFileOperation() async throws {
        try await withTemporaryDirectory { base in
            let fm = FileManager.default
            let root = base.appending(path: "models")
            let victim = base.appending(path: "outside/revision")
            try fm.createDirectory(at: root, withIntermediateDirectories: true)
            try fm.createDirectory(at: victim, withIntermediateDirectories: true)
            try Data("keep".utf8).write(to: victim.appending(path: "sentinel"))
            let store = ModelStore(root: root)
            let escaping = Self.ref(repo: "../outside", revision: "revision")
            await #expect(throws: AetherError.self) { try await store.remove(escaping) }
            #expect(fm.fileExists(atPath: victim.appending(path: "sentinel").path))
            #expect(await store.installed(escaping) == nil)
            await #expect(throws: AetherError.self) { _ = try await store.install(escaping) }
        }
        for bad in [Self.ref(repo: "owner"), Self.ref(repo: "owner/name/extra"), Self.ref(repo: "/abs/x"),
                    Self.ref(repo: "owner/.."), Self.ref(revision: ".."), Self.ref(revision: "a/b"), Self.ref(variant: "")] {
            #expect(throws: AetherError.self) { try bad.validateForStore() }
        }
        var asset = Self.ref()
        asset.assets = [.init(name: "main", repo: "owner/repo", revision: "rev", subtree: "../up", files: [])]
        #expect(throws: AetherError.self) { try asset.validateForStore() }
        asset.assets = [.init(name: "main", repo: "owner/repo", revision: "rev", subtree: "",
                              files: [CatalogFile(path: "a/../../b", size: 1, sha256: String(repeating: "0", count: 64))])]
        #expect(throws: AetherError.self) { try asset.validateForStore() }
        // Every catalog reference passes.
        for entry in Catalog.builtin.entries {
            for key in entry.variants.keys {
                let device = Device(platform: key.hasPrefix("ios") ? .iOS : .macOS, architecture: "h18p")
                if let ref = try? ModelRef.resolve(entry, for: device) { try ref.validateForStore() }
            }
        }
    }

    /// Item 8: an install was reused when only repo, revision and variant matched (here a different catalog id and
    /// driver at the same location).
    @Test func anInstallIsReusedOnlyForTheSameInstallation() async throws {
        try await withTemporaryDirectory { root in
            let store = ModelStore(root: root)
            let ref = Self.ref()
            let url = store.bundleURL(for: ref)
            try FileManager.default.createDirectory(at: InstallReceipt.url(in: url).deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(InstallReceipt(ref: ref, installedAt: .now)).write(to: InstallReceipt.url(in: url))
            #expect(await store.installed(ref)?.ref == ref)
            var other = ref
            other.catalogID = "new"
            other.driver = .kokoro
            #expect(await store.installed(other) == nil)
        }
    }

    /// Item 9: tokenizer paths weren't containment-checked, and stop ids above Int32.max passed and trapped later.
    @Test func tokenizerPathsAndStopIDsAreValidated() throws {
        let base = try Manifest.decode(Fixtures.data("sd-turbo/aether.json"))
        var source = base
        source.tokenizer?.source = .dir(AssetPath(asset: "main", path: "../../outside"))
        #expect(throws: AetherError.self) { try source.validate() }
        var template = base
        template.tokenizer?.chatTemplate = AssetPath(asset: "main", path: "../../template")
        #expect(throws: AetherError.self) { try template.validate() }
        var stop = base
        stop.tokenizer?.stop = [.id(Int(Int32.max) + 1)]
        #expect(throws: AetherError.self) { try stop.validate() }
        stop.tokenizer?.stop = [.id(Int(Int32.max))]
        try stop.validate()
    }

    /// Third review, P2-10: a reference with a negative size or a malformed hash passed validation, and sizes whose
    /// total overflows trapped in `bytes`.
    @Test func fileSizesAndHashesAreValidated() throws {
        let hash = String(repeating: "ab", count: 32)
        func ref(_ files: [CatalogFile]) -> ModelRef {
            ModelRef(catalogID: "test", repo: "owner/repo", revision: "rev", variant: "macos", driver: .decoderLM,
                     assets: [.init(name: "main", repo: "owner/repo", revision: "rev", subtree: "", files: files)])
        }
        try ref([CatalogFile(path: "a", size: 0, sha256: hash)]).validateForStore()
        for bad in [CatalogFile(path: "a", size: -1, sha256: hash), CatalogFile(path: "a", size: 1, sha256: ""),
                    CatalogFile(path: "a", size: 1, sha256: hash.uppercased()), CatalogFile(path: "a", size: 1, sha256: String(hash.dropLast())),
                    CatalogFile(path: "a", size: 1, sha256: String(hash.dropLast()) + "g")] {
            #expect(throws: AetherError.self, "\(bad)") { try ref([bad]).validateForStore() }
        }
        let huge = ref([CatalogFile(path: "a", size: .max, sha256: hash), CatalogFile(path: "b", size: 1, sha256: hash)])
        #expect(huge.bytes == .max, "saturates instead of trapping")
        #expect(throws: AetherError.self) { try huge.validateForStore() }
        #expect(CatalogAsset(subtree: "", files: huge.assets[0].files).bytes == .max)
    }

    /// Second review, item 15: the foreground download stream buffered without bound. The data task is suspended
    /// past the high-water mark of unread bytes and resumed below the low-water mark; nothing is dropped.
    @Test func foregroundDownloadsApplyBackpressure() {
        let task = URLSession(configuration: .ephemeral).dataTask(with: URL(string: "https://example.invalid/")!)
        let flow = FlowControl()
        flow.task = task
        flow.received(FlowControl.highWater)
        #expect(!flow.isSuspended)
        flow.received(1)
        #expect(flow.isSuspended)
        flow.consumed(FlowControl.highWater - FlowControl.lowWater)
        #expect(flow.isSuspended, "still above the low-water mark")
        flow.consumed(2)
        #expect(!flow.isSuspended && flow.bufferedBytes == FlowControl.lowWater - 1)
        task.cancel()
    }

    /// A `get` from an already-cancelled task (a download cancelled while it waited to start) cancels its data task
    /// before resuming it, and URLSession then completes the task on its delegate queue without waiting for the
    /// resume. On a loaded machine that completion could land before the response continuation was installed and was
    /// dropped, so `get` never returned (a cancelled install, and every later install of its ref, hung). Each attempt
    /// must throw; the time limit turns a regression into a failure rather than a hang. The window is narrow, so a
    /// regression shows only now and then, under load.
    @Test(.timeLimit(.minutes(1))) func aGetFromACancelledTaskThrowsRatherThanHanging() async throws {
        let site = StubSite()
        site.serve("/f.bin", Data(count: 16))
        let transport = ModelStoreTests.transport()
        for _ in 0..<200 {
            let attempt = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                _ = try await transport.get(site.endpoint.appending(path: "f.bin"), headers: [:])
            }
            await #expect(throws: (any Error).self) { try await attempt.value }
        }
    }

    /// Third review, P1-1: the suspend ran outside the lock that decided it, so a consumer could drain the buffer and
    /// resume the task before the suspend landed, leaving it stalled with nothing buffered and the bookkeeping saying
    /// it ran. The fake task gives the consumer a window inside `suspend()`: with the call serialized the consumer
    /// waits it out; without, its resume lands first and the suspend wins.
    @Test func backpressureCannotLeaveTheTaskSuspendedAfterTheConsumerDrains() {
        let task = RacingTask()
        let flow = FlowControl()
        flow.task = task
        task.duringSuspend = {
            let drained = DispatchSemaphore(value: 0)
            Thread.detachNewThread {
                flow.consumed(FlowControl.highWater + 1)
                drained.signal()
            }
            // Long enough for an unserialized consumer to run; a serialized one is blocked until this returns.
            _ = drained.wait(timeout: .now() + .milliseconds(200))
        }
        flow.received(FlowControl.highWater + 1)
        // Let the consumer finish if it was blocked.
        let deadline = Date().addingTimeInterval(5)
        while flow.bufferedBytes != 0, Date() < deadline { usleep(1_000) }
        #expect(flow.bufferedBytes == 0)
        #expect(!flow.isSuspended)
        #expect(task.isSuspended == flow.isSuspended, "the task's actual state must match the bookkeeping")
        #expect(task.log == ["suspend", "resume"])
    }
}

/// A data task whose `suspend()` runs a hook in the middle, standing in for the scheduler interleaving the consumer.
private final class RacingTask: FlowControlledTask, @unchecked Sendable {
    private let lock = NSLock()
    private var _suspended = false
    private var _log: [String] = []
    var duringSuspend: (() -> Void)?

    var isSuspended: Bool { lock.withLock { _suspended } }
    var log: [String] { lock.withLock { _log } }

    func suspend() {
        let hook = duringSuspend
        duringSuspend = nil
        hook?()
        lock.withLock { _suspended = true; _log.append("suspend") }
    }

    func resume() {
        lock.withLock { _suspended = false; _log.append("resume") }
    }
}
