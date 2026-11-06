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

    /// Second review, item 15: the foreground download stream buffered without bound. The data task is suspended
    /// past the high-water mark of unread bytes and resumed below the low-water mark; nothing is dropped.
    @Test func foregroundDownloadsApplyBackpressure() {
        let task = URLSession(configuration: .ephemeral).dataTask(with: URL(string: "https://example.invalid/")!)
        let flow = FlowControl()
        flow.task = task
        flow.received(FlowControl.highWater, by: task)
        #expect(!flow.isSuspended)
        flow.received(1, by: task)
        #expect(flow.isSuspended)
        flow.consumed(FlowControl.highWater - FlowControl.lowWater)
        #expect(flow.isSuspended, "still above the low-water mark")
        flow.consumed(2)
        #expect(!flow.isSuspended && flow.bufferedBytes == FlowControl.lowWater - 1)
        task.cancel()
    }
}
