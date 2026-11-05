import Foundation
import Synchronization
import Testing
@testable import AetherRuntime

@Suite("Bundle manifest provisioning")
struct ManifestProvisioningTests {
    /// Lays out a zoo bundle's small files (the graph directory is a stand-in: no weights).
    static func makeZooBundle(_ id: String, at root: URL) throws -> InstalledBundle {
        let entry = try Catalog.builtin.entry(id)
        let variant = try #require(entry.variants["macos"])
        let fm = FileManager.default
        for file in variant.files {
            let url = root.appending(path: file.path)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let name = url.lastPathComponent
            let source: Data? = switch (file.path, name) {
            case ("metadata.json", _): try Fixtures.data(id, "metadata.json")
            case (_, "tokenizer_config.json"): try Fixtures.data(id, "tokenizer_config.json")
            case (_, "chat_template.jinja"): try Fixtures.data(id, "chat_template.jinja")
            default: nil
            }
            try (source ?? Data()).write(to: url)
        }
        let ref = ModelRef(
            catalogID: id, repo: entry.repo, revision: entry.revision, subtree: variant.subtree, variant: "macos")
        return InstalledBundle(ref: ref, url: root, isStale: false)
    }

    @Test(arguments: ["qwen3-0.6b", "qwen3.5-0.8b"])
    func synthesizesOnceAndWritesTheSidecar(_ id: String) async throws {
        try await withTemporaryDirectory { root in
            let bundle = try Self.makeZooBundle(id, at: root)
            let calls = Mutex<[String]>([])
            let signature = try Fixtures.signature(id)
            let provider: BundleManifest.SignatureProvider = { url, function in
                calls.withLock { $0.append("\(url.lastPathComponent):\(function)") }
                return signature
            }
            let manifest = try await BundleManifest.loadOrSynthesize(bundle, signature: provider)
            #expect(manifest == (try Fixtures.manifest(id)))
            #expect(try Data(contentsOf: root.appending(path: "aether.json")) == Fixtures.data(id, "aether.json"))
            #expect(calls.withLock { $0.count } == 1)

            // From now on the sidecar is the contract; the heuristics do not run again.
            let again = try await BundleManifest.loadOrSynthesize(bundle, signature: provider)
            #expect(again == manifest)
            #expect(calls.withLock { $0.count } == 1)
        }
    }

    @Test func aBundleWithNeitherDocumentIsInvalid() async throws {
        try await withTemporaryDirectory { root in
            let ref = ModelRef(catalogID: "x", repo: "o/r", revision: "r", subtree: "s", variant: "macos")
            await #expect {
                _ = try await BundleManifest.loadOrSynthesize(InstalledBundle(ref: ref, url: root, isStale: false)) { _, _ in
                    Issue.record("must not inspect a graph")
                    return try Fixtures.signature("qwen3-0.6b")
                }
            } throws: { field(of: $0) == "aether.json" }
        }
    }
}
