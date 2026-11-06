import Foundation
import Synchronization
import Testing
@testable import AetherRuntime

@Suite("Bundle manifest provisioning")
struct ManifestProvisioningTests {
    /// Lays out an entry's macOS install with its small files (graphs are empty stand-ins: no weights).
    static func makeInstall(_ id: String, at root: URL) throws -> InstalledBundle {
        let ref = try ModelRef.resolve(id, for: Device(platform: .macOS, architecture: "h17c"))
        let bundle = InstalledBundle(ref: ref, url: root, isStale: false)
        let main = try #require(bundle.url(ofAsset: "main"))
        let fm = FileManager.default
        for file in try #require(ref.asset("main")).files {
            let url = main.appending(path: file.path)
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
        return bundle
    }

    @Test(arguments: Fixtures.goldenIDs)
    func synthesizesOnceIntoTheAetherDirectory(_ id: String) async throws {
        try await withTemporaryDirectory { root in
            let bundle = try Self.makeInstall(id, at: root)
            let calls = Mutex<[String]>([])
            let signatures = try Fixtures.signatures(id)
            let provider: BundleManifest.SignatureProvider = { url, function in
                calls.withLock { $0.append("\(url.lastPathComponent):\(function)") }
                return try #require(signatures[function])
            }
            let manifest = try await BundleManifest.loadOrSynthesize(bundle, signature: provider)
            #expect(manifest == (try Fixtures.manifest(id)))
            #expect(try Data(contentsOf: root.appending(path: ".aether/aether.json")) == Fixtures.data(id, "aether.json"))
            #expect(calls.withLock { $0.count } == signatures.count)

            // From now on the sidecar is the contract; the rules do not run again.
            let again = try await BundleManifest.loadOrSynthesize(bundle, signature: provider)
            #expect(again == manifest)
            #expect(calls.withLock { $0.count } == signatures.count)
        }
    }

    @Test func anInstallWithNeitherDocumentIsInvalid() async throws {
        try await withTemporaryDirectory { root in
            let ref = try ModelRef.resolve("qwen3-0.6b", for: Device(platform: .macOS, architecture: "h17c"))
            await #expect {
                _ = try await BundleManifest.loadOrSynthesize(InstalledBundle(ref: ref, url: root, isStale: false)) { _, _ in
                    Issue.record("must not inspect a graph")
                    return try #require(try Fixtures.signatures("qwen3-0.6b")["main"])
                }
            } throws: { field(of: $0) == "aether.json" }
        }
    }
}
