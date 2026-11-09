import AetherCore
import Foundation
import Testing
@testable import AetherRuntime

/// The compile-cache status of the Krea 2 fixture bundle, with Core AI's lookup replaced: which graphs are looked up,
/// with which options, and the source bytes still to compile. The real lookup runs in the gates.
@Suite("Compiled cache status")
struct CompiledCacheTests {
    static let manifestURL = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "AetherCoreTests/Fixtures/krea2/vanilla.json")
    static let mac = Device(platform: .macOS, architecture: "test")

    static func fixture() throws -> (InstalledBundle, Manifest) {
        let manifest = try Manifest.decode(Data(contentsOf: manifestURL))
        let files = manifest.graphs.values.compactMap { $0.file.resolve(architecture: "test") }.flatMap { file in
            [CatalogFile(path: file.path + "/main.mlirb", size: 1_000, sha256: String(repeating: "0", count: 64)),
             CatalogFile(path: file.path + "/weights/0.bin", size: 9_000, sha256: String(repeating: "0", count: 64))]
        } + [CatalogFile(path: "tokenizer.json", size: 5, sha256: String(repeating: "0", count: 64))]
        let ref = ModelRef(
            catalogID: "fixture", repo: "owner/model", revision: "r1", variant: "macos", driver: "krea2@1",
            assets: [.init(name: "main", repo: "owner/model", revision: "r1", subtree: "", files: files)])
        return (InstalledBundle(ref: ref, url: URL(filePath: "/bundles/krea"), isStale: false), manifest)
    }

    @Test func everyGraphIsLookedUpWithItsManifestOptions() throws {
        let (bundle, manifest) = try Self.fixture()
        var looked: [String: Bool] = [:]
        let status = try CompiledCache.status(of: bundle, manifest: manifest, device: Self.mac, roles: nil, cache: .default) {
            url, options in
            looked[url.lastPathComponent] = options.expectFrequentReshapes
            return url.lastPathComponent.contains("denoiser")
        }
        #expect(status.entries.map(\.role)
            == ["denoiser0", "denoiser1", "denoiser2", "denoiser3", "textEncoder", "vaeDecoder", "vaeEncoder"])
        #expect(looked["krea2_turbo-text.aimodel"] == true && looked["krea2_turbo-denoiser0.aimodel"] == false)
        #expect(status.entries.allSatisfy { $0.sourceBytes == 10_000 })
        #expect(!status.isComplete && status.uncompiledSourceBytes == 30_000)
    }

    @Test func aCompleteCacheNeedsNothingAndRolesNarrowTheCheck() throws {
        let (bundle, manifest) = try Self.fixture()
        let all = try CompiledCache.status(of: bundle, manifest: manifest, device: Self.mac, roles: nil, cache: .default) { _, _ in true }
        #expect(all.isComplete && all.uncompiledSourceBytes == 0)
        let text = try CompiledCache.status(
            of: bundle, manifest: manifest, device: Self.mac, roles: ["textEncoder"], cache: .default) { _, _ in false }
        #expect(text.entries.map(\.role) == ["textEncoder"] && text.uncompiledSourceBytes == 10_000)
    }

    @Test func sourceBytesCountOnlyFilesUnderTheGraph() {
        let ref = ModelRef(
            catalogID: "x", repo: "o/m", revision: "r", variant: "macos", driver: "krea2@1",
            assets: [.init(name: "main", repo: "o/m", revision: "r", subtree: "", files: [
                CatalogFile(path: "a.aimodel/x", size: 3, sha256: ""),
                CatalogFile(path: "a.aimodelz/x", size: 100, sha256: ""),
                CatalogFile(path: "a.aimodel", size: 0, sha256: ""),
            ])])
        #expect(CompiledCache.sourceBytes(of: AssetPath(asset: "main", path: "a.aimodel"), in: ref) == 3)
        #expect(CompiledCache.sourceBytes(of: AssetPath(asset: "other", path: "a.aimodel"), in: ref) == 0)
    }
}
