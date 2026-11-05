import Foundation
import Testing
@testable import AetherRuntime

enum EchoDriver: ModelDriver {
    static let id = DriverID(validated: "echo@1")
    static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> String {
        "echo:\(bundle.catalogID)@\(options.device.architecture)"
    }
}

enum EchoDriverV2: ModelDriver {
    static let id = DriverID(validated: "echo@2")
    static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> Int { 2 }
}

@Suite("Registry")
struct RegistryTests {
    static let options = LoadOptions(device: Device(platform: .macOS, architecture: "fake9"), cachePolicy: .default, cache: .default)

    static func bundle(in root: URL, driver: String) throws -> InstalledBundle {
        var manifest = try Fixtures.manifest("qwen3-0.6b")
        manifest.driver = DriverID(validated: driver)
        try manifest.write(toBundle: root)
        let ref = ModelRef(catalogID: "qwen3-0.6b", repo: "o/r", revision: "r", subtree: "s", variant: "macos")
        return InstalledBundle(ref: ref, url: root, isStale: false)
    }

    @Test func unsupportedDriverIsReportedBeforeAnyDownload() throws {
        let registry = Registry([EchoDriver.self])
        let entry = try Catalog.builtin.entry("qwen3-0.6b")
        #expect {
            try registry.require(entry)
        } throws: { error in
            if case AetherError.unsupported(let driver) = error { return driver == .decoderLM }
            return false
        }
        #expect(Registry.empty.ids.isEmpty)
        #expect(registry.supports(EchoDriver.id) && !registry.supports(EchoDriverV2.id))
    }

    @Test func versionsMatchExactly() async throws {
        try await withTemporaryDirectory { root in
            let bundle = try Self.bundle(in: root, driver: "echo@2")
            let registry = Registry([EchoDriver.self])
            await #expect {
                _ = try await registry.load(bundle, options: Self.options)
            } throws: { error in
                if case AetherError.unsupported(let driver) = error { return driver.description == "echo@2" }
                return false
            }
            let both = registry.registering(EchoDriverV2.self)
            #expect(try await both.load(bundle, options: Self.options) as? Int == 2)
        }
    }

    @Test func loadsThroughTheManifestDriver() async throws {
        try await withTemporaryDirectory { root in
            let bundle = try Self.bundle(in: root, driver: "echo@1")
            let registry = Registry().merging(Registry([EchoDriver.self]))
            #expect(try await registry.load(bundle, options: Self.options) as? String == "echo:qwen3-0.6b@fake9")
            #expect(try await registry.load(bundle, as: EchoDriver.self, options: Self.options) == "echo:qwen3-0.6b@fake9")
            await #expect(throws: AetherError.self) {
                _ = try await registry.load(bundle, as: EchoDriverV2.self, options: Self.options)
            }
        }
    }
}
