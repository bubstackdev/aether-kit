import Foundation
import Testing
@testable import AetherRuntime

enum EchoDriver: ModelDriver {
    static let id: DriverID = "decoder-lm@1"
    static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> String {
        "echo:\(bundle.catalogID)@\(options.device.architecture)"
    }
}

enum DetectorStub: ModelDriver {
    static let id: DriverID = "detector-dense@1"
    static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> Int { 2 }
}

@Suite("Registry")
struct RegistryTests {
    static let options = LoadOptions(device: Device(platform: .macOS, architecture: "fake9"), cachePolicy: .default, cache: .default)

    static func bundle(in root: URL, driver: DriverID) throws -> InstalledBundle {
        var manifest = try Fixtures.manifest("qwen3-0.6b")
        manifest.driver = driver
        try manifest.write(toBundle: root)
        let ref = try ModelRef.resolve("qwen3-0.6b", for: Device(platform: .macOS, architecture: "h17c"))
        return InstalledBundle(ref: ref, url: root, isStale: false)
    }

    @Test func unsupportedDriverIsReportedBeforeAnyDownload() throws {
        let registry = Registry([DetectorStub.self])
        let ref = try ModelRef.resolve("qwen3-0.6b", for: Device(platform: .macOS, architecture: "h17c"))
        #expect {
            try registry.require(ref)
        } throws: { error in
            if case AetherError.unsupported(let driver) = error { return driver == .decoderLM }
            return false
        }
        // The iOS variant of the same model uses another driver, which nobody implements yet.
        let ios = try ModelRef.resolve("qwen3-0.6b", for: Device(platform: .iOS, architecture: "h18p"))
        #expect(ios.driver.description == "static-chunked-lm@1")
        #expect(throws: AetherError.self) { try Registry([EchoDriver.self]).require(ios) }
        #expect(Registry.empty.ids.isEmpty)
    }

    @Test func unknownIdsAreNotUnsupported() throws {
        #expect {
            _ = try Registry.empty.driver(for: "telepathy@1")
        } throws: { error in
            if case AetherError.unknownDriver(let id) = error { return id.description == "telepathy@1" }
            return false
        }
        #expect {
            _ = try Registry.empty.driver(for: "decoder-lm@2")
        } throws: { error in
            if case AetherError.unknownDriver = error { return true }
            return false
        }
    }

    @Test func loadsThroughTheManifestDriver() async throws {
        try await withTemporaryDirectory { root in
            let bundle = try Self.bundle(in: root, driver: .decoderLM)
            let registry = Registry().merging(Registry([EchoDriver.self]))
            #expect(try await registry.load(bundle, options: Self.options) as? String == "echo:qwen3-0.6b@fake9")
            #expect(try await registry.load(bundle, as: EchoDriver.self, options: Self.options) == "echo:qwen3-0.6b@fake9")
            await #expect(throws: AetherError.self) {
                _ = try await registry.load(bundle, as: DetectorStub.self, options: Self.options)
            }
            let both = registry.registering(DetectorStub.self)
            #expect(both.supports(DetectorStub.id))
        }
    }
}
