import Foundation
import Testing
@testable import AetherCore
@_spi(Testing) import AetherHost

/// Second code review, batch C item 1: a pipeline over a bundle from a `ModelStore` holds the bundle's reader lease from
/// creation until `unload()` or deinit, so the store refuses to remove files the pipeline reads. (Before: only callers
/// that took `ModelStore.lease(_:)` themselves were protected.)
@Suite("Bundle reader leases")
struct BundleLeaseTests {
    static let ref = ModelRef(
        catalogID: "depth-test", repo: "owner/depth", revision: "r1", variant: "macos", driver: "depth-estimator@1",
        assets: [.init(name: "main", repo: "owner/depth", revision: "r1", subtree: "", files: [])])

    /// A temporary store root holding `ref` installed (its receipt written as the installer does).
    static func withInstalled(_ body: (ModelStore, InstalledBundle) async throws -> Void) async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "aether-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(root: root)
        let url = store.bundleURL(for: ref)
        let receipt = InstallReceipt.url(in: url)
        try FileManager.default.createDirectory(at: receipt.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(InstallReceipt(ref: ref, installedAt: Date())).write(to: receipt)
        let bundle = try #require(await store.installed(ref))
        try await body(store, bundle)
    }

    static func pipeline(_ bundle: InstalledBundle) throws -> StatelessPipeline {
        try StatelessPipeline(
            manifest: try StatelessPipelineTests.manifest(), device: Device(platform: .macOS, architecture: "test"),
            contract: StatelessPipelineTests.contract,
            options: PipelineOptions(memory: .bytes(1 << 40), gpu: .unadmitted, probe: GPUMemoryProbe { 0 }),
            bundle: bundle, load: { _ throws(AetherError) in StatelessPipelineTests.FakeEstimator(log: .init(), gate: nil) },
            fileBytes: { _ in 100 })
    }

    static func isInUse(_ error: any Error) -> Bool {
        if case AetherError.inUse = error { return true }
        return false
    }

    @Test func removalWaitsForUnload() async throws {
        try await Self.withInstalled { store, bundle in
            let pipeline = try Self.pipeline(bundle)
            await #expect { try await store.remove(Self.ref) } throws: { Self.isInUse($0) }
            // A request keeps it, and so does a failed one: only unload or deinit lets go.
            _ = try await pipeline.perform(job: nil) { _ in 1 }
            await #expect { try await store.remove(Self.ref) } throws: { Self.isInUse($0) }
            await pipeline.unload()
            try await store.remove(Self.ref)
            #expect(await store.installed(Self.ref) == nil)
            // The next request takes the lease again; the bundle is gone, so it refuses before loading anything.
            await #expect { try await pipeline.perform(job: nil) { _ in 1 } } throws: { error in
                if case AetherError.unknownModel = error { return true }
                return false
            }
        }
    }

    @Test func deinitReleasesTheLease() async throws {
        try await Self.withInstalled { store, bundle in
            var pipeline: StatelessPipeline? = try Self.pipeline(bundle)
            await #expect { try await store.remove(Self.ref) } throws: { Self.isInUse($0) }
            withExtendedLifetime(pipeline) {}
            pipeline = nil
            try await store.remove(Self.ref)
        }
    }

    @Test func unloadedThenUsedAgainLeasesAgain() async throws {
        try await Self.withInstalled { store, bundle in
            let pipeline = try Self.pipeline(bundle)
            await pipeline.unload()
            _ = try await pipeline.perform(job: nil) { _ in 1 }
            await #expect { try await store.remove(Self.ref) } throws: { Self.isInUse($0) }
        }
    }

    @Test func aBundleOutsideAStoreHasNoLease() throws {
        let bundle = InstalledBundle(ref: Self.ref, url: URL(filePath: "/nonexistent"), isStale: false)
        #expect(try bundle.readLease() == nil)
        _ = try Self.pipeline(bundle)
    }
}
