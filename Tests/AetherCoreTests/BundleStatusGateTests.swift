import Foundation
import Synchronization
import Testing
@testable import AetherCore

/// The bundle status on a real local export: the MiniMax H3 bundle (`aether/forge-outputs/minimax-h3/bundle`, or
/// AETHER_H3_EXPORT; not publicly hosted), about 94 GB in 1,375 declared files. Read-only, no GPU. Off by default:
/// `AETHER_GATES=1 swift test --filter BundleStatusGateTests`. The cheap status must be ready and stat every declared
/// file well under a second. The full verified status (every byte hashed) runs only with AETHER_GATES_FULL_VERIFY=1.
@Suite("Bundle status gate", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct BundleStatusGateTests {
    static let export: URL = {
        if let path = ProcessInfo.processInfo.environment["AETHER_H3_EXPORT"] { return URL(filePath: path) }
        var directory = URL(filePath: #filePath).deletingLastPathComponent()
        while directory.path != "/" {
            let candidate = directory.appending(path: "forge-outputs/minimax-h3")
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            directory = directory.deletingLastPathComponent()
        }
        return URL(filePath: "forge-outputs/minimax-h3")
    }()

    static var bundle: URL { export.appending(path: "bundle") }

    @Test func theLocalExportIsReadyByTheCheapStatus() throws {
        let manifest = try Manifest.load(fromBundle: Self.bundle)
        let declared = manifest.assets.values.flatMap { $0.files ?? [] }
        let clock = ContinuousClock()
        var status = BundleStatus.notPrepared
        let elapsed = clock.measure {
            status = InstalledBundle.status(ofLocalBundleAt: Self.bundle, supportedDrivers: [manifest.driver])
        }
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let bytes = declared.reduce(Int64(0)) { $0 + $1.bytes }
        print("bundle-status: \(status); \(declared.count) declared files, \(bytes) bytes; \(String(format: "%.3f", seconds)) s")
        #expect(status == .ready)
        #expect(seconds < 1)

        let ref = ModelRef(
            catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: manifest.variant.platform.rawValue,
            driver: manifest.driver, assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
        #expect(try InstalledBundle(ref: ref, url: Self.bundle, isStale: false).status() == .ready)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES_FULL_VERIFY"] == "1"))
    func theLocalExportVerifiesInFull() throws {
        let clock = ContinuousClock()
        let last = Mutex<BundleVerificationProgress?>(nil)
        var status = BundleStatus.notPrepared
        let elapsed = try clock.measure {
            status = try InstalledBundle.verifiedStatus(ofLocalBundleAt: Self.bundle) { p in last.withLock { $0 = p } }
        }
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let read = last.withLock { $0?.completedBytes ?? 0 }
        print("bundle-verified-status: \(status); \(read) bytes hashed in \(String(format: "%.1f", seconds)) s")
        #expect(status == .ready)
    }
}
