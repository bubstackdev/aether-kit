import AetherKit
import Darwin
import Foundation
import Testing
@testable import AetherProbe

// Private lease directories only; never the shared /tmp/atelier-gpu.lock.

@Test func gpuCommandsAreClassifiedByCommandAndMode() {
    for gpu in [["h3-profile-run", "p", "d", "-", "o", "--render"], ["h3-profile-decode", "p", "d", "r", "o"], ["inspect", "m"],
                ["smoke", "m", "preferGPU"], ["render-krea", "b"], ["render-krea-batch", "b"], ["parity-krea", "b"],
                ["qualify-h3", "b", "p", "o", "--run"], ["qualify-h3-chain", "b", "p", "c", "o", "--capture"],
                ["qualify-h3-denoising", "1", "2", "3", "4", "5", "6", "o", "--lifecycle"], ["benchmark-krea", "b", "p", "s", "o", "--run"]] {
        #expect(ProbeGPULease.usesGPU(gpu), "\(gpu)")
    }
    for cpu in [[], ["--help"], ["doctor"], ["verify-krea", "b"], ["prepare-krea", "b", "p"], ["h3-profile-dump", "p"],
                ["h3-weights-bench", "p", "1"], ["qualify-h3", "b", "p", "o", "--prepare-only"],
                ["benchmark-krea", "b", "p", "s", "o", "--prepare-only"], ["qualify-h3-phases", "b", "p", "c", "o", "--unknown"]] {
        #expect(!ProbeGPULease.usesGPU(cpu), "\(cpu)")
    }
}

@Test func probeTakesTheLeaseForGPUCommandsOnlyAndHonorsTheOverrides() async throws {
    let parent = FileManager.default.temporaryDirectory.appendingPathComponent("aether-probe-lease-\(UUID())", isDirectory: true)
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: parent) }
    let directory = parent.appendingPathComponent("gpu.lock")
    let environment = ["AETHER_GPU_LEASE_PATH": directory.path]
    #expect(try await ProbeGPULease.acquire(for: ["doctor"], environment: environment) == nil)
    #expect(!FileManager.default.fileExists(atPath: directory.path))
    let lease = try #require(try await ProbeGPULease.acquire(for: ["render-krea"], environment: environment))
    #expect(lease.holding == .owned && SharedGPULease.holder(of: directory) == getpid())
    // While held, a fail-fast probe refuses; disabled probes take nothing.
    await #expect(throws: SharedGPULeaseError.timedOut(holder: getpid())) {
        try await ProbeGPULease.acquire(for: ["inspect", "m"], environment: environment.merging(["AETHER_GPU_LEASE_TIMEOUT": "0"]) { $1 })
    }
    #expect(try await ProbeGPULease.acquire(for: ["inspect", "m"], environment: environment.merging(["AETHER_GPU_LEASE": "off"]) { $1 }) == nil)
    lease.release()
    #expect(!FileManager.default.fileExists(atPath: directory.path))
}
