import AetherKit
import Darwin
import Foundation
import Testing
@testable import AetherProbe

// Private lease directories only; never the shared /tmp/atelier-gpu.lock.

@Test func wellFormedGPUCommandsAreClassifiedByCommandAndMode() {
    for gpu in [["h3-profile-run", "p", "d", "-", "o", "--render"], ["h3-profile-decode", "p", "d", "r", "o"],
                ["h3-profile-decode", "p", "d", "r", "o", "m", "h"], ["inspect", "m"], ["smoke", "m", "preferGPU"],
                ["render-krea", "b", "o.png", "prompt", "512", "512", "7"], ["render-krea-batch", "b", "o", "prompt", "512", "512", "1,2"],
                ["parity-krea", "b", "l", "f", "s", "o", "encoded"], ["qualify-h3", "b", "p", "o", "--run"],
                ["qualify-h3-lifecycle", "b", "p", "o", "--run"], ["qualify-h3-phases", "b", "p", "c", "o", "--run"],
                ["qualify-h3-chain", "b", "p", "c", "o", "--capture"],
                ["qualify-h3-denoising", "1", "2", "3", "4", "5", "6", "o", "--lifecycle"], ["benchmark-krea", "b", "p", "s", "o", "--run"]] {
        #expect(ProbeGPULease.usesGPU(gpu), "\(gpu)")
    }
    for cpu in [[], ["--help"], ["doctor"], ["verify-krea", "b"], ["prepare-krea", "b", "p"], ["h3-profile-dump", "p"],
                ["h3-weights-bench", "p", "1"], ["qualify-h3", "b", "p", "o", "--prepare-only"],
                ["benchmark-krea", "b", "p", "s", "o", "--prepare-only"], ["qualify-h3-denoising", "1", "2", "3", "4", "5", "6", "o", "--prepare-only"]] {
        #expect(!ProbeGPULease.usesGPU(cpu), "\(cpu)")
    }
}

@Test func malformedGPUCommandsTakeNoLeaseSoTheirUsageErrorIsImmediate() {
    for typo in [["inspect"], ["smoke", "m", "fastest"], ["render-krea", "b", "o.png", "prompt", "512", "wide", "7"],
                 ["render-krea", "b"], ["render-krea-batch", "b", "o", " ", "512", "512", "1"], ["parity-krea", "b", "l", "f", "s", "o", "maybe"],
                 ["h3-profile-run", "p", "d", "-", "o", "--draw"], ["h3-profile-decode", "p", "d"], ["qualify-h3", "b", "p", "--run"],
                 ["qualify-h3-phases", "b", "p", "c", "o", "--unknown"], ["qualify-h3-chain", "b", "p", "c", "o", "--run", "extra"],
                 ["qualify-h3-denoising", "1", "2", "3", "4", "5", "-", "o", "--run"], ["benchmark-krea", "b", "p", "s", "--run"]] {
        #expect(!ProbeGPULease.usesGPU(typo), "\(typo)")
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
    let lease = try #require(try await ProbeGPULease.acquire(for: ["render-krea", "b", "o.png", "p", "512", "512", "7"], environment: environment))
    #expect(lease.holding == .owned && SharedGPULease.holder(of: directory) == getpid())
    // While held, a fail-fast probe refuses; disabled probes take nothing.
    await #expect(throws: SharedGPULeaseError.timedOut(holder: getpid())) {
        try await ProbeGPULease.acquire(for: ["inspect", "m"], environment: environment.merging(["AETHER_GPU_LEASE_TIMEOUT": "0"]) { $1 })
    }
    #expect(try await ProbeGPULease.acquire(for: ["inspect", "m"], environment: environment.merging(["AETHER_GPU_LEASE": "off"]) { $1 }) == nil)
    lease.release()
    #expect(!FileManager.default.fileExists(atPath: directory.path))
}
