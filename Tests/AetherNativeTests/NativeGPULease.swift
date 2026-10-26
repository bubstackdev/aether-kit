import AetherKit
import Foundation

// The opt-in native suites take the shared GPU lease themselves, like aether-probe: default on, with the same
// AETHER_GPU_LEASE / AETHER_GPU_LEASE_TIMEOUT / AETHER_GPU_LEASE_PATH overrides, and a lease held by an ancestor
// (a wrapper running `swift test`) counts as held. One GPU admission for this test process also keeps enabled native
// tests from running on the GPU at the same time. It is local to this test target, not a library default.
private let nativeGPU: Result<GPUAdmission?, any Error> = Result {
    let policy = try SharedGPULease.ToolPolicy.environment(nativeEnvironment)
    return try policy.configuration.map { try GPUAdmission(waiting: .queue(timeout: policy.timeout), crossProcessLease: $0) }
}

/// Hold the returned grant for the whole native test.
func nativeGPUGrant(_ test: String = #function) async throws -> GPUAdmissionGrant? {
    try await nativeGPU.get()?.admit(owner: test)
}
