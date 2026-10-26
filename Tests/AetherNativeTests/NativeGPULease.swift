import AetherKit
import Foundation

// The opt-in native suites take the shared GPU lease themselves, like aether-probe: default on, with the same
// AETHER_GPU_LEASE / AETHER_GPU_LEASE_TIMEOUT / AETHER_GPU_LEASE_PATH / AETHER_GPU_LEASE_HOLDER overrides, so a wrapper
// that runs `swift test` as its GPU job hands its lease over explicitly. One GPU admission for this test process also
// keeps enabled native tests from running on the GPU at the same time. It is local to this test target, not a library
// default.
private let nativeGPU: Result<GPUAdmission?, any Error> = Result {
    let policy = try SharedGPULease.ToolPolicy.environment(nativeEnvironment)
    return try policy.configuration.map { try GPUAdmission(waiting: .queue(timeout: policy.timeout), crossProcessLease: $0) }
}

/// Hold the returned grant for the whole native test.
func nativeGPUGrant(_ test: String = #function) async throws -> GPUAdmissionGrant? {
    try await nativeGPU.get()?.admit(owner: test)
}
