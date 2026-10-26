import AetherKit
import Foundation

/// aether-probe takes the shared GPU lease itself for every command that runs native GPU work, before dispatching it,
/// and holds it until the command returns. See docs/gpu-lease.md for the `AETHER_GPU_LEASE*` overrides.
enum ProbeGPULease {
    /// Commands that always run native work.
    static let gpuCommands: Set<String> = [
        "h3-profile-run", "h3-profile-decode", "inspect", "smoke", "render-krea", "render-krea-batch", "parity-krea",
    ]
    /// Commands whose `--prepare-only` mode is CPU-only and whose other modes run native work.
    static let modalCommands: Set<String> = [
        "qualify-h3-denoising", "qualify-h3-phases", "qualify-h3", "qualify-h3-lifecycle", "qualify-h3-chain", "benchmark-krea",
    ]
    static let gpuModes: Set<String> = ["--run", "--capture", "--lifecycle"]

    static func usesGPU(_ arguments: [String]) -> Bool {
        guard let command = arguments.first else { return false }
        if gpuCommands.contains(command) { return true }
        return modalCommands.contains(command) && arguments.dropFirst().contains(where: gpuModes.contains)
    }

    /// `nil` for CPU-only commands or when `AETHER_GPU_LEASE=off`; otherwise an owned or inherited lease.
    static func acquire(for arguments: [String],
                        environment: [String: String] = ProcessInfo.processInfo.environment) async throws -> SharedGPULease? {
        guard usesGPU(arguments) else { return nil }
        let policy = try SharedGPULease.ToolPolicy.environment(environment)
        let path = policy.configuration?.directory.path ?? ""
        return try await policy.acquire { holder in
            let owner = holder.map { "process \($0)" } ?? "an unidentified holder"
            FileHandle.standardError.write(Data("aether-probe: waiting for the GPU lease \(path) held by \(owner)\n".utf8))
        }
    }
}
