import AetherCoreAI
import AetherKit
import Foundation

/// aether-probe takes the shared GPU lease itself for every well-formed command that runs native GPU work, before
/// dispatching it, and holds it until the command returns. Malformed GPU commands take nothing, so a typo prints its
/// usage error at once instead of waiting for the lease. See docs/gpu-lease.md for the `AETHER_GPU_LEASE*` overrides.
enum ProbeGPULease {
    /// The argument shapes of GPU commands, mirroring each command's own usage check (`arguments[0]` is the command).
    /// Commands not listed here are CPU-only.
    static func usesGPU(_ arguments: [String]) -> Bool {
        guard let command = arguments.first else { return false }
        let count = arguments.count
        func mode(at index: Int, gpu: Set<String>) -> Bool { count > index && gpu.contains(arguments[index]) }
        switch command {
        case "inspect", "smoke":
            return (2...3).contains(count) && (count == 2 || ModelComputePolicy(rawValue: arguments[2]) != nil)
        case "render-krea":
            return (7...8).contains(count) && Int(arguments[4]) != nil && Int(arguments[5]) != nil && UInt64(arguments[6]) != nil
        case "render-krea-batch":
            return (7...8).contains(count) && Int(arguments[4]) != nil && Int(arguments[5]) != nil
                && !arguments[3].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case "parity-krea":
            return count == 7 && ["provided", "encoded"].contains(arguments[6])
        case "h3-profile-run":
            return count >= 6 && ["--graphs", "--render"].contains(arguments[5])
        case "h3-profile-decode":
            return count == 5 || count == 7
        case "qualify-h3", "qualify-h3-lifecycle":
            return count == 5 && mode(at: 4, gpu: ["--run"])
        case "qualify-h3-phases":
            return count == 6 && mode(at: 5, gpu: ["--run"])
        case "qualify-h3-chain":
            return count == 6 && mode(at: 5, gpu: ["--capture", "--run"])
        case "qualify-h3-denoising":
            return count == 9 && (arguments[8] == "--capture" || (["--run", "--lifecycle"].contains(arguments[8]) && arguments[6] != "-"))
        case "benchmark-krea":
            return count == 6 && mode(at: 5, gpu: ["--run"])
        default:
            return false
        }
    }

    /// `nil` for CPU-only or malformed commands, or when `AETHER_GPU_LEASE=off`; otherwise an owned or inherited lease.
    static func acquire(for arguments: [String],
                        environment: [String: String] = ProcessInfo.processInfo.environment) async throws -> SharedGPULease? {
        guard usesGPU(arguments) else { return nil }
        let policy = try SharedGPULease.ToolPolicy.environment(environment)
        return try await policy.acquire { busy in
            FileHandle.standardError.write(Data((policy.message(busy, tool: "aether-probe") + "\n").utf8))
        }
    }
}
