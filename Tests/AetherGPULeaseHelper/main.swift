import AetherKit
import Darwin
import Foundation
import Synchronization

// Test-only child process for SharedGPULeaseTests; not a product and never shipped. Every mode takes a lease on the
// private directory it is given, never the shared /tmp path.
//
//   default DIR   take the lease, print "ready", block (SIGINT/SIGTERM/SIGHUP left at their defaults)
//   handled DIR   first install an own graceful SIGTERM shutdown that releases and exits 42
//   ignored DIR   first ignore SIGHUP (as nohup does)
//   late DIR      take the lease, install a SIGTERM handler that exits 43, release, print "ready", block
//   try DIR [inherit]   one attempt: print "owned" or "inherited <pid>" and exit 0, or print "held <pid>" and exit 2
//   wait DIR SECONDS    wait up to SECONDS: print "acquired", release, exit 0; exit 3 on timeout
//   pair DIR      take DIR and its sibling DIR.second, release DIR, print "ready", block
private final class Held: Sendable { let lease = Mutex<SharedGPULease?>(nil) }

private func configuration(_ path: String, inherit: Bool = false) -> SharedGPULease.Configuration {
    SharedGPULease.Configuration(directory: URL(fileURLWithPath: path, isDirectory: true), inheritsAncestorLease: inherit,
                                 pollInterval: .milliseconds(20))
}

private func say(_ text: String) { print(text); fflush(stdout) }

private func describe(_ holder: Int32?) -> String { holder.map(String.init) ?? "unknown" }

private func run() async -> Never {
    let arguments = CommandLine.arguments
    guard arguments.count >= 3 else { exit(64) }
    let mode = arguments[1], path = arguments[2]
    switch mode {
    case "try":
        do {
            let lease = try SharedGPULease.tryAcquire(configuration(path, inherit: arguments.count > 3 && arguments[3] == "inherit"))
            switch lease.holding {
            case .owned: say("owned")
            case .inherited(let holder): say("inherited \(holder)")
            }
            lease.release()
            exit(0)
        } catch SharedGPULeaseError.held(let holder) {
            say("held \(describe(holder))"); exit(2)
        } catch { exit(65) }
    case "wait":
        guard arguments.count == 4, let seconds = Double(arguments[3]) else { exit(64) }
        do {
            let lease = try await SharedGPULease.acquire(configuration(path), timeout: .milliseconds(Int64(seconds * 1000)))
            say("acquired")
            lease.release()
            exit(0)
        } catch SharedGPULeaseError.timedOut(let holder) {
            say("timedOut \(describe(holder))"); exit(3)
        } catch { exit(65) }
    case "default", "handled", "ignored", "late":
        let held = Held()
        var shutdown: (any DispatchSourceSignal)?
        if mode == "handled" {
            signal(SIGTERM, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
            source.setEventHandler {
                // Long enough that a re-raised default termination would win if the lease had also handled SIGTERM.
                usleep(300_000)
                held.lease.withLock { $0?.release(); $0 = nil }
                exit(42)
            }
            source.resume()
            shutdown = source
        } else if mode == "ignored" {
            signal(SIGHUP, SIG_IGN)
        }
        do {
            let lease = try SharedGPULease.tryAcquire(configuration(path))
            held.lease.withLock { $0 = lease }
        } catch { exit(65) }
        if mode == "late" {
            signal(SIGTERM) { _ in _exit(43) }
            held.lease.withLock { $0?.release(); $0 = nil }
        }
        say("ready")
        while true { withExtendedLifetime(shutdown) { _ = sleep(60) } }
    case "pair":
        let second = URL(fileURLWithPath: path).deletingLastPathComponent().appendingPathComponent("gpu.lock.second").path
        let leases: [SharedGPULease]
        do { leases = [try SharedGPULease.tryAcquire(configuration(path)), try SharedGPULease.tryAcquire(configuration(second))] }
        catch { exit(65) }
        // The second lease must stay covered by signal release after the first, which installed it, is released.
        leases[0].release()
        say("ready")
        while true { withExtendedLifetime(leases) { _ = sleep(60) } }
    default:
        exit(64)
    }
}

await run()
