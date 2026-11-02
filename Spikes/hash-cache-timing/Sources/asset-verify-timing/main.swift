// Time spent in Krea's and H3's SHA-256 checks (CPU and I/O only; no Core AI work).
//
// usage: asset-verify-timing krea <bundle dir with aether.json> [rounds]
//        asset-verify-timing h3 <bundle.json> [rounds]
//        asset-verify-timing h3-weights <bundle.json> [rounds]
//        asset-verify-timing sha <MiB>          (SHA-256 throughput over an in-memory buffer)
import AetherH3
import AetherKrea
import CryptoKit
import Foundation

let args = CommandLine.arguments
let clock = ContinuousClock()
func seconds(_ d: Duration) -> Double { Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }
func timed(_ body: () throws -> Void) rethrows -> Double { let t = clock.now; try body(); return seconds(t.duration(to: clock.now)) }
let rounds = args.count > 3 ? Int(args[3])! : 2
switch args[1] {
case "krea":
    let url = URL(fileURLWithPath: args[2])
    for round in 1...rounds {
        var assets: KreaAssets?
        let open = try timed { assets = try KreaAssets.open(at: url) }
        let verify = try timed { try assets!.verify() }
        let references = try timed { try assets!.verifyIncludingReferences() }
        print(String(format: "krea round %d: open %.2f s, verify() %.2f s, verifyIncludingReferences() %.2f s", round, open, verify, references))
    }
case "h3":
    let url = URL(fileURLWithPath: args[2])
    for round in 1...rounds {
        var profile: H3Profile?
        let open = try timed { profile = try H3Profile(url: url) }
        let models = try timed { try profile!.verifyModels() }
        print(String(format: "h3 round %d: profile open %.2f s, verifyModels() %.2f s", round, open, models))
    }
case "h3-weights":
    // Every in-place weight binding of the body read into one buffer with verification, as a graph's first load does.
    let profile = try H3Profile(url: URL(fileURLWithPath: args[2]))
    let files = profile.parameters.flatMap { $0 }
    let largest = files.map(\.spec.byteCount).max() ?? 0
    let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: largest, alignment: 64)
    defer { buffer.deallocate() }
    for round in 1...rounds {
        var bytes = 0
        let t = try timed {
            for file in files {
                try file.read(into: UnsafeMutableRawBufferPointer(rebasing: buffer[..<file.spec.byteCount]), verify: true)
                bytes += file.spec.byteCount
            }
        }
        print(String(format: "h3 weights round %d: %d tensors, %.1f GB read with verify in %.2f s", round, files.count, Double(bytes) / 1e9, t))
    }
default:
    let count = Int(args[2])! << 20
    let data = Data(repeating: 7, count: count)
    var hasher = SHA256()
    let t = timed { hasher.update(data: data); _ = hasher.finalize() }
    print(String(format: "sha256: %d MiB in %.3f s, %.2f GB/s", count >> 20, t, Double(count) / t / 1e9))
}
