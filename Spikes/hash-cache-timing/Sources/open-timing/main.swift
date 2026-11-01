// Open-time breakdown of a registered language bundle with image input: the registration's verification alone
// (`verify`, `verifyVision`, `verifyTokenizer`: every hash a session open computes), then the full
// `VisionLanguageSession.open(bundle:)` and unload, for several rounds in one process.
//
// usage: open-timing <bundle> [rounds] [--verify-only]
import AetherKit
import AetherLanguageCoreAI
import Foundation

var args = Array(CommandLine.arguments.dropFirst())
let verifyOnly = args.contains("--verify-only")
args.removeAll { $0 == "--verify-only" }
let bundle = URL(fileURLWithPath: args[0])
let rounds = args.count > 1 ? Int(args[1])! : 3
let clock = ContinuousClock()
func seconds(_ d: Duration) -> Double { Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }
let manifest = try LanguageModelManifest.load(from: bundle)
for round in 1...rounds {
    var t = clock.now
    _ = try manifest.verify(in: bundle, platform: .macos)
    _ = try manifest.verifyVision(in: bundle)
    _ = try manifest.verifyTokenizer(in: bundle)
    let verify = seconds(t.duration(to: clock.now))
    var open = Double.nan
    if !verifyOnly {
        t = clock.now
        let session = try await VisionLanguageSession.open(bundle: bundle, platform: .macos)
        open = seconds(t.duration(to: clock.now))
        await session.unload()
    }
    print(String(format: "round %d: verification %.2f s, session open %.2f s", round, verify, open))
}
