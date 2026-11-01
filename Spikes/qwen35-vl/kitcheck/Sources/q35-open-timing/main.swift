// Open and first-request timing of a hybrid bundle's decoder: open, a 1-token text request, a second request, unload;
// three rounds in one process.
//
// usage: q35-open-timing <bundle>
import AetherKit
import AetherLanguageCoreAI
import Foundation

let bundle = URL(fileURLWithPath: CommandLine.arguments[1])
let clock = ContinuousClock()
func seconds(_ d: Duration) -> Double { Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }
for round in 1...3 {
    var t = clock.now
    let session = try await HybridLanguageSession.open(bundle: bundle, platform: .macos)
    let open = seconds(t.duration(to: clock.now))
    t = clock.now
    _ = try await session.generate(LanguageGenerationRequest(prompt: [248045, 846, 198], maximumTokens: 1), onToken: nil)
    let first = seconds(t.duration(to: clock.now))
    t = clock.now
    _ = try await session.generate(LanguageGenerationRequest(prompt: [248045, 846, 198], maximumTokens: 1), onToken: nil)
    let second = seconds(t.duration(to: clock.now))
    await session.unload()
    print(String(format: "round %d: open %.2f s, first request %.2f s, second %.3f s", round, open, first, second))
}
