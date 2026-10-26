import AetherAssets
import AetherCoreAI
import AetherH3
import AetherH3CoreAI
import AetherH3Tokenizers
import AetherKit
import AetherKrea
import AetherKreaCoreAI
import AetherKreaTokenizers
import AetherLanguageCoreAI
import Foundation

// A separate consumer package verifying every product's exported API. Running it
// prepares Krea inputs only; it loads no weights and performs no native work.
let configuration = try KreaConfiguration(width: 768, height: 1344, seed: 71344)
let schedule = try KreaSchedule(steps: configuration.steps)
let previewNoise = try KreaNoise.generate(seed: configuration.seed, count: 8)
print("Krea: \(configuration.imageTokenCount) image tokens, \(schedule.timesteps.count) steps")
print("Noise bit patterns: \(previewNoise.map(\.bitPattern))")
print("Products linked: \([String(describing: H3VideoSession.self), String(describing: LanguageGenerationSession.self)])")

// The README examples, type-checked here but never called.
func kreaReadmeExample(bundleURL: URL, outputURL: URL) async throws {
    let assets = try KreaAssets.open(at: bundleURL)
    let tokenizer = try await KreaTokenizer.open(assets: assets)
    let session = KreaGenerationSession(assets: assets)
    let configuration = try KreaConfiguration(width: 512, height: 512, seed: 17017)
    let image = try await session.generate(configuration: configuration,
                                           prompt: tokenizer.prepare("a red fox in fresh snow"))
    try image.savePNG(to: outputURL)
    let prepared = try await session.prepare(prompt: tokenizer.prepare("a red fox in fresh snow"))
    _ = try await session.generate(configuration: configuration, preparedPrompt: prepared)
    _ = KreaGenerationSession(assets: assets, residency: .resident)
    await session.unload()
}

func installedKreaExample(lease: AssetReadLease) throws {
    _ = try KreaGenerationSession(installed: lease)
}

func languageReadmeExample(bundleURL: URL, renderedTokenIDs: [Int]) async throws {
    let session = try await LanguageGenerationSession.open(bundle: bundleURL, platform: .macos)
    let reply = try await session.generate(prompt: renderedTokenIDs, maximumTokens: 512,
                                           prefillChunkSize: 64, onToken: { token in _ = token })
    _ = reply.tokens
    await session.unload()
}

func referenceExecutorExample(modelURL: URL) async throws {
    let session = try await ModelSession.open(at: modelURL)
    _ = try await session.run(inputs: ["x": FloatTensor(shape: [1, 4], values: [-2, 0, 1, 3])])
    await session.unload()
}
