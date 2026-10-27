import AetherAssets
import AetherFlux2
import AetherFlux2CoreAI
import AetherFlux2Tokenizers
import AetherCoreAI
import AetherH3
import AetherH3CoreAI
import AetherH3Tokenizers
import AetherKit
import AetherKrea
import AetherKreaCoreAI
import AetherKreaTokenizers
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import AetherSpeech
import AetherSpeechCoreAI
import AetherVision
import CoreGraphics
import Foundation

// A separate consumer package verifying every product's exported API. Running it
// prepares Krea inputs only; it loads no weights and performs no native work.
let configuration = try KreaConfiguration(width: 768, height: 1344, seed: 71344)
let schedule = try KreaSchedule(steps: configuration.steps)
let previewNoise = try KreaNoise.generate(seed: configuration.seed, count: 8)
print("Krea: \(configuration.imageTokenCount) image tokens, \(schedule.timesteps.count) steps")
print("Noise bit patterns: \(previewNoise.map(\.bitPattern))")
let flux2 = try Flux2Configuration(width: 512, height: 512, seed: 7)
let flux2Schedule = try Flux2Schedule(steps: flux2.steps, imageTokens: flux2.imageTokenCount)
let flux2Rotary = try Flux2Rotary.table(ids: Flux2Rotary.gridIDs(rows: flux2.rows, columns: flux2.columns))
print("FLUX.2: \(flux2.imageTokenCount) image tokens, sigmas \(flux2Schedule.sigmas), rotary \(flux2Rotary.shape)")
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

func flux2ReadmeExample(bundleURL: URL, picture: CGImage, outputURL: URL) async throws {
    let assets = try Flux2Assets.open(at: bundleURL)
    let tokenizer = try await Flux2Tokenizer.open(assets: assets)
    let session = Flux2GenerationSession(assets: assets)
    let image = try await session.generate(configuration: try Flux2Configuration(width: 512, height: 512, seed: 7),
                                           prompt: tokenizer.prepare("A lighthouse at dusk, watercolor"))
    try image.savePNG(to: outputURL)
    let references = try await session.prepare(references: [try Flux2ReferenceImage(image: picture, maxSide: 512)],
                                               generatedTokens: 1024)
    let prompt = try await session.prepare(prompt: tokenizer.prepare("Turn this photo into a watercolor painting"))
    _ = try await session.generate(configuration: try Flux2Configuration(width: 512, height: 512, seed: 8),
                                   preparedPrompt: prompt, references: references)
    await session.unload()
}
