import AetherKit
import AetherKrea
import AetherKreaCoreAI
import AetherKreaTokenizers
import Foundation
import Testing

/// Explicit native qualification of rebuilt text-only assets. No image is saved and no application evidence or
/// approval is created. Ordinary `swift test` never loads these models. The shared GPU lease joins other tools.
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_KREA_TEXT_QUALIFICATION"] == "1"
               && nativeEnvironment["AETHER_KREA_BUNDLE"] != nil), .timeLimit(.minutes(30)))
func rebuiltKreaTextBundleConditionsBoundaryPromptsAndCompletesGeneration() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let assets = try KreaAssets.open(at: URL(fileURLWithPath: try #require(nativeEnvironment["AETHER_KREA_BUNDLE"])))
    try assets.verify()
    let session = KreaGenerationSession(assets: assets, compute: .preferGPU, residency: .resident, lifecycle: .manual)
    do {
        // The encoder must preserve all prompt rows; the SDK alone removes the 34-row template prefix.
        for count in [35, 48, 546] {
            let prepared = try await session.prepare(prompt: KreaPrompt(tokenIDs: (1000..<(1000 + count)).map(Int32.init)))
            #expect(prepared.conditioning.rows == count - KreaPrompt.leadingRows)
        }
        let tokenizer = try await KreaTokenizer.open(assets: assets)
        let prompt = try tokenizer.prepare("A red ceramic mug on a white tabletop.")
        let prepared = try await session.prepare(prompt: prompt)
        #expect(prepared.conditioning.rows == prompt.conditioningRows)
        let image = try await session.generate(configuration: KreaConfiguration(width: 512, height: 512, steps: 8, seed: 18),
                                               preparedPrompt: prepared)
        #expect(image.pixelsCHW.shape == [3, 512, 512])
        #expect(image.pixelsCHW.values.allSatisfy { $0.isFinite })
        await session.unload()
    } catch {
        await session.unload()
        throw error
    }
}
