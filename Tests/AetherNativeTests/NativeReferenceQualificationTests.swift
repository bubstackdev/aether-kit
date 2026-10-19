import AetherKit
import AetherKrea
import AetherKreaTokenizers
import Foundation
import Testing
@testable import AetherKreaCoreAI

/// Opt-in native qualification of reference-image conditioning against the Krea 2 reference oracle
/// (`krea2-reference` fixture: Qwen3VLModel float32 text rows after the 34-row prefix, `text.f32`).
/// Loads the vision and text encoders on the GPU: run only with an explicit owner's approval, under the shared GPU
/// lock, one native suite at a time. Bars follow the prototype's T2 reference check: mean real-row cosine ≥ 0.999 and
/// no row below 0.99.
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_REFERENCE_QUALIFICATION"] == "1"
               && nativeEnvironment["AETHER_KREA_BUNDLE"] != nil && nativeEnvironment["AETHER_KREA_REFERENCE_FIXTURES"] != nil),
      .timeLimit(.minutes(30)))
func referenceConditioningMatchesTheOracleRows() async throws {
    let root = URL(fileURLWithPath: try #require(nativeEnvironment["AETHER_KREA_BUNDLE"]))
    let fixtures = URL(fileURLWithPath: try #require(nativeEnvironment["AETHER_KREA_REFERENCE_FIXTURES"]))
    let assets = try KreaAssets.open(at: root)
    try assets.verifyIncludingReferences()
    let tokenizer = try await KreaTokenizer.open(assets: assets)
    let session = KreaGenerationSession(assets: assets)
    struct Case: Decodable { let id: String; let prompt: String; let files: Files; struct Files: Decodable { let images: [String]; let text: String } }
    for line in try String(contentsOf: fixtures.appendingPathComponent("cases.jsonl"), encoding: .utf8).split(separator: "\n") {
        let item = try JSONDecoder().decode(Case.self, from: Data(line.utf8))
        let images = try item.files.images.map {
            try KreaReferencePreprocessing.prepare(KreaReferenceImage(encoded: Data(contentsOf: fixtures.appendingPathComponent($0))))
        }
        let prompt = try tokenizer.prepare(item.prompt, references: images)
        let prepared = try await session.prepare(referencePrompt: prompt, references: images)
        let oracle = try Data(contentsOf: fixtures.appendingPathComponent(item.files.text))
        let width = KreaConditioning.channels
        #expect(oracle.count == prepared.conditioning.rows * width * 4, "\(item.id) row count")
        var worst = Float.infinity, sum: Float = 0
        oracle.withUnsafeBytes { raw in
            for row in 0..<prepared.conditioning.rows {
                var dot: Float = 0, a2: Float = 0, b2: Float = 0
                for i in 0..<width {
                    let a = Float(prepared.conditioning.tensor.values[row * width + i])
                    let b = Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: (row * width + i) * 4, as: UInt32.self)))
                    dot += a * b; a2 += a * a; b2 += b * b
                }
                let cosine = dot / max((a2 * b2).squareRoot(), .leastNormalMagnitude)
                worst = min(worst, cosine); sum += cosine
            }
        }
        let mean = sum / Float(prepared.conditioning.rows)
        print("reference \(item.id): rows \(prepared.conditioning.rows) mean cosine \(mean) worst \(worst)")
        #expect(mean >= 0.999 && worst >= 0.99, "\(item.id)")
    }
    await session.unload()
}
