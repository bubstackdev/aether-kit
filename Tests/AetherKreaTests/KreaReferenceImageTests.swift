import AetherKit
import CryptoKit
import Foundation
import Testing
@testable import AetherKrea

// Parity with the reference Qwen-VL processor, forge's vision host tables and the Krea 2 reference oracle
// (Qwen3VLModel). Fixture provenance: Fixtures/reference/SOURCES.md. CPU only; no model is loaded.

private func referenceFixture(_ path: String) throws -> URL {
    let base = try #require(Bundle.module.url(forResource: "reference", withExtension: nil, subdirectory: "Fixtures"))
    return base.appendingPathComponent(path)
}

private func int32s(_ url: URL) throws -> [Int32] {
    let data = try Data(contentsOf: url)
    return data.withUnsafeBytes { raw in (0..<(data.count / 4)).map { Int32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self)) } }
}

private struct OracleCase: Decodable {
    struct Files: Decodable { let images: [String]; let ids: String; let rope: String; let pixels: String }
    let id: String
    let prompt: String
    let grids: [[Int]]
    let imageTokens: [Int]
    let patches: Int
    let prefix: Int
    let tokens: Int
    let files: Files
}

private func oracleCases() throws -> (cases: [OracleCase], sha256: [String: String]) {
    let root = try referenceFixture("krea2-c4f6229d")
    let lines = try String(contentsOf: root.appendingPathComponent("cases.jsonl"), encoding: .utf8).split(separator: "\n")
    let cases = try lines.map { try JSONDecoder().decode(OracleCase.self, from: Data($0.utf8)) }
    struct Manifest: Decodable { let sha256: [String: String] }
    let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: root.appendingPathComponent("fixture.json")))
    return (cases, manifest.sha256)
}

@Test func smartResizeMatchesTheReferenceProcessorForEverySize() throws {
    struct Case: Decodable { let src: [Int]; let grid: [Int]?; let error: String? }
    struct Document: Decodable { let cases: [Case] }
    let document = try JSONDecoder().decode(Document.self, from: Data(contentsOf: referenceFixture("grids.json")))
    #expect(document.cases.count >= 100)
    for item in document.cases {
        if item.error != nil {
            #expect(throws: AetherError.self, "\(item.src)") {
                _ = try KreaReferencePreprocessing.targetSize(width: item.src[0], height: item.src[1])
            }
        } else {
            let size = try KreaReferencePreprocessing.targetSize(width: item.src[0], height: item.src[1])
            #expect([1, size.height / 16, size.width / 16] == item.grid, "source \(item.src)")
        }
    }
}

@Test func patchTablesMatchForgeForEveryGrid() throws {
    struct Reference: Decodable { let grid: [Int]; let positionIndex: [Int32]; let positionWeight: [Float]; let coordinates: [Int32] }
    let references = try JSONDecoder().decode([Reference].self, from: Data(contentsOf: referenceFixture("patch-tables.json")))
    #expect(references.count >= 6)
    for reference in references {
        let tables = try KreaPatchTables(grid: (reference.grid[0], reference.grid[1], reference.grid[2]))
        #expect(tables.patches == reference.grid.reduce(1, *))
        #expect(tables.positionIndex == reference.positionIndex, "\(reference.grid)")
        #expect(tables.coordinates == reference.coordinates, "\(reference.grid)")
        let worst = zip(tables.positionWeight, reference.positionWeight).map { abs($0 - $1) }.max() ?? 0
        #expect(tables.positionWeight.count == reference.positionWeight.count && worst <= 1e-7, "\(reference.grid)")
    }
    #expect(throws: AetherError.self) { _ = try KreaPatchTables(grid: (1, 3, 4)) }
}

@Test func oraclePixelsAreReproducedBitExactlyFromThePNGs() throws {
    let (cases, sha256) = try oracleCases()
    #expect(cases.count == 6)
    let root = try referenceFixture("krea2-c4f6229d")
    for item in cases {
        var bytes = Data()
        var patches = 0
        for (index, name) in item.files.images.enumerated() {
            let image = try KreaReferenceImage(encoded: Data(contentsOf: root.appendingPathComponent(name)))
            let pixels = try KreaReferencePreprocessing.prepare(image)
            #expect([pixels.grid.t, pixels.grid.h, pixels.grid.w] == item.grids[index], "\(item.id) image \(index)")
            #expect(pixels.tokens == item.imageTokens[index])
            patches += pixels.patches
            pixels.values.withUnsafeBufferPointer { buffer in
                for value in buffer { withUnsafeBytes(of: value.bitPattern.littleEndian) { bytes.append(contentsOf: $0) } }
            }
        }
        #expect(patches == item.patches)
        let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        #expect(digest == sha256[item.files.pixels], "\(item.id) pixels.f32")
    }
}

@Test func oracleTokenExpansionAndMRoPEPositionsMatchExactly() throws {
    let (cases, _) = try oracleCases()
    let root = try referenceFixture("krea2-c4f6229d")
    for item in cases {
        let ids = try int32s(root.appendingPathComponent(item.files.ids))
        let rope = try int32s(root.appendingPathComponent(item.files.rope))
        #expect(ids.count == item.tokens && rope.count == 3 * item.tokens)
        let images = try item.files.images.map {
            try KreaReferencePreprocessing.prepare(KreaReferenceImage(encoded: Data(contentsOf: root.appendingPathComponent($0))))
        }
        // Collapse each expanded image span to the single pad the chat template renders.
        var collapsed: [Int] = []
        for id in ids where !(id == KreaReferenceContract.imageToken && collapsed.last == Int(KreaReferenceContract.imageToken)) {
            collapsed.append(Int(id))
        }
        let prompt = try KreaReferencePrompt(renderedTokenIDs: collapsed, textOnlyTokenCount: 100, images: images)
        #expect(prompt.tokenIDs == ids, "\(item.id) ids")
        #expect(prompt.ropePositions == rope, "\(item.id) rope")
        #expect(prompt.images.map(\.range.count) == item.imageTokens)
        #expect(prompt.images.first!.range.lowerBound >= item.prefix)
        #expect(!prompt.wasTruncated)
        let inputs = try prompt.encoderInputs(ropeShape: [3, 1, item.tokens])
        #expect(inputs["rope_positions"]?.shape == [3, 1, item.tokens])
        #expect(inputs["input_ids"]?.shape == [1, item.tokens])
    }
}

@Test func referencePromptRendersLabelledImagePartsBeforeTheText() {
    #expect(KreaReferencePrompt.render("the person in Picture 1", imageCount: 2) ==
        "<|im_start|>system\n\(KreaPrompt.systemMessage)<|im_end|>\n<|im_start|>user\n"
        + "Picture 1: <|vision_start|><|image_pad|><|vision_end|>Picture 2: <|vision_start|><|image_pad|><|vision_end|>"
        + "the person in Picture 1<|im_end|>\n<|im_start|>assistant\n")
}

@Test func truncationCutsOnlyTextAndKeepsTheTrailingTokens() throws {
    let image = try KreaReferencePreprocessing.prepare(KreaReferenceImage(rgb: Array(repeating: 128, count: 64 * 64 * 3), width: 64, height: 64))
    #expect(image.tokens == 64) // 64² smart-resizes up to 256² = 16×16 patches
    let pad = Int(KreaReferenceContract.imageToken)
    // 40 template/prefix tokens, one image part, then 600 text tokens and a 5-token suffix.
    let head = Array(1...40), part = [7, 8, pad, 9], text = Array(1000..<1600), tail = Array(2000..<2005)
    let rendered = head + part + text + tail
    let textOnly = head.count + text.count + tail.count // 645 > 546
    let prompt = try KreaReferencePrompt(renderedTokenIDs: rendered, textOnlyTokenCount: textOnly, images: [image])
    #expect(prompt.wasTruncated)
    let expectedKept = rendered.count - 5 - (textOnly - KreaPrompt.maximumTokens)
    #expect(prompt.tokenIDs.count == expectedKept + 5 - 1 + 64)
    #expect(prompt.tokenIDs.suffix(5).map(Int.init) == tail)
    #expect(prompt.tokenIDs.prefix(43).map(Int.init) == head + [7, 8, pad])
    #expect(prompt.images.count == 1 && prompt.images[0].range == 42..<106)
}

@Test func referenceLimitsRefuseBeforeAnyModelWork() throws {
    let big = try KreaReferencePreprocessing.prepare(KreaReferenceImage(rgb: Array(repeating: 0, count: 384 * 384 * 3), width: 384, height: 384))
    #expect(big.tokens == 144)
    let pad = Int(KreaReferenceContract.imageToken)
    let rendered = Array(1...40) + Array(repeating: pad, count: 5) + [1, 2, 3]
    #expect(throws: AetherError.self) { // 5 images exceed maxImages
        _ = try KreaReferencePrompt(renderedTokenIDs: rendered, textOnlyTokenCount: 43, images: Array(repeating: big, count: 5))
    }
    #expect(throws: AetherError.self) { // pads ≠ images
        _ = try KreaReferencePrompt(renderedTokenIDs: Array(1...40) + [pad, 1], textOnlyTokenCount: 41, images: [big, big])
    }
    #expect(throws: AetherError.self) { // an image inside the dropped system rows
        _ = try KreaReferencePrompt(renderedTokenIDs: [1, pad] + Array(1...40), textOnlyTokenCount: 41, images: [big])
    }
    let fourBig = Array(repeating: big, count: 4) // 576 tokens: at the limit
    #expect((try? KreaReferencePrompt(renderedTokenIDs: Array(1...40) + Array(repeating: pad, count: 4) + [1],
                                      textOnlyTokenCount: 41, images: fourBig)) != nil)
    #expect(throws: AetherError.self) { try KreaReferenceImage(encoded: Data("not an image".utf8)) }
    #expect(throws: AetherError.self) { try KreaReferenceImage(rgb: [0, 0], width: 1, height: 1) }
}

@Test func downscalingFitsTheLongSideAndDigestsTheSource() throws {
    var rgb = [UInt8](repeating: 0, count: 1000 * 300 * 3)
    for i in stride(from: 0, to: rgb.count, by: 3) { rgb[i] = UInt8(i % 251) }
    let image = try KreaReferenceImage(rgb: rgb, width: 1000, height: 300)
    #expect(KreaReferencePreprocessing.fitted(width: 1000, height: 300) == (384, 115))
    let pixels = try KreaReferencePreprocessing.prepare(image)
    #expect(pixels.grid.h * 16 * pixels.grid.w * 16 >= KreaReferenceContract.minimumPixels)
    #expect(pixels.grid.h % 2 == 0 && pixels.grid.w % 2 == 0)
    #expect(pixels.digest == image.digest && image.digest.count == 64)
    #expect(pixels.values.allSatisfy { $0 >= -1 && $0 <= 1 })
}
