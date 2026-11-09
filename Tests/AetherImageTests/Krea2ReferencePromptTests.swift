import AetherCore
import CryptoKit
import Foundation
import Testing
@testable import AetherImage

/// The reference-prompt rules, exactly as 0.0.x: the preprocessing (bit-exact against 0.0.7's own output and the
/// 0.0.x oracle's pixels), the smart-resize grids against the reference processor, the image-pad expansion and M-RoPE
/// positions against the oracle's ids and rope, truncation of text only, the extension ids and the refusals. No
/// model, no tokenizer: rendered prompts are rebuilt from the oracle's ids.
@Suite("Krea 2 reference prompts")
struct Krea2ReferencePromptTests {
    static let fixtures = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "AetherCoreTests/Fixtures/krea2-reference")

    /// The 0.0.x `visionTokens` contract as a `krea2.references` block.
    static let referencesJSON = """
    {"maxImages": 4, "maxSide": 384, "maxImageTokens": 576, "label": "Picture {n}: ",
     "tokens": {"pad": 151655, "start": 151652, "end": 151653}, "extensionBase": 151936, "rows": "1...1116",
     "positions": "mrope", "deepstackLevels": 3, "vision": "vision", "function": "references",
     "preprocess": {"fit": "longSideRounded", "maxSide": 384, "factor": 32, "minPixels": 65536, "maxPixels": 16777216,
                    "maxAspect": 200, "filter": "keysBicubic-0.5-antialias", "mean": 127.5, "std": 127.5, "patch": 16,
                    "temporalPatch": 2, "merge": 2, "order": "mergeWindow C,T,py,px"}}
    """

    static func references() throws -> Manifest.Krea2.References {
        try JSONDecoder().decode(Manifest.Krea2.References.self, from: Data(referencesJSON.utf8))
    }

    /// The exporter's real prompt rule (34 dropped rows, 546 tokens, 5 kept).
    static func promptRule() throws -> Manifest.Krea2.Prompt {
        try #require(Manifest.decode(Data(contentsOf: Krea2Tests.manifestURL)).krea2?.prompt)
    }

    static func hex(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

    static func floatsDigest(_ values: [Float]) -> String {
        var bytes = Data()
        for v in values { withUnsafeBytes(of: v.bitPattern.littleEndian) { bytes.append(contentsOf: $0) } }
        return hex(bytes)
    }

    /// 0.0.7's `KreaReferencePreprocessing.prepare` on these synthetic images (recorded 2026-11-08 from 0.0.7's
    /// own code): the patch grid and the SHA-256 of the float32 patch rows. Covers downscaling, the area floor, a thin
    /// image upscaled, an image already at its size, and an odd size.
    @Test(arguments: [
        (500, 300, 14, 24, "e72d52f15530e7c940fb1a809dd90cb08c022c3aa4153aca4d2393904772ed4c"),
        (1000, 1000, 24, 24, "c2215dad429f97415d348c1930541f3b1e1a77d7e01c43b1021ce666cb4171a7"),
        (100, 384, 32, 10, "e813b9904654828e857dc14ddb3ca5954081b7ebaf604ca94046727e4ef96cc4"),
        (384, 384, 24, 24, "4c86f7df3f084c6113e48652751a0f384d22e291ea2e0c3ea4db518cdeb77dbe"),
        (37, 900, 80, 4, "39da003d696f3b6fbdd90ee5a043f48a77399743c688962fa3529281d00f4909"),
        (2000, 900, 12, 24, "174fbf769bb1bda6ee6cae302e44abcb7b33bc78e739f59dcd89377184992ab0"),
        (383, 129, 10, 28, "41ead34a61661fbb68a5463400df8f55c11137d9601225492a62ca4ef3b72ca3"),
    ])
    func preprocessingMatches007BitForBit(width: Int, height: Int, rows: Int, columns: Int, digest: String) throws {
        let rgb = (0..<(width * height * 3)).map { UInt8(truncatingIfNeeded: ($0 &* 7919 &+ ($0 / 3) &* 13) % 256) }
        let patches = try Krea2ReferencePreprocessing.prepare(
            Krea2ReferenceImage(rgb: rgb, width: width, height: height), rule: Self.references().preprocess)
        #expect(patches.rows == rows && patches.columns == columns)
        #expect(Self.floatsDigest(patches.values) == digest)
    }

    @Test func smartResizeMatchesTheReferenceProcessor() throws {
        struct Document: Decodable {
            struct Case: Decodable { let src: [Int]; let grid: [Int]?; let error: String? }
            let cases: [Case]
        }
        let rule = try Self.references().preprocess
        let document = try JSONDecoder().decode(Document.self, from: Data(contentsOf: Self.fixtures.appending(path: "grids.json")))
        #expect(document.cases.count > 100)
        for item in document.cases {
            if item.error != nil {
                #expect(throws: AetherError.self, "\(item.src)") {
                    _ = try Krea2ReferencePreprocessing.targetSize(width: item.src[0], height: item.src[1], rule: rule)
                }
            } else {
                let size = try Krea2ReferencePreprocessing.targetSize(width: item.src[0], height: item.src[1], rule: rule)
                #expect([1, size.height / 16, size.width / 16] == item.grid, "source \(item.src)")
            }
        }
    }

    struct OracleCase: Decodable {
        struct Files: Decodable { let ids: String; let rope: String; let pixels: String; let images: [String] }
        let id: String
        let files: Files
        let grids: [[Int]]
        let imageTokens: [Int]
        let patches: Int
        let prefix: Int
        let tokens: Int
    }

    static func oracle() throws -> (cases: [OracleCase], sha256: [String: String]) {
        struct Fixture: Decodable { let sha256: [String: String] }
        let lines = try String(decoding: Data(contentsOf: fixtures.appending(path: "cases.jsonl")), as: UTF8.self)
            .split(separator: "\n")
        let cases = try lines.map { try JSONDecoder().decode(OracleCase.self, from: Data($0.utf8)) }
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtures.appending(path: "fixture.json")))
        return (cases, fixture.sha256)
    }

    static func int32s(_ url: URL) throws -> [Int32] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { raw in (0..<(data.count / 4)).map { Int32(littleEndian: raw.loadUnaligned(fromByteOffset: 4 * $0, as: Int32.self)) } }
    }

    static func patches(_ item: OracleCase) throws -> [Krea2ReferencePatches] {
        let rule = try references().preprocess
        return try item.files.images.map {
            try Krea2ReferencePreprocessing.prepare(
                Krea2ReferenceImage(encoded: Data(contentsOf: fixtures.appending(path: $0))), rule: rule)
        }
    }

    @Test func theOraclesPixelsAreReproducedFromItsPNGs() throws {
        let (cases, sha256) = try Self.oracle()
        #expect(cases.count == 6)
        for item in cases {
            let images = try Self.patches(item)
            #expect(images.map { [1, $0.rows, $0.columns] } == item.grids, "\(item.id)")
            #expect(images.map(\.tokens) == item.imageTokens && images.reduce(0) { $0 + $1.patches } == item.patches)
            #expect(Self.floatsDigest(images.flatMap(\.values)) == sha256[item.files.pixels], "\(item.id) pixels.f32")
        }
    }

    @Test func padExpansionAndMRoPEMatchTheOracle() throws {
        let (cases, _) = try Self.oracle()
        let rule = try Self.promptRule(), references = try Self.references()
        for item in cases {
            let ids = try Self.int32s(Self.fixtures.appending(path: item.files.ids))
            let rope = try Self.int32s(Self.fixtures.appending(path: item.files.rope))
            #expect(ids.count == item.tokens && rope.count == 3 * item.tokens)
            // Each expanded image span back to the single pad the template renders.
            var rendered: [Int32] = []
            for id in ids where !(id == references.tokens.pad && rendered.last == references.tokens.pad) { rendered.append(id) }
            let prompt = try Krea2ReferencePrompt(
                rendered: rendered, textOnlyCount: 100, images: try Self.patches(item), prompt: rule, references: references)
            #expect(prompt.ids == ids, "\(item.id) ids")
            #expect(prompt.ropePositions == rope, "\(item.id) rope")
            #expect(prompt.spans.map(\.range.count) == item.imageTokens && prompt.spans[0].range.lowerBound >= item.prefix)
            #expect(!prompt.truncated && prompt.conditioningRows == item.tokens - item.prefix)
            // Image rows go to the encoder as extension ids, one slot per row across the images.
            let slots = prompt.spans.flatMap { Array($0.range) }
            #expect(slots.enumerated().allSatisfy { prompt.encoderIDs[$0.element] == Int32(references.extensionBase + $0.offset) })
            #expect(prompt.encoderIDs.enumerated().allSatisfy { slots.contains($0.offset) || $0.element == ids[$0.offset] })
        }
    }

    @Test func truncationCutsOnlyTextAndKeepsTheTrailingTokens() throws {
        let references = try Self.references(), rule = try Self.promptRule()
        let image = try Krea2ReferencePreprocessing.prepare(
            Krea2ReferenceImage(rgb: Array(repeating: 128, count: 64 * 64 * 3), width: 64, height: 64), rule: references.preprocess)
        #expect(image.tokens == 64)  // 64² resizes up to 256²: 16 × 16 patches
        let pad = references.tokens.pad
        // 40 prefix tokens, one image part, then 600 text tokens and a 5-token suffix.
        let head = (1...40).map(Int32.init), part: [Int32] = [7, 8, pad, 9]
        let text = (1000..<1600).map(Int32.init), tail = (2000..<2005).map(Int32.init)
        let rendered = head + part + text + tail
        let textOnly = head.count + text.count + tail.count  // 645 > 546
        let prompt = try Krea2ReferencePrompt(
            rendered: rendered, textOnlyCount: textOnly, images: [image], prompt: rule, references: references)
        #expect(prompt.truncated)
        let kept = rendered.count - 5 - (textOnly - rule.maxTokens)
        #expect(prompt.ids.count == kept + 5 - 1 + 64)
        #expect(Array(prompt.ids.suffix(5)) == tail && Array(prompt.ids.prefix(43)) == head + [7, 8, pad])
        #expect(prompt.spans.count == 1 && prompt.spans[0].range == 42..<106)
    }

    @Test func referenceLimitsAreRefused() throws {
        let references = try Self.references(), rule = try Self.promptRule()
        let big = try Krea2ReferencePreprocessing.prepare(
            Krea2ReferenceImage(rgb: Array(repeating: 0, count: 384 * 384 * 3), width: 384, height: 384), rule: references.preprocess)
        #expect(big.tokens == 144)
        let pad = references.tokens.pad, head = (1...40).map(Int32.init)
        func build(_ rendered: [Int32], _ textOnly: Int, _ images: [Krea2ReferencePatches]) throws -> Krea2ReferencePrompt {
            try Krea2ReferencePrompt(rendered: rendered, textOnlyCount: textOnly, images: images, prompt: rule, references: references)
        }
        #expect(throws: AetherError.self) { _ = try build(head + Array(repeating: pad, count: 5) + [1], 41, Array(repeating: big, count: 5)) }
        #expect(throws: AetherError.self) { _ = try build(head + [pad, 1], 41, [big, big]) }  // pads ≠ images
        #expect(throws: AetherError.self) { _ = try build([1, pad] + head, 41, [big]) }  // inside the dropped prefix
        #expect((try? build(head + Array(repeating: pad, count: 4) + [1], 41, Array(repeating: big, count: 4))) != nil)  // 576
        #expect(throws: AetherError.self) { _ = try Krea2ReferenceImage(encoded: Data("not an image".utf8)) }
        #expect(throws: AetherError.self) { _ = try Krea2ReferenceImage(rgb: [0, 0], width: 1, height: 1) }
    }
}
