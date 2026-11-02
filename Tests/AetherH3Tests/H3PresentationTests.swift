import AetherKit
import AetherH3
import CryptoKit
import Foundation
import Testing

/// `Fixtures/presentation-oracle.json` (`Scripts/prepare-h3-presentation-oracle.py`): upstream's FL2VA presentation
/// from `MiniMaxH3FL2VATextEncoderStep` run verbatim, the processor's pixels and the tower's host tables for synthetic
/// keyframes, and the text rotary module's cos table and its value at the presentation's M-RoPE positions.
private struct PresentationOracle: Decodable {
    struct Case: Decodable {
        let canvas: String; let keyframes: Int; let anchors: [String]; let grids: [[Int]]; let count: Int
        let ids_sha256: String; let tags_sha256: String; let positions_sha256: String; let prompt_ids_sha256: String
    }
    struct Embedded: Decodable { let canvas: String; let prompt: [Int32]; let ids: [Int32]; let tags: [Int32]; let positions: [[Int32]] }
    struct Vision: Decodable {
        let width: Int; let height: Int; let grid: [Int]
        let pixels_sha256: String; let pos_index_sha256: String; let pos_weight_sha256: String; let patch_rc_sha256: String
    }
    struct Rotary: Decodable { let rows: Int; let mrope_section: [Int]; let cos_table_base64: String; let cos_sha256: String }
    let labels: [String: [Int32]]
    let presentations: [Case]
    let embedded: Embedded
    let vision: [Vision]
    let rotary: Rotary

    static func load() throws -> Self {
        let url = try #require(Bundle.module.url(forResource: "presentation-oracle", withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}

private func sha256<T>(_ values: [T]) -> String {
    values.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
}
private func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
private func canvas(_ text: String) -> (width: Int, height: Int) {
    let sides = text.split(separator: "x").map { Int($0)! }
    return (sides[0], sides[1])
}

@Test func h3PresentationMatchesUpstreamsFL2VAStep() throws {
    let oracle = try PresentationOracle.load()
    let tokenize: (String) throws -> [Int32] = { label in try #require(oracle.labels[label]) }
    let small = oracle.embedded
    let presentation = try H3Presentation(prompt: small.prompt, keyframes: [canvas(small.canvas)], tokenize: tokenize)
    #expect(presentation.ids == small.ids)
    #expect(presentation.tags == small.tags)
    #expect(presentation.positions == small.positions)
    // Every case, including 960x544 (567 rows: 6 label + 512 vision-block + 49 prompt), by hash. The embedded case's
    // prompt is the prompt of every case.
    for item in oracle.presentations {
        #expect(sha256(small.prompt) == item.prompt_ids_sha256)
        let sizes = Array(repeating: canvas(item.canvas), count: item.keyframes)
        let p = try H3Presentation(prompt: small.prompt, keyframes: sizes, tokenize: tokenize)
        #expect(p.count == item.count)
        #expect(sha256(p.ids) == item.ids_sha256, "\(item.canvas)")
        #expect(sha256(p.tags) == item.tags_sha256, "\(item.canvas)")
        #expect(sha256(p.positions.flatMap { $0 }) == item.positions_sha256, "\(item.canvas)")
        #expect(p.images.map { [$0.gridHeight, $0.gridWidth] } == item.grids)
    }
    // The cases the fixture must cover: first, last alone (labelled <Picture 1>), first plus last, and portrait.
    #expect(Set(oracle.presentations.map { "\($0.canvas) \($0.anchors.joined(separator: "+"))" })
            .isSuperset(of: ["288x288 first", "960x544 first", "544x960 first", "960x544 last", "960x544 first+last"]))
}

@Test func h3PresentationNumbersKeyframesInPackedOrder() throws {
    let oracle = try PresentationOracle.load()
    let tokenize: (String) throws -> [Int32] = { label in try #require(oracle.labels[label]) }
    let p = try H3Presentation(prompt: [9, 8, 7], keyframes: [(288, 288), (288, 288)], tokenize: tokenize)
    let block = 81 + 2
    #expect(Array(p.ids[0..<6]) == oracle.labels["<Picture 1>: "])
    #expect(Array(p.ids[(6 + block)..<(12 + block)]) == oracle.labels["<Picture 2>: "])
    #expect(p.images.map(\.start) == [7, 7 + block + 6])
    #expect(p.tags.filter { $0 == 0 }.count == 2 * block)
    // After a 9x9 merged image (t = 7, h and w = 7...15), the running position advances by 9: vision end at 16, the
    // second label at 17...22, its vision start at 23 and the second image from 24.
    #expect(p.positions[0][6 + block - 1] == 16)
    #expect((0..<3).map { p.positions[$0][6 + block] } == [17, 17, 17])
    let second = p.images[1].start
    #expect((0..<3).map { p.positions[$0][second] } == [24, 24, 24])
    #expect((0..<3).map { p.positions[$0][second + 80] } == [24, 32, 32])
    #expect((0..<3).map { p.positions[$0][second + 81 + 1] } == [34, 34, 34])   // the prompt, after vision end at 33
    #expect(Array(p.ids.suffix(3)) == [9, 8, 7])
}

@Test func h3PresentationRefusesCanvasesTheProcessorWouldResize() throws {
    let tokenize: (String) throws -> [Int32] = { _ in [1] }
    #expect(throws: AetherError.self) { try H3Presentation(prompt: [1], keyframes: [(96, 64)], tokenize: tokenize) }
    #expect(throws: AetherError.self) { try H3Presentation(prompt: [1], keyframes: [(300, 288)], tokenize: tokenize) }
}

@Test func h3VisionInputsMatchUpstreamsProcessorAndTables() throws {
    let oracle = try PresentationOracle.load()
    for item in oracle.vision {
        let rgb = (0..<(item.width * item.height * 3)).map { UInt8((7 * $0 + 3) % 256) }
        let inputs = try H3VisionInputs(rgb: rgb, width: item.width, height: item.height)
        #expect([inputs.gridHeight, inputs.gridWidth] == item.grid)
        #expect(inputs.pixels.spec.shape == [item.grid[0] * item.grid[1], 1536])
        #expect(sha256(inputs.pixels.bytes) == item.pixels_sha256, "\(item.width)x\(item.height) pixels")
        #expect(sha256(inputs.positionIndex.bytes) == item.pos_index_sha256, "\(item.width)x\(item.height) taps")
        #expect(sha256(inputs.positionWeight.bytes) == item.pos_weight_sha256, "\(item.width)x\(item.height) weights")
        #expect(sha256(inputs.patchCoordinates.bytes) == item.patch_rc_sha256, "\(item.width)x\(item.height) coordinates")
    }
}

@Test func h3PresentationRotaryIsAGatherOfTheTextTable() throws {
    let oracle = try PresentationOracle.load()
    let small = oracle.embedded
    let tokenize: (String) throws -> [Int32] = { label in try #require(oracle.labels[label]) }
    let presentation = try H3Presentation(prompt: small.prompt, keyframes: [canvas(small.canvas)], tokenize: tokenize)
    let table = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [oracle.rotary.rows, 128]),
                                 bytes: try #require(Data(base64Encoded: oracle.rotary.cos_table_base64)))
    let cos = try presentation.rotary(table, section: oracle.rotary.mrope_section)
    #expect(cos.spec.shape == [1, presentation.count, 128])
    #expect(sha256(cos.bytes) == oracle.rotary.cos_sha256)
    // Text rows (all axes equal) are the table's own rows.
    let rowBytes = 256
    #expect(cos.bytes.prefix(6 * rowBytes) == table.bytes.prefix(6 * rowBytes))
}

@Test func h3BFloat16RoundsToNearestEven() {
    #expect(h3BFloat16Bits(1) == 0x3F80)
    #expect(h3BFloat16Bits(Float(bitPattern: 0x3F80_8000)) == 0x3F80)   // tie, even stays
    #expect(h3BFloat16Bits(Float(bitPattern: 0x3F81_8000)) == 0x3F82)   // tie, odd rounds up
    #expect(h3BFloat16Bits(Float(bitPattern: 0x3F80_8001)) == 0x3F81)
    #expect(h3BFloat16Bits(-1) == 0xBF80)
}
