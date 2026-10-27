@testable import AetherVision
import CryptoKit
import Foundation
import Testing

private struct Fixture: Decodable {
    struct Case: Decodable {
        let grid: [Int]
        let pos_indexSHA256: String
        let pos_weightSHA256: String
        let patch_rcSHA256: String
        let pos_index: [Int32]?
        let pos_weight: [Float]?
        let patch_rc: [Int32]?
    }
    let cases: [Case]
}

private func sha256<T>(_ values: [T]) -> String {
    values.withUnsafeBytes { SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined() }
}

struct Qwen3VLVisionTablesTests {
    /// Bit-exact against the Python mirror that qualified the encoder (Forge/language).
    @Test func matchesTheForgeTablesBitForBit() throws {
        let url = try #require(Bundle.module.url(forResource: "qwen3vl-vision-tables", withExtension: "json", subdirectory: "Fixtures"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        #expect(fixture.cases.count == 8)
        for c in fixture.cases {
            let tables = try Qwen3VLVisionTables(t: c.grid[0], h: c.grid[1], w: c.grid[2])
            #expect(tables.positionIndex.count == tables.patchCount * 4 && tables.patchCoordinates.count == tables.patchCount * 2)
            #expect(sha256(tables.positionIndex) == c.pos_indexSHA256, "pos_index \(c.grid)")
            #expect(sha256(tables.positionWeight) == c.pos_weightSHA256, "pos_weight \(c.grid)")
            #expect(sha256(tables.patchCoordinates) == c.patch_rcSHA256, "patch_rc \(c.grid)")
            if let index = c.pos_index { #expect(tables.positionIndex == index) }
            if let weight = c.pos_weight { #expect(tables.positionWeight == weight) }
            if let rc = c.patch_rc { #expect(tables.patchCoordinates == rc) }
        }
    }

    @Test func mergeWindowOrderAndBilinearInvariants() throws {
        let tables = try Qwen3VLVisionTables(t: 1, h: 4, w: 4)
        // First window, raster inside; then the next window to the right.
        #expect(Array(tables.patchCoordinates[0..<8]) == [0, 0, 0, 1, 1, 0, 1, 1])
        #expect(Array(tables.patchCoordinates[8..<16]) == [0, 2, 0, 3, 1, 2, 1, 3])
        // Weights of each patch sum to one; the first patch samples table cell (0, 0) exactly,
        // the last raster patch the last cell.
        for p in 0..<tables.patchCount {
            #expect(abs(tables.positionWeight[(4 * p)..<(4 * p + 4)].reduce(0, +) - 1) < 1e-6)
        }
        #expect(tables.positionIndex[0] == 0 && tables.positionWeight[0] == 1)
        let big = try Qwen3VLVisionTables(t: 1, h: 6, w: 8)
        let last = big.patchCount - 1                         // the last window's last patch: grid (5, 7)
        #expect(Array(big.patchCoordinates[(2 * last)...]) == [5, 7])
        #expect(big.positionIndex[4 * last] == 47 * 48 + 47)
        // Temporal groups repeat the tables.
        let two = try Qwen3VLVisionTables(t: 2, h: 2, w: 2)
        #expect(Array(two.positionIndex[0..<16]) == Array(two.positionIndex[16..<32]))
    }

    @Test func rejectsGridsThatDoNotTileIntoWindows() {
        #expect(throws: Qwen3VLVisionTables.Failure.invalidGrid(t: 1, h: 3, w: 4, merge: 2)) { try Qwen3VLVisionTables(t: 1, h: 3, w: 4) }
        #expect(throws: Qwen3VLVisionTables.Failure.self) { try Qwen3VLVisionTables(t: 0, h: 2, w: 2) }
    }
}
