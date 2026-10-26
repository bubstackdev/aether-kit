@testable import AetherLanguageCoreAI
import Foundation
import Testing

struct LanguageCacheTests {
    @Test func growthAdmitsLargeBudgetsWithoutEagerAllocation() throws {
        #expect(try LanguageCachePlan.nextCapacity(current: 256, needed: 257, maximum: 16384) == 512)
        #expect(try LanguageCachePlan.nextCapacity(current: 256, needed: 8193, maximum: 16384) == 9216)
        #expect(try LanguageCachePlan.nextCapacity(current: 4096, needed: 4097, maximum: 16384) == 5120)
        #expect(try LanguageCachePlan.nextCapacity(current: 256, needed: 256, maximum: 16384) == 256)
        #expect(throws: LanguageGenerationError.self) { try LanguageCachePlan.nextCapacity(current: 256, needed: Int.max, maximum: 16384) }
    }
    @Test func growPreservesEachLayerHeadPrefixAndLeavesNewTailClear() {
        // Two rows, three tokens per row, two UInt16 channels per token. Only two
        // tokens are materialized. Stale third-token values must not be copied.
        let old: [UInt16] = [10, 11, 12, 13, 99, 99, 20, 21, 22, 23, 99, 99]
        var new = [UInt16](repeating: 0, count: 2 * 6 * 2)
        old.withUnsafeBytes { source in
            new.withUnsafeMutableBytes { target in
                LanguageCachePlan.copyPrefix(from: source.baseAddress!, to: target.baseAddress!, rows: 2,
                    oldCapacity: 3, newCapacity: 6, validTokens: 2, tokenBytes: 4)
            }
        }
        #expect(Array(new[0..<4]) == [10, 11, 12, 13])
        #expect(Array(new[12..<16]) == [20, 21, 22, 23])
        #expect(new[4..<12].allSatisfy { $0 == 0 })
        #expect(new[16..<24].allSatisfy { $0 == 0 })
    }
}
