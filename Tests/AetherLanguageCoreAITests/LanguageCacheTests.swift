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
        // tokens are materialized. Stale third-token values must not be copied, and
        // whatever the new buffer held before is overwritten.
        let old: [UInt16] = [10, 11, 12, 13, 99, 99, 20, 21, 22, 23, 99, 99]
        var new = [UInt16](repeating: 7777, count: 2 * 6 * 2 + 3)
        old.withUnsafeBytes { source in
            new.withUnsafeMutableBytes { target in
                LanguageCachePlan.relayout(from: source.baseAddress!, to: target.baseAddress!, destinationBytes: target.count, rows: 2,
                    oldCapacity: 3, newCapacity: 6, validTokens: 2, tokenBytes: 4)
            }
        }
        #expect(Array(new[0..<4]) == [10, 11, 12, 13])
        #expect(Array(new[12..<16]) == [20, 21, 22, 23])
        #expect(new[4..<12].allSatisfy { $0 == 0 })
        #expect(new[16...].allSatisfy { $0 == 0 }, "the tail past the last row is cleared too")
    }

    /// The one-pass relayout writes exactly the bytes of zeroing the new buffer and copying each row's prefix, in the
    /// serial path and the parallel one (8 MiB and up).
    @Test func relayoutEqualsZeroingThenCopying() {
        var seed: UInt64 = 3
        func next() -> UInt8 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return UInt8(truncatingIfNeeded: seed >> 33) }
        for (rows, old, new, valid, tokenBytes) in [(3, 4, 8, 2, 6), (1, 5, 5, 5, 2), (2, 4, 8, 0, 3), (7, 256, 512, 200, 256),
                                                    (224, 256, 512, 256, 256), (224, 256, 512, 0, 256)] {
            let source = (0..<(rows * old * tokenBytes)).map { _ in next() }
            var reference = [UInt8](repeating: 0, count: rows * new * tokenBytes + 5)
            for row in 0..<rows {
                for byte in 0..<(valid * tokenBytes) { reference[row * new * tokenBytes + byte] = source[row * old * tokenBytes + byte] }
            }
            var actual = (0..<reference.count).map { _ in next() }
            source.withUnsafeBytes { from in
                actual.withUnsafeMutableBytes { to in
                    LanguageCachePlan.relayout(from: from.baseAddress!, to: to.baseAddress!, destinationBytes: to.count, rows: rows,
                                               oldCapacity: old, newCapacity: new, validTokens: valid, tokenBytes: tokenBytes)
                }
            }
            #expect(actual == reference, "rows \(rows), \(old) -> \(new), valid \(valid)")
        }
    }

    @Test func paddedAndPermutedCachesPreserveUpdatesAndGrowth() {
        let shape = [2, 1, 2, 3, 3]
        let sourceStrides = [30, 30, 15, 5, 1]
        let nextShape = [2, 1, 2, 6, 3]
        // A noncontiguous channel axis as well as token padding; different layout after growth.
        let nextStrides = [96, 96, 48, 8, 2]
        var old = [Float16](repeating: -99, count: 60)
        let update: [Float16] = (0..<24).map { Float16($0 + 1) }
        update.withUnsafeBufferPointer { u in
            old.withUnsafeMutableBufferPointer { c in
                LanguageCachePlan.writeUpdateTensor(u.baseAddress!, shape: [2, 1, 2, 2, 3],
                    strides: [12, 12, 6, 3, 1], into: c.baseAddress!, capacity: 3, start: 0, cacheStrides: sourceStrides)
            }
        }
        var grown = [Float16](repeating: 777, count: 192)
        old.withUnsafeBufferPointer { source in
            grown.withUnsafeMutableBufferPointer { destination in
                LanguageCachePlan.relayoutStrided(from: source.baseAddress!, to: destination.baseAddress!,
                    destinationBytes: destination.count * MemoryLayout<Float16>.size, shape: nextShape,
                    sourceStrides: sourceStrides, destinationStrides: nextStrides, validTokens: 2)
            }
        }
        var expected = [Float16](repeating: 0, count: grown.count)
        for l in 0..<shape[0] { for h in 0..<shape[2] { for t in 0..<2 { for e in 0..<shape[4] {
            expected[l * nextStrides[0] + h * nextStrides[2] + t * nextStrides[3] + e * nextStrides[4]] = update[l * 12 + h * 6 + t * 3 + e]
        } } } }
        #expect(grown == expected, "valid tokens survive; stale rows and all padding are cleared")
    }

    /// Contiguous update rows take one copy per row; the result equals the per-token placement, including at the
    /// cache's first and last rows and for one-token (decode) updates with a padded token stride.
    @Test func contiguousUpdateRowsMatchPerTokenPlacement() {
        let rows = 6, capacity = 40, width = 8
        for (start, count, tokenStride) in [(13, 9, width), (0, 9, width), (31, 9, width), (39, 1, width + 5), (0, 1, width + 5)] {
            let update = (0..<(rows * count * tokenStride)).map { Float16(Float($0 % 97) / 7) }
            var fast = [Float16](repeating: 1, count: rows * capacity * width)
            var reference = fast
            update.withUnsafeBufferPointer { source in
                fast.withUnsafeMutableBufferPointer { cache in
                    LanguageCachePlan.writeUpdate(source.baseAddress!, rowStride: count * tokenStride, tokenStride: tokenStride,
                                                  elementStride: 1, into: cache.baseAddress!, rows: rows, capacity: capacity,
                                                  width: width, start: start, count: count)
                }
            }
            for row in 0..<rows { for token in 0..<count { for element in 0..<width {
                reference[(row * capacity + start + token) * width + element] = update[row * count * tokenStride + token * tokenStride + element]
            } } }
            #expect(fast == reference, "start \(start), count \(count), token stride \(tokenStride)")
        }
    }
}
