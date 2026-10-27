@testable import AetherLanguageCoreAI
import Foundation
import Testing

struct LanguageLogitsRowTests {
    /// The fast conversion equals `Float(Float16)` on every fp16 bit pattern (NaN stays NaN), so greedy and sampling see
    /// exactly the values they saw before.
    @Test func conversionIsExactOnEveryBitPattern() {
        let halves = (0...UInt16.max).map { Float16(bitPattern: $0) }
        let converted = halves.withUnsafeBufferPointer {
            LanguageLogitsRow.lastRow($0.baseAddress!, rows: 1, rowStride: halves.count, count: halves.count, stride: 1)
        }
        #expect(converted.count == halves.count)
        let mismatches = zip(halves, converted).filter { half, value in
            half.isNaN ? !value.isNaN : value.bitPattern != Float(half).bitPattern
        }
        #expect(mismatches.isEmpty, "first mismatches: \(mismatches.prefix(4).map { ($0.0.bitPattern, $0.1) })")
    }

    @Test func picksTheLastRowAndHonoursStrides() {
        let rows: [Float16] = [1, 2, 3, 4, 5, 6, 7, 8, 9]   // three rows of three
        rows.withUnsafeBufferPointer { pointer in
            #expect(LanguageLogitsRow.lastRow(pointer.baseAddress!, rows: 3, rowStride: 3, count: 3, stride: 1) == [7, 8, 9])
            #expect(LanguageLogitsRow.lastRow(pointer.baseAddress!, rows: 1, rowStride: 9, count: 3, stride: 3) == [1, 4, 7])
            #expect(LanguageLogitsRow.lastRow(pointer.baseAddress!, rows: 2, rowStride: 1, count: 4, stride: 2) == [2, 4, 6, 8])
        }
    }
}
