import Testing
@testable import AetherKit

@Test func rejectsMalformedTensorsBeforeNativeAllocation() throws {
    for shape in [[0], [-1, 2], [Int.max, 2], [Int.max / 2]] {
        #expect(throws: AetherError.self) { try FloatTensor.elementCount(for: shape) }
    }
    #expect(throws: AetherError.self) { try FloatTensor(shape: [2, 2], values: [1]) }
    #expect(throws: AetherError.self) { try FloatTensor(shape: [1], values: [.nan]) }
    #expect(throws: AetherError.self) { try FloatTensor(shape: [1], values: [.infinity]) }
    #expect(try FloatTensor(shape: [], values: [3]).values == [3])
}

@Test func specializedTensorValidationChecksEveryFloatingPointValue() throws {
    var floats = [Float](repeating: 0.25, count: 4096)
    var halves = [Float16](repeating: 0.25, count: 4096)
    #expect(try FloatTensor(shape: [64, 64], values: floats).values == floats)
    #expect(try Float16Tensor(shape: [64, 64], values: halves).values == halves)
    for invalid: Float in [.nan, .infinity, -.infinity] {
        floats[floats.count - 1] = invalid
        halves[halves.count - 1] = Float16(invalid)
        #expect(throws: AetherError.self) { try FloatTensor(shape: [64, 64], values: floats) }
        #expect(throws: AetherError.self) { try Float16Tensor(shape: [64, 64], values: halves) }
    }
    #expect(try Int32Tensor(shape: [2], values: [.min, .max]).values == [.min, .max])
}
