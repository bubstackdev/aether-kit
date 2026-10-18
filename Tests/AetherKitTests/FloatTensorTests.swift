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
