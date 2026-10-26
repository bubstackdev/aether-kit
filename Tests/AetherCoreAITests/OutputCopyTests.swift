import AetherKit
import CoreAI
import Testing
@testable import AetherCoreAI

@Test func nativeOutputBulkCopiesPreserveBitsAndIndependentOwnership() throws {
    let floats: [Float] = [0, -0.0, .leastNonzeroMagnitude, .greatestFiniteMagnitude, -3.5, 7]
    var array = NDArray(scalars: floats, shape: [2, 3])
    let output = try copyTensor(array, as: Float.self)
    array.mutableRawView().withUnsafeMutableBytes { pointer, _, _ in
        pointer.assumingMemoryBound(to: Float.self)[5] = 99
    }
    #expect(output.shape == [2, 3])
    #expect(output.values.map(\.bitPattern) == floats.map(\.bitPattern))
    let halves: [Float16] = [0, -0.0, .leastNonzeroMagnitude, .greatestFiniteMagnitude, -3.5, 7]
    let halfOutput = try copyTensor(NDArray(scalars: halves, shape: [2, 3]), as: Float16.self)
    #expect(halfOutput.values.map(\.bitPattern) == halves.map(\.bitPattern))
    let ints: [Int32] = [.min, -1, 0, .max]
    #expect(try copyTensor(NDArray(scalars: ints, shape: [2, 2]), as: Int32.self).values == ints)
}

@Test func nativeOutputBulkCopiesStillRejectNonfiniteValues() throws {
    for value: Float in [.infinity, -.infinity, .nan] {
        #expect(throws: AetherError.self) {
            try copyTensor(NDArray(scalars: [value], shape: [1]), as: Float.self)
        }
    }
    #expect(throws: AetherError.self) {
        try copyTensor(NDArray(scalars: [Float16.infinity], shape: [1]), as: Float16.self)
    }
}
