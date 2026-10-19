import AetherKit
import CoreAI
import Testing
@testable import AetherCoreAI

@Test func nativeInputCopiesPreserveScalarBitsAndOwnTheirStorage() throws {
    var floats: [Float] = [0, -0.0, .leastNonzeroMagnitude, .greatestFiniteMagnitude, -3.5, 7]
    let floatArray = TensorValue.float32(try FloatTensor(shape: [2, 3], values: floats)).nativeArray
    let expectedFloatBits = floats.map(\.bitPattern)
    floats[5] = 99
    #expect(floatArray.shape == [2, 3])
    floatArray.view(as: Float.self).withUnsafePointer { pointer, _, _ in
        #expect((0..<6).map { pointer[$0].bitPattern } == expectedFloatBits)
    }

    let halves: [Float16] = [0, -0.0, .leastNonzeroMagnitude, .greatestFiniteMagnitude, -3.5, 7]
    let halfArray = TensorValue.float16(try Float16Tensor(shape: [2, 3], values: halves)).nativeArray
    halfArray.view(as: Float16.self).withUnsafePointer { pointer, _, _ in
        #expect((0..<6).map { pointer[$0].bitPattern } == halves.map(\.bitPattern))
    }

    let ints: [Int32] = [.min, -1, 0, .max]
    let intArray = TensorValue.int32(try Int32Tensor(shape: [2, 2], values: ints)).nativeArray
    intArray.view(as: Int32.self).withUnsafePointer { pointer, _, _ in
        #expect(Array(UnsafeBufferPointer(start: pointer, count: 4)) == ints)
    }
}
