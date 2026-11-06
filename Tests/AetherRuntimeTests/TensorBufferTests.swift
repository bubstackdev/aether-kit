import Testing
@testable import AetherRuntime

@Suite("TensorBuffer")
struct TensorBufferTests {
    @Test func allocatesZeroedAndRoundTripsTypedValues() throws {
        let arena = TensorArena()
        let ids = try arena.allocate(name: "ids", shape: [1, 6], scalarType: .int32)
        #expect(try ids.read(Int32.self) == [0, 0, 0, 0, 0, 0])
        try ids.write([7, 8, 9] as [Int32], startingAt: 2)
        #expect(try ids.read(Int32.self) == [0, 0, 7, 8, 9, 0])
        try ids.write(Int32.self, count: 6) { Int32($0 * 10) }
        #expect(try ids.read(Int32.self, 1..<4) == [10, 20, 30])
        try ids.fill(Int32(-1))
        #expect(try ids.read(Int32.self).allSatisfy { $0 == -1 })
        #expect(ids.storageByteCount == 24 && arena.byteCount == 24)
    }

    @Test func paddedStridesAreRespected() throws {
        let arena = TensorArena()
        // 3 rows of 5 elements, each row padded to 8.
        let padded = try arena.allocate(name: "padded", shape: [3, 5], scalarType: .float16, strides: [8, 1])
        #expect(!padded.isContiguous)
        #expect(padded.storageByteCount == (2 * 8 + 4 + 1) * 2)
        #expect(padded.layout.offset(of: 7) == 10)
        try padded.write((0..<15).map { Float16($0) })
        #expect(try padded.read(Float16.self) == (0..<15).map { Float16($0) })
        #expect(try padded.readFloats(4..<9) == [4, 5, 6, 7, 8])
        // The padding stays untouched (zero) underneath.
        let raw = try arena.withMutableBytes(padded) { Array($0.bindMemory(to: Float16.self)) }
        #expect(raw[5] == 0 && raw[6] == 0 && raw[7] == 0 && raw[8] == 5)
    }

    @Test func readsFloatsFromEveryFloatType() throws {
        let arena = TensorArena()
        let f32 = try arena.allocate(name: "f32", shape: [4], scalarType: .float32)
        try f32.write([1.5, -2, 3, 4] as [Float])
        #expect(try f32.readFloats() == [1.5, -2, 3, 4])

        let bf16 = try arena.allocate(name: "bf16", shape: [3], scalarType: .bfloat16)
        let words = [Float(1), -2.5, 128].map { UInt16($0.bitPattern >> 16) }
        try arena.withMutableBytes(bf16) { bytes in
            for (i, w) in words.enumerated() { bytes.storeBytes(of: w, toByteOffset: 2 * i, as: UInt16.self) }
        }
        #expect(try bf16.readFloats() == [1, -2.5, 128])

        // The caller's array is reused, not reallocated, when it is large enough.
        var out = [Float]()
        out.reserveCapacity(64)
        let capacity = out.capacity
        try f32.readFloats(0..<4, into: &out)
        try f32.readFloats(1..<3, into: &out)
        #expect(out == [-2, 3] && out.capacity == capacity)
    }

    @Test func misuseIsReportedNotTrapped() throws {
        let arena = TensorArena()
        let buffer = try arena.allocate(name: "b", shape: [2, 2], scalarType: .float16)
        #expect(tensorFailure(of: Result { try buffer.read(Int32.self) }.error!)?.reason
                == .typeMismatch(expected: "float16", found: "int32"))
        #expect(tensorFailure(of: Result { try buffer.write([1, 2, 3] as [Float16], startingAt: 2) }.error!)?.reason
                == .outOfBounds(index: 5, count: 4))
        #expect(throws: AetherError.self) { try arena.allocate(name: "x", shape: [2, 0], scalarType: .float16) }
        #expect(throws: AetherError.self) { try arena.allocate(name: "x", shape: [2], scalarType: "int4") }

        let other = TensorArena()
        #expect(throws: AetherError.self) { try other.copyBytes(from: buffer, to: buffer) }

        try arena.release(buffer)
        #expect(arena.byteCount == 0)
        guard case .unavailable? = tensorFailure(of: Result { try buffer.zero() }.error!)?.reason else {
            Issue.record("a released handle must be unusable"); return
        }
        // The slot is reused by the next allocation, and the old handle stays invalid.
        let next = try arena.allocate(name: "n", shape: [1], scalarType: .int8)
        #expect(next.slot == buffer.slot)
        #expect(throws: AetherError.self) { try buffer.read(Float16.self) }
    }

    @Test func storageIsUnavailableWhileLentToAGraph() throws {
        let arena = TensorArena()
        let buffer = try arena.allocate(name: "logits", shape: [4], scalarType: .float32)
        try arena.beginRun("main")
        #expect(tensorFailure(of: Result { try buffer.readFloats() }.error!)?.reason == .unavailable("lent to the running graph 'main'"))
        #expect(throws: AetherError.self) { try arena.allocate(name: "x", shape: [1], scalarType: .float32) }
        #expect(throws: AetherError.self) { try arena.beginRun("again") }
        arena.endRun()
        #expect(try buffer.readFloats() == [0, 0, 0, 0])
    }

    @Test func copiesBetweenBuffersOfOneLayout() throws {
        let arena = TensorArena()
        let a = try arena.allocate(name: "a", shape: [2, 3], scalarType: .int32)
        let b = try arena.allocate(name: "b", shape: [2, 3], scalarType: .int32)
        try a.write((1...6).map(Int32.init))
        try b.copy(from: a)
        #expect(try b.read(Int32.self) == [1, 2, 3, 4, 5, 6])
        let c = try arena.allocate(name: "c", shape: [3, 2], scalarType: .int32)
        #expect(throws: AetherError.self) { try c.copy(from: a) }
    }
}

extension Result where Failure == any Error {
    init(_ body: () throws -> Success) {
        do { self = .success(try body()) } catch { self = .failure(error) }
    }

    var error: (any Error)? {
        if case .failure(let error) = self { return error }
        return nil
    }
}
