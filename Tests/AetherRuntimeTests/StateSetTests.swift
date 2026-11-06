import Testing
@testable import AetherRuntime

@Suite("StateSet")
struct StateSetTests {
    static func makeSet(_ arena: TensorArena) throws -> StateSet {
        let kv = try arena.allocate(name: "kv", shape: [2, 1, 4, 3], scalarType: .float16)
        let rec = try arena.allocate(name: "rec", shape: [5], scalarType: .float32, strides: [2])
        return try StateSet(arena: arena, entries: [("kv", kv), ("rec", rec)])
    }

    @Test func snapshotAndRestoreAreExact() throws {
        let arena = TensorArena()
        let states = try Self.makeSet(arena)
        let kv = try #require(states["kv"])
        let rec = try #require(states["rec"])
        try kv.write((0..<24).map { Float16($0) * 0.5 })
        try rec.write([1, 2, 3, 4, 5] as [Float])

        let snapshot = try states.snapshot()
        #expect(snapshot.byteCount == states.byteCount)
        #expect(states.byteCount == 24 * 2 + 9 * 4)

        try kv.fill(Float16(9))
        try rec.write([-1] as [Float], startingAt: 4)
        try states.restore(snapshot)
        #expect(try kv.read(Float16.self) == (0..<24).map { Float16($0) * 0.5 })
        #expect(try rec.read(Float.self) == [1, 2, 3, 4, 5])

        try states.zero()
        #expect(try kv.readFloats().allSatisfy { $0 == 0 })
        #expect(try rec.readFloats().allSatisfy { $0 == 0 })
        try states.restore(snapshot)
        #expect(try rec.read(Float.self) == [1, 2, 3, 4, 5])
    }

    @Test func snapshotIntoReusesStorage() throws {
        let arena = TensorArena()
        let states = try Self.makeSet(arena)
        var snapshot = try states.snapshot()
        let identity = snapshot.entries.map { $0.bytes.withUnsafeBufferPointer { $0.baseAddress } }
        try states["rec"]!.write([7] as [Float])
        try states.snapshot(into: &snapshot)
        #expect(snapshot.entries.map { $0.bytes.withUnsafeBufferPointer { $0.baseAddress } } == identity)
        try states.zero()
        try states.restore(snapshot)
        #expect(try states["rec"]!.read(Float.self, 0..<1) == [7])
    }

    @Test func restoreRejectsAForeignSnapshot() throws {
        let arena = TensorArena()
        let states = try Self.makeSet(arena)
        let other = try StateSet(arena: arena, entries: [
            ("kv", try arena.allocate(name: "kv", shape: [2, 1, 8, 3], scalarType: .float16)),
            ("rec", try arena.allocate(name: "rec", shape: [5], scalarType: .float32, strides: [2])),
        ])
        #expect(throws: AetherError.self) { try states.restore(other.snapshot()) }
        let single = try StateSet(arena: arena, entries: [("kv", states["kv"]!)])
        #expect(throws: AetherError.self) { try states.restore(single.snapshot()) }
    }

    @Test func namesAreUniqueAndBuffersBelongToTheArena() throws {
        let arena = TensorArena()
        let a = try arena.allocate(name: "a", shape: [1], scalarType: .float32)
        #expect(throws: AetherError.self) { try StateSet(arena: arena, entries: [("a", a), ("a", a)]) }
        #expect(throws: AetherError.self) { try StateSet(arena: TensorArena(), entries: [("a", a)]) }
    }
}
