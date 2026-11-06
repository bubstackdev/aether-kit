import Testing
@testable import AetherRuntime
import AetherTestSupport

/// Drives the fake decoder the way the M0 spike's Decoder drives a real graph: states allocated once at capacity
/// and shared by a prefill-chunk binding set and a single-step one; the positions input bound to a growing prefix
/// of one capacity-sized buffer; logits read per step; rewind by moving the position; snapshot/restore.
@Suite("Bindings and GraphBackend")
struct BindingsTests {
    final class Engine {
        let graph: FakeDecoder
        let arena = TensorArena()
        let states: StateSet
        let positions: TensorBuffer
        let chunkTokens: TensorBuffer
        let chunkLogits: TensorBuffer
        let stepTokens: TensorBuffer
        let stepLogits: TensorBuffer
        var chunk: Bindings
        var step: Bindings
        var position = 0
        let capacity: Int
        var logitsRow: [Float] = []

        init(graph: FakeDecoder, capacity: Int, chunkSize: Int) throws {
            self.graph = graph
            self.capacity = capacity
            states = try graph.makeStateSet(["keys", "values"], in: arena) { _ in [capacity] }
            positions = try graph.makeBuffer(for: "position_ids", role: .input, dimensions: [capacity], in: arena)
            try positions.write(Int32.self, count: capacity) { Int32($0) }  // 0..<capacity, written once
            chunkTokens = try graph.makeBuffer(for: "input_ids", role: .input, dimensions: [chunkSize], in: arena)
            chunkLogits = try graph.makeBuffer(for: "logits", role: .output, dimensions: [chunkSize], in: arena)
            stepTokens = try graph.makeBuffer(for: "input_ids", role: .input, dimensions: [1], in: arena)
            stepLogits = try graph.makeBuffer(for: "logits", role: .output, dimensions: [1], in: arena)
            chunk = Bindings(arena: arena)
            try chunk.bindStates(states)
            try chunk.bindInput("input_ids", to: chunkTokens)
            try chunk.bindOutput("logits", to: chunkLogits)
            step = Bindings(arena: arena)
            try step.bindStates(states)
            try step.bindInput("input_ids", to: stepTokens)
            try step.bindOutput("logits", to: stepLogits)
        }

        /// Feeds `tokens` (at most one chunk) and returns the last row of logits.
        func forward(_ tokens: [Int32]) async throws -> [Float] {
            let s = tokens.count
            let total = position + s
            let single = s == 1
            let ids = single ? stepTokens : chunkTokens
            let logits = single ? stepLogits : chunkLogits
            try ids.write(tokens)
            var bindings = single ? step : chunk
            try bindings.bindInput("input_ids", to: ids, extent: [1, s])
            try bindings.bindInput("position_ids", to: positions, extent: [1, total])
            try bindings.bindOutput("logits", to: logits, extent: [1, s, FakeDecoder.vocab])
            try await graph.run(&bindings)
            if single { step = bindings } else { chunk = bindings }
            position = total
            try logits.readFloats((s - 1) * FakeDecoder.vocab..<s * FakeDecoder.vocab, into: &logitsRow)
            return logitsRow
        }
    }

    /// Second review, item 2: a buffer released after binding, whose slot an allocation of another shape and type
    /// reused, still validated against the signature (stale shape and type: wrong storage on the encode path).
    @Test func aReleasedBindingNoLongerValidates() throws {
        let arena = TensorArena()
        let buffer = try arena.allocate(name: "input", shape: [16], scalarType: .float32)
        var bindings = Bindings(arena: arena)
        try bindings.bindInput("input", to: buffer)
        try arena.release(buffer)
        let replacement = try arena.allocate(name: "replacement", shape: [1], scalarType: .int8)
        #expect(replacement.slot == buffer.slot)
        let signature = GraphSignature(function: "main", inputs: [.init(name: "input", scalarType: .float32, shape: [16])],
                                       states: [], outputs: [])
        #expect(throws: AetherError.self) { try bindings.validate(against: signature) }
    }

    @Test func prefillThenStepSharesStateWithoutReallocating() async throws {
        let engine = try Engine(graph: FakeDecoder(), capacity: 16, chunkSize: 4)
        let bytes = engine.arena.byteCount
        #expect(try await engine.forward([1, 2, 3]) == [6, 7, 8, 9])
        #expect(try await engine.forward([10]) == [16, 17, 18, 19])
        #expect(try await engine.forward([100]) == [116, 117, 118, 119])
        #expect(engine.arena.byteCount == bytes, "no buffer was allocated after setup")
        #expect(try engine.states["values"]!.read(Float.self, 0..<5) == [0, 1, 2, 3, 4])
        #expect(engine.graph.runs == 3)
    }

    @Test func rewindOverwritesStaleSlots() async throws {
        let engine = try Engine(graph: FakeDecoder(), capacity: 16, chunkSize: 4)
        _ = try await engine.forward([1, 2, 3, 4])
        _ = try await engine.forward([50])
        engine.position = 2  // keep [1, 2]; slots 2... are stale
        #expect(try await engine.forward([7]) == [10, 11, 12, 13])
    }

    @Test func snapshotRestoreRewindsLikeAHybrid() async throws {
        let engine = try Engine(graph: FakeDecoder(), capacity: 16, chunkSize: 4)
        _ = try await engine.forward([1, 2])
        let snapshot = try engine.states.snapshot()
        let saved = engine.position
        let first = try await engine.forward([5])
        _ = try await engine.forward([9])
        try engine.states.restore(snapshot)
        engine.position = saved
        #expect(try await engine.forward([5]) == first)
    }

    @Test func validationNamesTheProblem() async throws {
        let graph = FakeDecoder()
        let arena = TensorArena()
        let states = try graph.makeStateSet(["keys", "values"], in: arena) { _ in [8] }
        let ids = try graph.makeBuffer(for: "input_ids", role: .input, dimensions: [2], in: arena)
        let positions = try graph.makeBuffer(for: "position_ids", role: .input, dimensions: [2], in: arena)

        var bindings = Bindings(arena: arena)
        try bindings.bindStates(states)
        try bindings.bindInput("input_ids", to: ids)
        await expectTensorFailure("position_ids", .unbound) { try await graph.run(&bindings) }

        try bindings.bindInput("position_ids", to: ids)
        await expectTensorFailure("position_ids", .unsupported("shares its buffer with 'input_ids'; bind distinct buffers")) {
            try await graph.run(&bindings)
        }

        let floats = try arena.allocate(name: "f", shape: [1, 2], scalarType: .float32)
        try bindings.bindInput("position_ids", to: floats)
        await expectTensorFailure("position_ids", .typeMismatch(expected: "int32", found: "float32")) {
            try await graph.run(&bindings)
        }

        let wrongRank = try arena.allocate(name: "w", shape: [2], scalarType: .int32)
        try bindings.bindInput("position_ids", to: wrongRank)
        await expectTensorFailure("position_ids", .shapeMismatch(expected: [1, -1], found: [2])) {
            try await graph.run(&bindings)
        }

        try bindings.bindInput("position_ids", to: positions)
        try bindings.bindInput("attention_mask", to: try arena.allocate(name: "m", shape: [1, 2], scalarType: .int32))
        await expectTensorFailure("attention_mask", .unknownName) { try await graph.run(&bindings) }

        #expect(throws: AetherError.self) { try bindings.bindInput("input_ids", to: ids, extent: [1, 3]) }
        #expect(throws: AetherError.self) { try bindings.bindInput("input_ids", to: ids, extent: [2]) }
        #expect(throws: AetherError.self) {
            try bindings.bindInput("input_ids", to: try TensorArena().allocate(name: "x", shape: [1, 2], scalarType: .int32))
        }
    }

    @Test func staticDimensionsMustMatch() throws {
        let arena = TensorArena()
        let spec = TensorSpec(name: "logits", scalarType: .float16, shape: [1, -1, 151936])
        #expect(spec.accepts([1, 7, 151936]))
        #expect(!spec.accepts([1, 7, 151935]))
        #expect(!spec.accepts([1, 0, 151936]))
        #expect(try spec.resolving([7]) == [1, 7, 151936])
        #expect(throws: AetherError.self) { try spec.resolving([]) }
        #expect(throws: AetherError.self) { try spec.resolving([1, 2]) }
        _ = arena
    }

    func expectTensorFailure(
        _ name: String, _ reason: AetherError.TensorFailure, sourceLocation: SourceLocation = #_sourceLocation,
        _ body: () async throws -> Void
    ) async {
        do {
            try await body()
            Issue.record("expected .tensor(\(name), \(reason))", sourceLocation: sourceLocation)
        } catch {
            let failure = tensorFailure(of: error)
            #expect(failure?.name == name && failure?.reason == reason, "\(error)", sourceLocation: sourceLocation)
        }
    }
}
