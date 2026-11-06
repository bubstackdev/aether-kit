import Foundation
import Testing
@testable import AetherRuntime

/// A block with a weight input: `hidden_out = 2 · hidden + w0` (broadcast over the last axis), on the CPU.
private final class FakeBlock: GraphBackend {
    let signature = GraphSignature(
        function: "block",
        inputs: [TensorSpec(name: "hidden", scalarType: .float32, shape: [1, -1, 4]),
                 TensorSpec(name: "w0", scalarType: .float32, shape: [4])],
        states: [],
        outputs: [TensorSpec(name: "hidden_out", scalarType: .float32, shape: [1, -1, 4])])
    var runs = 0

    func run(_ bindings: inout Bindings) async throws(AetherError) {
        try bindings.validate(against: signature)
        let x = try bindings.input("hidden")!.read(Float.self)
        let w = try bindings.input("w0")!.read(Float.self)
        try bindings.output("hidden_out")!.write(x.enumerated().map { 2 * $0.element + w[$0.offset % 4] })
        runs += 1
        await Task.yield()
    }
}

private func block(_ x: [Float], _ w: [Float]) -> [Float] { x.enumerated().map { 2 * $0.element + w[$0.offset % 4] } }

private func weights(_ block: Int) -> [Float] { (0..<4).map { Float(block * 10 + $0) } }

@Suite("Graph chain")
struct GraphChainTests {
    @Test func callsAlternateSlotsAndMatchTheUnchainedComposition() async throws {
        let graph = FakeBlock()
        let arena = TensorArena()
        let chain = try GraphChain(graph: graph, link: ("hidden", "hidden_out"), dimensions: [2], arena: arena)
        let w = try graph.makeBuffer(for: "w0", role: .input, dimensions: [], in: arena)
        let x0: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        try chain.head.write(x0)
        var heads: [Bool] = []
        try await chain.run { chain in
            for b in 0..<3 {
                try w.write(weights(b))
                try await chain.call(graph) { bindings throws(AetherError) in try bindings.bindInput("w0", to: w) }
                heads.append(chain.head.isSameStorage(as: chain.slots.1))
            }
        }
        #expect(heads == [true, false, true])
        #expect(chain.calls == 3 && graph.runs == 3)
        #expect(try chain.head.read(Float.self) == block(block(block(x0, weights(0)), weights(1)), weights(2)))
        #expect(!chain.isDeviceChained)
    }

    @Test func theLinkIsTheChainsToBind() async throws {
        let graph = FakeBlock()
        let arena = TensorArena()
        let chain = try GraphChain(graph: graph, link: ("hidden", "hidden_out"), dimensions: [2], arena: arena)
        let other = try graph.makeBuffer(for: "hidden", role: .input, dimensions: [2], in: arena)
        let w = try graph.makeBuffer(for: "w0", role: .input, dimensions: [], in: arena)
        await #expect(throws: AetherError.self) {
            try await chain.run { chain in
                try await chain.call(graph) { bindings throws(AetherError) in
                    try bindings.bindInput("w0", to: w)
                    try bindings.bindInput("hidden", to: other)
                }
            }
        }
        // Outside run(_:), a call is refused; two distinct slots are required.
        await #expect(throws: AetherError.self) { try await chain.call(graph) }
        #expect(throws: AetherError.self) {
            _ = try GraphChain(arena: arena, link: ("hidden", "hidden_out"), slots: (other, other))
        }
    }

    @Test @MainActor func cancellationStopsBetweenCallsAndLeavesTheChainUsable() async throws {
        let graph = FakeBlock()
        let arena = TensorArena()
        let chain = try GraphChain(graph: graph, link: ("hidden", "hidden_out"), dimensions: [1], arena: arena)
        let w = try graph.makeBuffer(for: "w0", role: .input, dimensions: [], in: arena)
        let task = Task { @MainActor in
            try await chain.run { chain in
                for _ in 0..<1000 {
                    try await chain.call(graph) { bindings throws(AetherError) in try bindings.bindInput("w0", to: w) }
                }
            }
        }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(graph.runs < 1000)
        try chain.reset()
        try await chain.run { chain in
            try await chain.call(graph) { bindings throws(AetherError) in try bindings.bindInput("w0", to: w) }
        }
        #expect(chain.calls == 1)
    }

    @Test @MainActor func busyRunsQueueInOrder() async throws {
        let gate = SerialGate()
        var order: [Int] = []
        let first = Task { @MainActor in
            try await gate.withTurn {
                order.append(1)
                for _ in 0..<5 { await Task.yield() }
                order.append(2)
            }
        }
        await Task.yield()
        let second = Task { @MainActor in try await gate.withTurn { order.append(3) } }
        try await first.value
        try await second.value
        #expect(order == [1, 2, 3])
        #expect(!gate.isBusy && gate.waiting == 0)
    }

    /// Second code review, batch C item 5: a run cancelled while it waits for the chain's turn leaves the queue at once
    /// (before: it took its turn when it came, then checked).
    @Test @MainActor func aRunCancelledWhileWaitingLeavesAtOnce() async throws {
        let gate = SerialGate()
        var release: CheckedContinuation<Void, Never>?
        let first = Task { @MainActor in try await gate.withTurn { await withCheckedContinuation { release = $0 } } }
        while release == nil { await Task.yield() }
        let second = Task { @MainActor in try await gate.withTurn { Issue.record("a cancelled run ran") } }
        while gate.waiting == 0 { await Task.yield() }
        second.cancel()
        await #expect(throws: CancellationError.self) { try await second.value }
        #expect(gate.waiting == 0 && gate.isBusy)
        release?.resume()
        try await first.value
    }

    // MARK: Weight banks

    @Test func banksLoadBlocksFromBundleFilesAndKeepTheSyncRule() async throws {
        try await withTemporaryDirectory { root in
            let blocks = 4
            var data = Data()
            for b in 0..<blocks { weights(b).withUnsafeBytes { data.append(contentsOf: $0) } }
            try data.write(to: root.appending(path: "weights.bin"))
            let table = try WeightTable(inputs: ["w0"], blocks: (0..<blocks).map { b in
                ["w0": .init(file: "main:weights.bin", offset: Int64(b * 16), bytes: 16)]
            })
            let graph = FakeBlock()
            let arena = TensorArena()
            let chain = try GraphChain(graph: graph, link: ("hidden", "hidden_out"), dimensions: [1], arena: arena)
            let banks = try WeightBanks(table: table, graph: graph, arena: arena) { root.appending(path: $0.path) }
            let x0: [Float] = [1, 0, -1, 2]
            try chain.head.write(x0)
            try await chain.run { chain in
                try banks.load(block: 0, into: 0, chain: chain)
                for b in 0..<blocks {
                    let bank = b % 2
                    try await chain.call(graph) { bindings throws(AetherError) in try banks.bind(bank: bank, into: &bindings) }
                    if b + 1 < blocks { try banks.load(block: b + 1, into: 1 - bank, chain: chain) }
                    // The bank just bound is in flight until the chain synchronizes.
                    #expect(throws: AetherError.self) { try banks.load(block: (b + 1) % blocks, into: bank, chain: chain) }
                    await chain.synchronize()
                }
            }
            var expected = x0
            for b in 0..<blocks { expected = block(expected, weights(b)) }
            #expect(try chain.head.read(Float.self) == expected)
            #expect(banks.loaded == [2, 3])
        }
    }

    /// Review finding 7: a run that ends without a last `synchronize()` has still drained, so a bank may be refilled
    /// before the next run (preloading the next generation's first block); before, it was refused as in flight.
    @Test func aBankMayBeRefilledBetweenRuns() async throws {
        try await withTemporaryDirectory { root in
            var data = Data()
            for b in 0..<2 { weights(b).withUnsafeBytes { data.append(contentsOf: $0) } }
            try data.write(to: root.appending(path: "weights.bin"))
            let table = try WeightTable(inputs: ["w0"], blocks: (0..<2).map { b in
                ["w0": .init(file: "main:weights.bin", offset: Int64(b * 16), bytes: 16)]
            })
            let graph = FakeBlock()
            let arena = TensorArena()
            let chain = try GraphChain(graph: graph, link: ("hidden", "hidden_out"), dimensions: [1], arena: arena)
            let banks = try WeightBanks(table: table, graph: graph, arena: arena) { root.appending(path: $0.path) }
            try chain.head.write([1, 0, -1, 2] as [Float])
            try await chain.run { chain in
                try banks.load(block: 0, into: 0, chain: chain)
                try await chain.call(graph) { bindings throws(AetherError) in try banks.bind(bank: 0, into: &bindings) }
            }
            try banks.load(block: 1, into: 0, chain: chain)
            #expect(banks.loaded[0] == 1)
        }
    }

    @Test func weightTablesAreValidated() throws {
        let extent = WeightTable.Extent(file: "main:w.bin", bytes: 16)
        #expect(throws: AetherError.self) { _ = try WeightTable(inputs: [], blocks: [[:]]) }
        #expect(throws: AetherError.self) { _ = try WeightTable(inputs: ["w0", "w0"], blocks: [["w0": extent]]) }
        #expect(throws: AetherError.self) { _ = try WeightTable(inputs: ["w0", "w1"], blocks: [["w0": extent]]) }
        #expect(throws: AetherError.self) {
            _ = try WeightTable(inputs: ["w0"], blocks: [["w0": .init(file: "no-asset", bytes: 16)]])
        }
        #expect(throws: AetherError.self) {
            _ = try WeightTable(inputs: ["w0"], blocks: [["w0": .init(file: "main:w.bin", offset: -1, bytes: 16)]])
        }
        // JSON decoding validates too.
        let bad = Data(#"{"inputs": ["w0"], "blocks": [{}]}"#.utf8)
        #expect(throws: (any Error).self) { _ = try JSONDecoder().decode(WeightTable.self, from: bad) }
        let good = Data(#"{"inputs": ["w0"], "blocks": [{"w0": {"file": "main:w.bin", "offset": 0, "bytes": 16}}]}"#.utf8)
        #expect(try JSONDecoder().decode(WeightTable.self, from: good).blockCount == 1)
        // An extent must hold exactly the input's bytes, and its file must resolve.
        let graph = FakeBlock()
        let wrong = try WeightTable(inputs: ["w0"], blocks: [["w0": .init(file: "main:w.bin", bytes: 12)]])
        expectField("weights.blocks[0].w0.bytes") {
            _ = try WeightBanks(table: wrong, graph: graph, arena: TensorArena()) { _ in URL(fileURLWithPath: "/") }
        }
        let table = try WeightTable(inputs: ["w0"], blocks: [["w0": extent]])
        expectField("weights.blocks[0].w0.file") {
            _ = try WeightBanks(table: table, graph: graph, arena: TensorArena()) { _ in nil }
        }
    }

    // MARK: Shape profiles

    @Test func profilesRefuseShapesTheyWereNotMeasuredAt() async throws {
        let profile = ShapeProfile(name: "2 rows", shapes: ["body": ["hidden": [[1, 2, 4]]]])
        let graph = FakeBlock()
        let arena = TensorArena()
        let w = try graph.makeBuffer(for: "w0", role: .input, dimensions: [], in: arena)
        let ok = try GraphChain(graph: graph, link: ("hidden", "hidden_out"), dimensions: [2], arena: arena, profile: profile)
        try await ok.run { chain in
            try await chain.call(graph, as: "body") { bindings throws(AetherError) in try bindings.bindInput("w0", to: w) }
        }
        let off = try GraphChain(graph: graph, link: ("hidden", "hidden_out"), dimensions: [3], arena: arena, profile: profile)
        await #expect(throws: AetherError.self) {
            try await off.run { chain in
                try await chain.call(graph, as: "body") { bindings throws(AetherError) in try bindings.bindInput("w0", to: w) }
            }
        }
        await #expect(throws: AetherError.self) {
            try await ok.run { chain in
                try await chain.call(graph, as: "conditioner") { bindings throws(AetherError) in try bindings.bindInput("w0", to: w) }
            }
        }
        #expect(graph.runs == 1)
    }

    @Test func profileIdentitiesAreCanonicalAndKeyTheLedger() {
        let a = ShapeProfile(name: "p", shapes: ["body": ["hidden": [[1, 2, 4], [1, 8, 4]], "mod": [[2, 4]]]])
        let b = ShapeProfile(name: "p", shapes: ["body": ["mod": [[2, 4]], "hidden": [[1, 8, 4], [1, 2, 4]]]])
        let c = ShapeProfile(name: "p", shapes: ["body": ["hidden": [[1, 2, 4]], "mod": [[2, 4]]]])
        #expect(a.identity == b.identity)
        #expect(a.identity != c.identity)
        #expect(a.identity == "p:body{hidden=1x2x4|1x8x4,mod=2x4}")
        let ledger = FootprintLedger()
        ledger.record(a.ledgerKey(model: "h3", phase: .perStep), PhaseFootprint(peakBytes: 10, heldBytes: 5))
        #expect(ledger.footprint(b.ledgerKey(model: "h3", phase: .perStep))?.peakBytes == 10)
        #expect(ledger.footprint(c.ledgerKey(model: "h3", phase: .perStep)) == nil)
    }
}
