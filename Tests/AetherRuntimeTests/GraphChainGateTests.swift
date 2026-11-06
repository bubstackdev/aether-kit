import Foundation
import Testing
@testable import AetherRuntime

extension GateTests {
    /// The device chain and weight banks on the GPU (DESIGN.md §6.6), with the tiny H3-shaped block of
    /// `Fixtures/chain` (weights as inputs; `Forge/chain/export_chain_fixture.py`):
    /// - chained, banked calls give the same bytes as the unchained path (`run()` per call, through host memory);
    /// - so do calls encoded back to back with one wait;
    /// - the graph specialized one shape, and a shape outside the chain's profile is refused before it runs.
    @Test func deviceChainMatchesTheUnchainedPath() async throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appending(path: "Fixtures/chain/lifted-block.aimodel")
        let graph = try await Graph(
            url: url, function: "block",
            options: GraphOptions(computeUnit: .gpu, expectFrequentReshapes: true, cachePolicy: .default, cache: .default))
        let rows = 256, dim = 64, ffn = 128, blocks = 6
        let link = (input: "hidden", output: "hidden_out")
        let inputs = ["mod", "w0", "w1", "w2"]
        func noise(_ seed: UInt64, _ count: Int, _ scale: Float) throws -> [Float16] {
            try SeededGaussianNoise.generate(seed: seed, count: count).map { Float16($0 * scale) }
        }
        let hidden = try noise(1, rows * dim, 1)
        let perBlock: [[String: [Float16]]] = try (0..<blocks).map { b in
            let s = UInt64(100 + 10 * b)
            return ["mod": try noise(s, 2 * dim, 0.5), "w0": try noise(s + 1, ffn * dim, 1 / Float(dim).squareRoot()),
                    "w1": try noise(s + 2, ffn * dim, 1 / Float(dim).squareRoot()),
                    "w2": try noise(s + 3, dim * ffn, 1 / Float(ffn).squareRoot())]
        }

        // A. Unchained: one run() per block, the hidden state copied back through the host each time.
        func unchained(_ count: Int) async throws -> [Float16] {
            let arena = TensorArena()
            var buffers: [String: TensorBuffer] = [:]
            for name in inputs { buffers[name] = try graph.makeBuffer(for: name, role: .input, dimensions: [], in: arena) }
            let x = try graph.makeBuffer(for: link.input, role: .input, dimensions: [rows], in: arena)
            let y = try graph.makeBuffer(for: link.output, role: .output, dimensions: [rows], in: arena)
            try x.write(hidden)
            for b in 0..<count {
                for name in inputs { try buffers[name]!.write(perBlock[b][name]!) }
                var bindings = Bindings(arena: arena)
                try bindings.bindInput(link.input, to: x)
                for name in inputs { try bindings.bindInput(name, to: buffers[name]!) }
                try bindings.bindOutput(link.output, to: y)
                try await graph.run(&bindings)
                try x.write(try y.read(Float16.self))
            }
            return try x.read(Float16.self)
        }

        try await withTemporaryDirectory { root in
            // The weights as one bundle file, block-major, with its table.
            var data = Data()
            var table: [[String: WeightTable.Extent]] = []
            for b in 0..<blocks {
                var entry: [String: WeightTable.Extent] = [:]
                for name in inputs {
                    let bytes = perBlock[b][name]!.withUnsafeBytes { Data($0) }
                    entry[name] = .init(file: "main:weights.bin", offset: Int64(data.count), bytes: Int64(bytes.count))
                    data.append(bytes)
                }
                table.append(entry)
            }
            try data.write(to: root.appending(path: "weights.bin"))
            let weights = try WeightTable(inputs: inputs, blocks: table)
            let profile = ShapeProfile(name: "\(rows) rows", shapes: ["block": [link.input: [[1, rows, dim]]]])

            guard let arena = TensorArena.metal() else { Issue.record("no GPU"); return }
            let chain = try GraphChain(graph: graph, link: link, dimensions: [rows], arena: arena, profile: profile)
            let banks = try WeightBanks(table: weights, graph: graph, arena: arena) { root.appending(path: $0.path) }

            // B. Device chain, the prototype's order: encode block i, load block i + 1 into the other bank while the
            // GPU runs, then synchronize.
            func chained() async throws -> [Float16] {
                try chain.reset()
                try chain.head.write(hidden)
                try await chain.run { chain in
                    try banks.load(block: 0, into: 0, chain: chain)
                    for b in 0..<blocks {
                        try await chain.call(graph, as: "block") { bindings throws(AetherError) in
                            try banks.bind(bank: b % 2, into: &bindings)
                        }
                        if b + 1 < blocks { try banks.load(block: b + 1, into: (b + 1) % 2, chain: chain) }
                        await chain.synchronize()
                    }
                }
                return try chain.head.read(Float16.self)
            }

            let reference = try await unchained(blocks)
            let device = try await chained()
            #expect(device == reference, "chained bytes differ from the unchained path")

            // C. Two blocks encoded back to back (both banks loaded first), one wait at the end.
            try chain.reset()
            try chain.head.write(hidden)
            try await chain.run { chain in
                try banks.load(block: 0, into: 0, chain: chain)
                try banks.load(block: 1, into: 1, chain: chain)
                for b in 0..<2 {
                    try await chain.call(graph, as: "block") { bindings throws(AetherError) in
                        try banks.bind(bank: b, into: &bindings)
                    }
                }
            }
            #expect(try chain.head.read(Float16.self) == (try await unchained(2)), "back-to-back bytes differ")

            // Timings (median of 5).
            func median(_ body: () async throws -> Void) async throws -> Double {
                var times: [Double] = []
                for _ in 0..<5 {
                    let start = ContinuousClock.now
                    try await body()
                    times.append(Double((ContinuousClock.now - start).components.attoseconds) / 1e15
                                 + Double((ContinuousClock.now - start).components.seconds) * 1e3)
                }
                return times.sorted()[2]
            }
            let a = try await median { _ = try await unchained(blocks) }
            let b = try await median { _ = try await chained() }
            let finite = reference.allSatisfy { $0.isFinite }
            print("gate: chain \(blocks) blocks × \(rows) rows: unchained \(String(format: "%.2f", a)) ms, "
                  + "device chain + banks \(String(format: "%.2f", b)) ms; bytes identical; finite \(finite)")
            #expect(finite)

            // One shape specialized; a shape outside the profile is refused before it runs.
            #expect(graph.distinctShapeCount == 1)
            let other = try GraphChain(graph: graph, link: link, dimensions: [rows / 2], arena: arena, profile: profile)
            await #expect(throws: AetherError.self) {
                try await other.run { chain in
                    try await chain.call(graph, as: "block") { bindings throws(AetherError) in
                        try banks.bind(bank: 0, into: &bindings)
                    }
                }
            }
            #expect(graph.distinctShapeCount == 1)
        }
    }
}
