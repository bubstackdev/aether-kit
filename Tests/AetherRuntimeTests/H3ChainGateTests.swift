import Foundation
import Testing
@testable import AetherCore
@testable import AetherRuntime

extension GateTests {
    /// The H3 export's own transformer-block graph (weights as inputs, Turbo LoRA slots included; `Forge/h3`) through
    /// `GraphChain` and `WeightBanks` on the GPU, against the CPU reference `Forge/h3/reference_chain_h3.py` wrote:
    /// - after each chained block, the output within 0.02 relative L2 of the CPU's (the 0.0.x per-graph limit); the
    ///   update (output − input) is recorded, not gated: with outputs this close to their inputs, bf16's rounding of
    ///   the output dominates it (about 0.04 a block);
    /// - the chained, banked bytes equal the unchained path's (`run()` per block through host memory);
    /// - one shape specialized for the profile.
    /// The export is `export_h3.py --random` (seeded random weights in the real shapes: the gate proves the runtime on
    /// the real graph, not the model) or a real export (AETHER_H3_CHAIN=<export dir>, after reference_chain_h3.py:
    /// 0.0093–0.0167 over 4 real blocks from seeded inputs). Default `../forge-outputs/minimax-h3-chain`.
    @Test(.enabled(if: FileManager.default.fileExists(atPath: Self.h3Chain.appending(path: "chain-reference/spec.json").path)))
    func h3BlocksChainOnTheGPU() async throws {
        struct Spec: Decodable {
            struct Tensor: Decodable { let dtype: String; let shape: [Int] }
            let rows: Int
            let blocks: Int
            let tensors: [String: Tensor]
        }
        let root = Self.h3Chain, bundle = root.appending(path: "bundle"), reference = root.appending(path: "chain-reference")
        let spec = try JSONDecoder().decode(Spec.self, from: Data(contentsOf: reference.appending(path: "spec.json")))
        func raw<T: BitwiseCopyable>(_ name: String, _: T.Type) throws -> [T] {
            let data = try Data(contentsOf: reference.appending(path: "\(name).raw"))
            return data.withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
        }
        let rows = spec.rows, hidden = 5376
        let graph = try await Graph(
            url: bundle.appending(path: "models/body.aimodel"), function: "main",
            options: GraphOptions(computeUnit: .gpu, expectFrequentReshapes: true, cachePolicy: .default, cache: .default))
        let table = try WeightTable(contentsOf: bundle.appending(path: "weights/body.json"))
        let blocks = min(spec.blocks, table.blockCount)
        let link = (input: "hidden", output: "hidden_out")
        // bf16 travels as its bit pattern.
        let x0 = try raw("hidden", UInt16.self), modulation = try raw("table", UInt16.self)
        let tags = try raw("tags", Int32.self), cos = try raw("cos", Float.self), sin = try raw("sin", Float.self)
        let expected = try (0..<blocks).map { try raw("block\($0)", UInt16.self) }
        func floats(_ bits: [UInt16]) -> [Double] { bits.map { Double(Float(bitPattern: UInt32($0) << 16)) } }
        func relativeL2(_ a: [Double], _ b: [Double]) -> Double {
            var d = 0.0, n = 0.0
            for i in a.indices { d += (a[i] - b[i]) * (a[i] - b[i]); n += b[i] * b[i] }
            return (d / max(n, 1e-30)).squareRoot()
        }
        let profile = ShapeProfile(name: "\(rows) rows", shapes: ["body": [link.input: [[1, rows, hidden]]]])
        let fileURL: (AssetPath) -> URL? = { bundle.appending(path: $0.path) }

        func fixed(_ arena: TensorArena) throws -> [String: TensorBuffer] {
            let buffers = [
                "table": try graph.makeBuffer(for: "table", role: .input, dimensions: [6], in: arena),
                "tags": try graph.makeBuffer(for: "tags", role: .input, dimensions: [rows], in: arena),
                "cos": try graph.makeBuffer(for: "cos", role: .input, dimensions: [rows], in: arena),
                "sin": try graph.makeBuffer(for: "sin", role: .input, dimensions: [rows], in: arena),
            ]
            try buffers["table"]!.writeBits(modulation)
            try buffers["tags"]!.write(tags)
            try buffers["cos"]!.write(cos)
            try buffers["sin"]!.write(sin)
            return buffers
        }

        // The device chain on a Metal arena (calls encoded on one stream), and the unchained path on a host arena (a
        // run() per call, through host memory), in the prototype's order: encode block i, load block i + 1 into the
        // other bank while the GPU runs it, synchronize.
        struct Rig { let chain: GraphChain; let banks: WeightBanks; let inputs: [String: TensorBuffer] }
        func rig(_ arena: TensorArena) throws -> Rig {
            Rig(chain: try GraphChain(graph: graph, link: link, dimensions: [rows], arena: arena, profile: profile),
                banks: try WeightBanks(table: table, graph: graph, arena: arena, resolve: fileURL), inputs: try fixed(arena))
        }
        func run(_ rig: Rig, _ count: Int) async throws -> [Float] {
            try rig.chain.reset()
            try rig.chain.head.writeBits(x0)
            try await rig.chain.run { chain in
                try rig.banks.load(block: 0, into: 0, chain: chain)
                for b in 0..<count {
                    try await chain.call(graph, as: "body") { bindings throws(AetherError) in
                        for (name, buffer) in rig.inputs { try bindings.bindInput(name, to: buffer) }
                        try rig.banks.bind(bank: b % 2, into: &bindings)
                    }
                    if b + 1 < count { try rig.banks.load(block: b + 1, into: (b + 1) % 2, chain: chain) }
                    await chain.synchronize()
                }
            }
            return try rig.chain.head.readFloats()
        }
        guard let metal = TensorArena.metal() else { Issue.record("no GPU"); return }
        let device = try rig(metal)
        func chained(_ count: Int) async throws -> [Float] { try await run(device, count) }

        var previous = floats(x0)
        var worst = 0.0
        for count in 1...blocks {
            let start = ContinuousClock.now
            let device = try await chained(count)
            let elapsed = ContinuousClock.now - start
            let gpu = device.map(Double.init), cpu = floats(expected[count - 1])
            let update = relativeL2(zip(gpu, previous).map { $0 - $1 }, zip(cpu, previous).map { $0 - $1 })
            let output = relativeL2(gpu, cpu)
            worst = max(worst, output)
            print("gate: h3 body \(count) block(s) × \(rows) rows: update rel L2 \(String(format: "%.5f", update)), "
                  + "output rel L2 \(String(format: "%.6f", output)), \(elapsed)")
            #expect(output < 0.02, "after \(count) block(s) the output differs from the CPU's by \(output)")
            let finite = gpu.allSatisfy { $0.isFinite }
            #expect(finite)
            previous = cpu
        }
        let host = try await run(try rig(TensorArena()), blocks)
        let again = try await chained(blocks)
        #expect(again == host, "chained bytes differ from the unchained path")
        #expect(graph.distinctShapeCount == 1)
        print("gate: h3 body chain: worst output rel L2 \(String(format: "%.5f", worst)); chained == unchained bytes")
    }

    static var h3Chain: URL {
        if let path = ProcessInfo.processInfo.environment["AETHER_H3_CHAIN"] { return URL(fileURLWithPath: path) }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent()   // Tests/AetherRuntimeTests
            .deletingLastPathComponent().deletingLastPathComponent()          // the checkout
            .deletingLastPathComponent().appending(path: "forge-outputs/minimax-h3-chain")
    }
}

extension TensorBuffer {
    /// bf16 written as its bit pattern (the runtime reads bf16 through ``readFloats(_:)``; tests feed raw bits).
    func writeBits(_ bits: [UInt16]) throws(AetherError) {
        precondition(scalarType == .bfloat16 && bits.count == count && layout.isContiguous)
        try arena.withMutableBytes(self) { bytes throws(AetherError) in
            bits.withUnsafeBytes { bytes.copyMemory(from: $0) }
        }
    }
}
