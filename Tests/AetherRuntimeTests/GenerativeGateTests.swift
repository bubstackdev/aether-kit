import Foundation
import Testing
@testable import AetherRuntime

extension GateTests {
    /// Measured footprints rest on `MTLDevice.currentAllocatedSize` seeing Core AI's graph memory, which arrives with
    /// the first run, not the load. Runs one qwen3-0.6b decoder step inside a measured ``GraphResidency`` phase and
    /// checks that the phase's peak covers the graph's weights; then that a released graph's memory comes back
    /// (seconds later: within 15 s).
    @Test func measuredPhaseSeesCoreAIGraphMemory() async throws {
        let id = "qwen3-0.6b"
        let manifest = try Fixtures.manifest(id)
        let bundle = try Self.labBundle(id)
        let files = GraphResidency.fileBytes(bundle: bundle, manifest: manifest, role: "decoder", device: Self.mac)
        #expect(files > 100 << 20)
        let ledger = FootprintLedger()
        let residency = GraphResidency(
            manifest: manifest, mode: .perPhase, ledger: ledger,
            load: GraphResidency.bundleLoader(bundle: bundle, manifest: manifest, device: Self.mac))
        let probe = GPUMemoryProbe.system
        let before = probe.allocatedBytes()

        residency.beginPhase(.perStep)
        _ = try await residency.graph("decoder")
        let afterLoad = probe.allocatedBytes()
        try await Self.runOneStep(try await residency.graph("decoder"), manifest: manifest)
        residency.sample()
        residency.endPhase(.perStep)  // the decoder has no residency: it stays loaded
        let footprint = try #require(ledger.footprint(.init(model: id, profile: "", phase: .perStep)))
        residency.releaseAll()
        var released = probe.allocatedBytes() - before
        for _ in 0..<15 where released > footprint.heldBytes / 4 {
            try await Task.sleep(for: .seconds(1))
            released = probe.allocatedBytes() - before
        }
        print("gate: qwen3-0.6b decoder files \(files >> 20) MiB; load +\((afterLoad - before) >> 20) MiB; "
              + "phase peak \(footprint.peakBytes >> 20) MiB, held \(footprint.heldBytes >> 20) MiB; "
              + "after release +\(released >> 20) MiB")
        #expect(footprint.peakBytes >= files, "the phase's Metal peak should cover the weights")
        #expect(released <= footprint.heldBytes / 4, "the released graph's memory should come back")
    }

    /// One decode step at position 0 (token 0) with small states.
    static func runOneStep(_ graph: any GraphBackend, manifest: Manifest) async throws {
        let decoder = try #require(manifest.graphs["decoder"])
        let tokens = try #require(decoder.inputs["tokens"]).name
        let positions = try #require(decoder.inputs["positions"]).name
        let logits = try #require(decoder.outputs["logits"]).name
        let isStatic = !(try #require(graph.signature.input(tokens))).isDynamic
        let arena = TensorArena()
        let states = try graph.makeStateSet(graph.signature.states.map(\.name), in: arena) { $0.isDynamic ? [16] : [] }
        let ids = try graph.makeBuffer(for: tokens, role: .input, dimensions: isStatic ? [] : [1], in: arena)
        let pos = try graph.makeBuffer(for: positions, role: .input, dimensions: [16], in: arena)
        let out = try graph.makeBuffer(for: logits, role: .output, dimensions: isStatic ? [] : [1], in: arena)
        try ids.write([Int32(0)])
        try pos.write(Int32.self, count: 16) { Int32($0) }
        var b = Bindings(arena: arena)
        try b.bindStates(states)
        try b.bindInput(tokens, to: ids)
        try b.bindInput(positions, to: pos, extent: [1, 1])
        try b.bindOutput(logits, to: out)
        try await graph.run(&b)
    }
}
