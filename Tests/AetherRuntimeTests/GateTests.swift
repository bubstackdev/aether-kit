import Foundation
import Testing
@testable import AetherRuntime

/// Parity gates on real bundles (DESIGN.md §12). Off by default: they load multi-hundred-megabyte graphs onto
/// the GPU. Run with `AETHER_GATES=1 swift test --filter GateTests`. Bundles are read from `lab/models` (a sibling of
/// this repo), or from `AETHER_LAB_MODELS` when set.
@Suite("Gates", .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct GateTests {
    static var models: URL {
        if let override = ProcessInfo.processInfo.environment["AETHER_LAB_MODELS"] {
            return URL(fileURLWithPath: override)
        }
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appending(path: "lab/models")
    }

    @Test func qwen3SignatureMatchesTheAdapterManifest() async throws {
        let bundle = Self.models.appending(path: "qwen3-0.6b/macos")
        let manifest = try Fixtures.manifest("qwen3-0.6b")
        let decoder = try #require(manifest.graphs["decoder"])
        let file = try #require(decoder.file.resolve(architecture: Device.current.architecture))
        let options = try GraphOptions(manifest: decoder, at: "graphs.decoder", cachePolicy: .default, cache: .default)

        let graph = try await Graph(url: bundle.appending(path: file), function: decoder.function, options: options)

        // The transcribed fixture is the real descriptor…
        #expect(graph.signature == (try Fixtures.signature("qwen3-0.6b")))
        // …the golden manifest validates against the real graph…
        try SignatureValidator.validate(manifest, signatures: ["decoder": graph.signature])
        // …and the adapter, given the real signature, reproduces the golden.
        let entry = try Catalog.builtin.entry("qwen3-0.6b")
        let produced = try ZooBundleAdapter.manifest(from: ZooBundleAdapter.Input(
            metadata: try Data(contentsOf: bundle.appending(path: "metadata.json")),
            signatures: [decoder.function: graph.signature],
            files: entry.variants["macos"]!.files.map(\.path),
            tokenizerConfig: try Data(contentsOf: bundle.appending(path: "tokenizer/tokenizer_config.json")),
            chatTemplate: try String(contentsOf: bundle.appending(path: "tokenizer/chat_template.jinja"), encoding: .utf8),
            catalogID: entry.id, repo: entry.repo, revision: entry.revision))
        #expect(produced == manifest)
    }

    // MARK: - First real Graph.run: greedy parity with the lab spike (itself token-exact vs llm-runner)

    /// The lab spike's recorded greedy run for the same bundle: prompt ids and generated ids.
    struct SpikeReference: Decodable { var promptTokens: [Int32]; var generated: [Int32] }

    static func spikeReference(_ id: String) throws -> SpikeReference {
        let url = models.deletingLastPathComponent().appending(path: "engine-spike/results/parity/\(id)-spike.json")
        return try JSONDecoder().decode(SpikeReference.self, from: Data(contentsOf: url))
    }

    static func argmaxLastRow(_ logits: TensorBuffer, rows: Int, vocab: Int) throws -> Int32 {
        let row = try logits.readFloats(((rows - 1) * vocab)..<(rows * vocab))
        var best = 0
        for i in row.indices where row[i] > row[best] { best = i }
        return Int32(best)
    }

    /// Prefill in one call (dynamic token axis) or token by token (static), then greedy-decode `count` tokens.
    /// Positions live in one capacity-sized buffer written once and bound by leading extent: this exercises the
    /// region binding against real Core AI for the first time.
    static func greedy(bundle: String, manifestID: String, count: Int) async throws -> (prompt: Int, tokens: [Int32]) {
        let manifest = try Fixtures.manifest(manifestID)
        let decoder = try #require(manifest.graphs["decoder"])
        let file = try #require(decoder.file.resolve(architecture: Device.current.architecture))
        let options = try GraphOptions(manifest: decoder, at: "graphs.decoder", cachePolicy: .default, cache: .default)
        let graph = try await Graph(url: models.appending(path: bundle).appending(path: file),
                                    function: decoder.function, options: options)
        try SignatureValidator.validate(manifest, signatures: ["decoder": graph.signature])
        #expect(graph.signature == (try Fixtures.signature(manifestID)))

        let tokensName = try #require(decoder.inputs["tokens"])
        let positionsName = try #require(decoder.inputs["positions"])
        let logitsName = try #require(decoder.outputs["logits"])
        let tokenSpec = try #require(graph.signature.input(tokensName))
        let logitsSpec = try #require(graph.signature.output(logitsName))
        let vocab = try #require(logitsSpec.shape.last)
        let staticTokens = !tokenSpec.shape.contains(where: { $0 < 0 })

        let capacity = 256
        let arena = TensorArena()
        let states = try graph.makeStateSet(graph.signature.states.map(\.name), in: arena) { spec in
            spec.shape.contains(where: { $0 < 0 }) ? [capacity] : []
        }
        let positions = try graph.makeBuffer(for: positionsName, role: .input, dimensions: [capacity], in: arena)
        try positions.write(Int32.self, count: capacity) { Int32($0) }

        func bindings(tokens: TensorBuffer, logits: TensorBuffer, total: Int) throws -> Bindings {
            var b = Bindings(arena: arena)
            try b.bindStates(states)
            try b.bindInput(tokensName, to: tokens)
            try b.bindInput(positionsName, to: positions, extent: [1, total])
            try b.bindOutput(logitsName, to: logits)
            return b
        }
        let stepTokens = try graph.makeBuffer(for: tokensName, role: .input, dimensions: staticTokens ? [] : [1], in: arena)
        let stepLogits = try graph.makeBuffer(for: logitsName, role: .output, dimensions: staticTokens ? [] : [1], in: arena)
        var position = 0
        func step(_ token: Int32) async throws -> Int32 {
            try stepTokens.write([token])
            position += 1
            var b = try bindings(tokens: stepTokens, logits: stepLogits, total: position)
            try await graph.run(&b)
            return try argmaxLastRow(stepLogits, rows: 1, vocab: vocab)
        }

        let reference = try spikeReference(manifestID)
        let prompt = reference.promptTokens
        var next: Int32
        if staticTokens {
            next = 0
            for t in prompt { next = try await step(t) }
        } else {
            let ids = try graph.makeBuffer(for: tokensName, role: .input, dimensions: [prompt.count], in: arena)
            let logits = try graph.makeBuffer(for: logitsName, role: .output, dimensions: [prompt.count], in: arena)
            try ids.write(prompt)
            position = prompt.count
            var b = try bindings(tokens: ids, logits: logits, total: position)
            try await graph.run(&b)
            next = try argmaxLastRow(logits, rows: prompt.count, vocab: vocab)
        }
        var out: [Int32] = [next]
        while out.count < count { out.append(try await step(out.last!)) }
        return (prompt.count, out)
    }

    @Test func qwen3GreedyMatchesSpike() async throws {
        let n = 32
        let (_, tokens) = try await Self.greedy(bundle: "qwen3-0.6b/macos", manifestID: "qwen3-0.6b", count: n)
        #expect(tokens == Array(try Self.spikeReference("qwen3-0.6b").generated.prefix(n)))
    }

    @Test func hybridQwen35GreedyMatchesSpike() async throws {
        let n = 32
        let (_, tokens) = try await Self.greedy(
            bundle: "qwen3.5-0.8b/gpu-pipelined/qwen3_5_0_8b_decode_int8hu_block32_sym",
            manifestID: "qwen3.5-0.8b", count: n)
        #expect(tokens == Array(try Self.spikeReference("qwen3.5-0.8b").generated.prefix(n)))
    }
}
