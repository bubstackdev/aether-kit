import Foundation
import Testing
@testable import AetherRuntime

/// Parity gates on real bundles (DESIGN.md §12). Off by default: they load multi-gigabyte graphs onto the GPU.
/// Run with `AETHER_GATES=1 swift test --filter GateTests` (serially). Bundles are read from `lab/models` (a sibling
/// of this repo), or from `AETHER_LAB_MODELS`. The lab layout `<models>/<id>/<subtree>` is the install layout, so
/// `<models>/<id>` serves as the revision root.
///
/// `AETHER_RECORD_SIGNATURES=1` additionally rewrites every fixture `signatures.json` from the real graphs.
@Suite("Gates", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct GateTests {
    static var models: URL {
        if let override = ProcessInfo.processInfo.environment["AETHER_LAB_MODELS"] {
            return URL(fileURLWithPath: override)
        }
        return labModels(from: #filePath)
    }

    static let mac = Device(platform: .macOS, architecture: Device.current.architecture)

    /// The lab copy of a catalog entry's macOS variant, as an installed bundle.
    static func labBundle(_ id: String) throws -> InstalledBundle {
        InstalledBundle(ref: try ModelRef.resolve(id, for: mac), url: models.appending(path: id), isStale: false)
    }

    /// Loads every function of the decoder graph that the adapter would inspect.
    static func realSignatures(_ id: String) async throws -> [String: GraphSignature] {
        let bundle = try labBundle(id)
        let entry = try Catalog.builtin.entry(id)
        let main = try #require(bundle.url(ofAsset: "main"))
        let metadata = try Data(contentsOf: main.appending(path: "metadata.json"))
        let request = try ZooBundleAdapter.graphRequest(entry: entry, variant: "macos", metadata: metadata)
        let url = try #require(bundle.url(of: request.file))
        // Signatures do not depend on specialization; GPU with the reshape hint is how every decoder loads.
        let options = GraphOptions(computeUnit: .gpu, expectFrequentReshapes: true, cachePolicy: .default, cache: .default)
        var out: [String: GraphSignature] = [:]
        for function in request.functions {
            out[function] = try await Graph(url: url, function: function, options: options).signature
        }
        return out
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["AETHER_RECORD_SIGNATURES"] == "1"))
    func recordSignatures() async throws {
        for id in Fixtures.goldenIDs {
            let signatures = try await Self.realSignatures(id)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            var data = try encoder.encode(signatures)
            data.append(0x0A)
            try data.write(to: Fixtures.zoo.appending(path: "\(id)/signatures.json"))
        }
    }

    /// Every golden: the recorded fixture is the real descriptor, the golden validates against the real graph,
    /// and the adapter, given the real signatures, reproduces the golden.
    @Test(arguments: Fixtures.goldenIDs)
    func goldenMatchesRealGraph(_ id: String) async throws {
        let real = try await Self.realSignatures(id)
        #expect(real == (try Fixtures.signatures(id)))
        let golden = try Fixtures.manifest(id)
        try SignatureValidator.validate(golden, signatures: ["decoder": real])

        let main = try #require(try Self.labBundle(id).url(ofAsset: "main"))
        let produced = try ZooBundleAdapter.manifest(from: ZooBundleAdapter.Input(
            entry: try Catalog.builtin.entry(id), variant: "macos",
            metadata: try Data(contentsOf: main.appending(path: "metadata.json")),
            signatures: real,
            tokenizerConfig: try? Data(contentsOf: main.appending(path: "tokenizer/tokenizer_config.json")),
            chatTemplate: try? String(contentsOf: main.appending(path: "tokenizer/chat_template.jinja"), encoding: .utf8)))
        #expect(produced == golden)
    }

    // MARK: - Greedy parity with the lab spike (itself token-exact vs llm-runner)

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
    /// Positions live in one capacity-sized buffer written once and bound by leading extent.
    static func greedy(_ id: String, count: Int) async throws -> [Int32] {
        let manifest = try Fixtures.manifest(id)
        let decoder = try #require(manifest.graphs["decoder"])
        let options = try GraphOptions(manifest: decoder, at: "graphs.decoder", cachePolicy: .default, cache: .default)
        let graph = try await Graph(
            bundle: try labBundle(id), manifest: manifest, architecture: mac.architecture, options: options)
        try SignatureValidator.validate(manifest, signatures: ["decoder": [graph.signature.function: graph.signature]])

        let tokensName = try #require(decoder.inputs["tokens"]).name
        let positionsName = try #require(decoder.inputs["positions"]).name
        let logitsName = try #require(decoder.outputs["logits"]).name
        let tokenSpec = try #require(graph.signature.input(tokensName))
        let vocab = try #require(graph.signature.output(logitsName)?.shape.last)
        let staticTokens = !tokenSpec.isDynamic

        let capacity = 256
        let arena = TensorArena()
        let states = try graph.makeStateSet(graph.signature.states.map(\.name), in: arena) { spec in
            spec.isDynamic ? [capacity] : []
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

        let prompt = try spikeReference(id).promptTokens
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
        return out
    }

    @Test func qwen3GreedyMatchesSpike() async throws {
        let n = 32
        #expect(try await Self.greedy("qwen3-0.6b", count: n) == Array(try Self.spikeReference("qwen3-0.6b").generated.prefix(n)))
    }

    @Test func hybridQwen35GreedyMatchesSpike() async throws {
        let n = 32
        #expect(try await Self.greedy("qwen3.5-0.8b", count: n) == Array(try Self.spikeReference("qwen3.5-0.8b").generated.prefix(n)))
    }
}

/// `lab/models` beside the repo: the first found walking up from `file`, so the gates find it from a worktree under
/// `.worktrees/` as from the main checkout (`AETHER_LAB_MODELS` overrides, at the call sites).
func labModels(from file: String) -> URL {
    var directory = URL(fileURLWithPath: file).deletingLastPathComponent()
    while directory.path != "/" {
        let candidate = directory.appending(path: "lab/models")
        if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        directory = directory.deletingLastPathComponent()
    }
    return URL(fileURLWithPath: file).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().appending(path: "lab/models")
}
