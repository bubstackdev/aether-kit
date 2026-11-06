import Foundation
import Testing
@testable import AetherCore

// Fixture provenance (Fixtures/zoo/<id>/), for the macOS variant of every chat entry downloaded in lab/models:
// - metadata.json, tokenizer_config.json and chat_template.jinja are verbatim copies from the pinned zoo bundle
//   (never weights; gemma-4-e2b ships no chat_template.jinja);
// - signatures.json is the decoder's InferenceFunctionDescriptor per function, recorded from the real graph by
//   `AETHER_GATES=1 AETHER_RECORD_SIGNATURES=1 swift test --filter GateTests/recordSignatures`;
// - aether.json is the golden: what the adapter must produce from those files, the builtin catalog entry and the
//   override table. The opt-in gate checks that it also holds against the real graphs.

@Suite("Zoo bundle adapter")
struct ZooAdapterTests {
    @Test(arguments: Fixtures.goldenIDs)
    func matchesGolden(_ id: String) throws {
        let produced = try ZooBundleAdapter.manifest(from: Fixtures.adapterInput(id))
        let golden = try Manifest.decode(Fixtures.zoo(id, "aether.json"))
        let rendered = String(decoding: try produced.encoded(), as: UTF8.self)
        #expect(produced == golden, "adapter output:\n\(rendered)")
        #expect(try produced.encoded() == Fixtures.zoo(id, "aether.json"), "the golden is in canonical form")
    }

    static func states(_ id: String) throws -> Manifest.States {
        try #require(try ZooBundleAdapter.manifest(from: Fixtures.adapterInput(id)).graphs["decoder"]?.states)
    }

    @Test func dynamicMeansSequenceAndStaticMeansRecurrent() throws {
        // recState [18×1×16×128×128] is static: a rank heuristic would read it as a 128-slot KV cache.
        let qwen35 = try Self.states("qwen3.5-0.8b")
        #expect(qwen35.sequence == [.init(tensors: ["keyCache", "valueCache"], seqAxis: 3)])
        #expect(qwen35.recurrent.map(\.name) == ["convState", "recState"])
        #expect(try Self.states("lfm2.5-1.2b").recurrent.map(\.name) == ["convState"])
        #expect(try Self.states("gemma-3-4b-it").recurrent.isEmpty)
    }

    @Test func youtuLatentsComeFromTheOverride() throws {
        #expect(try Self.states("youtu-llm-2b").sequence == [.init(tensors: ["kvCacheA", "kvCacheB"], kind: .latent, seqAxis: 3)])
        // Without the override the rule still finds the sequence axis, but calls the pair kv.
        var input = try Fixtures.adapterInput("youtu-llm-2b")
        input.overrides.entries["youtu-llm-2b"]!.variants["*"]!.graphs = nil
        let plain = try ZooBundleAdapter.manifest(from: input)
        #expect(plain.graphs["decoder"]?.states?.sequence.first?.kind == nil)
    }

    @Test func stopsAreTokenizerEOSUnionGenerationConfig() throws {
        func stops(_ id: String) throws -> [Manifest.StopToken] {
            try #require(try ZooBundleAdapter.manifest(from: Fixtures.adapterInput(id)).tokenizer?.stop)
        }
        #expect(try stops("gemma-4-e2b") == ["<turn|>", "<eos>", "<|tool_response>"])
        #expect(try stops("qwen3.5-0.8b") == ["<|im_end|>", "<|endoftext|>"])
        #expect(try stops("nemotron-3-nano-4b") == ["<|im_end|>", "</s>"])
        #expect(try stops("gemma-3-4b-it") == ["<eos>", "<end_of_turn>"])
    }

    @Test func pinnedStopsMustAgreeWithTheTokenizer() throws {
        var input = try Fixtures.adapterInput("qwen3-0.6b")
        input.overrides.entries["qwen3-0.6b"]!.variants["macos"]!.tokenizer!.stop!.tokens = ["<|im_end|>", "<|im_start|>"]
        expectField("tokenizer.stop") { _ = try ZooBundleAdapter.manifest(from: input) }
    }

    @Test func overridesArePinnedToTheCatalogRevision() throws {
        var input = try Fixtures.adapterInput("qwen3-0.6b")
        input.overrides.entries["qwen3-0.6b"]!.revision = String(repeating: "0", count: 40)
        expectField("zoo-overrides.qwen3-0.6b.revision") { _ = try ZooBundleAdapter.manifest(from: input) }
    }

    @Test func gemma4BindsItsTablesStatically() throws {
        let manifest = try ZooBundleAdapter.manifest(from: Fixtures.adapterInput("gemma-4-e2b"))
        let decoder = try #require(manifest.graphs["decoder"])
        #expect(decoder.inputs["pleTable"] == .init(name: "ple_table", binding: .static, source: .table("ple", storage: .auto)))
        #expect(decoder.warmup == .disabled)
        let entry = try Catalog.builtin.entry("gemma-4-e2b")
        let tableFile = try #require(entry.variants["macos"]?.assets["tables"]?.files.first { $0.path == "embed_per_layer.i8" })
        #expect(manifest.tables?["ple"]?.sha256 == tableFile.sha256)
        #expect(manifest.tables?["ple"]?.byteCount == tableFile.size)
        #expect(manifest.tokenizer?.chatTemplate == nil && manifest.tokenizer?.renderer == "gemma4-turns")
        #expect(manifest.tokenizer?.addBos == .renderer)
        #expect(manifest.assets["tables"]?.subtree == "ios-frontend/gemma4_qat_gather_raw")
    }

    @Test func gemma4WithoutItsTableOverrideCannotBindPLE() throws {
        var input = try Fixtures.adapterInput("gemma-4-e2b")
        input.overrides.entries["gemma-4-e2b"]!.variants["*"]!.graphs = nil
        expectField("graphs.decoder.inputs") { _ = try ZooBundleAdapter.manifest(from: input) }
    }

    @Test func embeddedTokenizerIsNeverTrusted() throws {
        // A bundle without tokenizer/ (mistral-7b's shape) needs a pinned source, whatever metadata claims.
        var input = try Fixtures.adapterInput("qwen3-0.6b")
        let variant = input.entry.variants["macos"]!
        var main = variant.assets["main"]!
        main.files.removeAll { $0.path.hasPrefix("tokenizer/") }
        input.entry.variants["macos"]!.assets["main"] = main
        expectField("tokenizer.source") { _ = try ZooBundleAdapter.manifest(from: input) }
    }

    @Test func thinkingIsInferredFromTheTemplateAndRefinedByOverrides() throws {
        func thinking(_ id: String) throws -> Manifest.Thinking? {
            try ZooBundleAdapter.manifest(from: Fixtures.adapterInput(id)).tokenizer?.thinking
        }
        #expect(try thinking("qwen3.5-0.8b") == .init(open: "<think>", close: "</think>", toggle: .templateVar("enable_thinking"), defaultOn: false))
        #expect(try thinking("nanbeige4.1-3b") == .init(open: "<think>", close: "</think>", toggle: Manifest.Thinking.Toggle.none))
        #expect(try thinking("lfm2.5-1.2b") == nil)
        #expect(try thinking("gemma-3-4b-it") == nil)
        #expect(try thinking("gemma-4-e2b")?.toggle == .emitEmpty("<|channel>thought\n<channel|>"))
    }

    @Test func staticTokenAxesBecomeStaticSequence() throws {
        func limits(_ id: String) throws -> Manifest.Limits? {
            try ZooBundleAdapter.manifest(from: Fixtures.adapterInput(id)).graphs["decoder"]?.limits
        }
        #expect(try limits("qwen3-0.6b") == .init(maxContext: 8192, prefillChunk: 512))
        #expect(try limits("granite-4.0-h-1b") == .init(maxContext: 4096, staticSequence: .one(1)))
        #expect(try limits("gemma-3-4b-it")?.maxContext == 131072, "the declared maximum, not a capacity")
    }

    @Test func prefillFunctionsBecomePhases() throws {
        var input = try Fixtures.adapterInput("qwen3.5-0.8b")
        var meta = try JSONSerialization.jsonObject(with: input.metadata) as! [String: Any]
        var language = meta["language"] as! [String: Any]
        language["function_map"] = ["main": ["main", "prefill"]]
        meta["language"] = language
        input.metadata = try JSONSerialization.data(withJSONObject: meta)
        var prefill = input.signatures["main"]!
        prefill.function = "prefill"
        prefill.inputs[0] = TensorSpec(name: "input_ids", scalarType: .int32, shape: [1, 64])
        prefill.outputs[0] = TensorSpec(name: "logits", scalarType: .float16, shape: [1, 64, 248320])
        input.signatures["prefill"] = prefill
        let graph = try #require(try ZooBundleAdapter.manifest(from: input).graphs["decoder"])
        #expect(graph.function == nil && graph.functions == ["step": "main", "prefill": "prefill"])
        #expect(graph.limits?.staticSequence == .perPhase(["step": .one(1), "prefill": .one(64)]))
        #expect(graph.limits?.prefillChunk == 64)
    }

    @Test func unknownFunctionMapsAndInputsAreNotGuessed() throws {
        var input = try Fixtures.adapterInput("qwen3-0.6b")
        var meta = try JSONSerialization.jsonObject(with: input.metadata) as! [String: Any]
        var language = meta["language"] as! [String: Any]
        language["function_map"] = ["main": ["extend_256_8"]]
        meta["language"] = language
        input.metadata = try JSONSerialization.data(withJSONObject: meta)
        expectField("graphs.decoder.functions") { _ = try ZooBundleAdapter.manifest(from: input) }

        input = try Fixtures.adapterInput("qwen3-0.6b")
        input.signatures["main"]!.inputs.append(TensorSpec(name: "per_layer", scalarType: .float16, shape: [1, -1, 8]))
        expectField("graphs.decoder.inputs") { _ = try ZooBundleAdapter.manifest(from: input) }
    }

    @Test func driverMustAgreeWithTheCatalogVariant() throws {
        var input = try Fixtures.adapterInput("qwen3-0.6b")
        input.variant = "ios"  // static-chunked-lm@1 in the catalog; the zoo metadata says llm
        #expect {
            _ = try ZooBundleAdapter.manifest(from: input)
        } throws: { field(of: $0) == "driver" }
    }

    @Test func malformedMetadataIsNamed() throws {
        var input = try Fixtures.adapterInput("qwen3-0.6b")
        input.metadata = Data(#"{"kind": "llm"}"#.utf8)
        expectField("metadata.assets") { _ = try ZooBundleAdapter.manifest(from: input) }
        input.metadata = Data(#"{"kind": "vision", "assets": {"main": "x"}}"#.utf8)
        expectField("metadata.kind") { _ = try ZooBundleAdapter.manifest(from: input) }
    }

    @Test func graphRequestComesFromMetadataAlone() throws {
        let request = try ZooBundleAdapter.graphRequest(
            entry: Catalog.builtin.entry("gemma-4-e2b"), variant: "macos",
            metadata: Fixtures.zoo("gemma-4-e2b", "metadata.json"))
        #expect(request.file == AssetPath(asset: "main", path: "gemma4_e2b_qat_decode_int4lin_tbl.aimodel"))
        #expect(request.functions == ["main"])
    }
}
