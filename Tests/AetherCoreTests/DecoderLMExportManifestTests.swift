import Foundation
import Testing
@testable import AetherCore

/// MANIFEST.md §1.1 (`decoder-lm@1`) for our own language-model exports. The fixture is the Qwen3 exporter's own output
/// (`Forge/qwen3/export_qwen3.py --sample-manifest`), so a change to what it emits shows up here.
@Suite("decoder-lm export manifest")
struct DecoderLMExportManifestTests {
    static func fixture() throws -> Manifest { try Manifest.decode(Fixtures.data("qwen3/aether.json")) }

    @Test func theQwen3ExportersManifestValidates() throws {
        let manifest = try Self.fixture()
        #expect(manifest.driver == .decoderLM && manifest.task == "text-generation")
        try DriverContract.decoderLM.validate(manifest)
        let decoder = try #require(manifest.graphs["decoder"])
        #expect(decoder.compute == .gpu && decoder.expectFrequentReshapes == true)
        #expect(decoder.limits?.maxContext == 32768 && decoder.limits?.prefillChunk == 512)
        let sequence = try #require(decoder.states?.sequence.first)
        #expect(sequence.tensors == ["keyCache", "valueCache"] && sequence.seqAxis == 3)
        // The checkpoint's own stops, in generation_config order, and no thinking block for an instruct-only template.
        let tokenizer = try #require(manifest.tokenizer)
        #expect(tokenizer.stop == ["<|im_end|>", "<|endoftext|>"])
        #expect(tokenizer.thinking == nil && tokenizer.toolDialect == "hermes-json")
        #expect(manifest.provenance?.source == "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated")
        // Every file the manifest names is listed in the asset, so a full verification covers it.
        let listed = Set(try #require(manifest.assets["main"]?.files).map(\.path))
        for name in ["decoder.aimodel/main.mlirb", "tokenizer/tokenizer.json", "tokenizer/chat_template.jinja", "LICENSE", "NOTICE"] {
            #expect(listed.contains(name), "\(name)")
        }
        #expect(try Manifest.decode(manifest.encoded()) == manifest)
    }
}
