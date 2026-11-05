import Foundation
import Testing
@testable import AetherCore

// Fixture provenance (Fixtures/zoo/<id>/):
// - metadata.json, tokenizer_config.json and chat_template.jinja are verbatim copies from the pinned zoo bundle in
//   lab/models (never weights).
// - signature.json is the graph's InferenceFunctionDescriptor, transcribed from the M0 spike's `inspect` findings
//   (lab/engine-spike/README.md). For qwen3-0.6b the opt-in gate checks it against the real graph. For
//   qwen3.5-0.8b the shapes and dtypes agree with the spike's recorded state sizes (35.3 MB at capacity 2048), but
//   the order of the states and the shape of position_ids are not yet confirmed by a gate.
// - aether.json is the golden: what the adapter must produce from those files plus the builtin catalog entry.

@Suite("Zoo bundle adapter")
struct ZooAdapterTests {
    @Test(arguments: ["qwen3-0.6b", "qwen3.5-0.8b"])
    func matchesGolden(_ id: String) throws {
        let produced = try ZooBundleAdapter.manifest(from: Fixtures.adapterInput(id))
        let golden = try Manifest.decode(Fixtures.zoo(id, "aether.json"))
        let rendered = String(decoding: try produced.encoded(), as: UTF8.self)
        #expect(produced == golden, "adapter output:\n\(rendered)")
        // The golden file is in canonical form, so a regenerated golden diffs cleanly.
        #expect(try produced.encoded() == Fixtures.zoo(id, "aether.json"))
    }

    @Test func hybridStatesComeFromNamesNotShapes() throws {
        // recState [18×1×16×128×128] is static: rank-based guessing would read it as a 128-slot KV cache.
        let manifest = try ZooBundleAdapter.manifest(from: Fixtures.adapterInput("qwen3.5-0.8b"))
        let states = try #require(manifest.graphs["decoder"]).states
        #expect(states.kv == ["keyCache", "valueCache"])
        #expect(states.recurrent == ["convState", "recState"])
        #expect(states.sequenceAxis == 3)
    }

    @Test func keysAreOrderedBeforeValues() throws {
        var input = try Fixtures.adapterInput("qwen3-0.6b")
        var signature = try #require(input.signatures["main"])
        signature.states.reverse()
        input.signatures["main"] = signature
        let manifest = try ZooBundleAdapter.manifest(from: input)
        #expect(manifest.graphs["decoder"]?.states.kv == ["keyCache", "valueCache"])
    }

    @Test func unknownKindIsRejected() throws {
        var input = try Fixtures.adapterInput("qwen3-0.6b")
        var json = try JSONSerialization.jsonObject(with: input.metadata) as! [String: Any]
        json["kind"] = "vision"
        input.metadata = try JSONSerialization.data(withJSONObject: json)
        expectField("metadata.kind") { _ = try ZooBundleAdapter.manifest(from: input) }
    }

    @Test func malformedMetadataIsNamed() throws {
        var input = try Fixtures.adapterInput("qwen3-0.6b")
        input.metadata = Data(#"{"kind": "llm"}"#.utf8)
        expectField("metadata.assets") { _ = try ZooBundleAdapter.manifest(from: input) }
    }

    @Test func extraInputsAreNotGuessed() throws {
        var input = try Fixtures.adapterInput("qwen3-0.6b")
        input.signatures["main"]!.inputs.append(TensorSpec(name: "per_layer", scalarType: .float16, shape: [1, -1, 8]))
        expectField("graphs.decoder.inputs") { _ = try ZooBundleAdapter.manifest(from: input) }
    }

    @Test func missingSignatureOrAssetIsNamed() throws {
        var input = try Fixtures.adapterInput("qwen3-0.6b")
        input.signatures = [:]
        expectField("graphs.decoder.function") { _ = try ZooBundleAdapter.manifest(from: input) }

        input = try Fixtures.adapterInput("qwen3-0.6b")
        input.files = ["metadata.json"]
        expectField("metadata.assets.main") { _ = try ZooBundleAdapter.manifest(from: input) }
    }

    @Test func staticKVWithoutASequenceAxisIsRefused() throws {
        var input = try Fixtures.adapterInput("qwen3-0.6b")
        input.signatures["main"]!.states = [
            TensorSpec(name: "keyCache", scalarType: .float16, shape: [28, 1, 8, 512, 128]),
            TensorSpec(name: "valueCache", scalarType: .float16, shape: [28, 1, 8, 512, 128]),
        ]
        expectField("graphs.decoder.states.sequenceAxis") { _ = try ZooBundleAdapter.manifest(from: input) }
    }

    @Test func graphRequestsComeFromMetadataAlone() throws {
        let request = try ZooBundleAdapter.graphRequests(metadata: Fixtures.zoo("qwen3-0.6b", "metadata.json"))
        #expect(request.file == "qwen3_0_6b_4bit_dynamic.aimodel")
        #expect(request.functions == ["main"])
    }

    @Test func fullyStaticGraphPrefersTheNeuralEngine() throws {
        var input = try Fixtures.adapterInput("qwen3-0.6b")
        var signature = try #require(input.signatures["main"])
        signature.inputs = signature.inputs.map { TensorSpec(name: $0.name, scalarType: $0.scalarType, shape: [1, 64]) }
        signature.states = []
        input.signatures["main"] = signature
        do {
            _ = try ZooBundleAdapter.manifest(from: input)
            Issue.record("no kv states must fail the decoder contract")
        } catch {
            #expect(field(of: error) == "graphs.decoder.states.kv")
        }
        let graph = try ZooBundleAdapter.decoderGraph(
            file: "x.aimodel", function: "main", signature: signature,
            meta: ZooBundleAdapter.decodeMetadata(input.metadata))
        #expect(graph.specialization == .init(computeUnit: .neuralEngine, expectFrequentReshapes: false))
        #expect(graph.limits.staticSequence == 64 && graph.limits.prefillChunk == nil)
    }
}
