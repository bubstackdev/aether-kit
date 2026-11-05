import Foundation
import Testing
@testable import AetherCore

@Suite("Manifest schema v1")
struct ManifestTests {
    /// The example from DESIGN.md §4, with a per-architecture file map and fields this build does not know.
    static let specExample = """
    {
      "schema": 1,
      "id": "qwen3-0.6b",
      "task": "text-generation",
      "driver": "decoder-lm@1",
      "futureTopLevelField": {"anything": [1, 2, 3]},
      "graphs": {
        "decoder": {
          "file": {"h18p": "x.h18p.aimodelc", "*": "x.aimodel"},
          "function": "main",
          "inputs":  { "tokens": "input_ids", "positions": "position_ids" },
          "outputs": { "logits": "logits" },
          "states":  { "kv": ["k_cache", "v_cache"], "recurrent": [], "sequenceAxis": 3 },
          "limits":  { "maxContext": 8192, "prefillChunk": null, "staticSequence": null },
          "specialization": { "computeUnit": "gpu", "expectFrequentReshapes": true, "unknownKnob": 7 }
        }
      },
      "tokenizer": { "dir": "tokenizer", "chatTemplate": "tokenizer/chat_template.jinja",
                     "stop": ["<|im_end|>"], "thinking": { "open": "<think>", "close": "</think>" } },
      "preprocess": null,
      "precision": { "weights": "int4", "activations": "fp16" },
      "provenance": { "source": "Qwen/Qwen3-0.6B", "gate": "token-exact vs fp16 ref", "zooRevision": "943eb6a" }
    }
    """

    static func mutated(_ edit: (inout [String: Any]) -> Void) throws -> Data {
        var json = try JSONSerialization.jsonObject(with: Data(specExample.utf8)) as! [String: Any]
        edit(&json)
        return try JSONSerialization.data(withJSONObject: json)
    }

    static func editGraph(_ json: inout [String: Any], _ edit: (inout [String: Any]) -> Void) {
        var graphs = json["graphs"] as! [String: Any]
        var decoder = graphs["decoder"] as! [String: Any]
        edit(&decoder)
        graphs["decoder"] = decoder
        json["graphs"] = graphs
    }

    @Test func decodesTheSpecExampleAndIgnoresUnknownFields() throws {
        let manifest = try Manifest.decode(Data(Self.specExample.utf8))
        #expect(manifest.driver == .decoderLM)
        let graph = try #require(manifest.graphs["decoder"])
        #expect(graph.inputs == ["tokens": "input_ids", "positions": "position_ids"])
        #expect(graph.states.kv == ["k_cache", "v_cache"])
        #expect(graph.states.sequenceAxis == 3)
        #expect(graph.limits.maxContext == 8192 && graph.limits.prefillChunk == nil)
        #expect(graph.specialization == .init(computeUnit: .gpu, expectFrequentReshapes: true))
        #expect(manifest.tokenizer?.thinking == .init(open: "<think>", close: "</think>"))
        #expect(manifest.preprocess == nil)
        try DriverContract.decoderLM.validate(manifest)
    }

    @Test func perArchitectureFileResolution() throws {
        let graph = try #require(try Manifest.decode(Data(Self.specExample.utf8)).graphs["decoder"])
        #expect(graph.file.resolve(architecture: "h18p") == "x.h18p.aimodelc")
        #expect(graph.file.resolve(architecture: "h17c") == "x.aimodel")
        #expect(Manifest.GraphFile.perArchitecture(["h18p": "a"]).resolve(architecture: "h17c") == nil)
        #expect(Manifest.GraphFile.single("a.aimodel").resolve(architecture: "anything") == "a.aimodel")
    }

    @Test func roundTripsThroughCanonicalJSON() throws {
        let manifest = try Manifest.decode(Data(Self.specExample.utf8))
        let again = try Manifest.decode(manifest.encoded())
        #expect(again == manifest)
        let text = String(decoding: try manifest.encoded(), as: UTF8.self)
        #expect(!text.contains("null"), "nil optionals are omitted")
        #expect(text.contains("\"*\" : \"x.aimodel\""))
    }

    @Test func missingRequiredFieldIsNamed() throws {
        expectField("graphs.decoder.function") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["function"] = nil } })
        }
        expectField("driver") {
            _ = try Manifest.decode(Self.mutated { $0["driver"] = nil })
        }
    }

    @Test func wrongTypeIsNamed() throws {
        expectField("graphs.decoder.limits.maxContext") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["limits"] = ["maxContext": "big"] } })
        }
    }

    @Test func malformedDriverIsNamed() throws {
        expectField("driver") {
            _ = try Manifest.decode(Self.mutated { $0["driver"] = "decoder-lm" })
        }
        #expect(throws: AetherError.self) { try DriverID(parsing: "Decoder@1") }
        #expect(try DriverID(parsing: "image-detector@12") == DriverID(name: "image-detector", version: 12))
    }

    @Test func unsupportedSchemaIsNamed() throws {
        expectField("schema") { _ = try Manifest.decode(Self.mutated { $0["schema"] = 2 }) }
    }

    @Test func structuralRulesNameTheirField() throws {
        expectField("graphs.decoder.limits.prefillChunk") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["limits"] = ["maxContext": 256, "prefillChunk": 512] } })
        }
        expectField("graphs.decoder.limits.prefillChunk") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["limits"] = ["prefillChunk": 3, "staticSequence": 2] } })
        }
        expectField("graphs.decoder.states") {
            _ = try Manifest.decode(Self.mutated {
                Self.editGraph(&$0) { $0["states"] = ["kv": ["a", "b"], "recurrent": ["a"], "sequenceAxis": 1] }
            })
        }
        expectField("graphs.decoder.inputs") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["inputs"] = ["tokens": "x", "positions": "x"] } })
        }
        expectField("graphs.decoder.file") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["file"] = [String: String]() } })
        }
        expectField("graphs") { _ = try Manifest.decode(Self.mutated { $0["graphs"] = [String: Any]() }) }
        expectField("tokenizer.stop.0") {
            _ = try Manifest.decode(Self.mutated { $0["tokenizer"] = ["dir": "tokenizer", "stop": [""]] })
        }
        expectField("preprocess.std") {
            _ = try Manifest.decode(Self.mutated { $0["preprocess"] = ["mean": [0.5, 0.5], "std": [0.5]] })
        }
    }

    @Test func decoderContractIsClosed() throws {
        let contract = DriverContract.decoderLM
        expectField("graphs.decoder.inputs.embeddings") {
            try contract.validate(Manifest.decode(Self.mutated {
                Self.editGraph(&$0) { $0["inputs"] = ["tokens": "a", "positions": "b", "embeddings": "c"] }
            }))
        }
        expectField("graphs.decoder.inputs.positions") {
            try contract.validate(Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["inputs"] = ["tokens": "a"] } }))
        }
        expectField("graphs.decoder.outputs.logits") {
            try contract.validate(Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["outputs"] = ["hidden": "h"] } }))
        }
        expectField("graphs.decoder.states.kv") {
            try contract.validate(Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["states"] = ["kv": [String]()] } }))
        }
        expectField("graphs.decoder.states.kv") {
            try contract.validate(Manifest.decode(Self.mutated {
                Self.editGraph(&$0) { $0["states"] = ["kv": ["a", "b", "c"], "sequenceAxis": 1] }
            }))
        }
        expectField("graphs.decoder.states.sequenceAxis") {
            try contract.validate(Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["states"] = ["kv": ["a", "b"]] } }))
        }
        expectField("graphs.encoder") {
            try contract.validate(Manifest.decode(Self.mutated {
                var graphs = $0["graphs"] as! [String: Any]
                graphs["encoder"] = graphs["decoder"]
                $0["graphs"] = graphs
            }))
        }
        expectField("tokenizer") {
            try contract.validate(Manifest.decode(Self.mutated { $0["tokenizer"] = nil }))
        }
        expectField("driver") {
            try contract.validate(Manifest.decode(Self.mutated { $0["driver"] = "decoder-lm@2" }))
        }
    }
}
