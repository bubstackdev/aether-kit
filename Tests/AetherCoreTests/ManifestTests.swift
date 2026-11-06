import Foundation
import Testing
@testable import AetherCore

@Suite("Manifest schema v1")
struct ManifestTests {
    /// DESIGN.md §4's example, extended with a per-arch file, a static table input, phases, `x-` fields and
    /// fields this build does not know.
    static let example = """
    { "schema": 1, "id": "qwen3.5-0.8b", "task": "text-generation", "driver": "decoder-lm@1",
      "variant": { "platform": "macos" },
      "assets": { "main": { "subtree": "gpu-pipelined/qwen3_5_0_8b_decode_int8hu_block32_sym" },
                  "tables": { "subtree": "tables", "files": [{ "path": "t.i8", "bytes": 8, "sha256": "00" }] } },
      "futureTopLevelField": { "anything": [1, 2, 3] },
      "x-notes": ["ignored"],
      "tables": { "t": { "asset": "tables", "file": "t.i8", "dtype": "int8", "shape": [2, 4] } },
      "graphs": { "decoder": {
          "file": { "h18p": "main:x.h18p.aimodelc", "*": "main:qwen3_5_0_8b_decode_int8hu_block32_sym.aimodel" },
          "functions": { "step": "main", "prefill": "prefill" },
          "inputs":  { "tokens": "input_ids", "positions": "position_ids",
                       "table": { "name": "t_in", "binding": "static", "source": { "table": "t", "storage": "auto" } } },
          "outputs": { "logits": { "name": "logits", "rows": "last" } },
          "states":  { "sequence":  [ { "tensors": ["keyCache", "valueCache"], "seqAxis": 3, "x-observed": "fp16" } ],
                       "recurrent": [ { "name": "convState", "kind": "conv", "init": "zeros" }, { "name": "recState" } ] },
          "limits":  { "maxContext": 4096, "staticSequence": { "step": 1, "prefill": 64 }, "prefillChunk": 64,
                       "kvCapacityCap": { "ios": 1024 } },
          "compute": "gpu", "expectFrequentReshapes": true, "warmup": false, "unknownKnob": 7 } },
      "tokenizer": { "source": { "dir": "main:tokenizer" }, "chatTemplate": "main:tokenizer/chat_template.jinja",
                     "addBos": "template", "stop": ["<|im_end|>", 248044],
                     "thinking": { "open": "<think>", "close": "</think>", "toggle": { "templateVar": "enable_thinking" },
                                   "defaultOn": false } },
      "decision": { "promptFormat": "decider", "maxOptions": 8, "slot": { "rule": "lastToken" } },
      "postprocess": { "kind": "denseNms", "boxFormat": "cxcywh", "boxSpace": "letterboxPixels",
                       "nms": { "type": "perClass", "iou": 0.45 }, "defaults": { "scoreThreshold": 0.3, "maxDetections": 50 } },
      "labels": { "set": "coco-91" },
      "calibration": { "temperature": 1.03, "source": "card" },
      "provenance": { "source": "Qwen/Qwen3.5-0.8B", "zooRepo": "mlboydaisuke/qwen3.5-0.8B-CoreAI", "zooRevision": "1b8c020" } }
    """

    static func mutated(_ edit: (inout [String: Any]) -> Void) throws -> Data {
        var json = try JSONSerialization.jsonObject(with: Data(example.utf8)) as! [String: Any]
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

    @Test func decodesTheFullSchemaAndIgnoresUnknownAndXFields() throws {
        let manifest = try Manifest.decode(Data(Self.example.utf8))
        let graph = try #require(manifest.graphs["decoder"])
        #expect(graph.phases == ["step": "main", "prefill": "prefill"])
        #expect(graph.inputs["tokens"] == "input_ids")
        #expect(graph.inputs["table"]?.source == .table("t", storage: .auto) && graph.inputs["table"]?.isStatic == true)
        #expect(graph.outputs["logits"] == .init(name: "logits", rows: .last))
        #expect(graph.states?.sequence == [.init(tensors: ["keyCache", "valueCache"], seqAxis: 3)])
        #expect(graph.states?.recurrent.map(\.name) == ["convState", "recState"])
        #expect(graph.states?.recurrent.first?.initial == "zeros")
        #expect(graph.limits?.staticSequence?.values(forPhase: "prefill") == [64])
        #expect(graph.limits?.kvCapacityCap == ["ios": 1024])
        #expect(graph.compute == .gpu && graph.warmup == .disabled)
        #expect(manifest.tokenizer?.stop == [.text("<|im_end|>"), .id(248044)])
        #expect(manifest.tokenizer?.thinking?.toggle == .templateVar("enable_thinking"))
        #expect(manifest.decision?.maxOptions == 8)
        #expect(manifest.postprocess?.kind == .denseNms)
        #expect(manifest.tables?["t"]?.byteCount == 8)
        try DriverContract.decoderLM.validate(manifest)
    }

    @Test func assetPathsAndPerArchitectureFiles() throws {
        let graph = try #require(try Manifest.decode(Data(Self.example.utf8)).graphs["decoder"])
        #expect(graph.file.resolve(architecture: "h18p") == AssetPath(asset: "main", path: "x.h18p.aimodelc"))
        #expect(graph.file.resolve(architecture: "h17c")?.path == "qwen3_5_0_8b_decode_int8hu_block32_sym.aimodel")
        #expect(throws: AetherError.self) { try AssetPath(parsing: "no-asset") }
        #expect(try AssetPath(parsing: "tables:.") == AssetPath(asset: "tables", path: "."))
        expectField("graphs.decoder.file.*") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["file"] = ["*": "towers:x.aimodel"] } })
        }
        expectField("graphs.decoder.file") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["file"] = "main:../escape" } })
        }
    }

    @Test func roundTripsThroughCanonicalJSON() throws {
        let manifest = try Manifest.decode(Data(Self.example.utf8))
        #expect(try Manifest.decode(manifest.encoded()) == manifest)
        let text = String(decoding: try manifest.encoded(), as: UTF8.self)
        #expect(!text.contains("null") && !text.contains("x-notes") && !text.contains("unknownKnob"))
        #expect(text.contains("\"tokens\" : \"input_ids\""), "a name-only binding encodes as a bare string")
        #expect(text.contains("\"warmup\" : false"))
    }

    @Test func refsAreRejectedWithTheirPath() throws {
        expectField("tokenizer.text") {
            _ = try Manifest.decode(Self.mutated {
                var t = $0["tokenizer"] as! [String: Any]
                t["text"] = ["$ref": "main:reference.json#/text"]
                $0["tokenizer"] = t
            })
        }
    }

    @Test func missingRequiredFieldsAreNamed() throws {
        expectField("graphs.decoder.compute") { _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["compute"] = nil } }) }
        expectField("variant") { _ = try Manifest.decode(Self.mutated { $0["variant"] = nil }) }
        expectField("assets.main") { _ = try Manifest.decode(Self.mutated { $0["assets"] = ["other": ["subtree": "x"]] }) }
        expectField("graphs.decoder.function") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["functions"] = nil } })
        }
        expectField("graphs.decoder.function") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["function"] = "main" } })
        }
        expectField("graphs.decoder.states.sequence.0.seqAxis") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["states"] = ["sequence": [["tensors": ["k"]]]] } })
        }
        expectField("decision.maxOptions") {
            _ = try Manifest.decode(Self.mutated { $0["decision"] = ["promptFormat": "slot"] })
        }
        expectField("postprocess.kind") { _ = try Manifest.decode(Self.mutated { $0["postprocess"] = ["nms": 1] }) }
        expectField("tokenizer.source") {
            _ = try Manifest.decode(Self.mutated { $0["tokenizer"] = ["stop": ["x"]] })
        }
    }

    @Test func structuralRulesNameTheirField() throws {
        expectField("schema") { _ = try Manifest.decode(Self.mutated { $0["schema"] = 2 }) }
        expectField("driver") { _ = try Manifest.decode(Self.mutated { $0["driver"] = "vision-tower@1" }) }
        expectField("driver") { _ = try Manifest.decode(Self.mutated { $0["driver"] = "decoder-lm" }) }
        expectField("variant.platform") { _ = try Manifest.decode(Self.mutated { $0["variant"] = ["platform": "android"] }) }
        expectField("graphs.decoder.limits.prefillChunk") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["limits"] = ["maxContext": 256, "prefillChunk": 512] } })
        }
        expectField("graphs.decoder.limits.staticSequence.decode") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["limits"] = ["staticSequence": ["decode": 1]] } })
        }
        expectField("graphs.decoder.limits.kvCapacityCap.android") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["limits"] = ["kvCapacityCap": ["android": 1]] } })
        }
        expectField("graphs.decoder.states.recurrent.0.name") {
            _ = try Manifest.decode(Self.mutated {
                Self.editGraph(&$0) { $0["states"] = ["sequence": [["tensors": ["a"], "seqAxis": 1]], "recurrent": [["name": "a"]]] }
            })
        }
        expectField("graphs.decoder.inputs.table.source.table") {
            _ = try Manifest.decode(Self.mutated { $0["tables"] = nil })
        }
        expectField("graphs.decoder.inputs.tokens") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["inputs"] = ["tokens": "x", "positions": "x"] } })
        }
        expectField("graphs.decoder.inputsByFunction.decode") {
            _ = try Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["inputsByFunction"] = ["decode": ["tokens": "a"]] } })
        }
        expectField("tables.t.asset") {
            _ = try Manifest.decode(Self.mutated { $0["tables"] = ["t": ["asset": "nope", "file": "t.i8", "dtype": "int8", "shape": [2]]] })
        }
        expectField("tokenizer.source.revision") {
            _ = try Manifest.decode(Self.mutated {
                var t = $0["tokenizer"] as! [String: Any]
                t["source"] = ["hf": "mistralai/Mistral-7B-Instruct-v0.3", "revision": "main"]
                $0["tokenizer"] = t
            })
        }
        expectField("labels") { _ = try Manifest.decode(Self.mutated { $0["labels"] = ["set": "coco-91", "file": "main:labels.json"] }) }
        expectField("calibration") { _ = try Manifest.decode(Self.mutated { $0["calibration"] = ["source": "x"] }) }
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
        expectField("graphs.decoder.states.sequence") {
            try contract.validate(Manifest.decode(Self.mutated {
                Self.editGraph(&$0) { $0["states"] = ["recurrent": [["name": "c"]]] }
            }))
        }
        expectField("graphs.decoder.states.host") {
            try contract.validate(Manifest.decode(Self.mutated {
                Self.editGraph(&$0) { $0["states"] = ["sequence": [["tensors": ["k"], "seqAxis": 3]], "host": [["in": "a"]]] }
            }))
        }
        expectField("graphs.decoder.functions.decode") {
            try contract.validate(Manifest.decode(Self.mutated {
                Self.editGraph(&$0) { $0["functions"] = ["step": "m", "decode": "d"]; $0["limits"] = ["maxContext": 4096] }
            }))
        }
        expectField("graphs.decoder.limits.maxContext") {
            try contract.validate(Manifest.decode(Self.mutated { Self.editGraph(&$0) { $0["limits"] = ["prefillChunk": 64] } }))
        }
        expectField("graphs.encoder") {
            try contract.validate(Manifest.decode(Self.mutated {
                var graphs = $0["graphs"] as! [String: Any]
                graphs["encoder"] = graphs["decoder"]
                $0["graphs"] = graphs
            }))
        }
        expectField("tokenizer") { try contract.validate(Manifest.decode(Self.mutated { $0["tokenizer"] = nil })) }
        expectField("tokenizer.stop") {
            try contract.validate(Manifest.decode(Self.mutated {
                $0["tokenizer"] = ["source": ["dir": "main:tokenizer"]]
            }))
        }
        expectField("driver") { try contract.validate(Manifest.decode(Self.mutated { $0["driver"] = "decision-lm@1" })) }
    }

    @Test func driverCatalogKnowsEveryDriverOfTheSpec() throws {
        // The 18 of §1, plus the generative drivers of §5 as they become normative (latent-diffusion@1, G1;
        // flux2-klein@1, G2; krea2@1, G3; kokoro@1, G5; minimax-h3@1, G4).
        #expect(DriverCatalog.drivers.count == 23)
        try DriverCatalog.requireKnown("gliner-classifier@1")
        try DriverCatalog.requireKnown(.latentDiffusion)
        try DriverCatalog.requireKnown(.flux2Klein)
        try DriverCatalog.requireKnown(.krea2)
        #expect(throws: AetherError.self) { try DriverCatalog.requireKnown("vision-tower@1") }
        #expect(throws: AetherError.self) { try DriverCatalog.requireKnown("decoder-static-chunks@1") }
    }
}
