import Testing
@testable import AetherRuntime

@Suite("Manifest ↔ signature validation")
struct SignatureValidationTests {
    @Test(arguments: Fixtures.goldenIDs)
    func goldenManifestsFitTheirGraphs(_ id: String) throws {
        try SignatureValidator.validate(Fixtures.manifest(id), signatures: ["decoder": Fixtures.signatures(id)])
    }

    static func check(
        _ id: String = "qwen3-0.6b", manifest edit: (inout Manifest) -> Void = { _ in },
        signature editSignature: (inout GraphSignature) -> Void = { _ in }
    ) throws {
        var manifest = try Fixtures.manifest(id)
        edit(&manifest)
        var signatures = try Fixtures.signatures(id)
        editSignature(&signatures["main"]!)
        try SignatureValidator.validate(manifest, signatures: ["decoder": signatures])
    }

    static func decoder(_ m: inout Manifest, _ edit: (inout Manifest.Graph) -> Void) { edit(&m.graphs["decoder"]!) }

    @Test func roleNamesMustExist() throws {
        expectField("graphs.decoder.inputs.tokens") { try Self.check(manifest: { Self.decoder(&$0) { $0.inputs["tokens"] = "ids" } }) }
        expectField("graphs.decoder.outputs.logits") {
            try Self.check(manifest: { Self.decoder(&$0) { $0.outputs["logits"] = "scores" } })
        }
        expectField("graphs.decoder.states.sequence.0.tensors") {
            try Self.check(manifest: { Self.decoder(&$0) { $0.states!.sequence[0].tensors[1] = "vCache" } })
        }
        expectField("graphs.decoder.functions.step") { try Self.check(manifest: { Self.decoder(&$0) { $0.function = "prefill" } }) }
    }

    @Test func everyGraphInputAndStateNeedsARole() throws {
        expectField("graphs.decoder.inputs") {
            try Self.check(signature: { $0.inputs.append(TensorSpec(name: "mask", scalarType: .int32, shape: [1, -1])) })
        }
        expectField("graphs.decoder.states") {
            try Self.check("qwen3.5-0.8b", manifest: { Self.decoder(&$0) { $0.states!.recurrent.removeLast() } })
        }
    }

    @Test func ranksScalarTypesAndRowsMustFit() throws {
        expectField("graphs.decoder.outputs.logits") {
            try Self.check(signature: { $0.outputs[0] = TensorSpec(name: "logits", scalarType: .float16, shape: [-1, 151936]) })
        }
        expectField("graphs.decoder.inputs.tokens") {
            try Self.check(signature: { $0.inputs[0] = TensorSpec(name: "input_ids", scalarType: .float16, shape: [1, -1]) })
        }
        expectField("graphs.decoder.states.sequence.0.tensors") {
            try Self.check(signature: { $0.states[0] = TensorSpec(name: "keyCache", scalarType: "int4", shape: [28, 1, 8, -1, 128]) })
        }
        expectField("graphs.decoder.states.sequence.0.seqAxis") {
            try Self.check(manifest: { Self.decoder(&$0) { $0.states!.sequence[0].seqAxis = 5 } })
        }
        expectField("graphs.decoder.outputs.logits.rows") {
            try Self.check(manifest: { Self.decoder(&$0) { $0.outputs["logits"]!.rows = .last } })
        }
    }

    @Test func tokenAxisAgreesWithStaticSequence() throws {
        expectField("graphs.decoder.limits.staticSequence") {
            try Self.check("qwen3.5-0.8b", manifest: { Self.decoder(&$0) { $0.limits!.staticSequence = nil } })
        }
        expectField("graphs.decoder.limits.staticSequence") {
            try Self.check("qwen3.5-0.8b", manifest: { Self.decoder(&$0) { $0.limits!.staticSequence = .one(4) } })
        }
        expectField("graphs.decoder.limits.staticSequence") {
            try Self.check(manifest: { Self.decoder(&$0) { $0.limits!.staticSequence = .one(1); $0.limits!.prefillChunk = nil } })
        }
    }

    @Test func tablesMustMatchTheInputsTheyFeed() throws {
        expectField("tables.ple") {
            try Self.check("gemma-4-e2b", manifest: { $0.tables!["ple"]!.shape = [262144, 4480] })
        }
        expectField("graphs.decoder.inputs") {
            try Self.check("gemma-4-e2b", manifest: { Self.decoder(&$0) { $0.inputs["pleScale"] = nil } })
        }
    }

    @Test func knownButUnimplementedDriversAreUnsupported() throws {
        var manifest = try Fixtures.manifest("qwen3-0.6b")
        manifest.driver = "detector-dense@1"
        #expect {
            try SignatureValidator.validate(manifest, signatures: ["decoder": Fixtures.signatures("qwen3-0.6b")])
        } throws: { error in
            if case AetherError.unsupported(let driver) = error { return driver.description == "detector-dense@1" }
            return false
        }
        manifest.driver = "image-detector@7"
        #expect {
            try SignatureValidator.validate(manifest, signatures: ["decoder": Fixtures.signatures("qwen3-0.6b")])
        } throws: { error in
            if case AetherError.unknownDriver(let driver) = error { return driver.description == "image-detector@7" }
            return false
        }
    }
}
