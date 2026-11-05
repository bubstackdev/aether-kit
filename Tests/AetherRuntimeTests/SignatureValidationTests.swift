import Testing
@testable import AetherRuntime

@Suite("Manifest ↔ signature validation")
struct SignatureValidationTests {
    @Test(arguments: ["qwen3-0.6b", "qwen3.5-0.8b"])
    func goldenManifestsFitTheirGraphs(_ id: String) throws {
        try SignatureValidator.validate(Fixtures.manifest(id), signatures: ["decoder": Fixtures.signature(id)])
    }

    static func check(
        _ id: String = "qwen3-0.6b", manifest edit: (inout Manifest.Graph) -> Void = { _ in },
        signature editSignature: (inout GraphSignature) -> Void = { _ in }
    ) throws {
        var manifest = try Fixtures.manifest(id)
        edit(&manifest.graphs["decoder"]!)
        var signature = try Fixtures.signature(id)
        editSignature(&signature)
        try SignatureValidator.validate(manifest, signatures: ["decoder": signature])
    }

    @Test func roleNamesMustExist() throws {
        expectField("graphs.decoder.inputs.tokens") { try Self.check(manifest: { $0.inputs["tokens"] = "ids" }) }
        expectField("graphs.decoder.outputs.logits") { try Self.check(manifest: { $0.outputs["logits"] = "scores" }) }
        expectField("graphs.decoder.states.kv.1") { try Self.check(manifest: { $0.states.kv[1] = "vCache" }) }
        expectField("graphs.decoder.function") { try Self.check(manifest: { $0.function = "prefill" }) }
    }

    @Test func everyGraphInputAndStateNeedsARole() throws {
        expectField("graphs.decoder.inputs") {
            try Self.check(signature: { $0.inputs.append(TensorSpec(name: "mask", scalarType: .int32, shape: [1, -1])) })
        }
        expectField("graphs.decoder.states") {
            try Self.check("qwen3.5-0.8b", manifest: { $0.states.recurrent = ["convState"] })
        }
    }

    @Test func ranksAndScalarTypesMustFit() throws {
        expectField("graphs.decoder.outputs.logits") {
            try Self.check(signature: { $0.outputs[0] = TensorSpec(name: "logits", scalarType: .float16, shape: [-1, 151936]) })
        }
        expectField("graphs.decoder.inputs.tokens") {
            try Self.check(signature: { $0.inputs[0] = TensorSpec(name: "input_ids", scalarType: .float16, shape: [1, -1]) })
        }
        expectField("graphs.decoder.states.kv.0") {
            try Self.check(signature: { $0.states[0] = TensorSpec(name: "keyCache", scalarType: "int4", shape: [28, 1, 8, -1, 128]) })
        }
        expectField("graphs.decoder.states.sequenceAxis") { try Self.check(manifest: { $0.states.sequenceAxis = 5 }) }
    }

    @Test func tokenAxisAgreesWithStaticSequence() throws {
        expectField("graphs.decoder.limits.staticSequence") {
            try Self.check("qwen3.5-0.8b", manifest: { $0.limits.staticSequence = nil })
        }
        expectField("graphs.decoder.limits.staticSequence") {
            try Self.check("qwen3.5-0.8b", manifest: { $0.limits.staticSequence = 4 })
        }
        expectField("graphs.decoder.limits.staticSequence") {
            try Self.check(manifest: { $0.limits.staticSequence = 1; $0.limits.prefillChunk = nil })
        }
    }

    @Test func unknownDriverIsUnsupported() throws {
        var manifest = try Fixtures.manifest("qwen3-0.6b")
        manifest.driver = DriverID(validated: "image-detector@1")
        #expect {
            try SignatureValidator.validate(manifest, signatures: ["decoder": Fixtures.signature("qwen3-0.6b")])
        } throws: { error in
            if case AetherError.unsupported(let driver) = error { return driver.description == "image-detector@1" }
            return false
        }
    }
}
