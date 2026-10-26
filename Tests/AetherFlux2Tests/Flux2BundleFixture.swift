import CryptoKit
import Foundation
import Testing

/// A weight-free FLUX.2 bundle with the exact recipe declarations and tiny placeholder streams.
struct Flux2BundleFixture {
    let root: URL
    static let roles = ["textEncoder", "denoiser0", "denoiser1", "denoiser2", "vaeDecoder", "vaeEncoder"]

    init(references: Bool = true) throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("flux2-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var assets: [String: String] = [:]
        for role in Self.roles where references || role != "vaeEncoder" {
            let directory = root.appendingPathComponent(role + ".aimodel")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let data = Data("fixture model: \(role)".utf8)
            try data.write(to: directory.appendingPathComponent("main.mlirb"))
            try Data(SHA256.hash(data: data)).write(to: directory.appendingPathComponent("main.hash"))
            try Data("{\"assetVersion\":\"2.0\"}".utf8).write(to: directory.appendingPathComponent("metadata.json"))
            assets[role] = role + ".aimodel"
        }
        let metadata: [String: Any] = ["name": "flux2_klein_4b_reference", "kind": "diffusion", "assets": assets]
        try JSONSerialization.data(withJSONObject: metadata).write(to: root.appendingPathComponent("metadata.json"))
        let tokenizer = root.appendingPathComponent("tokenizer")
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            try Data("{\"name\":\"\(name)\"}".utf8).write(to: tokenizer.appendingPathComponent(name))
        }
        try JSONSerialization.data(withJSONObject: Self.manifest(references: references), options: [.sortedKeys])
            .write(to: root.appendingPathComponent("aether.json"))
    }

    static func manifest(references: Bool) -> [String: Any] {
        var functions: [String: Any] = [
            "textEncoder/encode": ["role": "encodeText"], "vaeDecoder/decode": ["role": "decodeLatents"],
            "denoiser0/segment": ["role": "denoiseSegment"], "denoiser1/segment": ["role": "denoiseSegment"],
            "denoiser2/segment": ["role": "denoiseSegment"],
        ]
        var inputs: [String: Any] = [
            "input_ids": ["bind": "tokens", "dtype": "int32"], "attention_mask": ["bind": "tokenMask", "dtype": "int32"],
            "position_ids": ["bind": "positions", "dtype": "int32", "mode": "running"],
            "latents": ["bind": "latents", "dtype": "float16"], "text_embeds": ["bind": "textEmbeddings", "dtype": "float16"],
            "sigma": ["bind": "timestep", "dtype": "float32"], "rope_text": ["bind": "textRotary", "dtype": "float32"],
            "rope_image": ["bind": "imageRotary", "dtype": "float32"],
        ]
        var outputs: [String: Any] = [
            "text_embeds": ["role": "textEmbeddings", "dtype": "float16"], "velocity": ["role": "velocity", "dtype": "float16"],
            "image": ["role": "pixels", "dtype": "float32"],
        ]
        var diffusion: [String: Any] = [
            "latentChannels": 128, "pixelsPerToken": 16, "steps": 4, "maxSteps": 50, "trainTimesteps": 1000,
            "guidance": false, "textEncoderAsset": "textEncoder", "decoderAsset": "vaeDecoder", "denoiserAsset": "denoiser",
            "denoiserSegments": ["denoiser0", "denoiser1", "denoiser2"],
            "sigmaShift": ["rule": "empirical", "lowSteps": 10, "lowSlope": 8.73809524e-05, "lowIntercept": 1.89833333,
                           "highSteps": 200, "highSlope": 0.00016927, "highIntercept": 0.45666666, "tokenThreshold": 4300],
            "rotary": ["axes": [32, 32, 32, 32], "theta": 2000.0],
            "prompt": ["template": "chat", "thinking": "off", "maxLength": 512, "padding": "right", "padToken": 151643],
            "expectFrequentReshapes": true,
            "assetReshapes": references ? ["vaeDecoder": false, "vaeEncoder": false] : ["vaeDecoder": false],
        ]
        if references {
            functions["vaeEncoder/encode"] = ["role": "encodeLatents"]
            inputs["reference_pixels"] = ["bind": "image", "dtype": "float32"]
            outputs["reference_latents"] = ["role": "latents", "dtype": "float32"]
            diffusion["referenceImages"] = ["kind": "packedLatents", "encoderAsset": "vaeEncoder", "positionStride": 10,
                                            "maxImages": 2, "maxSide": 512, "maxTotalTokens": 4096]
        }
        return [
            "schema": 1, "driver": "diffusion", "functions": functions, "inputs": inputs, "outputs": outputs,
            "diffusion": diffusion,
            "target": ["backend": "coreai", "platform": "macos", "arch": "any", "compiled": false, "compute": "gpu", "minOS": "27.0"],
            "text": ["tokenizer": "tokenizer/", "chatTemplate": "embedded"],
            "provenance": ["source": ["hfRepo": "black-forest-labs/FLUX.2-klein-4B",
                                      "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294"], "files": [String: String]()],
        ]
    }

    func cleanup() { try? FileManager.default.removeItem(at: root) }

    func edit(_ name: String, _ edit: (inout [String: Any]) -> Void) throws {
        let url = root.appendingPathComponent(name)
        var json = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        edit(&json)
        try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys]).write(to: url)
    }
}
