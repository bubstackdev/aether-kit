import AetherKit
import AetherCoreAI
import AetherKrea

@available(macOS 27, iOS 27, *)
enum KreaContracts {
    static let roots: Set<String> = ["latents", "text_embeds", "sigma", "rope_image"]

    static func binding(_ name: String, tokens: Int, text: Int) throws -> (String, [Int]) {
        switch name {
        case "latents", "velocity": ("float16", [tokens, 64])
        case "text_embeds": ("float16", [text, 30720])
        case "rope_image": ("float32", [tokens, 2, 128])
        case "sigma": ("float32", [1])
        case "cond": ("float32", [8, 6144])
        case "image_1", "image_2", "image_3": ("float32", [tokens, 6144])
        case "text_1", "text_2", "text_3": ("float32", [text, 6144])
        default: throw AetherError.unsupported("unknown Krea binding '\(name)'")
        }
    }

    static func declaration(_ value: ModelValueDescription, type: String, shape: [Int]) throws {
        guard value.kind == "tensor", value.scalarType == type, let declared = value.shape,
              declared.count == shape.count,
              zip(declared, shape).allSatisfy({ $0 < 0 || $0 == $1 }) else {
            throw AetherError.invalidModel("incompatible Krea declaration; expected \(type) \(shape)")
        }
    }

    static func segments(_ functions: [ModelFunctionDescription], tokens: Int, text: Int) throws {
        guard functions.count == 4 else { throw AetherError.invalidModel("expected four denoiser segments") }
        var available = roots
        var usedRoots: Set<String> = []
        for (index, function) in functions.enumerated() {
            guard function.name == "segment", function.states.isEmpty, !function.outputs.isEmpty else {
                throw AetherError.invalidModel("invalid denoiser segment function")
            }
            for (name, input) in function.inputs {
                guard available.contains(name) else {
                    throw AetherError.invalidModel("segment \(index) has missing upstream '\(name)'")
                }
                if roots.contains(name) { usedRoots.insert(name) }
                let expected = try binding(name, tokens: tokens, text: text)
                try declaration(input, type: expected.0, shape: expected.1)
            }
            for (name, output) in function.outputs {
                guard !available.contains(name), name != "velocity" || index == 3 else {
                    throw AetherError.invalidModel("invalid or overwritten denoiser output '\(name)'")
                }
                let expected = try binding(name, tokens: tokens, text: text)
                try declaration(output, type: expected.0, shape: expected.1)
                available.insert(name)
            }
        }
        guard functions[3].outputs["velocity"] != nil, usedRoots == roots else {
            throw AetherError.invalidModel("denoiser chain omits velocity or a required root input")
        }
    }

    static func decoder(_ function: ModelFunctionDescription, configuration: KreaConfiguration) throws {
        guard function.name == "decode", function.states.isEmpty,
              Set(function.inputs.keys) == ["latents"], Set(function.outputs.keys) == ["image"],
              let input = function.inputs["latents"], let output = function.outputs["image"] else {
            throw AetherError.invalidModel("invalid Krea decoder signature")
        }
        try declaration(input, type: "float16", shape: [configuration.height / 16, configuration.width / 16, 64])
        try declaration(output, type: "float32", shape: [3, configuration.height, configuration.width])
    }

    /// The vision tower's `encode` for one image of `patches` patch rows and `tokens` merged tokens.
    static func vision(_ function: ModelFunctionDescription, patches: Int, tokens: Int) throws {
        guard function.name == "encode", function.states.isEmpty,
              Set(function.inputs.keys) == ["pixels", "pos_index", "pos_weight", "patch_rc"],
              Set(function.outputs.keys) == ["image_embeds", "deepstack"] else {
            throw AetherError.invalidModel("invalid Krea vision encoder signature")
        }
        let hidden = KreaReferenceContract.hiddenSize
        try declaration(function.inputs["pixels"]!, type: "float16", shape: [patches, KreaReferenceContract.rowLength])
        try declaration(function.inputs["pos_index"]!, type: "int32", shape: [patches, 4])
        try declaration(function.inputs["pos_weight"]!, type: "float32", shape: [patches, 4])
        try declaration(function.inputs["patch_rc"]!, type: "int32", shape: [patches, 2])
        try declaration(function.outputs["image_embeds"]!, type: "float16", shape: [tokens, hidden])
        try declaration(function.outputs["deepstack"]!, type: "float16", shape: [KreaReferenceContract.deepstackLevels, tokens, hidden])
    }

    /// The text encoder's `encodeWithImages` for `rows` prompt tokens and `images` image tokens. Returns the declared
    /// `rope_positions` rank-3 flag (`[3, 1, L]` rather than `[3, L]`).
    static func encodeWithImages(_ function: ModelFunctionDescription, rows: Int, images: Int) throws -> Bool {
        guard function.name == "encodeWithImages", function.states.isEmpty,
              Set(function.inputs.keys) == ["input_ids", "attention_mask", "position_ids", "rope_positions", "image_embeds", "deepstack"],
              Set(function.outputs.keys) == ["text_embeds"], let rope = function.inputs["rope_positions"]?.shape else {
            throw AetherError.invalidModel("invalid Krea encodeWithImages signature")
        }
        let hidden = KreaReferenceContract.hiddenSize
        for name in ["input_ids", "attention_mask", "position_ids"] {
            try declaration(function.inputs[name]!, type: "int32", shape: [1, rows])
        }
        let rank3 = rope.count == 3
        try declaration(function.inputs["rope_positions"]!, type: "int32", shape: rank3 ? [3, 1, rows] : [3, rows])
        try declaration(function.inputs["image_embeds"]!, type: "float16", shape: [images, hidden])
        try declaration(function.inputs["deepstack"]!, type: "float16", shape: [KreaReferenceContract.deepstackLevels, images, hidden])
        try declaration(function.outputs["text_embeds"]!, type: "float16", shape: [rows, KreaConditioning.channels])
        return rank3
    }

    static func value(_ value: TensorValue, type: String, shape: [Int]) throws {
        let actual: String
        switch value {
        case .float16: actual = "float16"
        case .float32: actual = "float32"
        case .int32: actual = "int32"
        }
        guard actual == type, value.shape == shape else {
            throw AetherError.invalidTensor("Krea result must be \(type) \(shape)")
        }
    }
}
