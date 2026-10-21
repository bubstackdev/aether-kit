import AetherKit
import AetherCoreAI
import AetherKrea

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
