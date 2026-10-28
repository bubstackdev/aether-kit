import AetherCoreAI
import AetherFlux2
import AetherKit

/// Declared and runtime tensor contracts of the klein export, checked before every native call.
@available(macOS 27, iOS 27, *)
enum Flux2Contracts {
    static let hidden = 3072
    static let conditionRows = 17
    static let roots: Set<String> = ["latents", "text_embeds", "sigma", "rope_text", "rope_image"]

    /// `tokens` counts generated plus reference image tokens; `text` is always 512.
    static func binding(_ name: String, tokens: Int, text: Int = Flux2Prompt.length) throws -> (String, [Int]) {
        switch name {
        case "latents", "velocity": ("float16", [tokens, Flux2Configuration.latentChannels])
        case "text_embeds": ("float16", [text, Flux2Conditioning.width])
        case "sigma": ("float32", [1])
        case "rope_text": ("float32", [text, 2, Flux2Rotary.headDimension])
        case "rope_image": ("float32", [tokens, 2, Flux2Rotary.headDimension])
        case "cond": ("float32", [conditionRows, hidden])
        case _ where name.hasPrefix("image_"): ("float32", [tokens, hidden])
        case _ where name.hasPrefix("text_"): ("float32", [text, hidden])
        default: throw AetherError.unsupported("unknown FLUX.2 binding '\(name)'")
        }
    }

    static func declaration(_ value: ModelValueDescription, type: String, shape: [Int]) throws {
        guard value.kind == "tensor", value.scalarType == type, let declared = value.shape,
              declared.count == shape.count, zip(declared, shape).allSatisfy({ $0 < 0 || $0 == $1 }) else {
            throw AetherError.invalidModel("incompatible FLUX.2 declaration; expected \(type) \(shape)")
        }
    }

    static func value(_ value: TensorValue, type: String, shape: [Int]) throws {
        let actual: String
        switch value {
        case .float16: actual = "float16"
        case .float32: actual = "float32"
        case .int32: actual = "int32"
        }
        guard actual == type, value.shape == shape else {
            throw AetherError.invalidTensor("FLUX.2 result must be \(type) \(shape)")
        }
    }

    /// The segment chain: the first takes the host roots, later ones carries by name, the last emits velocity.
    static func segments(_ functions: [ModelFunctionDescription], tokens: Int) throws {
        guard functions.count == Flux2Assets.segmentCount else {
            throw AetherError.invalidModel("expected \(Flux2Assets.segmentCount) denoiser segments")
        }
        var available = roots, used: Set<String> = []
        for (index, function) in functions.enumerated() {
            guard function.name == "segment", function.states.isEmpty, !function.outputs.isEmpty else {
                throw AetherError.invalidModel("invalid denoiser segment function")
            }
            for (name, input) in function.inputs {
                guard available.contains(name) else { throw AetherError.invalidModel("segment \(index) has missing upstream '\(name)'") }
                if roots.contains(name) { used.insert(name) }
                let expected = try binding(name, tokens: tokens)
                try declaration(input, type: expected.0, shape: expected.1)
            }
            for (name, output) in function.outputs {
                guard !available.contains(name), name != "velocity" || index == functions.count - 1 else {
                    throw AetherError.invalidModel("invalid or overwritten denoiser output '\(name)'")
                }
                let expected = try binding(name, tokens: tokens)
                try declaration(output, type: expected.0, shape: expected.1)
                available.insert(name)
            }
        }
        guard functions.last?.outputs["velocity"] != nil, used == roots else {
            throw AetherError.invalidModel("denoiser chain omits velocity or a required root input")
        }
    }

    static func textEncoder(_ function: ModelFunctionDescription) throws {
        guard function.name == "encode", function.states.isEmpty,
              Set(function.inputs.keys) == ["input_ids", "position_ids", "attention_mask"],
              Set(function.outputs.keys) == ["text_embeds"] else {
            throw AetherError.invalidModel("invalid FLUX.2 text encoder signature")
        }
        for input in function.inputs.values { try declaration(input, type: "int32", shape: [1, Flux2Prompt.length]) }
        try declaration(function.outputs["text_embeds"]!, type: "float16", shape: [Flux2Prompt.length, Flux2Conditioning.width])
    }

    static func decoder(_ function: ModelFunctionDescription, rows: Int, columns: Int) throws {
        guard function.name == "decode", function.states.isEmpty,
              Set(function.inputs.keys) == ["latents"], Set(function.outputs.keys) == ["image"] else {
            throw AetherError.invalidModel("invalid FLUX.2 decoder signature")
        }
        try declaration(function.inputs["latents"]!, type: "float16", shape: [rows, columns, Flux2Configuration.latentChannels])
        try declaration(function.outputs["image"]!, type: "float32", shape: [3, rows * 16, columns * 16])
    }

    static func referenceEncoder(_ function: ModelFunctionDescription, rows: Int, columns: Int) throws {
        guard function.name == "encode", function.states.isEmpty,
              Set(function.inputs.keys) == ["reference_pixels"], Set(function.outputs.keys) == ["reference_latents"] else {
            throw AetherError.invalidModel("invalid FLUX.2 reference encoder signature")
        }
        try declaration(function.inputs["reference_pixels"]!, type: "float32", shape: [3, rows * 16, columns * 16])
        try declaration(function.outputs["reference_latents"]!, type: "float32", shape: [rows, columns, Flux2Configuration.latentChannels])
    }
}
