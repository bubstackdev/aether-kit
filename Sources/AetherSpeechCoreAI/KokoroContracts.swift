import AetherCoreAI
import AetherKit
import AetherSpeech

/// Declared signatures of the two exported graphs. Dynamic dimensions are reported as negative.
enum KokoroContracts {
    static let prosodyChannels = 640
    static let textChannels = 512

    static func durations(_ function: ModelFunctionDescription) throws {
        guard function.name == "durations", function.states.isEmpty,
              Set(function.inputs.keys) == ["input_ids", "style"],
              Set(function.outputs.keys) == ["duration", "prosody", "text"] else {
            throw AetherError.invalidModel("invalid Kokoro durations signature")
        }
        try declaration(function.inputs["input_ids"], "int32", [1, -1])
        try declaration(function.inputs["style"], "float32", [1, KokoroVoiceStyles.dimension])
        try declaration(function.outputs["duration"], "float32", [1, -1])
        try declaration(function.outputs["prosody"], "float32", [1, -1, prosodyChannels])
        try declaration(function.outputs["text"], "float32", [1, textChannels, -1])
    }

    /// A bucketed decoder also takes `frames` (int32 [1]), the valid prefix of padded inputs.
    static func decoder(_ function: ModelFunctionDescription, bucketed: Bool) throws {
        let inputs: Set<String> = bucketed ? ["prosody", "text", "style", "noise", "frames"] : ["prosody", "text", "style", "noise"]
        guard function.name == "decode", function.states.isEmpty,
              Set(function.inputs.keys) == inputs,
              Set(function.outputs.keys) == ["audio"] else {
            throw AetherError.invalidModel("invalid Kokoro decode signature for this bundle's bucket contract")
        }
        if bucketed { try declaration(function.inputs["frames"], "int32", [1]) }
        try declaration(function.inputs["prosody"], "float32", [1, prosodyChannels, -1])
        try declaration(function.inputs["text"], "float32", [1, textChannels, -1])
        try declaration(function.inputs["style"], "float32", [1, KokoroVoiceStyles.dimension])
        try declaration(function.inputs["noise"], "float32", [1, -1, KokoroAssets.samplesPerFrame, KokoroAssets.harmonics])
        try declaration(function.outputs["audio"], "float32", [1, -1])
    }

    /// -1 accepts any declared extent (dynamic or fixed); other values must match a fixed or dynamic dimension.
    private static func declaration(_ value: ModelValueDescription?, _ type: String, _ shape: [Int]) throws {
        guard let value, value.kind == "tensor", value.scalarType == type, let declared = value.shape,
              declared.count == shape.count,
              zip(declared, shape).allSatisfy({ $1 < 0 || $0 < 0 || $0 == $1 }) else {
            throw AetherError.invalidModel("incompatible Kokoro declaration; expected \(type) \(shape)")
        }
    }

    static func float(_ outputs: [String: TensorValue], _ name: String, shape: [Int]) throws -> FloatTensor {
        guard case .float32(let tensor) = outputs[name], tensor.shape == shape else {
            throw AetherError.invalidTensor("Kokoro output '\(name)' must be float32 \(shape)")
        }
        return tensor
    }
}
