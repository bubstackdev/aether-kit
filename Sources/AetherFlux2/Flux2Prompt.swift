// Derived from the Apache-2.0 Aether prototype (11 §B2). See NOTICE.
import AetherKit

/// Text-encoder input for klein: the bundle's Qwen3 chat template (one user message, generation prompt,
/// thinking off; rendered by `Flux2Tokenizer`), right-padded with `<|endoftext|>` to exactly 512 tokens
/// with an attention mask.
/// All 512 encoder rows condition the denoiser, including the padding rows: they cannot be trimmed.
public struct Flux2Prompt: Sendable, Equatable {
    public static let length = 512
    public static let padToken: Int32 = 151643

    public let tokenIDs: [Int32]
    public let attentionMask: [Int32]
    public let wasTruncated: Bool
    public var realTokens: Int { attentionMask.reduce(0) { $0 + Int($1) } }

    /// Pads or truncates the rendered sequence (the tokenizer's right truncation keeps the prefix).
    public init(renderedTokenIDs: [Int]) throws {
        guard !renderedTokenIDs.isEmpty, renderedTokenIDs.allSatisfy({ $0 >= 0 && $0 <= Int32.max }) else {
            throw AetherError.invalidRequest("invalid FLUX.2 rendered token IDs")
        }
        wasTruncated = renderedTokenIDs.count > Self.length
        let kept = renderedTokenIDs.prefix(Self.length).map(Int32.init)
        tokenIDs = kept + Array(repeating: Self.padToken, count: Self.length - kept.count)
        attentionMask = Array(repeating: 1, count: kept.count) + Array(repeating: 0, count: Self.length - kept.count)
    }

    /// Replays already padded IDs and mask (fixture path). The mask must be a nonempty prefix of ones.
    public init(paddedTokenIDs: [Int32], attentionMask: [Int32]) throws {
        let real = attentionMask.prefix { $0 == 1 }.count
        guard paddedTokenIDs.count == Self.length, attentionMask.count == Self.length, real > 0,
              attentionMask.dropFirst(real).allSatisfy({ $0 == 0 }), paddedTokenIDs.allSatisfy({ $0 >= 0 }) else {
            throw AetherError.invalidRequest("FLUX.2 needs 512 nonnegative token IDs and a right-padded mask")
        }
        tokenIDs = paddedTokenIDs; self.attentionMask = attentionMask; wasTruncated = false
    }

    public var encoderInputs: [String: TensorValue] {
        get throws {
            [
                "input_ids": .int32(try Int32Tensor(shape: [1, Self.length], values: tokenIDs)),
                "position_ids": .int32(try Int32Tensor(shape: [1, Self.length], values: (0..<Self.length).map(Int32.init))),
                "attention_mask": .int32(try Int32Tensor(shape: [1, Self.length], values: attentionMask)),
            ]
        }
    }
}

/// The text encoder's float16 rows `[512, 7680]`: the residual stream after Qwen3 layers 9, 18 and 27.
public struct Flux2Conditioning: Sendable, Equatable {
    public static let width = 7680
    public let tensor: Float16Tensor
    public var rows: Int { tensor.shape[0] }

    public init(tensor: Float16Tensor) throws {
        guard tensor.shape == [Flux2Prompt.length, Self.width] else {
            throw AetherError.invalidTensor("FLUX.2 conditioning must be [512,7680]")
        }
        self.tensor = tensor
    }

    /// Fixture path: narrows reference float32 rows once to float16, refusing overflow.
    public init(provided tensor: FloatTensor) throws {
        guard tensor.shape == [Flux2Prompt.length, Self.width] else {
            throw AetherError.invalidTensor("FLUX.2 conditioning must be [512,7680]")
        }
        try self.init(tensor: Float16Tensor(shape: tensor.shape, values: tensor.values.map(Float16.init)))
    }

    /// Tensor payload only.
    public var byteCount: Int { tensor.values.count * MemoryLayout<Float16>.stride }
}
