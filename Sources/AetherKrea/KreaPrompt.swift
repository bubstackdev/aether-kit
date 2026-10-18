// Derived from the Apache-2.0 Aether prototype. See NOTICE.
import AetherKit

/// Exact, unpadded Qwen encoder input for Krea's text-only recipe.
public struct KreaPrompt: Sendable, Equatable {
    public static let systemMessage = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"
    public static let leadingRows = 34
    public static let maximumTokens = 546
    public static let trailingTokens = 5

    public let tokenIDs: [Int32]
    public let wasTruncated: Bool
    public var conditioningRows: Int { tokenIDs.count - Self.leadingRows }

    /// Replay already prepared IDs without applying truncation or a chat template.
    public init(tokenIDs: [Int32]) throws {
        guard (35...Self.maximumTokens).contains(tokenIDs.count), tokenIDs.allSatisfy({ $0 >= 0 }) else {
            throw AetherError.invalidRequest("Krea needs 35...546 nonnegative token IDs")
        }
        self.tokenIDs = tokenIDs
        self.wasTruncated = false
    }

    /// Prepare the complete rendered token sequence, preserving its final five tokens.
    public init(renderedTokenIDs: [Int]) throws {
        guard renderedTokenIDs.count >= 35,
              renderedTokenIDs.allSatisfy({ $0 >= 0 && $0 <= Int32.max }) else {
            throw AetherError.invalidRequest("invalid Krea rendered token IDs")
        }
        wasTruncated = renderedTokenIDs.count > Self.maximumTokens
        let selected = wasTruncated
            ? Array(renderedTokenIDs.prefix(Self.maximumTokens - Self.trailingTokens)) + renderedTokenIDs.suffix(Self.trailingTokens)
            : renderedTokenIDs
        tokenIDs = selected.map(Int32.init)
    }

    public static func render(_ text: String) -> String {
        "<|im_start|>system\n\(systemMessage)<|im_end|>\n<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n"
    }

    public var encoderInputs: [String: TensorValue] {
        // The private construction rules bound n to 35...546; all shapes are valid.
        get throws {
            let n = tokenIDs.count
            return [
                "input_ids": .int32(try Int32Tensor(shape: [1, n], values: tokenIDs)),
                "position_ids": .int32(try Int32Tensor(shape: [1, n], values: (0..<n).map(Int32.init))),
                "attention_mask": .int32(try Int32Tensor(shape: [1, n], values: Array(repeating: 1, count: n))),
            ]
        }
    }
}
