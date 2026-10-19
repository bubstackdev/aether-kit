import AetherKit

/// Selected finite fp16 conditioning rows in packed `[rows, 30720]` order.
public struct KreaConditioning: Sendable, Equatable {
    public static let channels = 30720
    public let tensor: Float16Tensor
    public var rows: Int { tensor.shape[0] }

    public init(tensor: Float16Tensor) throws {
        guard tensor.shape.count == 2, (1...KreaReferencePrompt.maximumConditioningRows).contains(tensor.shape[0]),
              tensor.shape[1] == Self.channels else {
            throw AetherError.invalidTensor("Krea conditioning must be [1...1116,30720]")
        }
        self.tensor = tensor
    }

    /// Selects the model's rows after the 34-row system prefix without changing precision.
    public init(encoderOutput: Float16Tensor, prompt: KreaPrompt) throws {
        guard encoderOutput.shape == [prompt.tokenIDs.count, Self.channels] else {
            throw AetherError.invalidTensor("Krea encoder output must match the prepared prompt rows")
        }
        let selected = Array(encoderOutput.values.dropFirst(KreaPrompt.leadingRows * Self.channels))
        try self.init(tensor: Float16Tensor(shape: [prompt.conditioningRows, Self.channels], values: selected))
    }

    /// Selects the rows after the 34-row system prefix of a reference prompt, image rows included.
    public init(encoderOutput: Float16Tensor, referencePrompt: KreaReferencePrompt) throws {
        guard encoderOutput.shape == [referencePrompt.tokenIDs.count, Self.channels] else {
            throw AetherError.invalidTensor("Krea encoder output must match the prepared reference prompt rows")
        }
        let selected = Array(encoderOutput.values.dropFirst(KreaPrompt.leadingRows * Self.channels))
        try self.init(tensor: Float16Tensor(shape: [referencePrompt.conditioningRows, Self.channels], values: selected))
    }

    /// Reference-fixture path: narrow once to fp16, rejecting overflow/nonfinite results.
    public init(provided tensor: FloatTensor) throws {
        guard tensor.shape.count == 2, (1...512).contains(tensor.shape[0]),
              tensor.shape[1] == Self.channels else {
            throw AetherError.invalidTensor("Krea conditioning must be [1...512,30720]")
        }
        try self.init(tensor: Float16Tensor(shape: tensor.shape, values: tensor.values.map(Float16.init)))
    }
}
