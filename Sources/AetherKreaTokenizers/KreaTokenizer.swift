import AetherKrea
import Tokenizers

/// Optional local Hugging Face tokenizer adapter. Never downloads tokenizer files.
public struct KreaTokenizer: Sendable {
    private let tokenizer: any Tokenizer

    public static func open(assets: KreaAssets) async throws -> Self {
        try assets.verifyTokenizer()
        let tokenizer = try await AutoTokenizer.from(modelFolder: assets.tokenizerDirectory)
        try assets.verifyTokenizer()
        return Self(tokenizer: tokenizer)
    }

    public func prepare(_ text: String) throws -> KreaPrompt {
        try KreaPrompt(renderedTokenIDs: tokenizer.encode(text: KreaPrompt.render(text), addSpecialTokens: false))
    }

    /// A `visionTokens` reference prompt: labelled image parts before the text, each pad expanded to its image's
    /// tokens. Truncation is decided by the text-only rendering, so images and labels are never cut.
    public func prepare(_ text: String, references: [KreaReferencePixels]) throws -> KreaReferencePrompt {
        let textOnly = tokenizer.encode(text: KreaPrompt.render(text), addSpecialTokens: false).count
        let ids = tokenizer.encode(text: KreaReferencePrompt.render(text, imageCount: references.count), addSpecialTokens: false)
        return try KreaReferencePrompt(renderedTokenIDs: ids, textOnlyTokenCount: textOnly, images: references)
    }
}
