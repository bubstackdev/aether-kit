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
}
