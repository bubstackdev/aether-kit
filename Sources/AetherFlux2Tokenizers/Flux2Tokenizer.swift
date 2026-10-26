import AetherFlux2
import Tokenizers

/// Optional local Hugging Face tokenizer adapter for the bundle's Qwen3 tokenizer. Never downloads files.
public struct Flux2Tokenizer: Sendable {
    private let tokenizer: any Tokenizer

    public static func open(assets: Flux2Assets) async throws -> Self {
        try assets.verifyTokenizer()
        let tokenizer = try await AutoTokenizer.from(modelFolder: assets.tokenizerDirectory)
        try assets.verifyTokenizer()
        return Self(tokenizer: tokenizer)
    }

    public func prepare(_ text: String) throws -> Flux2Prompt {
        try Flux2Prompt(renderedTokenIDs: tokenizer.encode(text: Flux2Prompt.render(text), addSpecialTokens: false))
    }
}
