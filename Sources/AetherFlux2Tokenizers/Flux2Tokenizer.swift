import AetherFlux2
import AetherKit
import AetherLanguageTokenizers

/// Optional local adapter for the bundle's Qwen3 tokenizer and chat template, through `AetherLanguageTokenizers`.
/// Never downloads files. The template is one of the hashed tokenizer files, so it is part of the bundle fingerprint.
public struct Flux2Tokenizer: Sendable {
    private let tokenizer: LanguageTokenizer

    public static func open(assets: Flux2Assets) async throws -> Self {
        try assets.verifyTokenizer()
        let tokenizer = try await LanguageTokenizer.open(directory: assets.tokenizerDirectory)
        try assets.verifyTokenizer()
        guard tokenizer.hasChatTemplate else { throw AetherError.invalidModel("FLUX.2 tokenizer has no chat template") }
        return Self(tokenizer: tokenizer)
    }

    /// One user message rendered by the bundle's template with the generation prompt and thinking off, then padded or
    /// truncated by `Flux2Prompt`.
    public func prepare(_ text: String) throws -> Flux2Prompt {
        try Flux2Prompt(renderedTokenIDs: tokenizer.render([.user(text)], thinking: false).tokenIDs)
    }
}
