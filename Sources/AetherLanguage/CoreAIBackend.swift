import AetherKit
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import Foundation

extension AetherLanguageModel {
    /// Whether the Core AI backend can run on this OS. The package floor is macOS 14 and iOS 17; Core AI needs 27.
    static var coreAIAvailable: Bool {
        if #available(macOS 27, iOS 27, *) { true } else { false }
    }

    /// The Core AI backend and the bundle's tokenizer, each fully verified (checksums included) as it opens.
    static func loadCoreAI(bundle: URL, platform: LanguageModelManifest.Platform) async throws -> LoadedLanguageModel {
        guard #available(macOS 27, iOS 27, *) else { throw AetherLanguageError.unavailable(.unsupportedOS) }
        let tokenizer = try await LanguageTokenizer.open(bundle: bundle)
        let bytes = try LanguageTokenizer.tokenBytes(bundle: bundle)
        let session = try await LanguageGenerationSession.open(bundle: bundle, platform: platform)
        return LoadedLanguageModel(backend: session, codec: TokenizerCodec(tokenizer: tokenizer, bytes: bytes))
    }
}
