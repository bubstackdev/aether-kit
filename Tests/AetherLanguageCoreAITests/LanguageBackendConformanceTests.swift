@_spi(Qualification) import AetherLanguageCoreAI
import Testing

struct LanguageBackendConformanceTests {
    /// The Core AI session is a `LanguageBackend` (and qualification backend), seen through `AetherLanguageCoreAI` alone.
    @Test func coreAISessionIsALanguageBackend() {
        let backend: any LanguageBackend.Type = LanguageGenerationSession.self
        let qualification: any LanguageBackendQualification.Type = LanguageGenerationSession.self
        #expect(backend == LanguageGenerationSession.self && qualification == LanguageGenerationSession.self)
        #expect(LanguageGenerationRequest(prompt: [1], maximumTokens: 1).sampling == .greedy)
    }
}
