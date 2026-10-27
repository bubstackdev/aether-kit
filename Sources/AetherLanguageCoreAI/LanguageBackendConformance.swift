@_spi(Qualification) import AetherLanguageRuntime

/// The Core AI backend of the app-facing language runtime. Image prompts stay on this session's own API.
extension LanguageGenerationSession: LanguageBackend {}

/// Teacher-forced logits for backend-neutral parity harnesses.
@_spi(Qualification) extension LanguageGenerationSession: LanguageBackendQualification {}
