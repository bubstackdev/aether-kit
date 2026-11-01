@_spi(Qualification) import AetherLanguageRuntime

/// The Core AI backend of the app-facing language runtime. Image prompts stay on this session's own API.
@available(macOS 27, iOS 27, *)
extension LanguageGenerationSession: LanguageBackend {}

/// Teacher-forced logits for backend-neutral parity harnesses.
@available(macOS 27, iOS 27, *)
@_spi(Qualification) extension LanguageGenerationSession: LanguageBackendQualification {}

/// The hybrid recurrent-state backend (Qwen3.5, Granite 4.0-H).
@available(macOS 27, iOS 27, *)
extension HybridLanguageSession: LanguageBackend {}

@available(macOS 27, iOS 27, *)
@_spi(Qualification) extension HybridLanguageSession: LanguageBackendQualification {}
