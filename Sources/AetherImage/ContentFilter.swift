import AetherCore
public import AetherRuntime

/// The app's content filter for a generator whose license requires one (Krea 2: the Krea 2 Community License §4.2,
/// "any deployment must implement content filters").
///
/// **The filter is the app's responsibility.** The license binds the deployer, which is the app, not Aether-Kit, and
/// the kit can't judge content: it can only make sure every generation passes through the filter the app supplies.
/// There is no default and no built-in pass-through. A filter that allows everything is the app's own choice, and the
/// app owns it. Pick a classifier or moderation service suited to the Acceptable Use Policy that ships with the model.
///
/// A pipeline calls the filter twice per generation:
/// 1. ``review(prompt:)`` before anything runs (no graph loads, no GPU time for a refused prompt);
/// 2. ``review(image:prompt:)`` on the finished image before it is returned. A refused image is discarded.
///
/// A refusal throws ``AetherError/GenerationFailure/contentRefused(stage:reason:)``. A filter that throws fails the
/// generation with its own error: an unreachable moderation service blocks output rather than letting it through.
public protocol ContentFilter: Sendable {
    /// Reviews a prompt before it is encoded.
    func review(prompt: String) async throws -> ContentVerdict
    /// Reviews a generated image before it is returned. `prompt` is the request's.
    func review(image: GeneratedImage, prompt: String) async throws -> ContentVerdict
}

/// A ``ContentFilter``'s decision.
public enum ContentVerdict: Sendable, Hashable {
    case allow
    /// The reason is passed on in the error.
    case refuse(reason: String)
}

extension ContentFilter {
    /// Throws `contentRefused(stage: .prompt, …)` unless the filter allows `prompt`.
    func requireAllowed(prompt: String) async throws {
        if case .refuse(let reason) = try await review(prompt: prompt) {
            throw AetherError.generation(.contentRefused(stage: .prompt, reason: reason))
        }
    }

    /// Throws `contentRefused(stage: .output, …)` unless the filter allows `image`.
    func requireAllowed(image: GeneratedImage, prompt: String) async throws {
        if case .refuse(let reason) = try await review(image: image, prompt: prompt) {
            throw AetherError.generation(.contentRefused(stage: .output, reason: reason))
        }
    }
}
