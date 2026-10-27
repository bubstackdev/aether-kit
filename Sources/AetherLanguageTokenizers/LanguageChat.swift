import AetherKit

/// One chat message for a language model's own chat template. A part is text or an image; each image
/// renders as the model's image placeholder (for Qwen3-VL, `<|vision_start|><|image_pad|><|vision_end|>`),
/// in the order given, which is the order `VisionLanguageSession` expects its images.
public struct LanguageChatMessage: Sendable, Equatable {
    public enum Role: String, Sendable, Equatable { case system, user, assistant }
    public enum Part: Sendable, Equatable {
        case text(String)
        case image
    }

    public let role: Role
    public let parts: [Part]

    /// Only user messages carry images, and every message has at least one part. A message without images has
    /// exactly one text part: it renders as the template's plain string content, which has no part boundaries.
    public init(role: Role, parts: [Part]) throws {
        guard !parts.isEmpty else { throw AetherError.invalidRequest("a chat message needs at least one part") }
        guard parts.contains(.image) || parts.count == 1 else {
            throw AetherError.invalidRequest("a message without images has one text part")
        }
        guard role == .user || !parts.contains(.image) else {
            throw AetherError.invalidRequest("only user messages carry images")
        }
        self.role = role
        self.parts = parts
    }

    public static func system(_ text: String) -> Self { Self(checked: .system, [.text(text)]) }
    public static func assistant(_ text: String) -> Self { Self(checked: .assistant, [.text(text)]) }
    /// A user message: `images` placeholders first, then the text (Qwen-VL's usual layout).
    public static func user(_ text: String, images: Int = 0) -> Self {
        Self(checked: .user, Array(repeating: .image, count: max(0, images)) + [.text(text)])
    }

    private init(checked role: Role, _ parts: [Part]) { self.role = role; self.parts = parts }

    public var imageCount: Int { parts.filter { $0 == .image }.count }

    /// The template's message: plain string content when there are no images (every template's common path),
    /// typed content parts otherwise, as Qwen3-VL's template reads them.
    var templateMessage: [String: Any] {
        guard imageCount > 0 else {
            guard case .text(let text)? = parts.first else { return ["role": role.rawValue, "content": ""] }
            return ["role": role.rawValue, "content": text]
        }
        let content: [[String: Any]] = parts.map { part in
            switch part {
            case .text(let text): ["type": "text", "text": text]
            case .image: ["type": "image"]
            }
        }
        return ["role": role.rawValue, "content": content]
    }
}

/// A chat rendered by the model's template with the generation prompt, ready for
/// `LanguageGenerationSession.generate(prompt:)` or `VisionLanguageSession.generate(renderedTokenIDs:images:)`.
public struct LanguageChatPrompt: Sendable, Equatable {
    public let tokenIDs: [Int]
    /// How many image placeholders the prompt holds: pass exactly that many images, in message order.
    public let imageCount: Int
}
