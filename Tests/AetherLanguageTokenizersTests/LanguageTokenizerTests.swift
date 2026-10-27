import AetherKit
@testable import AetherLanguageTokenizers
import CryptoKit
import Foundation
import Testing

private let environment = ProcessInfo.processInfo.environment

@Test func chatMessagesKeepTheirPartsAndRefuseImagesOutsideUserTurns() throws {
    let user = LanguageChatMessage.user("Describe her look.", images: 2)
    #expect(user.parts == [.image, .image, .text("Describe her look.")] && user.imageCount == 2)
    #expect(LanguageChatMessage.user("Hello").parts == [.text("Hello")])
    #expect(LanguageChatMessage.system("Rules").role == .system && LanguageChatMessage.assistant("Teal.").role == .assistant)
    let after = try LanguageChatMessage(role: .user, parts: [.text("Describe the outfit."), .image])
    #expect(after.imageCount == 1)
    #expect(throws: AetherError.self) { try LanguageChatMessage(role: .system, parts: [.image, .text("x")]) }
    #expect(throws: AetherError.self) { try LanguageChatMessage(role: .user, parts: []) }
    // Without images a message is one plain string; two text parts would lose their boundary.
    #expect(throws: AetherError.self) { try LanguageChatMessage(role: .user, parts: [.text("a"), .text("b")]) }
    #expect(try LanguageChatMessage(role: .user, parts: [.text("a"), .image, .text("b")]).imageCount == 1)
}

@Test func templatesResolveLikeTheLoaderAndDropOneTrailingNewlineAsJinjaDoes() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("language-template-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    func write(_ name: String, _ object: Any) throws {
        try JSONSerialization.data(withJSONObject: object).write(to: folder.appendingPathComponent(name))
    }
    #expect(try LanguageTokenizer.template(in: folder) == nil)
    // tokenizer_config.json: a string, or the named list's "default".
    try write("tokenizer_config.json", ["chat_template": [["name": "tool_use", "template": "T"], ["name": "default", "template": "D\n\n"]]])
    #expect(try LanguageTokenizer.template(in: folder) == "D\n")
    // chat_template.json wins over tokenizer_config.json, and chat_template.jinja over both.
    try write("chat_template.json", ["chat_template": "J\r\n"])
    #expect(try LanguageTokenizer.template(in: folder) == "J")
    try Data("{{- 'X' }}\n".utf8).write(to: folder.appendingPathComponent("chat_template.jinja"))
    #expect(try LanguageTokenizer.template(in: folder) == "{{- 'X' }}")
}

@Test func aBundleOpensOnlyWithItsRegisteredTokenizer() async throws {
    let bundle = FileManager.default.temporaryDirectory.appendingPathComponent("language-tokenizer-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: bundle) }
    let tokenizer = bundle.appendingPathComponent("tokenizer")
    try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
    let contents = Data(#"{"not":"a tokenizer"}"#.utf8)
    try contents.write(to: tokenizer.appendingPathComponent("tokenizer.json"))
    func register(tokenizerDirectory: String, hash: String, files: [String: String]? = nil) throws {
        var manifest: [String: Any] = [
            "schemaVersion": 1, "modelDirectory": "model.aimodel", "tokenizerDirectory": tokenizerDirectory,
            "sourceRepository": "Qwen/Qwen3-0.6B", "sourceRevision": String(repeating: "a", count: 40),
            "modelSHA256": String(repeating: "b", count: 64), "tokenizerSHA256": hash, "maximumContextLength": 512,
            "stopTokenIDs": [151645], "platform": "macos", "quantization": "fp16",
        ]
        if let files { manifest["tokenizerFiles"] = files }
        try JSONSerialization.data(withJSONObject: manifest).write(to: bundle.appendingPathComponent("language-model.json"))
    }
    let digest = SHA256.hash(data: contents).map { String(format: "%02x", $0) }.joined()
    // The registered hash and a path inside the bundle verify (no model asset is hashed).
    try register(tokenizerDirectory: "tokenizer", hash: digest)
    let manifest = try LanguageModelManifest.load(from: bundle)
    #expect(try manifest.verifyTokenizer(in: bundle).lastPathComponent == "tokenizer")
    // A mismatched hash or an escaping path refuses before any tokenizer loads.
    try register(tokenizerDirectory: "tokenizer", hash: String(repeating: "0", count: 64))
    await #expect(throws: AetherError.self) { try await LanguageTokenizer.open(bundle: bundle) }
    try register(tokenizerDirectory: "../tokenizer", hash: digest)
    await #expect(throws: AetherError.self) { try await LanguageTokenizer.open(bundle: bundle) }
    // With `tokenizerFiles`, every tokenizer file is covered: the chat template too.
    func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    let template = Data("{{- 'instruct' }}".utf8)
    try template.write(to: tokenizer.appendingPathComponent("chat_template.jinja"))
    let files = ["tokenizer.json": digest, "chat_template.jinja": hash(template)]
    try register(tokenizerDirectory: "tokenizer", hash: digest, files: files)
    #expect(try LanguageModelManifest.load(from: bundle).verifyTokenizer(in: bundle).lastPathComponent == "tokenizer")
    // A swapped template, an unregistered extra file, or a list that disagrees with tokenizerSHA256 refuses.
    try Data("{{- '<think>' }}".utf8).write(to: tokenizer.appendingPathComponent("chat_template.jinja"))
    #expect(throws: AetherError.self) { try LanguageModelManifest.load(from: bundle).verifyTokenizer(in: bundle) }
    await #expect(throws: AetherError.self) { try await LanguageTokenizer.open(bundle: bundle) }
    try template.write(to: tokenizer.appendingPathComponent("chat_template.jinja"))
    try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("chat_template.json"))
    #expect(throws: AetherError.self) { try LanguageModelManifest.load(from: bundle).verifyTokenizer(in: bundle) }
    try FileManager.default.removeItem(at: tokenizer.appendingPathComponent("chat_template.json"))
    #expect(try LanguageModelManifest.load(from: bundle).verifyTokenizer(in: bundle).lastPathComponent == "tokenizer")
    try register(tokenizerDirectory: "tokenizer", hash: digest, files: files.merging(["tokenizer.json": String(repeating: "c", count: 64)]) { $1 })
    #expect(throws: AetherError.self) { try LanguageModelManifest.load(from: bundle).verifyTokenizer(in: bundle) }
}

/// Opt-in: every case in the upstream oracle (`Forge/language/reference_chat_tokens.py`, Transformers' own
/// `apply_chat_template`) from the pinned local snapshots under `AETHER_LANGUAGE_TOKENIZER_HUB` (a Hugging Face hub
/// cache). Reads tokenizer files only; loads no model.
@Test(.enabled(if: environment["AETHER_LANGUAGE_TOKENIZER_HUB"] != nil))
func renderingMatchesUpstreamTokenIDsForEveryModel() async throws {
    let hub = URL(fileURLWithPath: try #require(environment["AETHER_LANGUAGE_TOKENIZER_HUB"]))
    let fixture = try #require(Bundle.module.url(forResource: "chat-oracle", withExtension: "json", subdirectory: "Fixtures"))
    let oracle = try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: fixture))
    var cases = 0
    for model in oracle.models {
        let snapshot = hub.appendingPathComponent("models--" + model.repository.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent("snapshots/\(model.revision)")
        let tokenizer = try await LanguageTokenizer.open(directory: snapshot)
        #expect(tokenizer.hasChatTemplate, "\(model.repository)")
        for entry in model.cases {
            let messages = try entry.messages.map { try $0.message() }
            let prompt = try tokenizer.render(messages, thinking: entry.thinking)
            #expect(prompt.tokenIDs == entry.tokenIDs, "\(model.repository) \(entry.name)")
            #expect(tokenizer.decode(prompt.tokenIDs, skipSpecialTokens: false) == entry.text, "\(model.repository) \(entry.name)")
            #expect(tokenizer.encode(entry.text) == entry.tokenIDs, "\(model.repository) \(entry.name)")
            #expect(prompt.imageCount == messages.reduce(0) { $0 + $1.imageCount })
            cases += 1
        }
        #expect(tokenizer.decode(model.decode.tokenIDs) == model.decode.text, "\(model.repository) decode")
        #expect(tokenizer.tokenID("<|im_end|>") == 151645)
    }
    #expect(cases == 24)
}

/// Opt-in: the hybrid Qwen3 template with thinking off is exactly FLUX.2's hand-written render, so
/// `AetherFlux2Tokenizers` can delegate here without changing a token.
@Test(.enabled(if: environment["AETHER_LANGUAGE_TOKENIZER_HUB"] != nil))
func thinkingOffMatchesFlux2sRenderedPrompt() async throws {
    let hub = URL(fileURLWithPath: try #require(environment["AETHER_LANGUAGE_TOKENIZER_HUB"]))
    let snapshot = hub.appendingPathComponent("models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    let tokenizer = try await LanguageTokenizer.open(directory: snapshot)
    for text in ["a red fox in the snow", "Café — 🌙\nsecond line"] {
        let flux = "<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        #expect(try tokenizer.render([.user(text)]).tokenIDs == tokenizer.encode(flux))
    }
}

private struct Oracle: Decodable {
    let models: [Model]
    struct Model: Decodable {
        let repository: String
        let revision: String
        let cases: [Case]
        let decode: Decode
    }
    struct Case: Decodable {
        let name: String
        let messages: [Message]
        let thinking: Bool
        let text: String
        let tokenIDs: [Int]
    }
    struct Decode: Decodable {
        let tokenIDs: [Int]
        let text: String
    }
    struct Message: Decodable {
        let role: String
        let parts: [Part]
        struct Part: Decodable {
            let text: String?
            let image: Bool?
        }
        func message() throws -> LanguageChatMessage {
            let role = try #require(LanguageChatMessage.Role(rawValue: role))
            return try LanguageChatMessage(role: role, parts: parts.map { $0.image == true ? .image : .text($0.text ?? "") })
        }
    }
}
