import Foundation
import Testing
@testable import AetherText

/// Where the lab keeps the installed bundles (`<models>/<id>` is the revision root, as in the runtime gates).
enum Lab {
    static var models: URL {
        if let override = ProcessInfo.processInfo.environment["AETHER_LAB_MODELS"] { return URL(fileURLWithPath: override) }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appending(path: "lab/models")
    }

    static let mac = Device(platform: .macOS, architecture: "h17c")

    static func bundle(_ id: String) throws -> InstalledBundle {
        InstalledBundle(ref: try ModelRef.resolve(id, for: mac), url: models.appending(path: id), isStale: false)
    }

    static func manifest(_ id: String) throws -> Manifest {
        try Manifest.decode(Data(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "AetherCoreTests/Fixtures/zoo/\(id)/aether.json")))
    }

    /// Whether the tokenizer files of the chat goldens are present. They are 5–33 MB each (163 MB for the nine), too
    /// large to check in; the HF reference ids are checked in instead, and each tokenizer.json is verified against
    /// the catalog sha256 on load.
    static var hasTokenizers: Bool {
        FileManager.default.fileExists(atPath: models.appending(path: "qwen3-0.6b/macos/tokenizer/tokenizer.json").path)
    }
}

/// TokenizerAdapter + JinjaChatTemplate against HF `transformers` ids (Fixtures/tokenizer-parity-hf.json, recorded by
/// scripts/tokenizer-parity-reference): every chat golden × 3 message shapes, token for token. Hermetic (no network,
/// no GPU); it needs the installed tokenizer files from lab/models.
@Suite("Tokenizer parity with HF transformers", .enabled(if: Lab.hasTokenizers))
struct TokenizerParityTests {
    struct Reference: Decodable {
        struct Case: Decodable { var prompt: String; var system: String?; var thinking: Bool? }
        struct Model: Decodable { var ids: [[Int32]]; var tokenizerSha256: String }
        var cases: [Case]
        var models: [String: Model]
    }

    static let reference: Reference = {
        let url = Bundle.module.url(forResource: "Fixtures/tokenizer-parity-hf", withExtension: "json")!
        return try! JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
    }()

    static let ids = reference.models.keys.sorted()

    @Test(arguments: ids)
    func matchesHF(_ id: String) async throws {
        let bundle = try Lab.bundle(id)
        let manifest = try Lab.manifest(id)
        let tokenizer = try await TokenizerAdapter(bundle: bundle, manifest: manifest)
        let renderer = try PromptRenderers.make(manifest: manifest, bundle: bundle, tokenizer: tokenizer)
        let expected = try #require(Self.reference.models[id])
        for (i, c) in Self.reference.cases.enumerated() {
            let messages = (c.system.map { [ChatMessage.system($0)] } ?? []) + [.user(c.prompt)]
            let got = try renderer.render(messages, options: RenderOptions(thinking: c.thinking)).tokens
            let want = expected.ids[i]
            let diff = zip(got, want).enumerated().first { $0.element.0 != $0.element.1 }?.offset
            #expect(got == want, "\(id) case \(i): first difference at \(diff ?? min(got.count, want.count)) (\(got.count) vs \(want.count) tokens)")
        }
        #expect(!tokenizer.stopTokenIDs.isEmpty)
    }

    @Test func stopsComeFromTheManifest() async throws {
        let tokenizer = try await TokenizerAdapter(bundle: try Lab.bundle("qwen3-0.6b"), manifest: try Lab.manifest("qwen3-0.6b"))
        #expect(tokenizer.stopTokenIDs == [151645, 151643])  // <|im_end|>, <|endoftext|>
        var manifest = try Lab.manifest("qwen3-0.6b")
        manifest.tokenizer?.stop = ["<|not a token|>"]
        await #expect(throws: AetherError.self) { _ = try await TokenizerAdapter(bundle: try Lab.bundle("qwen3-0.6b"), manifest: manifest) }
    }

    /// The "template ignores enable_thinking" fallback: an empty block is appended when thinking is off.
    @Test func thinkingFallbackClosesTheBlock() async throws {
        let manifest = try Lab.manifest("qwen3-0.6b")
        let tokenizer = try await TokenizerAdapter(bundle: try Lab.bundle("qwen3-0.6b"), manifest: manifest)
        let ignoring = JinjaChatTemplate(
            tokenizer: tokenizer,
            template: "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n",
            thinking: manifest.tokenizer?.thinking)
        let off = try ignoring.render([.user("Hi")], options: RenderOptions(thinking: false))
        #expect(tokenizer.decode(off.tokens).hasSuffix("<|im_start|>assistant<think>\n\n</think>\n\n"))  // Jinja drops the template's final newline
        #expect(!off.startsInThinking)
        let on = try ignoring.render([.user("Hi")], options: RenderOptions(thinking: true))
        #expect(tokenizer.decode(on.tokens).hasSuffix("<|im_start|>assistant"))
        // The real template honours the flag, so nothing is appended twice.
        let real = try PromptRenderers.make(manifest: manifest, bundle: try Lab.bundle("qwen3-0.6b"), tokenizer: tokenizer)
        let realOff = tokenizer.decode(try real.render([.user("Hi")], options: RenderOptions(thinking: false)).tokens)
        #expect(realOff.components(separatedBy: "</think>").count == 2)
        // Qwen3.5's template opens the block itself when thinking is on: generation starts inside it.
        let q35 = try Lab.manifest("qwen3.5-0.8b")
        let q35Tokenizer = try await TokenizerAdapter(bundle: try Lab.bundle("qwen3.5-0.8b"), manifest: q35)
        let q35Renderer = try PromptRenderers.make(manifest: q35, bundle: try Lab.bundle("qwen3.5-0.8b"), tokenizer: q35Tokenizer)
        #expect(try q35Renderer.render([.user("Hi")], options: RenderOptions(thinking: true)).startsInThinking)
    }

    /// Gemma 4 (`renderer: gemma4-turns`, `addBos: renderer`) is refused until its renderer exists.
    @Test func gemma4RendererIsNotImplementedYet() async throws {
        let manifest = try Lab.manifest("gemma-4-e2b")
        #expect(manifest.tokenizer?.addBos == .renderer)
        await #expect(throws: AetherError.self) { _ = try await TokenizerAdapter(bundle: try Lab.bundle("gemma-4-e2b"), manifest: manifest) }
    }
}
