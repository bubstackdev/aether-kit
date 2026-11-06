import CryptoKit
import Foundation
import Testing
@testable import AetherText
import Tokenizers

/// The 0.0.x tokenizer oracles, folded into the parity suite: `chat-oracle.json` (Transformers 4.57.3's own
/// `apply_chat_template`, `Forge/language/reference_chat_tokens.py`; 24 cases over four Qwen3-family checkpoints) and
/// `olmo2-chat-oracle.json` (5 OLMo2 cases). Each checkpoint's small tokenizer files (`tokenizer_config.json`, and a
/// `chat_template.jinja`/`.json` where it keeps its template) are checked in under Fixtures/chat-oracle; the
/// heavy `tokenizer.json` is found by its SHA-256:
/// - in lab/models (qwen3-0.6b's is Qwen3's, which the Qwen3-0.6B and Qwen3-4B-2507 checkpoints share), or
/// - in a Hugging Face hub cache at `AETHER_TOKENIZER_HUB` (`models--org--name/snapshots/<revision>/tokenizer.json`).
/// A checkpoint whose tokenizer.json is in neither place is skipped (and reported).
///
/// The 4 image cases of the two Qwen3-VL checkpoints are not run: chat messages are text-only until M4.
@Suite("Tokenizer parity: 0.0.x oracle")
struct TokenizerOracleTests {
    struct Oracle: Decodable {
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
            let tokenIDs: [Int32]
        }
        struct Decode: Decodable {
            let tokenIDs: [Int32]
            let text: String
        }
        struct Message: Decodable {
            let role: String
            let parts: [Part]
            struct Part: Decodable {
                let text: String?
                let image: Bool?
            }
        }
        let models: [Model]
    }

    struct Olmo: Decodable {
        let repository: String
        let revision: String
        let tokenizerSHA256: String
        let cases: [Oracle.Case]
        let decode: Oracle.Decode
    }

    /// Each oracle checkpoint's tokenizer.json, by SHA-256 (recorded from the pinned revisions).
    static let tokenizerSHA256: [String: String] = [
        "Qwen/Qwen3-0.6B": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "Qwen/Qwen3-VL-2B-Instruct": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
        "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "allenai/OLMo-2-0425-1B-Instruct": "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca",
    ]

    /// Checkpoints whose template has a thinking switch (the manifest would declare `thinking`).
    static let hybrid: Set<String> = ["Qwen/Qwen3-0.6B"]

    static var fixtures: URL { Bundle.module.url(forResource: "Fixtures/chat-oracle", withExtension: nil)! }

    static func sha256(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The checkpoint's tokenizer.json, wherever a copy with the recorded hash is.
    static func tokenizerJSON(_ repository: String, revision: String) -> URL? {
        guard let expected = tokenizerSHA256[repository] else { return nil }
        var candidates: [URL] = []
        if let hub = ProcessInfo.processInfo.environment["AETHER_TOKENIZER_HUB"] {
            candidates.append(URL(fileURLWithPath: hub).appending(path: "models--" + repository.replacingOccurrences(of: "/", with: "--"))
                .appending(path: "snapshots/\(revision)/tokenizer.json"))
        }
        candidates.append(Lab.models.appending(path: "qwen3-0.6b/macos/tokenizer/tokenizer.json"))
        return candidates.first { sha256($0) == expected }
    }

    /// A tokenizer folder for the checkpoint: its checked-in small files plus its tokenizer.json (symlinked).
    static func folder(_ repository: String, revision: String) throws -> URL? {
        guard let json = tokenizerJSON(repository, revision: revision) else { return nil }
        let source = fixtures.appending(path: repository.replacingOccurrences(of: "/", with: "--"))
        let folder = FileManager.default.temporaryDirectory.appending(path: "aether-oracle-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for name in try FileManager.default.contentsOfDirectory(atPath: source.path) {
            try FileManager.default.copyItem(at: source.appending(path: name), to: folder.appending(path: name))
        }
        try FileManager.default.createSymbolicLink(at: folder.appending(path: "tokenizer.json"), withDestinationURL: json)
        return folder
    }

    /// The chat template from the same sources, in the same order, as the swift-transformers loader:
    /// `chat_template.jinja`, then `chat_template.json`, then `tokenizer_config.json` (a string, or the list's
    /// "default"). Verbatim: the trailing newline is swift-jinja's to drop, as Jinja does.
    static func template(in folder: URL) throws -> String? {
        func json(_ name: String) throws -> [String: Any]? {
            let url = folder.appending(path: name)
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            return try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        }
        let jinja = folder.appending(path: "chat_template.jinja")
        if FileManager.default.fileExists(atPath: jinja.path) { return try String(contentsOf: jinja, encoding: .utf8) }
        if let value = try json("chat_template.json")?["chat_template"] as? String { return value }
        if let value = try json("tokenizer_config.json")?["chat_template"] {
            return value as? String ?? (value as? [[String: Any]])?.first { $0["name"] as? String == "default" }?["template"] as? String
        }
        return nil
    }

    struct Loaded {
        let adapter: TokenizerAdapter
        let renderer: JinjaChatTemplate
    }

    static func load(_ repository: String, revision: String) async throws -> Loaded? {
        guard let folder = try folder(repository, revision: revision) else { return nil }
        defer { try? FileManager.default.removeItem(at: folder) }
        let tokenizer = try await AutoTokenizer.from(modelFolder: folder)
        let bytes = try TokenBytes(tokenizerJSON: Data(contentsOf: folder.appending(path: "tokenizer.json")))
        let block = Manifest.Tokenizer(source: .dir(AssetPath(asset: "main", path: "tokenizer")))
        let adapter = try TokenizerAdapter(tokenizer: tokenizer, block: block, forGeneration: false, tokenBytes: bytes)
        let thinking = hybrid.contains(repository)
            ? Manifest.Thinking(open: "<think>", close: "</think>", toggle: .templateVar("enable_thinking")) : nil
        let template = try #require(try Self.template(in: folder), "\(repository) has a chat template")
        return Loaded(adapter: adapter, renderer: JinjaChatTemplate(tokenizer: adapter, template: template, thinking: thinking))
    }

    /// Every text case: the rendered ids, their decode and their bytes equal upstream's, and encoding upstream's text
    /// gives its ids. The decode case: the byte table and the tokenizer decode alike.
    static func check(_ loaded: Loaded, _ repository: String, _ cases: [Oracle.Case], _ decode: Oracle.Decode) throws -> Int {
        var checked = 0
        for entry in cases {
            guard entry.messages.allSatisfy({ $0.parts.allSatisfy { $0.image != true } }) else { continue }
            let messages = entry.messages.map { ChatMessage(ChatMessage.Role(rawValue: $0.role)!, $0.parts.compactMap(\.text).joined()) }
            let ids = try loaded.renderer.render(messages, options: RenderOptions(thinking: entry.thinking)).tokens
            #expect(ids == entry.tokenIDs, "\(repository) \(entry.name)")
            #expect(loaded.adapter.decode(entry.tokenIDs) == entry.text, "\(repository) \(entry.name) decode")
            #expect(loaded.adapter.encode(entry.text, addSpecialTokens: false) == entry.tokenIDs, "\(repository) \(entry.name) encode")
            let bytes = try #require(loaded.adapter.tokenBytes).bytes(of: entry.tokenIDs.map(Int.init), skipSpecialTokens: false)
            #expect(String(decoding: bytes, as: UTF8.self) == entry.text, "\(repository) \(entry.name) bytes")
            checked += 1
        }
        let table = try #require(loaded.adapter.tokenBytes)
        #expect(String(decoding: table.bytes(of: decode.tokenIDs.map(Int.init)), as: UTF8.self) == decode.text, "\(repository) decode bytes")
        return checked
    }

    @Test func chatOracleMatchesUpstreamForEveryTextCase() async throws {
        let oracle = try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: Self.fixtures.appending(path: "chat-oracle.json")))
        #expect(oracle.models.reduce(0) { $0 + $1.cases.count } == 24)
        var checked = 0, skipped: [String] = []
        for model in oracle.models {
            guard let loaded = try await Self.load(model.repository, revision: model.revision) else {
                skipped.append(model.repository)
                continue
            }
            checked += try Self.check(loaded, model.repository, model.cases, model.decode)
            #expect(loaded.adapter.tokenID("<|im_end|>") == 151645)
        }
        print("ORACLE chat: \(checked) text cases checked; skipped (no tokenizer.json with the recorded hash): \(skipped)")
        if skipped.isEmpty { #expect(checked == 20) }
    }

    @Test func olmo2OracleMatchesUpstream() async throws {
        let oracle = try JSONDecoder().decode(Olmo.self, from: Data(contentsOf: Self.fixtures.appending(path: "olmo2-chat-oracle.json")))
        #expect(oracle.tokenizerSHA256 == Self.tokenizerSHA256[oracle.repository])
        guard let loaded = try await Self.load(oracle.repository, revision: oracle.revision) else {
            print("ORACLE olmo2: skipped (no tokenizer.json with the recorded hash; set AETHER_TOKENIZER_HUB)")
            return
        }
        #expect(try Self.check(loaded, oracle.repository, oracle.cases, oracle.decode) == 5)
    }

    /// The 0.0.x trailing-newline fix: upstream compiles templates with `keep_trailing_newline=False`, dropping one
    /// trailing newline from the source. swift-jinja does the same, so the source goes in verbatim (stripping it
    /// again would drop a second one).
    @Test(.enabled(if: Lab.hasTokenizers)) func templatesDropOneTrailingNewlineAsJinjaDoes() async throws {
        let manifest = try Lab.manifest("qwen3-0.6b")
        let adapter = try await TokenizerAdapter(bundle: try Lab.bundle("qwen3-0.6b"), manifest: manifest)
        func render(_ template: String) throws -> String {
            adapter.decode(try adapter.applyChatTemplate([.user("x")], template: template))
        }
        #expect(try render("{{- 'D' }}\n\n") == "D\n")
        #expect(try render("{{- 'D' }}\n") == "D")
        #expect(try render("{{- 'J' }}\r\n") == "J")
        #expect(try render("{{- 'K' }}") == "K")
    }
}
