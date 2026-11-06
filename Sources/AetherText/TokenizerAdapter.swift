public import AetherRuntime
import CryptoKit
import Foundation
import Tokenizers

/// Turns token ids back into text. The Generator needs only this (tests drive it with a fake).
public protocol TokenDecoding: Sendable {
    func decode(_ ids: [Int32]) -> String
}

/// The one tokenizer adapter (DESIGN.md §7), over swift-transformers. Three rules from measurement:
/// - **BOS:** the chat template owns BOS; the runtime adds none (`addBos: template`, 33/33 identical to HF).
/// - **Stops:** the manifest's `tokenizer.stop` set (the source's `generation_config` eos list ∪ the tokenizer eos,
///   pinned per entry). Nothing is hardcoded; a stop that is not a single token of this vocab fails the load.
/// - **Source:** only the installed, hashed asset. Never a hub fetch: files are read from the bundle (and checked
///   against the catalog's sha256 when the install lists them).
public struct TokenizerAdapter: TokenDecoding {
    let tokenizer: any Tokenizer
    public let stopTokenIDs: Set<Int32>
    public let addBos: Manifest.AddBos
    public let addSpecialTokens: Bool

    /// Loads the manifest's tokenizer from an installed bundle.
    public init(bundle: InstalledBundle, manifest: Manifest) async throws {
        guard let block = manifest.tokenizer else {
            throw AetherError.manifestInvalid(field: "tokenizer", reason: "the manifest has no tokenizer")
        }
        let (directory, asset) = try Self.directory(for: block.source, in: bundle)
        try Self.verify(directory: directory, asset: asset)
        try await self.init(directory: directory, block: block)
    }

    /// Loads tokenizer files already verified in `directory` (local files only).
    init(directory: URL, block: Manifest.Tokenizer) async throws {
        let loaded: any Tokenizer
        do {
            loaded = try await AutoTokenizer.from(modelFolder: directory)
        } catch {
            throw AetherError.runtime(underlying: error)
        }
        try self.init(tokenizer: loaded, block: block)
    }

    /// `forGeneration`: the tokenizer serves a generating driver, so `tokenizer.stop` is required (MANIFEST §2.4).
    init(tokenizer: any Tokenizer, block: Manifest.Tokenizer, forGeneration: Bool = true) throws {
        self.tokenizer = tokenizer
        addBos = block.addBos ?? .template
        switch addBos {
        case .template, .never:
            // The chat template owns BOS; the runtime adds none (with `never` there is nothing to add either).
            break
        case .always, .renderer:
            throw AetherError.unsupportedOperation("tokenizer.addBos \(addBos) is not implemented; only template (or false)")
        }
        if forGeneration, (block.stop ?? []).isEmpty {
            throw AetherError.manifestInvalid(field: "tokenizer.stop", reason: "a generating driver needs at least one stop marker")
        }
        addSpecialTokens = block.addSpecialTokens ?? true
        var stops = Set<Int32>()
        for (i, stop) in (block.stop ?? []).enumerated() {
            switch stop {
            case .id(let id):
                stops.insert(Int32(id))
            case .text(let text):
                guard let id = Self.singleToken(text, tokenizer) else {
                    throw AetherError.manifestMismatch(
                        field: "tokenizer.stop.\(i)", expected: "a single token of the vocabulary", found: "'\(text)'")
                }
                stops.insert(id)
            }
        }
        stopTokenIDs = stops
    }

    /// The id of `text` if it is exactly one token (the id maps back to the same string; an UNK fallback does not).
    static func singleToken(_ text: String, _ tokenizer: any Tokenizer) -> Int32? {
        guard let id = tokenizer.convertTokenToId(text), tokenizer.convertIdToToken(id) == text else { return nil }
        return Int32(id)
    }

    public func tokenID(_ text: String) -> Int32? { Self.singleToken(text, tokenizer) }

    public func decode(_ ids: [Int32]) -> String {
        tokenizer.decode(tokens: ids.map(Int.init), skipSpecialTokens: false)
    }

    /// Plain-text encoding. Chat prompts go through the template instead; this follows `addSpecialTokens`.
    public func encode(_ text: String, addSpecialTokens: Bool? = nil) -> [Int32] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens ?? self.addSpecialTokens).map(Int32.init)
    }

    /// Renders `messages` with `template` (the template owns BOS) and tokenizes the result.
    public func applyChatTemplate(
        _ messages: [ChatMessage], template: String, context: [String: any Sendable] = [:], addGenerationPrompt: Bool = true
    ) throws -> [Int32] {
        let rendered: [[String: any Sendable]] = messages.map { ["role": $0.role.rawValue, "content": $0.content] }
        do {
            return try tokenizer.applyChatTemplate(
                messages: rendered, chatTemplate: .literal(template), addGenerationPrompt: addGenerationPrompt,
                truncation: false, maxLength: nil, tools: nil, additionalContext: context.isEmpty ? nil : context
            ).map(Int32.init)
        } catch {
            throw AetherError.runtime(underlying: error)
        }
    }

    // MARK: - Source

    /// The installed directory that holds the tokenizer files, and the bundle asset it belongs to.
    static func directory(for source: Manifest.TokenizerSource, in bundle: InstalledBundle) throws -> (URL, ModelRef.Asset?) {
        switch source {
        case .dir(let path):
            guard let url = bundle.url(of: path) else {
                throw AetherError.manifestInvalid(field: "tokenizer.source.dir", reason: "'\(path)' is not installed")
            }
            return (url, bundle.ref.asset(path.asset).map { scoped($0, to: path.path) })
        case .file(let path):
            guard let url = bundle.url(of: path), url.lastPathComponent == "tokenizer.json" else {
                throw AetherError.unsupportedOperation("tokenizer.source.file other than a tokenizer.json is not implemented")
            }
            let parent = (path.path as NSString).deletingLastPathComponent
            return (url.deletingLastPathComponent(), bundle.ref.asset(path.asset).map { scoped($0, to: parent) })
        case .hf(let repo, let revision):
            // A foreign tokenizer is a pinned, hashed catalog asset of the same install (MANIFEST §0).
            guard let asset = bundle.ref.assets.first(where: { $0.repo == repo && $0.revision == revision }),
                  let url = bundle.url(ofAsset: asset.name)
            else {
                throw AetherError.manifestInvalid(
                    field: "tokenizer.source.hf", reason: "no installed asset of \(repo)@\(revision); a tokenizer is never fetched")
            }
            return (url, asset)
        case .embedded:
            throw AetherError.unsupportedOperation("tokenizer.source.embedded is not implemented")
        }
    }

    /// The asset's files under `prefix`, with paths made relative to it.
    private static func scoped(_ asset: ModelRef.Asset, to prefix: String) -> ModelRef.Asset {
        let p = prefix.isEmpty || prefix == "." ? "" : prefix.hasSuffix("/") ? prefix : prefix + "/"
        var copy = asset
        copy.files = asset.files.filter { $0.path.hasPrefix(p) }.map {
            var f = $0
            f.path = String($0.path.dropFirst(p.count))
            return f
        }
        return copy
    }

    /// Checks the tokenizer files the install's catalog entry hashes (tokenizer.json, its config, the template).
    static func verify(directory: URL, asset: ModelRef.Asset?) throws {
        guard let asset else { return }
        for file in asset.files where ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"].contains(file.path) {
            let url = directory.appending(path: file.path)
            guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else {
                throw AetherError.integrity(file: file.path, expected: file.sha256, actual: "missing")
            }
            let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard actual == file.sha256 else { throw AetherError.integrity(file: file.path, expected: file.sha256, actual: actual) }
        }
    }
}
