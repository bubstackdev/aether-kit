import Foundation
import Testing
import AetherHost
@testable import AetherEmbeddings

/// The EmbeddingGemma 300M gate (DESIGN.md §12): the exported `text-embedder@1` bundle, tokenized by the driver and run
/// on the GPU, against the upstream model on the CPU in float32 (`Forge/embeddinggemma/reference_embeddinggemma.py`,
/// sentence-transformers semantics). Off by default; run with
/// `AETHER_GATES=1 swift test --no-parallel --filter TextEmbedderGateTests`.
///
/// - **Tokenization:** each case's ids and attention mask exactly the reference's (prompt, strip, `<bos>` … `<eos>`,
///   truncation to 256 keeping both specials, right padding with 0), and its truncation flag.
/// - **Embeddings** (M4.md's bar): cosine ≥ 0.999 per vector, unit norm, a repeat bit-identical.
/// - **Retrieval:** each query's best document is one the reference names (`expectedBest`).
///
/// The bundle and references default to the Forge outputs beside this repo (`aether/forge-outputs/embeddinggemma-300m`),
/// or `AETHER_EMBEDDINGGEMMA_BUNDLE` and `AETHER_EMBEDDINGGEMMA_REFERENCE`.
@Suite("Text embedder gates", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct TextEmbedderGateTests {
    static let environment = ProcessInfo.processInfo.environment
    static let outputs = forgeOutputs().appending(path: "embeddinggemma-300m")
    static let bundleURL = environment["AETHER_EMBEDDINGGEMMA_BUNDLE"].map { URL(filePath: $0) } ?? outputs.appending(path: "bundle")
    static let referenceURL = environment["AETHER_EMBEDDINGGEMMA_REFERENCE"].map { URL(filePath: $0) }
        ?? outputs.appending(path: "reference")
    static let mac = Device(platform: .macOS, architecture: Device.current.architecture)
    static let bar: Float = 0.999

    struct Index: Decodable {
        struct Entry: Decodable { var name: String }
        struct Similarity: Decodable {
            var queries: [String], documents: [String]
            var cosine: [String: [String: Double]]
            var expectedBest: [String: [String]]
        }
        var cases: [Entry]
        var similarity: Similarity
    }

    struct Case: Decodable {
        var name: String, role: String, text: String
        var tokens: Int, keptTokens: Int, truncated: Bool
    }

    /// A Forge export is laid out as an installed revision root; until it is catalogued its ref is local. The
    /// tokenizer's catalog hashes are absent, so the adapter reads its files unverified.
    static func localBundle(_ url: URL) throws -> InstalledBundle {
        let manifest = try Manifest.load(fromBundle: url)
        let ref = ModelRef(
            catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: manifest.variant.platform.rawValue,
            driver: manifest.driver, assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
        return InstalledBundle(ref: ref, url: url, isStale: false)
    }

    static func load<T: BitwiseCopyable>(_ url: URL, as: T.Type) throws -> [T] {
        try Data(contentsOf: url).withUnsafeBytes { raw in
            (0..<(raw.count / MemoryLayout<T>.size)).map { raw.loadUnaligned(fromByteOffset: $0 * MemoryLayout<T>.size, as: T.self) }
        }
    }

    @Test func embeddingGemma300mMatchesUpstream() async throws {
        let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: Self.referenceURL.appending(path: "references.json")))
        let model = try await Registry.embeddings.load(
            try Self.localBundle(Self.bundleURL), as: TextEmbedderDriver.self,
            options: LoadOptions(device: Self.mac, cachePolicy: .default, cache: .default))
        let embedder = try await model.embedder()
        #expect(embedder.postprocess == .init(normalized: .inGraph, pooling: .inGraph, similarity: .cosine))
        #expect(embedder.sequenceLength == 256)

        var embeddings: [String: Embedding] = [:]
        for entry in index.cases {
            let directory = Self.referenceURL.appending(path: entry.name)
            let c = try JSONDecoder().decode(Case.self, from: Data(contentsOf: directory.appending(path: "case.json")))
            let role: EmbeddingRole = c.role == "query" ? .query : .document
            let ids = try Self.load(directory.appending(path: "input_ids.i32"), as: Int32.self)
            let mask = try Self.load(directory.appending(path: "attention_mask.i32"), as: Int32.self)
            let reference = try Self.load(directory.appending(path: "embedding.f32"), as: Float.self)

            // Tokenization: exact.
            let tokenized = try embedder.tokenize(c.text, role: role)
            #expect(tokenized.ids == ids, "\(c.name): token ids")
            #expect(tokenized.attentionMask == mask, "\(c.name): attention mask")
            #expect(tokenized.tokenCount == c.keptTokens && tokenized.isTruncated == c.truncated, "\(c.name): truncation")
            if tokenized.ids != ids, let first = zip(tokenized.ids, ids).enumerated().first(where: { $1.0 != $1.1 }) {
                print("gate: embeddinggemma \(c.name): first id difference at \(first.offset): \(first.element)")
            }

            // Embedding: the driver's own tokenization end to end.
            let started = ContinuousClock.now
            let embedding = try await embedder.embed(c.text, role: role)
            let elapsed = ContinuousClock.now - started
            #expect(embedding.dimensions == reference.count)
            let cosine = embedding.similarity(Embedding(reference))
            let norm = embedding.vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
            let worst = zip(embedding.vector, reference).map { abs($0 - $1) }.max() ?? 0
            print("gate: embeddinggemma \(c.name) (\(c.role), \(c.tokens) tokens\(c.truncated ? ", truncated" : "")): cosine "
                  + "\(cosine), norm \(norm), max |Δ| \(worst), \(elapsed)")
            #expect(cosine >= Self.bar, "\(c.name): cosine")
            #expect(abs(norm - 1) < 1e-3, "\(c.name): unit norm")
            #expect(try await embedder.embed([tokenized]) == [embedding], "\(c.name): repeat bit-identical")
            embeddings[c.name] = embedding
        }

        // Retrieval: the same best document per query, and the cosine table within 0.005.
        let s = index.similarity
        let documents = try s.documents.map { try #require(embeddings[$0]) }
        for query in s.queries {
            let q = try #require(embeddings[query])
            let scores = documents.map { q.similarity($0) }
            let best = s.documents[scores.indices.max { scores[$0] < scores[$1] }!]
            let worst = zip(s.documents, scores).map { abs(Double($1) - (s.cosine[query]?[$0] ?? .nan)) }.max() ?? .nan
            print("gate: embeddinggemma \(query): best \(best), scores \(scores.map { ($0 * 1e4).rounded() / 1e4 }), "
                  + "max |Δ cosine| to the reference \(worst)")
            #expect(s.expectedBest[query]?.contains(best) == true, "\(query): best document \(best)")
            #expect(worst <= 0.005, "\(query): cosine table")
        }

        // A batch is one request: the same vectors in order.
        let batch = try await embedder.embed(["Which planet is known as the Red Planet?", "Paris"], role: .query)
        #expect(batch.count == 2 && batch[0] == embeddings["query-short"])
        for (key, footprint) in embedder.options.ledger.snapshot.sorted(by: { $0.key.description < $1.key.description }) {
            print("gate: embeddinggemma \(key): peak \(footprint.peakBytes >> 20) MiB, held \(footprint.heldBytes >> 20) MiB")
        }
        await embedder.unload()
        #expect(await embedder.residentRoles.isEmpty)
    }
}

/// The Forge outputs directory beside the repo: the first `forge-outputs` found walking up from this file, so the gates
/// find it from a worktree under `.worktrees/` as from the main checkout (`AETHER_FORGE_OUTPUTS` overrides).
func forgeOutputs(from file: String = #filePath) -> URL {
    if let path = ProcessInfo.processInfo.environment["AETHER_FORGE_OUTPUTS"] { return URL(filePath: path) }
    var directory = URL(filePath: file).deletingLastPathComponent()
    while directory.path != "/" {
        let candidate = directory.appending(path: "forge-outputs")
        if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        directory = directory.deletingLastPathComponent()
    }
    return URL(filePath: file).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().appending(path: "forge-outputs")
}
