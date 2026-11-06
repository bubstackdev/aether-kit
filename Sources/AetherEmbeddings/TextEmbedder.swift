public import AetherCore
public import AetherHost
public import AetherRuntime
import AetherText
import Foundation

/// `text-embedder@1` (MANIFEST.md §1.3, §2.4 `tokenizer.text`, §2.8 `embedding`): one vector per text from one static
/// encoder graph (EmbeddingGemma). Loads a ``TextEmbedderModel``: set its options (a shared coordinator, GPU admission,
/// a ledger), then make the ``TextEmbedder``.
public enum TextEmbedderDriver: ModelDriver {
    public static let id: DriverID = .textEmbedder

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> TextEmbedderModel {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        _ = try TextEmbedder.Configuration(manifest)
        return TextEmbedderModel(
            bundle: bundle, manifest: manifest, device: options.device,
            options: .init(cachePolicy: options.cachePolicy, cache: options.cache))
    }
}

/// An opened text-embedding bundle. Nothing is loaded yet; ``options`` start from the `LoadOptions` cache settings.
public struct TextEmbedderModel: Sendable {
    public let bundle: InstalledBundle
    public let manifest: Manifest
    public let device: Device
    public var options: PipelineOptions

    /// The embedder: loads the tokenizer now and the graph on the first request.
    public func embedder() async throws -> TextEmbedder {
        try await TextEmbedder(bundle: bundle, manifest: manifest, device: device, options: options)
    }
}

extension Registry {
    /// The drivers AetherEmbeddings links: `text-embedder@1`.
    public static let embeddings = Registry([TextEmbedderDriver.self])
}

/// What a text is embedded as: the key of its prompt in the manifest's `tokenizer.text.prompts`. Retrieval models
/// such as EmbeddingGemma embed queries and documents asymmetrically; compare a query's embedding with documents'.
public struct EmbeddingRole: Sendable, Hashable, RawRepresentable {
    public let rawValue: String
    public init(rawValue: String) { self.rawValue = rawValue }

    /// A search query (EmbeddingGemma: `task: search result | query: `).
    public static let query = EmbeddingRole(rawValue: "query")
    /// A document to retrieve (EmbeddingGemma: `title: none | text: `).
    public static let document = EmbeddingRole(rawValue: "document")
}

/// One embedding request's settings.
public struct TextEmbeddingRequest: Sendable, Hashable {
    /// The request's GPU priority (absent: `.medium`). Keep the job to change it while the request waits.
    public var job: GPUJob?

    public init(job: GPUJob? = nil) { self.job = job }
}

/// A text as the graph takes it: exactly `padTo` ids and their attention mask.
public struct TokenizedText: Sendable, Hashable {
    /// The ids, padded to the graph's sequence length.
    public let ids: [Int32]
    /// 1 for each real id (prompt and special tokens included), 0 for padding.
    public let attentionMask: [Int32]
    /// Real ids, special tokens included, after truncation.
    public let tokenCount: Int
    /// Whether the text was longer than the graph's sequence and lost its tail.
    public let isTruncated: Bool
}

/// One loaded text-embedding model (DESIGN.md §8): text → ``TokenizedText`` (CPU, the manifest's `tokenizer.text`)
/// → the `encoder` graph (GPU, on AetherHost's ``StatelessPipeline``), one call per text → ``Embedding``.
///
/// - **Tokenization** (`tokenizer.text`): the role's prompt is prepended and the result stripped of leading and
///   trailing whitespace, then encoded with the tokenizer's special tokens (EmbeddingGemma: `<bos>` … `<eos>`). A text
///   longer than `padTo` ids keeps its first content ids and every special token (`truncation: right`); then it is
///   padded on the right with `padId` to exactly `padTo`, with a mask of 1 for every real id.
/// - **Postprocess** (`embedding`): pooling must be in the graph; `normalized: host` L2-normalizes on the CPU. Compare
///   embeddings with ``Embedding/similarity(_:)`` (cosine).
/// - **Memory and the GPU:** the graph stays loaded until ``unload()``. Each request (a batch of texts) is admitted
///   against the measured footprint (``FootprintLedger``) or the file's size, with the shared `ResourceCoordinator`
///   when the options set one, and through the options' GPU admission (by default ``GPUAdmission/shared``), one grant
///   per request.
/// - **One at a time:** requests run in GPU priority order (the request's `job`), in the order made within a priority;
///   ``unload()`` waits for the running one.
/// - **Cancellation** stops between graph calls with `CancellationError`. Any failure releases the graph; the embedder
///   stays usable.
public actor TextEmbedder {
    public typealias Options = PipelineOptions
    public typealias Request = TextEmbeddingRequest

    /// What the driver reads from a manifest, checked before anything loads.
    struct Configuration: Sendable {
        var tokens: String, attentionMask: String, embedding: String
        var text: TextRules
        var postprocess: Manifest.Postprocess.Embedding

        init(_ manifest: Manifest) throws(AetherError) {
            try DriverContract.textEmbedder.validate(manifest)
            let id = TextEmbedderDriver.id
            guard case .embedding(let embedding)? = manifest.postprocess else {
                throw .manifestInvalid(field: "postprocess", reason: "\(id) requires kind embedding")
            }
            guard (embedding.pooling ?? .inGraph) == .inGraph else {
                throw .unsupportedOperation("\(id): postprocess.embedding.pooling \(embedding.pooling!) (only inGraph)")
            }
            guard embedding.similarity == .cosine || embedding.normalized != .none else {
                throw .unsupportedOperation("\(id): similarity dot over unnormalized embeddings (Embedding.similarity is cosine)")
            }
            guard let tokenizer = manifest.tokenizer else {
                throw .manifestInvalid(field: "tokenizer", reason: "\(id) requires a tokenizer")
            }
            guard let rules = tokenizer.text else {
                throw .manifestInvalid(field: "tokenizer.text", reason: "\(id) requires padId and padTo")
            }
            text = try TextRules(rules, addSpecialTokens: tokenizer.addSpecialTokens ?? true)
            let graph = manifest.graphs["encoder"]!
            if let declared = graph.limits?.staticSequence?.values(forPhase: "step"), declared != [text.length] {
                throw .manifestMismatch(
                    field: "graphs.encoder.limits.staticSequence", expected: "\(text.length) (tokenizer.text.padTo)",
                    found: "\(declared.sorted())")
            }
            tokens = graph.inputs["tokens"]!.name
            attentionMask = graph.inputs["attentionMask"]!.name
            self.embedding = graph.outputs["embedding"]!.name
            postprocess = embedding
        }
    }

    /// Plain text and a role into ids for the graph (the manifest's `tokenizer.text`). `encode` is the tokenizer:
    /// text and whether to add its special tokens.
    typealias Encode = @Sendable (String, Bool) -> [Int32]

    public nonisolated let manifest: Manifest
    public nonisolated let options: Options
    /// The manifest's `postprocess.embedding`.
    public nonisolated let postprocess: Manifest.Postprocess.Embedding
    let configuration: Configuration
    nonisolated let encode: Encode
    private let pipeline: StatelessPipeline

    /// Opens an installed bundle and loads its tokenizer; the graph loads on the first request. `options.memory`
    /// absent means ``MemoryBudget/platformDefault``.
    public init(bundle: InstalledBundle, manifest: Manifest, device: Device, options: Options = Options()) async throws {
        let configuration = try Configuration(manifest)
        let tokenizer = try await TokenizerAdapter(bundle: bundle, manifest: manifest, forGeneration: false)
        self.configuration = configuration
        encode = { text, specials in tokenizer.encode(text, addSpecialTokens: specials) }
        pipeline = try StatelessPipeline(
            bundle: bundle, manifest: manifest, device: device, contract: .textEmbedder, options: options)
        self.manifest = manifest
        self.options = options
        postprocess = configuration.postprocess
    }

    /// The embedder over any tokenizer and graph loader (tests drive it with fakes).
    init(
        manifest: Manifest, device: Device, options: Options, encode: @escaping Encode,
        load: @escaping GraphResidency.Loader, fileBytes: @escaping (String) -> Int64
    ) throws(AetherError) {
        configuration = try Configuration(manifest)
        self.encode = encode
        pipeline = try StatelessPipeline(
            manifest: manifest, device: device, contract: .textEmbedder, options: options, load: load, fileBytes: fileBytes)
        self.manifest = manifest
        self.options = options
        postprocess = configuration.postprocess
    }

    /// Roles currently loaded.
    public var residentRoles: [String] { pipeline.residentRoles }

    /// The graph's sequence length (`tokenizer.text.padTo`).
    public nonisolated var sequenceLength: Int { configuration.text.length }

    /// Tokenizes `text` as `role` for the graph: the CPU half of ``embed(_:role:_:)``. A role without a prompt in the
    /// manifest is embedded as the bare text.
    public nonisolated func tokenize(_ text: String, role: EmbeddingRole) throws(AetherError) -> TokenizedText {
        try configuration.text.tokenize(text, role: role, encode: encode)
    }

    /// Embeds one text as `role`.
    public nonisolated func embed(_ text: String, role: EmbeddingRole, _ request: Request = Request()) async throws -> Embedding {
        try await embed([text], role: role, request)[0]
    }

    /// Embeds `texts` as `role`, in order: one request, one graph call per text. Tokenizing runs on the caller's
    /// executor, before the request queues.
    public nonisolated func embed(
        _ texts: [String], role: EmbeddingRole, _ request: Request = Request()
    ) async throws -> [Embedding] {
        let tokenized = try texts.map { text throws(AetherError) in try tokenize(text, role: role) }
        return try await embed(tokenized, request)
    }

    /// Embeds texts this embedder's ``tokenize(_:role:)`` made.
    public func embed(_ texts: [TokenizedText], _ request: Request = Request()) async throws -> [Embedding] {
        guard !texts.isEmpty else { return [] }
        let c = configuration
        let length = c.text.length
        guard texts.allSatisfy({ $0.ids.count == length && $0.attentionMask.count == length }) else {
            throw AetherError.invalidInput("tokenized text must be \(length) ids and mask values (tokenize(_:role:))")
        }
        let vectors = try await pipeline.perform(job: request.job) { residency in
            let graph = try await residency.graph("encoder")
            let arena = TensorArena()
            var bindings = Bindings(arena: arena)
            let tokens = try Self.buffer(c.tokens, role: .input, of: graph, shape: [1, length], in: arena)
            let mask = try Self.buffer(c.attentionMask, role: .input, of: graph, shape: [1, length], in: arena)
            let dimensions = graph.signature.output(c.embedding)?.shape.last ?? 0
            let output = try Self.buffer(c.embedding, role: .output, of: graph, shape: [1, dimensions], in: arena)
            try bindings.bindInput(c.tokens, to: tokens)
            try bindings.bindInput(c.attentionMask, to: mask)
            try bindings.bindOutput(c.embedding, to: output)
            var vectors: [[Float]] = []
            vectors.reserveCapacity(texts.count)
            for text in texts {
                try tokens.write(text.ids)
                try mask.write(text.attentionMask)
                try await residency.run(graph, &bindings)
                vectors.append(try output.readFloats())
            }
            return vectors
        }
        return vectors.map { Embedding(c.postprocess.normalized == .host ? Self.normalized($0) : $0) }
    }

    /// Releases the graph, waiting for a running request. The next request loads it again.
    public func unload() async { await pipeline.unload() }

    /// A buffer for `name`, which must be `shape` in the graph.
    static func buffer(
        _ name: String, role: TensorRole, of graph: any GraphBackend, shape: [Int], in arena: TensorArena
    ) throws(AetherError) -> TensorBuffer {
        let spec = role == .input ? graph.signature.input(name) : graph.signature.output(name)
        guard let spec, spec.shape == shape, shape.allSatisfy({ $0 > 0 }) else {
            throw .manifestMismatch(
                field: role == .input ? "graphs.encoder.inputs" : "graphs.encoder.outputs",
                expected: "'\(name)' of shape \(shape)" + (role == .input ? " (tokenizer.text.padTo)" : ""),
                found: spec.map { "\($0.shape)" } ?? "no such tensor")
        }
        return try graph.makeBuffer(for: name, role: role, dimensions: [], in: arena)
    }

    /// `vector` scaled to unit L2 norm (unchanged when zero).
    static func normalized(_ vector: [Float]) -> [Float] {
        let norm = vector.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        return norm > 0 ? vector.map { Float(Double($0) / norm) } : vector
    }
}

/// The manifest's `tokenizer.text`, for one fixed-length encoder input (``TextEmbedder``).
struct TextRules: Sendable, Hashable {
    let padId: Int32
    let length: Int
    let prompts: [String: String]
    let addSpecialTokens: Bool

    init(_ text: Manifest.Tokenizer.Text, addSpecialTokens: Bool) throws(AetherError) {
        let id = TextEmbedderDriver.id
        guard let padId = text.padId, let length = text.padTo else {
            throw .manifestInvalid(field: "tokenizer.text", reason: "\(id) requires padId and padTo")
        }
        guard (text.padSide ?? .right) == .right else {
            throw .unsupportedOperation("\(id): tokenizer.text.padSide \(text.padSide!) (only right)")
        }
        guard (text.truncation ?? .right) == .right else {
            throw .unsupportedOperation("\(id): tokenizer.text.truncation \(text.truncation!) (only right)")
        }
        guard (text.maskRule ?? .realTokens) == .realTokens else {
            throw .unsupportedOperation("\(id): tokenizer.text.maskRule \(text.maskRule!) (only realTokens)")
        }
        self.init(padId: Int32(padId), length: length, prompts: text.prompts ?? [:], addSpecialTokens: addSpecialTokens)
    }

    init(padId: Int32, length: Int, prompts: [String: String], addSpecialTokens: Bool) {
        self.padId = padId
        self.length = length
        self.prompts = prompts
        self.addSpecialTokens = addSpecialTokens
    }

    /// The prompt, the text, stripped; encoded; truncated keeping the special tokens; padded on the right.
    func tokenize(_ text: String, role: EmbeddingRole, encode: TextEmbedder.Encode) throws(AetherError) -> TokenizedText {
        let input = ((prompts[role.rawValue] ?? "") + text).trimmingCharacters(in: .whitespacesAndNewlines)
        var ids = encode(input, addSpecialTokens)
        let truncated = ids.count > length
        if truncated { ids = try truncate(ids, content: addSpecialTokens ? encode(input, false) : ids) }
        let real = ids.count
        let mask = [Int32](repeating: 1, count: real) + [Int32](repeating: 0, count: length - real)
        ids += [Int32](repeating: padId, count: length - real)
        return TokenizedText(ids: ids, attentionMask: mask, tokenCount: real, isTruncated: truncated)
    }

    /// `full` is `content` with the tokenizer's special tokens around it; keeps those and the first content ids that
    /// fit (HF `truncation=longest_first` on one sequence).
    func truncate(_ full: [Int32], content: [Int32]) throws(AetherError) -> [Int32] {
        let specials = full.count - content.count
        guard specials >= 0, specials < length,
              let prefix = (0...specials).first(where: { full[$0..<($0 + content.count)].elementsEqual(content) })
        else {
            throw .unsupportedOperation(
                "\(TextEmbedderDriver.id): the tokenizer's special tokens are not around the text, or leave no room for it; "
                + "tokenizer.text.truncation right cannot keep them")
        }
        let keep = length - specials
        return Array(full[..<prefix]) + content.prefix(keep) + full[(prefix + content.count)...]
    }
}

extension Embedding {
    /// A Matryoshka embedding's first `dimension` values, re-normalized to unit length (EmbeddingGemma: 512, 256 or
    /// 128 of its 768 are valid smaller embeddings; which prefixes a model supports is its model card's).
    public func truncated(to dimension: Int) -> Embedding {
        precondition(dimension > 0 && dimension <= vector.count, "truncated(to: \(dimension)) of dimension \(vector.count)")
        return Embedding(TextEmbedder.normalized(Array(vector.prefix(dimension))))
    }
}
