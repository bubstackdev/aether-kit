public import AetherCore
public import AetherHost
public import AetherRuntime
import Foundation

/// `clip-joint@1` (MANIFEST.md §1.3, §2.8): CLIP's image and text towers into one embedding space. Loads an
/// ``ImageTextEncoderModel``: set its options (a shared coordinator, GPU admission, a ledger), then make the
/// ``ImageTextEncoder``.
public enum ImageTextEncoderDriver: ModelDriver {
    public static let id: DriverID = .clipJoint

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> ImageTextEncoderModel {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        _ = try ImageTextEncoder.Configuration(manifest)
        return ImageTextEncoderModel(
            bundle: bundle, manifest: manifest, device: options.device,
            options: .init(cachePolicy: options.cachePolicy, cache: options.cache))
    }
}

/// An opened CLIP bundle. Nothing is loaded yet; ``options`` start from the `LoadOptions` cache settings.
public struct ImageTextEncoderModel: Sendable {
    public let bundle: InstalledBundle
    public let manifest: Manifest
    public let device: Device
    public var options: PipelineOptions

    public func encoder() throws -> ImageTextEncoder {
        try ImageTextEncoder(bundle: bundle, manifest: manifest, device: device, options: options)
    }
}

/// One image-text request's settings.
public struct ImageTextRequest: Sendable, Hashable {
    /// The request's GPU priority (absent: `.medium`). Keep the job to change it while the request waits.
    public var job: GPUJob?

    public init(job: GPUJob? = nil) { self.job = job }
}

/// One loaded CLIP model (DESIGN.md §8): images through ``ImagePreprocessor`` (CPU) and the `imageEncoder` graph, texts
/// through ``CLIPTokenizer`` (CPU) and the `textEncoder` graph (GPU, on AetherHost's ``StatelessPipeline``), each to an
/// ``Embedding``; zero-shot classification as `softmax(logitScale · similarity)` over the labels' text embeddings.
///
/// - **Embeddings:** the manifest's `postprocess.embedding` says whether the graphs normalize (`inGraph`: CLIP ViT-B/32's
///   are unit length, so cosine similarity is their dot product) or the host does (`host`). Compare them with
///   ``Embedding/similarity(_:)``.
/// - **Text batches:** the text graph has a static batch. Texts run in groups of the largest `buckets` value on axis
///   `batch`, each group at the smallest bucket that holds it (CLIP ViT-B/32: 1 and 8). A bucket's rows beyond the texts
///   are `tokenizer.text.unusedBatchRows` then the pad id, and their outputs are discarded. Each bucket is loaded once
///   it is first used and stays resident.
/// - **Memory and the GPU:** the graphs stay loaded until ``unload()``. Each request is admitted against the measured
///   footprint (``FootprintLedger``) or the files' size, with the shared `ResourceCoordinator` when the options set one,
///   and through the options' GPU admission (by default ``GPUAdmission/shared``), one grant per request (a
///   classification's texts and image run under one).
/// - **One at a time:** requests run in GPU priority order (the request's `job`), in the order made within a priority;
///   ``unload()`` waits for the running one.
/// - **Cancellation** stops after a graph call with `CancellationError`. Any failure releases the graphs; the encoder
///   stays usable.
public actor ImageTextEncoder {
    public typealias Options = PipelineOptions
    public typealias Request = ImageTextRequest

    /// What the driver reads from a manifest, checked before anything loads.
    struct Configuration: Sendable {
        var image: String, imageEmbedding: String
        var tokens: String, attentionMask: String?, textEmbedding: String
        var embedding: Manifest.Postprocess.Embedding
        var logitScale: Float
        var text: Manifest.Tokenizer.Text
        /// `padTo`: the token rows' length.
        var length: Int
        /// `padId`.
        var padId: Int32
        /// `maskRule` (default `realTokens`), read when the text graph binds `attentionMask`.
        var maskRule: Manifest.Tokenizer.Text.MaskRule
        /// The text graph's `buckets` on axis `batch`, ascending; empty: the default function's own static batch.
        var batches: [Int]
        var preprocessor: ImagePreprocessor

        init(_ manifest: Manifest) throws(AetherError) {
            let id = ImageTextEncoderDriver.id
            try DriverContract.clipJoint.validate(manifest)
            guard case .embedding(let embedding)? = manifest.postprocess else {
                throw .manifestInvalid(field: "postprocess", reason: "\(id) requires kind embedding")
            }
            guard (embedding.pooling ?? .inGraph) == .inGraph else {
                throw .manifestInvalid(field: "postprocess.embedding.pooling", reason: "\(id) takes pooled (inGraph) outputs")
            }
            guard let logitScale = embedding.logitScale else {
                throw .manifestInvalid(field: "postprocess.embedding.logitScale", reason: "required by \(id)")
            }
            preprocessor = try ImagePreprocessor(manifest: manifest)
            guard (preprocessor.configuration.layout ?? .NCHW) == .NCHW else {
                throw .manifestInvalid(field: "preprocess.image.layout", reason: "\(id) takes NCHW")
            }
            guard let tokenizer = manifest.tokenizer, tokenizer.renderer == "clip-bpe" else {
                throw .manifestInvalid(field: "tokenizer.renderer", reason: "\(id) requires a clip-bpe tokenizer")
            }
            guard let text = tokenizer.text else {
                throw .manifestInvalid(field: "tokenizer.text", reason: "required by \(id)")
            }
            guard let padId = text.padId else {
                throw .manifestInvalid(field: "tokenizer.text.padId", reason: "required by \(id)")
            }
            guard let length = text.padTo else {
                throw .manifestInvalid(field: "tokenizer.text.padTo", reason: "required by \(id)")
            }
            guard (text.padSide ?? .right) == .right, (text.truncation ?? .right) == .right else {
                throw .manifestInvalid(field: "tokenizer.text", reason: "\(id) pads and truncates on the right")
            }
            let imageGraph = manifest.graphs["imageEncoder"]!, textGraph = manifest.graphs["textEncoder"]!
            image = imageGraph.inputs["image"]!.name
            imageEmbedding = imageGraph.outputs["imageEmbedding"]!.name
            tokens = textGraph.inputs["tokens"]!.name
            attentionMask = textGraph.inputs["attentionMask"]?.name
            textEmbedding = textGraph.outputs["textEmbedding"]!.name
            maskRule = text.maskRule ?? .realTokens
            if attentionMask != nil, maskRule == .noMask {
                throw .manifestInvalid(field: "tokenizer.text.maskRule", reason: "the text graph binds attentionMask: none takes no mask")
            }
            if let buckets = textGraph.buckets {
                guard buckets.axis == "batch", buckets.rule == "smallestFitting" else {
                    throw .manifestInvalid(field: "graphs.textEncoder.buckets", reason: "\(id) takes smallestFitting buckets on axis batch")
                }
                for value in buckets.values where value < 1 || textGraph.bucket(value) == nil {
                    throw .manifestInvalid(field: "graphs.textEncoder.buckets", reason: "bucket \(value) needs a function or file")
                }
                batches = buckets.values
            } else {
                batches = []
            }
            self.embedding = embedding
            self.logitScale = Float(logitScale)
            self.text = text
            self.length = length
            self.padId = Int32(padId)
        }
    }

    public nonisolated let manifest: Manifest
    public nonisolated let options: Options
    /// The manifest's `preprocess.image`, as a transform.
    public nonisolated let preprocessor: ImagePreprocessor
    /// The bundle's tokenizer, padding to `tokenizer.text.padTo` with its `padId`.
    public nonisolated let tokenizer: CLIPTokenizer
    /// The manifest's `postprocess.embedding`.
    public nonisolated let postprocess: Manifest.Postprocess.Embedding
    let configuration: Configuration
    private let pipeline: StatelessPipeline

    /// Opens an installed bundle; the graphs load on first use. `options.memory` absent means
    /// ``MemoryBudget/platformDefault``.
    public init(bundle: InstalledBundle, manifest: Manifest, device: Device, options: Options = Options()) throws(AetherError) {
        let configuration = try Configuration(manifest)
        guard case .dir(let path)? = manifest.tokenizer?.source, let directory = bundle.url(of: path) else {
            throw .manifestInvalid(field: "tokenizer.source", reason: "\(ImageTextEncoderDriver.id) needs a tokenizer directory")
        }
        try self.init(
            configuration: configuration,
            tokenizer: CLIPTokenizer(directory: directory, length: configuration.length, pad: configuration.padId),
            pipeline: StatelessPipeline(bundle: bundle, manifest: manifest, device: device, contract: .clipJoint, options: options))
    }

    /// The encoder over any graph loader and tokenizer (tests drive it with fakes).
    init(
        manifest: Manifest, tokenizer: CLIPTokenizer, device: Device, options: Options,
        load: @escaping GraphResidency.Loader, fileBytes: @escaping (String) -> Int64
    ) throws(AetherError) {
        try self.init(
            configuration: Configuration(manifest), tokenizer: tokenizer,
            pipeline: StatelessPipeline(
                manifest: manifest, device: device, contract: .clipJoint, options: options, load: load, fileBytes: fileBytes))
    }

    private init(configuration: Configuration, tokenizer: CLIPTokenizer, pipeline: StatelessPipeline) throws(AetherError) {
        guard tokenizer.length == configuration.length, tokenizer.pad == configuration.padId else {
            throw .manifestMismatch(
                field: "tokenizer.text", expected: "padTo \(configuration.length), padId \(configuration.padId)",
                found: "a tokenizer of length \(tokenizer.length), pad \(tokenizer.pad)")
        }
        self.configuration = configuration
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        manifest = pipeline.manifest
        options = pipeline.options
        preprocessor = configuration.preprocessor
        postprocess = configuration.embedding
    }

    /// Roles currently loaded (a bucket as `textEncoder[8]`).
    public var residentRoles: [String] { pipeline.residentRoles }

    /// The zero-shot temperature: `postprocess.embedding.logitScale`.
    public nonisolated var logitScale: Float { configuration.logitScale }

    // MARK: - Images

    /// Decodes and prepares `image` for the image graph: the CPU half of ``embed(image:_:)-(ImageInput,_)``.
    public nonisolated func prepare(_ image: ImageInput) throws(AetherError) -> PreparedImage {
        try preprocessor.prepare(image)
    }

    /// Embeds `image`. Preparing it runs on the caller's executor, before the request queues.
    public nonisolated func embed(image: ImageInput, _ request: Request = Request()) async throws -> Embedding {
        try await embed(image: prepare(image), request)
    }

    /// Embeds an image this encoder's ``prepare(_:)`` made.
    public func embed(image prepared: PreparedImage, _ request: Request = Request()) async throws -> Embedding {
        try await pipeline.perform(job: request.job) { residency in try await embedImage(prepared, residency) }
    }

    // MARK: - Texts

    /// The token rows `texts` run as: BOS, at most `padTo − 2` pieces, EOS, then the pad id, `padTo` each.
    public nonisolated func tokenize(_ texts: [String]) throws(AetherError) -> [[Int32]] {
        var rows: [[Int32]] = []
        for text in texts { rows.append(try tokenizer.encode(text)) }
        return rows
    }

    /// Embeds `texts`, in order (none for none). Tokenizing runs before the request queues; every group of texts runs
    /// under the one GPU grant.
    public func embed(texts: [String], _ request: Request = Request()) async throws -> [Embedding] {
        let rows = try tokenize(texts)
        guard !rows.isEmpty else { return [] }
        return try await pipeline.perform(job: request.job) { residency in try await embedTexts(rows, residency) }
    }

    // MARK: - Zero-shot

    /// Scores `labels` for `image`: `softmax(logitScale · similarity(image, text(label)))`, in the labels' order. Each
    /// label is embedded as `template` with `{}` replaced by it: by default OpenAI's prompt for ImageNet-style labels,
    /// `"a photo of a {}."`; pass `"{}"` to embed the labels as they are. Image and texts run under one GPU grant.
    public nonisolated func classify(
        _ image: ImageInput, labels: [String], template: String = "a photo of a {}.", _ request: Request = Request()
    ) async throws -> [LabelScore] {
        try await classify(prepare(image), labels: labels, template: template, request)
    }

    /// ``classify(_:labels:template:_:)-(ImageInput,_,_,_)`` for an image this encoder's ``prepare(_:)`` made.
    public func classify(
        _ prepared: PreparedImage, labels: [String], template: String = "a photo of a {}.", _ request: Request = Request()
    ) async throws -> [LabelScore] {
        guard !labels.isEmpty else { throw AetherError.invalidInput("classify needs at least one label") }
        let rows = try tokenize(labels.map { template.replacingOccurrences(of: "{}", with: $0) })
        let (image, texts) = try await pipeline.perform(job: request.job) { residency in
            (try await embedImage(prepared, residency), try await embedTexts(rows, residency))
        }
        return scores(image: image, labels: labels, texts: texts)
    }

    /// Zero-shot scores from embeddings already made (label embeddings computed once, say), in the labels' order:
    /// `softmax(logitScale · similarity)`. `labels` and `texts` pair up by index.
    public nonisolated func scores(image: Embedding, labels: [String], texts: [Embedding]) -> [LabelScore] {
        precondition(labels.count == texts.count, "\(labels.count) labels for \(texts.count) text embeddings")
        let logits = texts.map { logitScale * similarity(image, $0) }
        let top = logits.max() ?? 0
        let weights = logits.map { exp(Double($0 - top)) }
        let total = weights.reduce(0, +)
        return zip(labels, weights).map { LabelScore(label: $0, score: Float($1 / total)) }
    }

    /// The manifest's similarity: cosine, or the dot product.
    public nonisolated func similarity(_ a: Embedding, _ b: Embedding) -> Float {
        configuration.embedding.similarity == .dot ? a.dot(b) : a.similarity(b)
    }

    /// Releases the graphs, waiting for a running request. The next request loads them again.
    public func unload() async { await pipeline.unload() }

    // MARK: - Graph calls

    private func embedImage(_ prepared: PreparedImage, _ residency: GraphResidency) async throws -> Embedding {
        let c = configuration
        let graph = try await residency.graph("imageEncoder")
        let arena = TensorArena()
        var bindings = Bindings(arena: arena)
        try bindings.bindInput(c.image, to: graph.makeBuffer(for: c.image, holding: prepared, in: arena))
        let output = try Self.output(c.imageEmbedding, of: graph, role: "imageEncoder", rows: 1, dimension: c.embedding.dimension)
        let buffer = try graph.makeBuffer(for: output, role: .output, dimensions: [], in: arena)
        try bindings.bindOutput(output, to: buffer)
        try await residency.run(graph, &bindings)
        return finish(try buffer.readFloats())
    }

    /// Runs `rows` in groups of the largest batch, each at the smallest batch that holds it.
    private func embedTexts(_ rows: [[Int32]], _ residency: GraphResidency) async throws -> [Embedding] {
        let c = configuration
        var embeddings: [Embedding] = []
        embeddings.reserveCapacity(rows.count)
        var start = 0
        while start < rows.count {
            let graph: any GraphBackend, batch: Int
            if let largest = c.batches.last {
                batch = c.batches.first { $0 >= rows.count - start } ?? largest
                graph = try await residency.graph(textKey(batch))
            } else {
                graph = try await residency.graph("textEncoder")
                batch = graph.signature.input(c.tokens)?.shape.first ?? 1
            }
            let group = Array(rows[start..<min(rows.count, start + batch)])
            embeddings += try await embedGroup(group, batch: batch, graph: graph, residency)
            start += group.count
        }
        return embeddings
    }

    /// The loader key of text bucket `batch`: the role itself when the bucket is the graph's own function.
    private func textKey(_ batch: Int) -> String {
        let graph = manifest.graphs["textEncoder"]!
        return graph.bucket(batch) == graph ? "textEncoder" : GraphResidency.key("textEncoder", bucket: batch)
    }

    private func embedGroup(
        _ group: [[Int32]], batch: Int, graph: any GraphBackend, _ residency: GraphResidency
    ) async throws -> [Embedding] {
        let c = configuration
        let expected = [batch, c.length]
        guard let spec = graph.signature.input(c.tokens), spec.shape == expected else {
            throw AetherError.manifestMismatch(
                field: "graphs.textEncoder.inputs.tokens", expected: "'\(c.tokens)' of shape \(expected)",
                found: graph.signature.input(c.tokens).map { "\($0.shape)" } ?? "no such input")
        }
        let unused = (c.text.unusedBatchRows ?? []).map(Int32.init)
        let filler = unused + Array(repeating: c.padId, count: c.length - unused.count)
        let ids = group.flatMap { $0 } + Array(repeating: filler, count: batch - group.count).flatMap { $0 }
        let arena = TensorArena()
        var bindings = Bindings(arena: arena)
        let tokens = try graph.makeBuffer(for: c.tokens, role: .input, dimensions: [], in: arena)
        try tokens.write(ids)
        try bindings.bindInput(c.tokens, to: tokens)
        if let name = c.attentionMask {
            let mask = try graph.makeBuffer(for: name, role: .input, dimensions: [], in: arena)
            try mask.write(Self.mask(ids, length: c.length, rule: c.maskRule, pad: c.padId))
            try bindings.bindInput(name, to: mask)
        }
        let output = try Self.output(c.textEmbedding, of: graph, role: "textEncoder", rows: batch, dimension: c.embedding.dimension)
        let buffer = try graph.makeBuffer(for: output, role: .output, dimensions: [], in: arena)
        try bindings.bindOutput(output, to: buffer)
        try await residency.run(graph, &bindings)
        let values = try buffer.readFloats(), width = values.count / batch
        return (0..<group.count).map { finish(Array(values[($0 * width)..<(($0 + 1) * width)])) }
    }

    /// The postprocess: the host normalizes when the manifest says so.
    private func finish(_ vector: [Float]) -> Embedding {
        let embedding = Embedding(vector)
        return configuration.embedding.normalized == .host ? embedding.normalized : embedding
    }

    /// `name`, which must be `[rows, D]` (`D` the manifest's dimension when it gives one).
    static func output(
        _ name: String, of graph: any GraphBackend, role: String, rows: Int, dimension: Int?
    ) throws(AetherError) -> String {
        let spec = graph.signature.output(name)
        guard let spec, spec.rank == 2, spec.shape[0] == rows, spec.shape[1] >= 1, dimension.map({ $0 == spec.shape[1] }) ?? true else {
            throw .manifestMismatch(
                field: "graphs.\(role).outputs", expected: "'\(name)' of shape [\(rows), \(dimension.map(String.init) ?? "D")]",
                found: spec.map { "\($0.shape)" } ?? "no such output")
        }
        return name
    }

    /// The attention mask of `ids` (rows of `length`) under `rule`, as int32. A CLIP row's real ids are BOS through its
    /// EOS (the tokenizer always ends a text with EOS, truncated or not, and an unused row starts `[BOS, EOS]`), so
    /// `realTokens` is `throughFirstEos`.
    static func mask(_ ids: [Int32], length: Int, rule: Manifest.Tokenizer.Text.MaskRule, pad: Int32) -> [Int32] {
        switch rule {
        case .noMask:
            return Array(repeating: 1, count: ids.count)
        case .nonPad:
            return ids.map { $0 == pad ? 0 : 1 }
        case .realTokens, .throughFirstEos:
            var mask = [Int32](repeating: 0, count: ids.count)
            for row in stride(from: 0, to: ids.count, by: length) {
                for i in row..<(row + length) {
                    mask[i] = 1
                    if ids[i] == CLIPTokenizer.eos { break }
                }
            }
            return mask
        }
    }
}
