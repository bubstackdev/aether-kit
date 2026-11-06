import CoreGraphics
import Foundation
import Synchronization
import Testing
@_spi(Testing) import AetherHost
@testable import AetherVision

/// `clip-joint@1` over fake graphs (DESIGN.md §12): the contract, the text batch buckets and their unused rows, the
/// embedding postprocess and zero-shot scores. A constant probe and no GPU admission, so nothing depends on the machine.
@Suite("Image-text encoder")
struct ImageTextEncoderTests {
    /// A CLIP-shaped manifest: a 2x2 image graph, a text graph of 8-id rows with batch buckets 1 (its own function) and
    /// 4, 4-dimensional embeddings. `edit` changes the JSON object before decoding.
    static func manifest(_ edit: (inout [String: Any]) -> Void = { _ in }) throws -> Manifest {
        var json: [String: Any] = [
            "schema": 1, "id": "clip-test", "task": "image-text-embedding", "driver": "clip-joint@1",
            "variant": ["platform": "macos"], "assets": ["main": ["subtree": ""]],
            "graphs": [
                "imageEncoder": [
                    "file": "main:image.aimodel", "function": "main", "inputs": ["image": "image"],
                    "outputs": ["imageEmbedding": "image_embedding"], "compute": "gpu",
                ] as [String: Any],
                "textEncoder": textGraph(),
            ],
            "tokenizer": [
                "source": ["dir": "main:tokenizer"], "renderer": "clip-bpe",
                "text": ["padId": 49407, "padSide": "right", "padTo": 8, "truncation": "right", "maskRule": "none",
                         "unusedBatchRows": [49406, 49407]] as [String: Any],
            ] as [String: Any],
            "preprocess": ["image": [
                "resize": ["mode": "stretch", "size": [2, 2], "filter": "vImage"] as [String: Any], "colorSpace": "sRGB",
                "scale": 1.0 / 255, "layout": "NCHW",
            ] as [String: Any]],
            "postprocess": ["kind": "embedding", "embedding": [
                "normalized": "inGraph", "similarity": "cosine", "dimension": 4, "logitScale": 100,
            ] as [String: Any]],
        ]
        edit(&json)
        return try Manifest.decode(JSONSerialization.data(withJSONObject: json))
    }

    static func textGraph(buckets: [String: Any]? = [
        "axis": "batch", "rule": "smallestFitting",
        "entries": [["value": 1, "function": "batch1"], ["value": 4, "function": "batch4"]],
    ], inputs: [String: Any] = ["tokens": "tokens"]) -> [String: Any] {
        var graph: [String: Any] = [
            "file": "main:text.aimodel", "function": "batch1", "inputs": inputs,
            "outputs": ["textEmbedding": "text_embedding"], "compute": "gpu", "limits": ["staticSequence": 8],
        ]
        graph["buckets"] = buckets
        return graph
    }

    /// `r`, `g`, `b` and `a` with their end-of-word forms, BOS and EOS, and no config files: `!` is not added and the
    /// pad is EOS, as OpenAI CLIP's.
    static func tokenizer(length: Int = 8) throws -> CLIPTokenizer {
        let directory = FileManager.default.temporaryDirectory.appending(path: "aether-clip-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var vocab: [String: Int] = ["!": 0, "<|startoftext|>": 49406, "<|endoftext|>": 49407]
        for (i, letter) in "abcdefghijklmnopqrstuvwxyz".enumerated() {
            vocab[String(letter)] = 100 + i
            vocab[String(letter) + "</w>"] = 200 + i
        }
        try JSONSerialization.data(withJSONObject: vocab).write(to: directory.appending(path: "vocab.json"))
        try "#version: 0.2\n".write(to: directory.appending(path: "merges.txt"), atomically: true, encoding: .utf8)
        return try CLIPTokenizer(directory: directory, length: length)
    }

    static let r: Int32 = 217, g: Int32 = 206, b: Int32 = 201

    final class Log: Sendable {
        private let state = Mutex<[String]>([])
        var entries: [String] { state.withLock { $0 } }
        func append(_ entry: String) { state.withLock { $0.append(entry) } }
    }

    /// The image tower: each channel's mean, then 0.
    final class FakeImage: GraphBackend {
        let signature: GraphSignature
        let log: Log
        init(log: Log, dimension: Int = 4) {
            signature = GraphSignature(
                function: "main", inputs: [TensorSpec(name: "image", scalarType: .float16, shape: [1, 3, 2, 2])], states: [],
                outputs: [TensorSpec(name: "image_embedding", scalarType: .float32, shape: [1, dimension])])
            self.log = log
        }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            log.append("run image")
            let image = try bindings.input("image")!.read(Float16.self).map(Float.init)
            let means = (0..<3).map { c in image[(4 * c)..<(4 * c + 4)].reduce(0, +) / 4 }
            try bindings.output("image_embedding")!.write(means + [Float](repeating: 0, count: signature.outputs[0].shape[1] - 3))
        }
    }

    /// The text tower at a static batch: per row, how many `r`, `g`, `b` pieces and BOS ids it holds. Logs its rows.
    final class FakeText: GraphBackend {
        let signature: GraphSignature
        let log: Log
        let batch: Int
        init(log: Log, batch: Int, mask: Bool = false) {
            var inputs = [TensorSpec(name: "tokens", scalarType: .int32, shape: [batch, 8])]
            if mask { inputs.append(TensorSpec(name: "attention_mask", scalarType: .int32, shape: [batch, 8])) }
            signature = GraphSignature(
                function: "batch\(batch)", inputs: inputs, states: [],
                outputs: [TensorSpec(name: "text_embedding", scalarType: .float32, shape: [batch, 4])])
            self.log = log
            self.batch = batch
        }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            let ids = try bindings.input("tokens")!.read(Int32.self)
            let rows = (0..<batch).map { Array(ids[($0 * 8)..<(($0 + 1) * 8)]) }
            log.append("run batch\(batch) \(rows)")
            if let mask = bindings.input("attention_mask") { log.append("mask \(try mask.read(Int32.self))") }
            try bindings.output("text_embedding")!.write(rows.flatMap { row in
                [r, g, b, CLIPTokenizer.bos].map { id in Float(row.filter { $0 == id }.count) }
            })
        }
    }

    static let device = Device(platform: .macOS, architecture: "test")

    static func encoder(_ manifest: Manifest, log: Log = Log(), tokenizer: CLIPTokenizer? = nil, imageDimension: Int = 4) throws -> ImageTextEncoder {
        let mask = manifest.graphs["textEncoder"]?.inputs["attentionMask"] != nil
        return try ImageTextEncoder(
            manifest: manifest, tokenizer: tokenizer ?? Self.tokenizer(), device: device,
            options: PipelineOptions(memory: .bytes(1 << 40), gpu: .unadmitted, probe: GPUMemoryProbe { 0 }),
            load: { key throws(AetherError) in
                log.append("load \(key)")
                switch key {
                case "imageEncoder": return FakeImage(log: log, dimension: imageDimension)
                case "textEncoder": return FakeText(log: log, batch: 1, mask: mask)
                case "textEncoder[4]": return FakeText(log: log, batch: 4, mask: mask)
                default: throw .manifestInvalid(field: "graphs.\(key)", reason: "no fake")
                }
            },
            fileBytes: { _ in 100 })
    }

    /// A 2x2 opaque sRGB image of one colour.
    static func image(_ rgb: (UInt8, UInt8, UInt8)) throws -> ImageInput {
        let bytes: [UInt8] = (0..<4).flatMap { _ in [rgb.0, rgb.1, rgb.2, 255] }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = try #require(CGImage(
            width: 2, height: 2, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 8,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        return ImageInput(image)
    }

    static func isInvalid(_ field: String) -> (any Error) -> Bool {
        { error in
            switch error as? AetherError {
            case .manifestInvalid(let f, _)?: f == field
            default: false
            }
        }
    }

    static func isMismatch(_ field: String) -> (any Error) -> Bool {
        { error in
            switch error as? AetherError {
            case .manifestMismatch(let f, _, _)?: f == field
            default: false
            }
        }
    }

    // MARK: - Contract

    @Test func theDriverIsRegisteredWithItsContract() throws {
        #expect(Registry.vision.supports(.clipJoint))
        #expect(try Registry.vision.driver(for: .clipJoint) == ImageTextEncoderDriver.self)
        #expect(DriverCatalog.contracts[.clipJoint] == .clipJoint)
        try DriverContract.clipJoint.validate(Self.manifest())
    }

    @Test func theManifestIsCheckedBeforeAnythingLoads() throws {
        func embedding(_ fields: [String: Any]) -> (inout [String: Any]) -> Void {
            { json in json["postprocess"] = ["kind": "embedding", "embedding": fields] }
        }
        func tokenizer(_ edit: @escaping (inout [String: Any]) -> Void) -> (inout [String: Any]) -> Void {
            { json in
                var tokenizer = json["tokenizer"] as! [String: Any]
                edit(&tokenizer)
                json["tokenizer"] = tokenizer
            }
        }
        func textGraph(_ graph: [String: Any]) -> (inout [String: Any]) -> Void {
            { json in
                var graphs = json["graphs"] as! [String: Any]
                graphs["textEncoder"] = graph
                json["graphs"] = graphs
            }
        }
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("graphs.imageEncoder", { json in
                var graphs = json["graphs"] as! [String: Any]
                graphs["imageEncoder"] = nil
                json["graphs"] = graphs
            }),
            ("postprocess", { json in json["postprocess"] = ["kind": "depth", "depth": ["polarity": "largerIsNearer"]] }),
            ("postprocess.embedding.logitScale", embedding(["normalized": "inGraph", "similarity": "cosine"])),
            ("postprocess.embedding.pooling", embedding(["normalized": "inGraph", "pooling": "mean", "similarity": "cosine", "logitScale": 1])),
            ("tokenizer.renderer", tokenizer { $0["renderer"] = nil }),
            ("tokenizer.text", tokenizer { $0["text"] = nil }),
            ("tokenizer.text.padTo", tokenizer { $0["text"] = ["padId": 49407] }),
            ("tokenizer.text", tokenizer { $0["text"] = ["padId": 49407, "padTo": 8, "padSide": "left"] }),
            ("graphs.textEncoder.buckets", textGraph(Self.textGraph(buckets: [
                "axis": "sequence", "rule": "smallestFitting", "entries": [["value": 4, "function": "batch4"]]]))),
            ("graphs.textEncoder.buckets", textGraph(Self.textGraph(buckets: [
                "axis": "batch", "rule": "smallestFitting", "entries": [["value": 4, "asset": "main"]]]))),
            ("tokenizer.text.maskRule", textGraph(Self.textGraph(inputs: ["tokens": "tokens", "attentionMask": "attention_mask"]))),
            ("preprocess.image", { json in json["preprocess"] = nil }),
        ]
        for (field, edit) in cases {
            let manifest = try Self.manifest(edit)
            #expect("\(field)", performing: { _ = try Self.encoder(manifest) }, throws: Self.isInvalid(field))
        }
        // The tokenizer must pad and truncate as the text block says.
        let manifest = try Self.manifest()
        #expect(performing: { _ = try Self.encoder(manifest, tokenizer: Self.tokenizer(length: 6)) },
                throws: Self.isMismatch("tokenizer.text"))
    }

    // MARK: - Texts

    @Test func textsRunInTheSmallestBucketWithUnusedRowsFilled() async throws {
        let log = Log()
        let encoder = try Self.encoder(Self.manifest(), log: log)
        let filler: [Int32] = [49406, 49407] + [Int32](repeating: 49407, count: 6)
        let row = { (pieces: [Int32]) in [CLIPTokenizer.bos] + pieces + [CLIPTokenizer.eos] + [Int32](repeating: 49407, count: 6 - pieces.count) }
        #expect(try encoder.tokenize(["R", "g  b"]) == [row([Self.r]), row([Self.g, Self.b])])

        // One text: the graph's own function (bucket 1).
        let one = try await encoder.embed(texts: ["r"])
        #expect(one == [Embedding([1, 0, 0, 1])])
        #expect(log.entries == ["load textEncoder", "run batch1 \([row([Self.r])])"])

        // Three: bucket 4, its last row unused (BOS, EOS, then pad) and dropped.
        let three = try await encoder.embed(texts: ["r", "g b", "b"])
        #expect(three == [Embedding([1, 0, 0, 1]), Embedding([0, 1, 1, 1]), Embedding([0, 0, 1, 1])])
        #expect(Array(log.entries.dropFirst(2)) == [
            "load textEncoder[4]", "run batch4 \([row([Self.r]), row([Self.g, Self.b]), row([Self.b]), filler])"])

        // Six: groups of the largest bucket, in order; both bucket 4, now resident.
        let six = try await encoder.embed(texts: ["r", "g", "b", "r r", "g g", "b b"])
        #expect(six.map(\.vector) == [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [2, 0, 0, 1], [0, 2, 0, 1], [0, 0, 2, 1]])
        #expect(Array(log.entries.dropFirst(4)) == [
            "run batch4 \([row([Self.r]), row([Self.g]), row([Self.b]), row([Self.r, Self.r])])",
            "run batch4 \([row([Self.g, Self.g]), row([Self.b, Self.b]), filler, filler])"])
        #expect(await encoder.residentRoles == ["textEncoder", "textEncoder[4]"])

        // None: no graph runs.
        #expect(try await encoder.embed(texts: []).isEmpty)
        #expect(log.entries.count == 6)
    }

    @Test func withoutBucketsTextsRunAtTheGraphsStaticBatch() async throws {
        let manifest = try Self.manifest { json in
            var graphs = json["graphs"] as! [String: Any]
            graphs["textEncoder"] = Self.textGraph(buckets: nil)
            json["graphs"] = graphs
        }
        let log = Log()
        let encoder = try Self.encoder(manifest, log: log)
        #expect(try await encoder.embed(texts: ["r", "g"]).map(\.vector) == [[1, 0, 0, 1], [0, 1, 0, 1]])
        #expect(log.entries.filter { $0.hasPrefix("run") }.count == 2)
    }

    @Test func theAttentionMaskFollowsTheManifestsRule() async throws {
        let manifest = try Self.manifest { json in
            var graphs = json["graphs"] as! [String: Any]
            graphs["textEncoder"] = Self.textGraph(buckets: nil, inputs: ["tokens": "tokens", "attentionMask": "attention_mask"])
            json["graphs"] = graphs
            var tokenizer = json["tokenizer"] as! [String: Any]
            tokenizer["text"] = ["padId": 49407, "padTo": 8, "maskRule": "throughFirstEos"]
            json["tokenizer"] = tokenizer
        }
        let log = Log()
        _ = try await Self.encoder(manifest, log: log).embed(texts: ["r g"])
        #expect(log.entries.last == "mask \([1, 1, 1, 1, 0, 0, 0, 0])")
        let ids: [Int32] = [49406, 5, 49407, 0, 49407, 49407]
        #expect(ImageTextEncoder.mask(ids, length: 3, rule: .throughFirstEos, pad: 49407) == [1, 1, 1, 1, 1, 0])
        #expect(ImageTextEncoder.mask(ids, length: 3, rule: .nonPad, pad: 49407) == [1, 1, 0, 1, 0, 0])
    }

    // MARK: - Images and zero-shot

    @Test func anImageEmbedsThroughThePreprocessor() async throws {
        let log = Log()
        let encoder = try Self.encoder(Self.manifest(), log: log)
        let red = try await encoder.embed(image: Self.image((255, 0, 0)))
        #expect(red == Embedding([1, 0, 0, 0]))
        #expect(log.entries == ["load imageEncoder", "run image"])
        // The prepared (CPU) and the graph (GPU) halves separately give the same embedding.
        #expect(try await encoder.embed(image: encoder.prepare(Self.image((255, 0, 0)))) == red)
    }

    @Test func classificationIsTheSoftmaxOfScaledSimilarities() async throws {
        let log = Log()
        let encoder = try Self.encoder(Self.manifest(), log: log)
        let scores = try await encoder.classify(Self.image((255, 0, 0)), labels: ["r", "g", "b"], template: "a {}")
        #expect(scores.map(\.label) == ["r", "g", "b"])
        // cos(red, text "a r") = 1/√2, the others 0: softmax of [100/√2, 0, 0].
        let top = exp(Double(100 / Float(2).squareRoot()))
        #expect(abs(Double(scores[0].score) - top / (top + 2)) < 1e-6)
        #expect(scores[1].score == scores[2].score && abs(scores.map(\.score).reduce(0, +) - 1) < 1e-6)
        // The image and the label texts under one request; the template applied to each label.
        #expect(log.entries.contains("load imageEncoder") && log.entries.contains("load textEncoder[4]"))
        #expect(log.entries.last?.contains("[49406, 200, \(Self.r), 49407,") == true)   // "a r": a</w> 200, r</w> 217

        // From embeddings already made, the same scores.
        let image = try await encoder.embed(image: Self.image((255, 0, 0)))
        let texts = try await encoder.embed(texts: ["a r", "a g", "a b"])
        #expect(encoder.scores(image: image, labels: ["r", "g", "b"], texts: texts) == scores)
        await #expect(throws: AetherError.self) { _ = try await encoder.classify(Self.image((0, 0, 0)), labels: []) }
    }

    @Test func theHostNormalizesWhenTheManifestSaysSo() async throws {
        let manifest = try Self.manifest { json in
            json["postprocess"] = ["kind": "embedding", "embedding": ["normalized": "host", "similarity": "dot", "logitScale": 1]]
        }
        let encoder = try Self.encoder(manifest)
        let embedding = try await encoder.embed(image: Self.image((255, 255, 0)))
        #expect(abs(embedding.norm - 1) < 1e-6 && abs(embedding.vector[0] - Float(0.5).squareRoot()) < 1e-6)
        let texts = try await encoder.embed(texts: ["r"])
        #expect(abs(texts[0].norm - 1) < 1e-6)
        #expect(encoder.similarity(embedding, texts[0]) == embedding.dot(texts[0]))
    }

    @Test func anEmbeddingOfTheWrongDimensionIsRefused() async throws {
        let log = Log()
        let encoder = try Self.encoder(Self.manifest(), log: log, imageDimension: 5)
        await #expect(performing: { _ = try await encoder.embed(image: Self.image((0, 0, 0))) },
                      throws: Self.isMismatch("graphs.imageEncoder.outputs"))
        #expect(!log.entries.contains("run image"))
    }

    @Test func embeddingsCompareByCosine() {
        let a = Embedding([3, 4]), b = Embedding([4, 3])
        #expect(a.dimension == 2 && a.norm == 5 && a.dot(b) == 24)
        #expect(abs(a.similarity(b) - 0.96) < 1e-6 && a.normalized == Embedding([0.6, 0.8]))
        #expect(Embedding([0, 0]).similarity(a) == 0 && Embedding([0, 0]).normalized == Embedding([0, 0]))
    }
}
