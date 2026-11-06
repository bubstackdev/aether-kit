import CoreGraphics
import CryptoKit
import Foundation
import Testing
import AetherHost
@testable import AetherRuntime
@testable import AetherVision

/// The CLIP ViT-B/32 gate (DESIGN.md §12): the exported `clip-joint@1` bundle, run through the driver on the GPU,
/// against transformers' `CLIPModel` on the CPU in float32 (`Forge/clip/reference_clip.py`) on the same inputs. Off by
/// default; run with `AETHER_GATES=1 swift test --no-parallel --filter ImageTextEncoderGateTests`.
///
/// - **Tokenizer:** the 17 reference texts (edge cases: empty, `!`, whitespace, accents, CJK, digits, truncation) id for
///   id against transformers' `CLIPTokenizer` (`texts/tokens.i32`).
/// - **Text tower:** all 17 in one request (buckets 8, 8, 1), cosine ≥ 0.999 per vector; one text alone (bucket 1)
///   matches its batched row.
/// - **Image tower:** each case's `input.f32` straight into the graph, cosine ≥ 0.999.
/// - **Zero-shot:** the ten labels with OpenAI's `"a photo of a {}."`, the argmax of each case matching the reference's;
///   the logits' largest difference is reported.
/// - **End to end from the source files** (`ImageInput`): decode, `centerCrop` with `pillowBicubic` and the towers. The
///   prepared tensor's largest difference from `input.f32` and the cosine are reported; the argmax is gated. The images
///   come from `Forge/clip/fetch_upstream.py`'s directory (`~/.cache/aether-kit/clip/upstream/clip-<commit>/images`) or
///   `AETHER_CLIP_IMAGES`, checked by SHA-256; absent, that part is skipped with a note.
///
/// The bundle and references default to the Forge outputs beside this repo (`aether/forge-outputs/clip-vit-b32`), or
/// `AETHER_CLIP_BUNDLE` and `AETHER_CLIP_REFERENCE`.
@Suite("Image-text encoder gates", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct ImageTextEncoderGateTests {
    static let environment = ProcessInfo.processInfo.environment
    static let outputs = forgeOutputs().appending(path: "clip-vit-b32")
    static let bundleURL = environment["AETHER_CLIP_BUNDLE"].map { URL(filePath: $0) } ?? outputs.appending(path: "bundle")
    static let referenceURL = environment["AETHER_CLIP_REFERENCE"].map { URL(filePath: $0) } ?? outputs.appending(path: "reference")
    static let mac = Device(platform: .macOS, architecture: Device.current.architecture)
    static let bar: Float = 0.999

    struct Index: Decodable {
        struct Texts: Decodable { var strings: [String] }
        struct ZeroShot: Decodable { var labels: [String]; var prompt: String; var logitScale: Double }
        struct Entry: Decodable { var name: String }
        var shapes: [String: [Int]]
        var texts: Texts
        var zeroShot: ZeroShot
        var cases: [Entry]
    }

    struct Case: Decodable {
        struct Source: Decodable { var kind: String; var upstreamFile: String?; var sha256: String? }
        struct ZeroShot: Decodable { var argmax: Int; var logits: [Float] }
        var name: String
        var source: Source
        var zeroShot: ZeroShot
    }

    static func ints(_ url: URL) throws -> [Int32] {
        try Data(contentsOf: url).withUnsafeBytes { raw in
            (0..<(raw.count / 4)).map { Int32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self)) }
        }
    }

    static func rows<T>(_ values: [T], width: Int) -> [[T]] {
        stride(from: 0, to: values.count, by: width).map { Array(values[$0..<($0 + width)]) }
    }

    static func argmax(_ values: [Float]) -> Int { values.indices.max { values[$0] < values[$1] }! }

    /// The images' directory: `AETHER_CLIP_IMAGES`, else the first `clip-*` fetch under the cache.
    static func imagesDirectory() -> URL? {
        if let path = environment["AETHER_CLIP_IMAGES"] { return URL(filePath: path) }
        let root = FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/aether-kit/clip/upstream")
        let fetched = (try? FileManager.default.contentsOfDirectory(atPath: root.path))?.filter { $0.hasPrefix("clip-") }.sorted()
        return fetched?.first.map { root.appending(path: $0).appending(path: "images") }
    }

    @Test func clipViTB32MatchesTransformers() async throws {
        let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: Self.referenceURL.appending(path: "references.json")))
        let model = try await Registry.vision.load(
            try DepthEstimatorGateTests.localBundle(Self.bundleURL), as: ImageTextEncoderDriver.self,
            options: LoadOptions(device: Self.mac, cachePolicy: .default, cache: .default))
        let encoder = try model.encoder()
        #expect(encoder.postprocess.dimension == 512 && abs(Double(encoder.logitScale) - index.zeroShot.logitScale) < 1e-4)
        let dimension = 512, length = try #require(index.shapes["tokens"])[1]

        // Tokenizer: id for id.
        let texts = index.texts.strings
        let expectedIDs = Self.rows(try Self.ints(Self.referenceURL.appending(path: "texts/tokens.i32")), width: length)
        let ids = try encoder.tokenize(texts)
        #expect(expectedIDs.count == texts.count)
        for (i, text) in texts.enumerated() { #expect(ids[i] == expectedIDs[i], "tokenizer: \(text.debugDescription)") }
        print("gate: clip tokenizer \(zip(ids, expectedIDs).filter { $0 == $1 }.count)/\(texts.count) texts id-exact")

        // Text tower: every text in one request.
        let expectedTexts = Self.rows(try DepthEstimatorGateTests.floats(Self.referenceURL.appending(path: "texts/embedding.f32")),
                                      width: dimension).map(Embedding.init)
        let started = ContinuousClock.now
        let textEmbeddings = try await encoder.embed(texts: texts)
        let textElapsed = ContinuousClock.now - started
        let textCosines = zip(textEmbeddings, expectedTexts).map { $0.similarity($1) }
        print("gate: clip text tower (\(texts.count) texts, buckets 8+8+1, \(textElapsed)): cosine min \(textCosines.min()!), "
              + "mean \(textCosines.reduce(0, +) / Float(textCosines.count)); norms \(textEmbeddings.map(\.norm).min()!)–"
              + "\(textEmbeddings.map(\.norm).max()!)")
        for (i, cosine) in textCosines.enumerated() { #expect(cosine >= Self.bar, "text \(texts[i].debugDescription): cosine \(cosine)") }
        let alone = try await encoder.embed(texts: [texts[1]])
        print("gate: clip text tower bucket 1 vs bucket 8 row: cosine \(alone[0].similarity(textEmbeddings[1]))")
        #expect(alone[0].similarity(textEmbeddings[1]) >= Self.bar)
        #expect(await encoder.residentRoles == ["textEncoder", "textEncoder[8]"])

        // Labels for zero-shot, embedded once.
        let labels = index.zeroShot.labels
        let labelTexts = try await encoder.embed(texts: labels.map { index.zeroShot.prompt.replacingOccurrences(of: "{}", with: $0) })
        let images = Self.imagesDirectory()
        let shape = try #require(index.shapes["image"])

        for entry in index.cases {
            let directory = Self.referenceURL.appending(path: "images/\(entry.name)")
            let c = try JSONDecoder().decode(Case.self, from: Data(contentsOf: directory.appending(path: "case.json")))
            let input = try DepthEstimatorGateTests.floats(directory.appending(path: "input.f32"))
            let expected = Embedding(try DepthEstimatorGateTests.floats(directory.appending(path: "embedding.f32")))

            // Image tower: the reference tensor straight in.
            let (height, width) = (shape[2], shape[3])
            let geometry = ImageGeometry(sourceWidth: width, sourceHeight: height, width: width, height: height,
                                         content: CGRect(x: 0, y: 0, width: width, height: height))
            let prepared = PreparedImage(shape: shape, values: .float32(input), geometry: geometry)
            let imageStarted = ContinuousClock.now
            let embedding = try await encoder.embed(image: prepared)
            let imageElapsed = ContinuousClock.now - imageStarted
            let cosine = embedding.similarity(expected)
            #expect(cosine >= Self.bar, "\(entry.name): image cosine \(cosine)")
            #expect(try await encoder.embed(image: prepared) == embedding, "\(entry.name): repeat bit-identical")

            // Zero-shot from the label embeddings, and through classify (one request).
            let logits = labelTexts.map { encoder.logitScale * embedding.similarity($0) }
            let worst = zip(logits, c.zeroShot.logits).map { abs($0 - $1) }.max()!
            let scores = try await encoder.classify(prepared, labels: labels, template: index.zeroShot.prompt)
            let chosen = Self.argmax(scores.map(\.score))
            #expect(chosen == c.zeroShot.argmax && Self.argmax(logits) == c.zeroShot.argmax, "\(entry.name): zero-shot argmax")
            print("gate: clip \(entry.name) graph: cosine \(cosine), \(imageElapsed); zero-shot \(labels[chosen]) "
                  + "(\(scores[chosen].score)), reference \(labels[c.zeroShot.argmax]), logits max |Δ| \(worst)")

            // End to end from the source file.
            guard c.source.kind == "image", let file = c.source.upstreamFile, let sha = c.source.sha256 else { continue }
            guard let url = images?.appending(path: file), let data = try? Data(contentsOf: url) else {
                print("gate: clip \(entry.name): no source image (Forge/clip/fetch_upstream.py); skipped")
                continue
            }
            #expect(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == sha, "\(file): SHA-256")
            let source = ImageInput(contentsOf: url)
            let ours = try encoder.prepare(source)
            let inputΔ = zip(ours.floats(), input).map { abs($0 - $1) }.max()!
            let fromFile = try await encoder.embed(image: source)
            let fileScores = try await encoder.classify(source, labels: labels, template: index.zeroShot.prompt)
            let fileChosen = Self.argmax(fileScores.map(\.score))
            #expect(fileChosen == c.zeroShot.argmax, "\(entry.name): end-to-end zero-shot argmax")
            print("gate: clip \(entry.name) end to end \(ours.geometry.sourceWidth)x\(ours.geometry.sourceHeight): prepared "
                  + "input max |Δ| \(inputΔ), cosine \(fromFile.similarity(expected)), zero-shot \(labels[fileChosen]) "
                  + "(\(fileScores[fileChosen].score))")
        }
        for (key, footprint) in encoder.options.ledger.snapshot.sorted(by: { $0.key.description < $1.key.description }) {
            print("gate: clip \(key): peak \(footprint.peakBytes >> 20) MiB, held \(footprint.heldBytes >> 20) MiB")
        }
        await encoder.unload()
        #expect(await encoder.residentRoles.isEmpty)
    }
}
