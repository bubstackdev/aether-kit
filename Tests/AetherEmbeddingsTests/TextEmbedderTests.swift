import Foundation
import Synchronization
import Testing
@_spi(Testing) import AetherHost
@testable import AetherEmbeddings

/// `text-embedder@1` over a fake tokenizer and fake graphs (DESIGN.md §12): the contract, `tokenizer.text` (prompts,
/// strip, special tokens, truncation, padding and mask), the `embedding` postprocess, requests and cancellation. A
/// constant probe and no GPU admission, so nothing depends on the machine.
@Suite("Text embedder")
struct TextEmbedderTests {
    static let length = 8

    /// An EmbeddingGemma-shaped manifest with an 8-token graph; `edit` changes the JSON object before decoding.
    static func manifest(_ edit: (inout [String: Any]) -> Void = { _ in }) throws -> Manifest {
        var json: [String: Any] = [
            "schema": 1, "id": "embedder-test", "task": "text-embedding", "driver": "text-embedder@1",
            "variant": ["platform": "macos"], "assets": ["main": ["subtree": ""]],
            "graphs": ["encoder": [
                "file": "main:encoder.aimodel", "function": "main",
                "inputs": ["tokens": "input_ids", "attentionMask": "attention_mask"], "outputs": ["embedding": "embedding"],
                "compute": "gpu", "limits": ["staticSequence": length],
            ] as [String: Any]],
            "tokenizer": [
                "source": ["dir": "main:tokenizer"], "addSpecialTokens": true,
                "text": ["padId": 0, "padSide": "right", "padTo": length, "truncation": "right",
                         "prompts": ["query": "q: ", "document": "d: "]] as [String: Any],
            ] as [String: Any],
            "postprocess": ["kind": "embedding", "embedding": ["normalized": "inGraph", "pooling": "inGraph", "similarity": "cosine"]],
        ]
        edit(&json)
        return try Manifest.decode(JSONSerialization.data(withJSONObject: json))
    }

    /// A fake tokenizer: each character's scalar value is one id; the special tokens are `<bos>` 2 and `<eos>` 1.
    static let encode: TextEmbedder.Encode = { text, specials in
        let ids = text.unicodeScalars.map { Int32($0.value) }
        return specials ? [2] + ids + [1] : ids
    }

    static func ids(_ text: String) -> [Int32] { text.unicodeScalars.map { Int32($0.value) } }

    final class Log: Sendable {
        private let state = Mutex<[String]>([])
        var entries: [String] { state.withLock { $0 } }
        func append(_ entry: String) { state.withLock { $0.append(entry) } }
    }

    /// Holds the first run until released.
    actor Gate {
        private var entered = false, released = false
        private var waiting: CheckedContinuation<Void, Never>?
        private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
        func enter() async {
            guard !entered else { return }
            entered = true
            enteredWaiters.forEach { $0.resume() }
            enteredWaiters = []
            if !released { await withCheckedContinuation { waiting = $0 } }
        }
        func waitUntilEntered() async { if !entered { await withCheckedContinuation { enteredWaiters.append($0) } } }
        func release() {
            released = true
            waiting?.resume()
            waiting = nil
        }
    }

    /// The encoder's I/O at `sequence`: the embedding is (real ids, first id, last real id, 1), unnormalized.
    final class FakeEncoder: GraphBackend {
        let signature: GraphSignature
        let log: Log
        let gate: Gate?
        init(log: Log, gate: Gate?, sequence: Int) {
            signature = GraphSignature(
                function: "main",
                inputs: [TensorSpec(name: "input_ids", scalarType: .int32, shape: [1, sequence]),
                         TensorSpec(name: "attention_mask", scalarType: .int32, shape: [1, sequence])],
                states: [], outputs: [TensorSpec(name: "embedding", scalarType: .float32, shape: [1, 4])])
            self.log = log
            self.gate = gate
        }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            log.append("run")
            await gate?.enter()
            let ids = try bindings.input("input_ids")!.read(Int32.self)
            let mask = try bindings.input("attention_mask")!.read(Int32.self)
            let real = Int(mask.reduce(0, +))
            try bindings.output("embedding")!.write([Float(real), Float(ids[0]), Float(ids[real - 1]), 1])
        }
    }

    static let device = Device(platform: .macOS, architecture: "test")

    static func embedder(_ manifest: Manifest, log: Log = Log(), gate: Gate? = nil, sequence: Int = length) throws -> TextEmbedder {
        try TextEmbedder(
            manifest: manifest, device: device,
            options: PipelineOptions(memory: .bytes(1 << 40), gpu: .unadmitted, probe: GPUMemoryProbe { 0 }),
            encode: encode,
            load: { role throws(AetherError) in
                log.append("load \(role)")
                return FakeEncoder(log: log, gate: gate, sequence: sequence)
            },
            fileBytes: { _ in 100 })
    }

    static func fails(_ check: @escaping (AetherError) -> Bool) -> (any Error) -> Bool {
        { error in (error as? AetherError).map(check) ?? false }
    }

    static func isInvalid(_ field: String) -> (any Error) -> Bool {
        fails { if case .manifestInvalid(let f, _) = $0 { f == field } else { false } }
    }

    static func setText(_ json: inout [String: Any], _ edit: (inout [String: Any]) -> Void) {
        var tokenizer = json["tokenizer"] as! [String: Any]
        var text = tokenizer["text"] as! [String: Any]
        edit(&text)
        tokenizer["text"] = text
        json["tokenizer"] = tokenizer
    }

    // MARK: - Contract and manifest

    @Test func theDriverIsRegisteredWithItsContract() throws {
        #expect(Registry.embeddings.supports(.textEmbedder))
        #expect(try Registry.embeddings.driver(for: .textEmbedder) == TextEmbedderDriver.self)
        #expect(DriverCatalog.contracts[.textEmbedder] == .textEmbedder)
        try DriverContract.textEmbedder.validate(Self.manifest())
    }

    @Test func tokenizerTextIsTyped() throws {
        let text = try #require(try Self.manifest().tokenizer?.text)
        #expect(text == .init(padId: 0, padSide: .right, padTo: 8, truncation: .right,
                              prompts: ["query": "q: ", "document": "d: "]))
        #expect(throws: AetherError.self) { _ = try Self.manifest { Self.setText(&$0) { $0["padTo"] = 0 } } }
        #expect(throws: AetherError.self) { _ = try Self.manifest { Self.setText(&$0) { $0["padId"] = -1 } } }
        #expect(throws: AetherError.self) { _ = try Self.manifest { Self.setText(&$0) { $0["truncation"] = "middle" } } }
    }

    @Test func theManifestIsCheckedBeforeAnythingLoads() throws {
        let invalid: [(String, (inout [String: Any]) -> Void)] = [
            ("graphs.encoder.inputs.attentionMask", { json in
                json["graphs"] = ["encoder": ["file": "main:e.aimodel", "function": "main", "inputs": ["tokens": "input_ids"],
                                              "outputs": ["embedding": "embedding"], "compute": "gpu"] as [String: Any]]
            }),
            ("postprocess", { json in json["postprocess"] = ["kind": "gliner"] }),
            ("postprocess", { json in json["postprocess"] = nil }),
            ("tokenizer.text", { json in
                var tokenizer = json["tokenizer"] as! [String: Any]
                tokenizer["text"] = nil
                json["tokenizer"] = tokenizer
            }),
            ("tokenizer.text", { json in Self.setText(&json) { $0["padTo"] = nil } }),
            ("tokenizer", { json in json["tokenizer"] = nil }),
        ]
        for (field, edit) in invalid {
            let manifest = try Self.manifest(edit)
            #expect("\(field)", performing: { _ = try Self.embedder(manifest) }, throws: Self.isInvalid(field))
        }
        let unsupported: [(inout [String: Any]) -> Void] = [
            { json in Self.setText(&json) { $0["padSide"] = "left" } },
            { json in Self.setText(&json) { $0["truncation"] = "word" } },
            { json in json["postprocess"] = ["kind": "embedding", "embedding": ["normalized": "host", "pooling": "mean", "similarity": "cosine"]] },
            { json in json["postprocess"] = ["kind": "embedding", "embedding": ["normalized": "none", "similarity": "dot"]] },
        ]
        for edit in unsupported {
            let manifest = try Self.manifest(edit)
            #expect(performing: { _ = try Self.embedder(manifest) },
                    throws: Self.fails { if case .unsupportedOperation = $0 { true } else { false } })
        }
        let mismatched = try Self.manifest { json in
            var graphs = json["graphs"] as! [String: [String: Any]]
            graphs["encoder"]!["limits"] = ["staticSequence": 16]
            json["graphs"] = graphs
        }
        #expect(performing: { _ = try Self.embedder(mismatched) }, throws: Self.fails {
            if case .manifestMismatch(let field, _, _) = $0 { field == "graphs.encoder.limits.staticSequence" } else { false }
        })
    }

    @Test func aGraphWhoseSequenceIsNotPadToIsRefused() async throws {
        let log = Log()
        let embedder = try Self.embedder(Self.manifest(), log: log, sequence: 16)
        await #expect(performing: { _ = try await embedder.embed("a", role: .query) }, throws: Self.fails {
            // The loaded graph's token axis is checked against limits.staticSequence (= padTo).
            if case .manifestMismatch(let field, _, _) = $0 { field == "graphs.encoder.limits.staticSequence" } else { false }
        })
        #expect(await embedder.residentRoles.isEmpty && !log.entries.contains("run"))
    }

    // MARK: - Tokenization

    @Test func theRolesPromptIsPrependedStrippedAndPaddedOnTheRight() throws {
        let embedder = try Self.embedder(Self.manifest())
        let query = try embedder.tokenize("hi \n", role: .query)
        // "q: hi" stripped, between <bos> and <eos>, then pad 0.
        #expect(query.ids == [2] + Self.ids("q: hi") + [1, 0])
        #expect(query.attentionMask == [Int32](repeating: 1, count: 7) + [0])
        #expect(query.tokenCount == 7 && !query.isTruncated)
        let document = try embedder.tokenize("x", role: .document)
        #expect(document.ids == [2] + Self.ids("d: x") + [1, 0, 0])
        #expect(document.attentionMask == [1, 1, 1, 1, 1, 1, 0, 0])
        // A role without a prompt is the bare text; leading whitespace of the prompt is stripped too.
        let bare = try embedder.tokenize(" y ", role: EmbeddingRole(rawValue: "classification"))
        #expect(bare.ids == [2] + Self.ids("y") + [1, 0, 0, 0, 0, 0])
        #expect(bare.tokenCount == 3)
        #expect(embedder.sequenceLength == 8)
    }

    @Test func truncationKeepsTheFirstContentIdsAndBothSpecialTokens() throws {
        let embedder = try Self.embedder(Self.manifest())
        // Exactly 8 with the specials: not truncated.
        let full = try embedder.tokenize("abc", role: .query)
        #expect(full.ids == [2] + Self.ids("q: abc") + [1] && !full.isTruncated && full.attentionMask.allSatisfy { $0 == 1 })
        // One more: the tail goes, the specials stay, no padding.
        let long = try embedder.tokenize("abcdefgh", role: .query)
        #expect(long.ids == [2] + Self.ids("q: abc") + [1])
        #expect(long.isTruncated && long.tokenCount == 8 && long.attentionMask.allSatisfy { $0 == 1 })
        // Without special tokens it is a plain prefix.
        let plain = try Self.embedder(Self.manifest { json in
            var tokenizer = json["tokenizer"] as! [String: Any]
            tokenizer["addSpecialTokens"] = false
            json["tokenizer"] = tokenizer
        })
        #expect(try plain.tokenize("abcdefgh", role: .query).ids == Self.ids("q: abcde"))
    }

    @Test func truncationRefusesSpecialTokensItCannotFind() throws {
        let rules = TextRules(padId: 0, length: 4, prompts: [:], addSpecialTokens: true)
        #expect(throws: AetherError.self) { _ = try rules.truncate([2, 9, 9, 9, 9, 1], content: [8, 8, 8, 8]) }
        #expect(throws: AetherError.self) { _ = try rules.truncate([2, 3, 4, 5, 9, 1], content: [9]) }
        #expect(try rules.truncate([2, 9, 8, 7, 6, 1], content: [9, 8, 7, 6]) == [2, 9, 8, 1])
    }

    // MARK: - Embeddings

    @Test func embeddingsComeBackInOrderFromOneRequest() async throws {
        let log = Log()
        let embedder = try Self.embedder(Self.manifest(), log: log)
        #expect(try await embedder.embed([], role: .query).isEmpty && log.entries.isEmpty)
        let vectors = try await embedder.embed(["a", "bb", "abcdefgh"], role: .document)
        // (real ids, <bos>, last real id = <eos>, 1); inGraph normalization leaves them as the graph wrote them.
        #expect(vectors.map(\.vector) == [[6, 2, 1, 1], [7, 2, 1, 1], [8, 2, 1, 1]])
        #expect(log.entries == ["load encoder", "run", "run", "run"])
        #expect(await embedder.residentRoles == ["encoder"])
        #expect(try await embedder.embed("a", role: .document) == vectors[0])
        let tokenized = try embedder.tokenize("bb", role: .document)
        #expect(try await embedder.embed([tokenized]) == [vectors[1]])
        await #expect(throws: AetherError.self) {
            _ = try await embedder.embed([TokenizedText(ids: [1], attentionMask: [1], tokenCount: 1, isTruncated: false)])
        }
    }

    @Test func hostNormalizationMakesUnitVectors() async throws {
        let embedder = try Self.embedder(Self.manifest { json in
            json["postprocess"] = ["kind": "embedding", "embedding": ["normalized": "host", "similarity": "cosine"]]
        })
        let e = try await embedder.embed("a", role: .query)
        let norm = e.vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
        #expect(abs(norm - 1) < 1e-6)
        #expect(abs(e.similarity(Embedding([6, 2, 1, 1])) - 1) < 1e-6)
    }

    @Test func similarityIsCosineAndMatryoshkaPrefixesRenormalize() {
        let a = Embedding([3, 4, 0, 0]), b = Embedding([6, 8, 0, 0]), c = Embedding([0, 0, 1, 0])
        #expect(abs(a.similarity(b) - 1) < 1e-6 && a.similarity(c) == 0 && a.similarity(Embedding([0, 0, 0, 0])) == 0)
        #expect(abs(a.similarity(Embedding([-3, -4, 0, 0])) + 1) < 1e-6)
        let short = Embedding([0.6, 0.0, 0.8, 0]).truncated(to: 2)
        #expect(short.dimension == 2 && short.vector == [1, 0])
    }

    // MARK: - Requests

    @Test func cancellationStopsBetweenGraphCallsAndTheEmbedderRecovers() async throws {
        let log = Log(), gate = Gate()
        let embedder = try Self.embedder(Self.manifest(), log: log, gate: gate)
        let task = Task { try await embedder.embed(["a", "b", "c"], role: .query) }
        await gate.waitUntilEntered()
        // A second request waits its turn and runs once the first ends.
        let queued = Task { try await embedder.embed(["d"], role: .query) }
        task.cancel()
        await gate.release()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(try await queued.value.count == 1)
        // The first stopped after its first graph call; the failure released the graph, so the second loaded it again.
        #expect(log.entries == ["load encoder", "run", "load encoder", "run"])
        await embedder.unload()
        #expect(await embedder.residentRoles.isEmpty)
    }
}
