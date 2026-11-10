import Foundation
import Testing
@testable import AetherText

/// The gate of the Forge Qwen3 `decoder-lm@1` exporter (`Forge/qwen3/export_qwen3.py`): a local export of
/// `huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated@c9bd464`, opened from its folder (not the catalog), must give
/// the 0.0.7 export's greedy replies (Fixtures/decoder-lm-export, recorded with 0.0.7's own tokenizer and generation
/// session; synthetic prompts). Off by default; run with
/// `AETHER_GATES=1 scripts/gpu-lease -- swift test --no-parallel --filter DecoderLMExportGateTests`.
///
/// **The bar.** A reply is token-exact, or equal up to a position where the reference itself is at a near tie, where it
/// may take the reference's runner-up (INTEGRATION.md's near-tie rule: the reference's top-1 minus top-2 probability
/// below 0.1, and our token in its top five); after such a fork the replies are different texts and are not compared.
/// Both exports keep float16 logits, so a near tie can be an exact tie in the reference broken by id order, and a
/// one-ulp drift between the two graphs (the 0.0.x bucketed attention over a padded cache, ours over the written rows)
/// moves it. Teacher-forced on the reference's replies, the rule holds at every position. Exact and forked replies are
/// counted in the output (`GATE ... token-exact`).
///
/// The usage it covers, on the bundle as an app opens it:
/// - **an open-only pre-warm**: the model opened and released without a request, then opened again;
/// - **system + user turns** (and a user-only turn), each its own conversation, rendered by the bundle's chat template
///   to the reference's prompt ids exactly;
/// - **greedy decode with a `maxTokens` limit**: the replies (without the stop token) and why they ended (a stop token,
///   or the limit) equal the reference's, including prompts longer than one 512-token prefill chunk;
/// - **several requests on one opened model**, in the reference's order (the last repeats the first), with the
///   Generator's prefix reuse between them, then each case again on a fresh model with nothing to reuse.
///
/// The export defaults to the Forge outputs beside this repo (`forge-outputs/huihui-qwen3-4b-instruct-2507-abliterated`),
/// or `AETHER_QWEN3_EXPORT`.
@Suite("Qwen3 decoder-lm export gates", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct DecoderLMExportGateTests {
    struct Reference: Decodable {
        struct Case: Decodable {
            var id: String
            var system: String?
            var user: String
            var maximumTokens: Int
            var promptTokenIDs: [Int32]
            var tokens: [Int32]
            var stopReason: String
            var margins: [Float]
            var probabilityMargins: [Float]
            var top5: [[Int32]]
        }
        var cases: [Case]
    }

    static let exportURL = ProcessInfo.processInfo.environment["AETHER_QWEN3_EXPORT"].map { URL(filePath: $0) }
        ?? textForgeOutputs().appending(path: "huihui-qwen3-4b-instruct-2507-abliterated")
    static let mac = Device(platform: .macOS, architecture: Device.current.architecture)

    static let reference: Reference = {
        let url = Bundle.module.url(forResource: "Fixtures/decoder-lm-export/qwen3-4b-instruct-2507-abliterated", withExtension: "json")!
        return try! JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
    }()

    /// A Forge export is laid out as an installed revision root; it is not catalogued, so its ref is local.
    static func localBundle() throws -> (InstalledBundle, Manifest) {
        let manifest = try Manifest.load(fromBundle: exportURL)
        let ref = ModelRef(
            catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: manifest.variant.platform.rawValue,
            driver: manifest.driver, assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
        return (InstalledBundle(ref: ref, url: exportURL, isStale: false), manifest)
    }

    static func open() async throws -> Generator {
        let (bundle, manifest) = try localBundle()
        return try await ChatSession.generator(bundle: bundle, manifest: manifest, device: mac)
    }

    static func messages(_ c: Reference.Case) -> [ChatMessage] {
        (c.system.map { [ChatMessage.system($0)] } ?? []) + [.user(c.user)]
    }

    /// The reply's tokens without a final stop token, and the stop reason in the reference's words.
    static func reply(_ generator: Generator, _ c: Reference.Case) async throws -> (tokens: [Int32], stop: String, summary: GenerationSummary) {
        var summary: GenerationSummary?
        for try await event in generator.generate(messages(c), parameters: GenerationParameters(maxTokens: c.maximumTokens)) {
            if case .finished(let s) = event { summary = s }
        }
        let s = try #require(summary)
        switch s.stopReason {
        case .stopToken: return (Array(s.tokens.dropLast()), "eos", s)
        case .maxTokens: return (s.tokens, "output-limit", s)
        default: return (s.tokens, "\(s.stopReason)", s)
        }
    }

    /// The first position where two replies part, nil when they are identical.
    static func fork(_ got: [Int32], _ c: Reference.Case) -> Int? {
        zip(got, c.tokens).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset
            ?? (got.count == c.tokens.count ? nil : min(got.count, c.tokens.count))
    }

    /// INTEGRATION.md's near-tie rule at reference position `i`: our token may differ from the reference's only when
    /// its top-1 minus top-2 probability is below 0.1 and ours is in its top five.
    static func nearTie(_ token: Int32, at i: Int, _ c: Reference.Case) -> Bool {
        i < c.tokens.count && c.probabilityMargins[i] < 0.1 && c.top5[i].contains(token)
    }

    static func describe(_ got: [Int32], _ c: Reference.Case) -> String {
        guard let i = fork(got, c) else { return "identical" }
        let margins = i < c.margins.count ? "logit \(c.margins[i]), probability \(c.probabilityMargins[i])" : "n/a"
        return "forks at \(i) of \(c.tokens.count): got \(i < got.count ? "\(got[i])" : "end"), reference "
            + "\(i < c.tokens.count ? "\(c.tokens[i])" : "end"), its top five \(i < c.top5.count ? "\(c.top5[i])" : "n/a") "
            + "(reference top-1 minus top-2: \(margins))"
    }

    /// Token-exact, or forked at a reference near tie; records an issue otherwise. Returns whether it was exact.
    static func check(_ got: [Int32], _ stop: String, _ c: Reference.Case) -> Bool {
        guard let i = fork(got, c) else {
            #expect(stop == c.stopReason, "\(c.id): stopped by \(stop), the reference by \(c.stopReason)")
            return stop == c.stopReason
        }
        #expect(i < got.count && nearTie(got[i], at: i, c), "\(c.id): \(describe(got, c))")
        return false
    }

    @Test func greedyRepliesMatchThe007Export() async throws {
        // An open-only pre-warm, as an app does before its first request: open, release, open again.
        var started = ContinuousClock.now
        do { _ = try await Self.open() }
        let prewarm = ContinuousClock.now - started
        started = ContinuousClock.now
        let generator = try await Self.open()
        let reopen = ContinuousClock.now - started
        print("GATE decoder-lm export: pre-warm open \(prewarm), reopen \(reopen), \(Self.reference.cases.count) cases")

        let (bundle, manifest) = try Self.localBundle()
        let tokenizer = try await TokenizerAdapter(bundle: bundle, manifest: manifest)
        let renderer = try PromptRenderers.make(manifest: manifest, bundle: bundle, tokenizer: tokenizer)
        var exact = 0
        for c in Self.reference.cases {
            let prompt = try renderer.render(Self.messages(c), options: RenderOptions()).tokens
            #expect(prompt == c.promptTokenIDs, "\(c.id): the prompt renders differently from the reference")
            let (tokens, stop, summary) = try await Self.reply(generator, c)
            print("GATE \(c.id): \(summary.promptTokens) prompt (\(summary.reusedTokens) reused), \(tokens.count) generated "
                  + "(\(stop), \(summary.decodeMode)), \(Self.describe(tokens, c))")
            if Self.check(tokens, stop, c) { exact += 1 }
        }
        print("GATE decoder-lm export, one opened model: \(exact)/\(Self.reference.cases.count) token-exact")
    }

    @Test func freshModelRepliesMatchToo() async throws {
        // Each case on a model with nothing to reuse: what a one-request-per-open caller sees.
        var exact = 0
        for c in Self.reference.cases {
            let generator = try await Self.open()
            let (tokens, stop, summary) = try await Self.reply(generator, c)
            #expect(summary.reusedTokens == 0)
            print("GATE fresh \(c.id): \(Self.describe(tokens, c))")
            if Self.check(tokens, stop, c) { exact += 1 }
        }
        print("GATE decoder-lm export, fresh model per case: \(exact)/\(Self.reference.cases.count) token-exact")
    }
}

extension DecoderLMExportGateTests {
    /// The engine fed each reference reply after its prompt: at every position our argmax is the reference's token or
    /// within the near-tie rule. Also reports how far our logit margin between the reference's top two moves from
    /// the reference's own (the drift between the two exports, in logits), and prints the export's
    /// ``DecoderEngine/qualifiedModelBytesTable`` entry from the engine's measured model bytes over every case
    /// (`GATE qualified model bytes entry:`), ready to add to `Sources/AetherRuntime/Decoder/QualifiedModelBytes.swift`;
    /// an entry the kit already has for the export on this chip must cover the measurement.
    @Test func teacherForcedRepliesFollowTheReference() async throws {
        let (bundle, manifest) = try Self.localBundle()
        let engine = try await DecoderEngine(bundle: bundle, manifest: manifest, device: Self.mac)
        var positions = 0, differing = 0, worstDrift: Float = 0, drift: Float = 0
        for c in Self.reference.cases {
            try await engine.reset()
            var logits = try #require(try await engine.prefill(c.promptTokenIDs, logits: .last))
            for i in c.tokens.indices {
                let row = logits.last
                let ours = try Logits.argmax(row)
                let base = row.startIndex
                let margin = row[base + Int(c.top5[i][0])] - row[base + Int(c.top5[i][1])]
                let d = abs(margin - c.margins[i])
                worstDrift = max(worstDrift, d)
                drift += d
                positions += 1
                if ours != c.tokens[i] {
                    differing += 1
                    print("GATE teacher-forced \(c.id) at \(i): ours \(ours), reference \(c.tokens[i]) (top five \(c.top5[i]), "
                          + "reference margins logit \(c.margins[i]) probability \(c.probabilityMargins[i]); our margin \(margin))")
                    #expect(Self.nearTie(ours, at: i, c), "\(c.id) at \(i): ours \(ours), reference \(c.tokens[i])")
                }
                if i + 1 < c.tokens.count { logits = try await engine.step(c.tokens[i]) }
            }
        }
        print("GATE decoder-lm export, teacher-forced: \(positions - differing)/\(positions) positions take the reference's "
              + "token; top-two margin drift mean \(drift / Float(positions)), worst \(worstDrift) (logits)")
        let memory = await engine.memory
        print("GATE decoder-lm export memory: charged \(memory.chargedModelBytes >> 20) MiB, measured model "
              + "\(memory.measuredModelBytes >> 20) MiB, peak \((memory.peakBytes - memory.baselineBytes) >> 20) MiB above the baseline")
        print("GATE qualified model bytes entry: " + DecoderEngine.qualifiedEntry(
            measured: memory.measuredModelBytes, manifest: manifest, device: Self.mac,
            note: "Qwen3 export gate, teacher-forced over \(Self.reference.cases.count) cases"))
        if let qualified = DecoderEngine.qualifiedModelBytes(for: manifest, device: Self.mac) {
            #expect(qualified >= memory.measuredModelBytes, "the qualified entry no longer covers the measurement; replace it")
        }
    }
}

/// The Forge outputs directory beside the repo: the first `forge-outputs` found walking up from this file
/// (`AETHER_FORGE_OUTPUTS` overrides).
func textForgeOutputs(from file: String = #filePath) -> URL {
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
