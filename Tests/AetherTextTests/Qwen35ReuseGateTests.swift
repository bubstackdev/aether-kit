import AetherCore
import AetherHost
@_spi(Diagnostics) import AetherRuntime
import Foundation
import Testing
@testable import AetherText

/// The Generator's prefix reuse on a real Qwen3.5 / Qwen3.6 hybrid, opt-in: a tool chat (the system prompt, tools and
/// user turns of `AETHER_QWEN35_CHAT`, thinking off, a repetition penalty as Atelier's) with a one-picture side
/// request between its turns (`AETHER_QWEN35_PICTURE`, optional), then a new conversation. After every text request
/// the engine's state (the live state the request left, built from restored snapshots, live states and decoded tokens)
/// is checked against a fresh one: the same tokens fed from an empty state in the same calls (a recording engine
/// keeps each state's calls, and a snapshot carries its own). The next row's logits must be equal bit for bit. Also
/// printed, not asserted: against one prefill call of every token, which differs without any reuse as well (decoded
/// tokens go through one-token calls; the export's rows round differently per call shape).
///
///     AETHER_GATES=1 AETHER_QWEN35_BUNDLE=<export> AETHER_QWEN35_CHAT=<chat.json> [AETHER_QWEN35_PICTURE=<jpg>] \
///         scripts/gpu-lease -- swift test --filter Qwen35ReuseGateTests
///
/// `chat.json` as `Qwen35ScheduleGateTests`'; `users` (optional) adds later user turns. Prints `GATE qwen35-reuse …`.
@Suite("Qwen35ReuseGateTests", .serialized, .enabled(if: Qwen35ScheduleGate.enabled))
struct Qwen35ReuseGateTests {
    struct Chat: Decodable {
        let system: String
        let user: String
        let users: [String]?
        let tools: [Qwen35ScheduleGateTests.Chat.Tool]
    }

    @Test func reusedStatesEqualAFreshPrefill() async throws {
        let env = Qwen35ScheduleGate.env
        let chat = try JSONDecoder().decode(Chat.self, from: Data(contentsOf: URL(fileURLWithPath: env["AETHER_QWEN35_CHAT"]!)))
        let url = URL(fileURLWithPath: env["AETHER_QWEN35_BUNDLE"]!)
        let bundle = try Qwen35GateTests.localBundle(url)
        let manifest = try Manifest.load(fromBundle: url)
        let inner = try await DecoderEngine(bundle: bundle, manifest: manifest, device: .current, options: .init(canary: false))
        let engine = RecordingEngine(inner)
        let base = try await ChatSession.generator(engine: inner, bundle: bundle, manifest: manifest)
        let generator = Generator(engine: engine, decoder: base.decoder, renderer: base.renderer, stopTokens: base.stopTokens,
                                  profile: base.profile, gpu: base.gpu)
        let tools = chat.tools.map { ToolDefinition(name: $0.name, description: $0.description, parameters: $0.parameters) }
        let dialect = try ToolDialects.make(generator.profile.toolDialect)
        let parameters = GenerationParameters(
            maxTokens: 48, sampling: SamplingParameters(repetitionPenalty: 1.05, repetitionWindow: 256), thinking: false, tools: tools)
        var failures: [String] = []

        func reply(_ messages: [ChatMessage], _ label: String) async throws -> (ChatMessage, GenerationSummary) {
            var text = "", calls: [ToolCall] = []
            var summary: GenerationSummary?
            for try await event in generator.generate(messages, parameters: parameters) {
                switch event {
                case .text(let d): text += d
                case .toolCall(let body): calls += (try? dialect.parse(body)) ?? []
                case .finished(let s): summary = s
                default: break
                }
            }
            let s = try #require(summary)
            print("GATE qwen35-reuse \(label): prompt \(s.promptTokens), reused \(s.reusedTokens), prefill \(String(format: "%.3f", s.prefillSeconds)) s, "
                  + "\(s.tokens.count) tokens, \(calls.count) tool calls")
            return (.assistant(text.trimmingCharacters(in: .whitespacesAndNewlines), toolCalls: calls), s)
        }

        /// The state the request left against a fresh prefill of the tokens the Generator says the engine holds; the
        /// state is put back after, so the next request reuses it as it would have.
        func check(_ label: String) async throws {
            let held = await generator.mirroredTokens
            let calls = await engine.calls
            #expect(await inner.position == held.count)
            #expect(calls.flatMap(\.tokens) == held, "the recorded calls feed the tokens the Generator holds")
            let saved = try await inner.snapshot()
            let probe = held.first { $0 < Int32(inner.vocab) && $0 >= 0 } ?? 0
            let reused = Array(try #require(try await inner.prefill([probe], logits: .last)).last)
            // A fresh state: reset (which unbinds the images), the bound images bound again (encoded again), the calls.
            let rebind = await inner.boundImages.isEmpty ? nil : await engine.lastBind
            func freshState() async throws {
                try await inner.reset()
                if let rebind { try await inner.bindImages(rebind.plan, prepared: rebind.prepared) }
            }
            try await freshState()
            for call in calls { _ = try await inner.prefill(call.tokens, logits: call.select) }
            let replayed = Array(try #require(try await inner.prefill([probe], logits: .last)).last)
            try await freshState()
            let oneCall = Array(try #require(try await inner.prefill(held + [probe], logits: .last)).last)
            try await inner.restore(saved)
            let diff = zip(reused, oneCall).map { abs($0 - $1) }.max() ?? 0
            print("GATE qwen35-reuse \(label): \(held.count) held tokens in \(calls.count) calls; next row vs a fresh state fed the "
                  + "same calls: \(reused == replayed ? "bit-identical" : "DIFFERS"); vs one prefill call: "
                  + (reused == oneCall ? "bit-identical" : "max |diff| \(diff) (argmax \(try Logits.argmax(reused[...]) == Logits.argmax(oneCall[...]) ? "same" : "differs"))"))
            if reused != replayed { failures.append(label) }
        }

        func toolOutput(_ call: ToolCall) -> String { "ok: \(call.name) done" }

        var history: [ChatMessage] = [.system(chat.system), .user(chat.user)]
        let users = chat.users ?? ["Thanks. What else?", "One more change, please."]
        var summaries: [GenerationSummary] = []
        for (turn, user) in ([chat.user] + users).enumerated() {
            if turn > 0 { history.append(.user(user)) }
            for step in 0..<3 {
                let (message, summary) = try await reply(history, "turn \(turn + 1) step \(step + 1)")
                summaries.append(summary)
                history.append(message)
                try await check("turn \(turn + 1) step \(step + 1)")
                if message.toolCalls.isEmpty { break }
                for call in message.toolCalls { history.append(.tool(toolOutput(call), name: call.name)) }
            }
            if turn == 0, let picture = env["AETHER_QWEN35_PICTURE"] {
                var look: GenerationSummary?
                let image = ChatImage(ImageInput(contentsOf: URL(fileURLWithPath: picture)))
                for try await event in generator.generate([.system("Describe only the hair."), .user("Describe the hair.", images: [image])],
                                                           parameters: GenerationParameters(maxTokens: 12, sampling: parameters.sampling, thinking: false)) {
                    if case .finished(let s) = event { look = s }
                }
                print("GATE qwen35-reuse look (picture side request): prompt \(look?.promptTokens ?? 0), reused \(look?.reusedTokens ?? 0)")
            }
        }
        // Every request after the first reused something.
        #expect(summaries.dropFirst().allSatisfy { $0.reusedTokens > 0 })
        // A new conversation with the same system turn starts from its snapshot.
        let (_, fresh) = try await reply([.system(chat.system), .user(users.last!)], "new conversation")
        try await check("new conversation")
        #expect(fresh.reusedTokens > 0)
        // Studio's outfit describer: questions about one picture, the picture and shared instructions first, each
        // question in a turn of its own. The tower runs once and the shared part is prefilled once.
        if let picture = env["AETHER_QWEN35_PICTURE"] {
            let image = ChatImage(ImageInput(contentsOf: URL(fileURLWithPath: picture)))
            let instructions = "You describe the outfit in a picture for an image generator: garments with colors, materials and cut, accessories and footwear, as short comma-separated phrases. Never describe the person."
            var towerCalls = 0  // the requests' own (a check encodes again on its fresh state)
            var reusedCounts: [Int] = []
            for (k, part) in ["the footwear", "the belt and anything worn at the waist", "the gloves or bracers", "the headwear"].enumerated() {
                let messages: [ChatMessage] = [.system(instructions), .user("This is the outfit to describe.", images: [image]),
                                               .user("Describe only \(part), in one short phrase.")]
                var summary: GenerationSummary?
                let before = await inner.imageEncodes
                for try await event in generator.generate(messages, parameters: GenerationParameters(maxTokens: 16, sampling: parameters.sampling, thinking: false)) {
                    if case .finished(let s) = event { summary = s }
                }
                towerCalls += await inner.imageEncodes - before
                let s = try #require(summary)
                reusedCounts.append(s.reusedTokens)
                print("GATE qwen35-reuse describe \(k + 1) (\(part)): prompt \(s.promptTokens), reused \(s.reusedTokens), prefill "
                      + "\(String(format: "%.3f", s.prefillSeconds)) s, tower calls so far \(towerCalls)")
                try await check("describe \(k + 1)")
            }
            #expect(towerCalls == 1, "the picture is encoded once")
            #expect(reusedCounts.dropFirst().allSatisfy { $0 > 0 && $0 == reusedCounts[1] })
        }
        print("GATE qwen35-reuse states that differ from a fresh prefill: \(failures)")
        #expect(failures.isEmpty)
    }
}

/// Forwards to a ``DecoderEngine`` and keeps the calls that fed its current state: a reset empties them, a snapshot
/// carries them, a restore puts them back.
actor RecordingEngine: ImageConditionedEngine {
    struct Call: Sendable {
        var tokens: [Int32]
        var select: LogitsSelection
    }

    struct Payload: Sendable {
        var inner: EngineSnapshot
        var calls: [Call]
        var images: [ImagePromptPlan.Image]
    }

    let inner: DecoderEngine
    private(set) var calls: [Call] = []
    /// The last bind, so a check can bind the same images on a fresh state.
    private(set) var lastBind: (plan: ImagePromptPlan, prepared: [PreparedPromptImage])?

    init(_ inner: DecoderEngine) { self.inner = inner }

    nonisolated var vocab: Int { inner.vocab }
    nonisolated var maxCapacity: Int { inner.maxCapacity }
    nonisolated var hasRecurrentState: Bool { inner.hasRecurrentState }
    nonisolated var imageInput: ImageInputSpec? { inner.imageInput }
    /// The inner engine's, kept in step (host decoding only: the state never runs ahead).
    var position: Int { calls.reduce(0) { $0 + $1.tokens.count } }
    var stateRanAhead: Bool { false }
    var decodeModeDescription: String { "host (recording)" }
    private(set) var boundImages: [ImagePromptPlan.Image] = []

    func prefill(_ tokens: [Int32], logits: LogitsSelection) async throws(AetherError) -> Logits? {
        let out = try await inner.prefill(tokens, logits: logits)
        calls.append(Call(tokens: tokens, select: logits))
        return out
    }

    func step(_ token: Int32) async throws(AetherError) -> Logits {
        let out = try await inner.step(token)
        calls.append(Call(tokens: [token], select: .last))
        return out
    }

    func rewind(to position: Int) async throws(AetherError) {
        guard position == 0 else { throw .unsupportedOperation("the recording engine rewinds only to 0") }
        try await reset()
    }

    func reset() async throws(AetherError) {
        try await inner.reset()
        calls = []
        boundImages = []
    }

    func snapshot() async throws(AetherError) -> EngineSnapshot {
        let s = try await inner.snapshot()
        return EngineSnapshot(position: s.position, byteCount: s.byteCount, payload: Payload(inner: s, calls: calls, images: boundImages))
    }

    func restore(_ snapshot: EngineSnapshot) async throws(AetherError) {
        guard let payload = snapshot.payload as? Payload else { throw .unsupportedOperation("not a recording engine's snapshot") }
        try await inner.restore(payload.inner)
        calls = payload.calls
        boundImages = payload.images
    }

    func beginGreedy(first: Int32, limit: Int) async throws(AetherError) {
        throw .unsupportedOperation("the recording engine decodes on the host only (a repetition penalty)")
    }

    func nextGreedy() async throws(AetherError) -> Int32? { nil }

    @discardableResult
    func endGreedy() async throws(AetherError) -> (fed: Int, overshoot: Int) { (0, 0) }

    func bindImages(_ plan: ImagePromptPlan, prepared: [PreparedPromptImage]) async throws(AetherError) {
        lastBind = (plan, prepared)
        do {
            try await inner.bindImages(plan, prepared: prepared)
        } catch {
            boundImages = await inner.boundImages
            throw error
        }
        boundImages = await inner.boundImages
    }
}
