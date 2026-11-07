public import AetherHost
public import AetherRuntime
import Foundation

public struct GenerationParameters: Sendable {
    public var maxTokens: Int
    public var sampling: SamplingParameters
    /// Text-level stops, besides the manifest's stop tokens. The matched text is not emitted.
    public var stopSequences: [String]
    /// Thinking on or off; nil leaves the template's default.
    public var thinking: Bool?
    /// Token constraints, intersected, applied to every next-token row on the CPU (structured output plugs in here).
    /// Each one's `accept` runs concurrently with the step whose logits it constrains.
    public var logitsProcessors: [any LogitsProcessor]
    /// Tools the model may call, rendered into the template. Needs a manifest `tokenizer.toolDialect` this build
    /// implements (`hermes-json`); otherwise the request is refused.
    public var tools: [ToolDefinition]
    /// The reply's GPU priority (absent: `.medium`). Keep the job to change it while the reply waits for the GPU.
    public var job: GPUJob?

    public init(
        maxTokens: Int = 512, sampling: SamplingParameters = .greedy, stopSequences: [String] = [], thinking: Bool? = nil,
        logitsProcessors: [any LogitsProcessor] = [], tools: [ToolDefinition] = [], job: GPUJob? = nil
    ) {
        self.maxTokens = maxTokens
        self.sampling = sampling
        self.stopSequences = stopSequences
        self.thinking = thinking
        self.logitsProcessors = logitsProcessors
        self.tools = tools
        self.job = job
    }
}

public enum GenerationEvent: Sendable, Equatable {
    /// Response text (deltas).
    case text(String)
    /// Text inside the manifest's thinking tags (deltas).
    case thinking(String)
    /// The raw body of one complete tool call (its dialect is M3's).
    case toolCall(String)
    case finished(GenerationSummary)
}

public struct GenerationSummary: Sendable, Equatable {
    public enum StopReason: Sendable, Equatable {
        case stopToken(Int32)
        case stopSequence(String)
        case maxTokens
        case capacity
    }

    public var promptTokens: Int
    /// Prompt tokens served from the engine's state (rewind or snapshot) instead of prefilled.
    public var reusedTokens: Int
    /// Every token chosen, including a final stop token.
    public var tokens: [Int32]
    /// Of `tokens`, those inside a thinking block (its tags included).
    public var thinkingTokens: Int
    public var stopReason: StopReason
    public var prefillSeconds: Double
    public var decodeSeconds: Double
    public var decodeMode: String
}

/// The single decode loop (DESIGN.md §7), shared by ChatSession, the FoundationModels executor and future
/// multimodal front ends. It owns a `LanguageEngine` (``DecoderEngine`` for `decoder-lm@1`) and keeps a token mirror of
/// the engine's state:
/// - attention-only models reuse the longest common prefix by `rewind` (O(1));
/// - models with recurrent state reuse the live state when the new prompt extends it, else the snapshot taken at
///   the end of the previous prompt (with that prompt's last logits), else start over.
///
/// Plain greedy decoding with no logits processor runs on the engine's pipelined GPU path; anything else feeds one
/// token at a time and samples on the CPU. Each request (prefill and decode of one reply) holds one GPU grant from
/// ``gpu`` (by default ``GPUAdmission/shared``) and requests are served one at a time: in GPU priority order
/// (`GenerationParameters.job`), in arrival order within a priority.
public actor Generator {
    /// The engine. Cast it (`as? DecoderEngine`) for a backend's own diagnostics; generate only through the Generator,
    /// which keeps its token mirror in step with the engine's state.
    public nonisolated let engine: any LanguageEngine
    /// The GPU admission every request takes before it renders or prefills.
    public nonisolated let gpu: GPUAdmissionOwner
    public nonisolated let decoder: any TokenDecoding
    public nonisolated let renderer: any PromptRenderer
    public nonisolated let stopTokens: Set<Int32>
    public nonisolated let profile: OutputProfile

    private var mirror: [Int32] = []
    /// False once the engine's live state no longer matches `mirror` (an error, or recurrent state that ran ahead).
    private var liveValid = true
    /// Hybrids: the snapshots of the last prompt, at the end of its rendered history (the prefix the next turn's
    /// prompt shares) and at its end (with its last logits, for a regenerate). At most ``maxSnapshots`` at once.
    private var snapshots: [(tokens: [Int32], state: EngineSnapshot, logits: Logits?)] = []

    /// The most engine snapshots a Generator holds at once (hybrids only): the two of the last prompt, and while a
    /// prompt takes their replacements, only the one it reused. What `DecoderEngine.Options.reservedSnapshots`
    /// reserves room for by default; size another engine's memory the same way.
    public static let maxSnapshots = 3
    /// One request at a time, in arrival order (after the GPU grant).
    private nonisolated let gate = SerialGate()
    /// The vocabulary indexed for JSON constraints, built on the first structured request (63 ms for Qwen3 in 0.0.x).
    private var jsonVocabulary: JSONVocabulary?

    public init(
        engine: any LanguageEngine, decoder: any TokenDecoding, renderer: any PromptRenderer, stopTokens: Set<Int32>,
        profile: OutputProfile, gpu: GPUAdmissionOwner = GPUAdmission.shared.owner("text")
    ) {
        self.engine = engine
        self.gpu = gpu
        self.decoder = decoder
        self.renderer = renderer
        self.stopTokens = stopTokens
        self.profile = profile
    }

    /// The prompt tokens the engine currently holds (diagnostics and tests).
    public var mirroredTokens: [Int32] { mirror }

    /// Renders `messages` and generates. Cancelling the consuming task stops the decode and leaves the engine
    /// consistent.
    public nonisolated func generate(_ messages: [ChatMessage], parameters: GenerationParameters = GenerationParameters())
        -> AsyncThrowingStream<GenerationEvent, any Error>
    {
        stream(job: parameters.job) { generator, continuation in
            if !parameters.tools.isEmpty { _ = try ToolDialects.make(generator.profile.toolDialect) }
            let rendered = try generator.renderer.render(
                messages, options: RenderOptions(thinking: parameters.thinking, tools: parameters.tools))
            // The history alone (no generation prompt): where the next turn's prompt will diverge from this one. Only a
            // recurrent model uses it (to snapshot there); an attention-only model reuses any common prefix.
            var boundary: Int?
            if generator.engine.hasRecurrentState {
                let history = try generator.renderer.render(
                    messages, options: RenderOptions(thinking: parameters.thinking, addGenerationPrompt: false, tools: parameters.tools)).tokens
                boundary = rendered.tokens.starts(with: history) && history.count < rendered.tokens.count ? history.count : nil
            }
            try await generator.run(rendered, parameters, continuation, boundary: boundary)
        }
    }

    /// Generates from prompt tokens already rendered.
    public nonisolated func generate(prompt: RenderedPrompt, parameters: GenerationParameters = GenerationParameters())
        -> AsyncThrowingStream<GenerationEvent, any Error>
    {
        stream(job: parameters.job) { generator, continuation in try await generator.run(prompt, parameters, continuation) }
    }

    private nonisolated func stream(
        job: GPUJob?,
        _ body: @escaping @Sendable (isolated Generator, AsyncThrowingStream<GenerationEvent, any Error>.Continuation) async throws -> Void
    ) -> AsyncThrowingStream<GenerationEvent, any Error> {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: GenerationEvent.self)
        let task = Task {
            // The GPU before this generator's own turn, so priority also orders this generator's requests. The grant
            // is released after the turn, once the decode's GPU work has joined.
            let grant: GPUAdmissionGrant
            do {
                grant = try await self.gpu.admit(job: job)
            } catch {
                continuation.finish(throwing: error)
                return
            }
            defer { grant.release() }
            do {
                // A request cancelled while it waits for its turn leaves the queue at once: it renders and prefills
                // nothing.
                try await self.gate.withTurn {
                    do {
                        try await body(self, continuation)
                        continuation.finish()
                    } catch AetherError.runtime(let underlying) where underlying is CancellationError {
                        continuation.finish(throwing: CancellationError())
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    // MARK: - Prefix reuse

    /// Brings the engine to `prompt`'s end, reusing what it can. Returns the prompt's last logits and the reused count.
    private func prepare(_ prompt: [Int32], boundary: Int?) async throws -> (Logits, Int) {
        let recurrent = engine.hasRecurrentState
        if !recurrent {
            let lcp = liveValid ? Self.commonPrefix(mirror, prompt) : 0
            let keep = min(lcp, prompt.count - 1)
            try await engine.rewind(to: keep)
            mirror = Array(prompt.prefix(keep))
            let logits = try await engine.prefill(Array(prompt[keep...]), logits: .last)!
            mirror = prompt
            return (logits, keep)
        }
        // The longest reusable state: the live one if the prompt extends it, else a snapshot that is a prefix.
        var reused = 0
        var stored: Logits?
        let ahead = await engine.stateRanAhead
        let live = liveValid && !ahead && !mirror.isEmpty && mirror.count < prompt.count && prompt.starts(with: mirror)
        let best = snapshots.filter { snap in
            snap.tokens.count <= prompt.count && prompt.starts(with: snap.tokens) && (snap.tokens.count < prompt.count || snap.logits != nil)
        }.max { $0.tokens.count < $1.tokens.count }
        // Only the snapshot reused here outlives the replacement (so a failed prefill can still fall back on it): the
        // engine's memory budget reserves `maxSnapshots`, the two retained ones plus this one.
        snapshots = best.map { [$0] } ?? []
        if live, mirror.count >= best?.tokens.count ?? 0 {
            reused = mirror.count
        } else if let best {
            try await engine.restore(best.state)
            reused = best.tokens.count
            stored = best.logits
        } else {
            try await engine.reset()
        }
        var fresh: [(tokens: [Int32], state: EngineSnapshot, logits: Logits?)] = []
        let logits: Logits
        if reused == prompt.count, let stored {
            logits = stored
        } else {
            if let boundary, boundary > reused, boundary < prompt.count {
                _ = try await engine.prefill(Array(prompt[reused..<boundary]), logits: .none)
                fresh.append((Array(prompt[..<boundary]), try await engine.snapshot(), nil))
                logits = try await engine.prefill(Array(prompt[boundary...]), logits: .last)!
            } else {
                logits = try await engine.prefill(Array(prompt[reused...]), logits: .last)!
            }
        }
        mirror = prompt
        // DESIGN §6.2: a snapshot at the end of each prompt replaces partial rewind on hybrids.
        fresh.append((prompt, try await engine.snapshot(), logits))
        snapshots = fresh
        return (logits, reused)
    }

    static func commonPrefix(_ a: [Int32], _ b: [Int32]) -> Int {
        var n = 0
        while n < a.count, n < b.count, a[n] == b[n] { n += 1 }
        return n
    }

    // MARK: - Constraints

    /// A JSON-schema constraint for this model (structured output). With `allowThinking`, the reply may open a
    /// thinking block first and is constrained from its close tag on; a prompt that already opened one (a template
    /// that can't switch thinking off) is handled either way. Needs a byte-level tokenizer (``TokenBytes``).
    public func jsonConstraint(_ schema: OutputSchema, allowThinking: Bool = false) throws -> JSONSchemaConstraint {
        guard let bytes = decoder.tokenBytes else {
            throw AetherError.unsupportedOperation("structured output needs a byte-level tokenizer (no token byte table)")
        }
        let tag = profile.thinkingTag
        let open = tag.flatMap { decoder.tokenID($0.open) }, close = tag.flatMap { decoder.tokenID($0.close) }
        if jsonVocabulary == nil {
            jsonVocabulary = JSONVocabulary(bytes: bytes, headRows: engine.vocab, excluded: Set([open, close].compactMap { $0 }.map(Int.init)))
        }
        let reasoning = open.flatMap { open in close.map { JSONSchemaConstraint.Reasoning(open: open, close: $0) } }
        return try JSONSchemaConstraint(schema: schema, vocabulary: jsonVocabulary!, stopTokenIDs: stopTokens, reasoning: reasoning,
                                        allowThinking: allowThinking)
    }

    /// Whether structured output can run here (a byte-level tokenizer).
    public nonisolated var supportsStructuredOutput: Bool { decoder.tokenBytes != nil }

    /// The first constraint, from every processor (run beside the prefill).
    nonisolated static func begin(_ processors: [any LogitsProcessor], _ prompt: [Int32], base: TokenConstraint) async throws -> TokenConstraint {
        var constraint = base
        for processor in processors { constraint = constraint.intersection(try await processor.begin(prompt: prompt)) }
        return constraint
    }

    /// The constraint after `token`, from every processor (run beside the step that feeds `token`).
    nonisolated static func accept(_ processors: [any LogitsProcessor], _ token: Int32, base: TokenConstraint) async throws -> TokenConstraint {
        var constraint = base
        for processor in processors { constraint = constraint.intersection(try await processor.accept(token)) }
        return constraint
    }

    // MARK: - The loop

    private func run(
        _ rendered: RenderedPrompt, _ parameters: GenerationParameters,
        _ continuation: AsyncThrowingStream<GenerationEvent, any Error>.Continuation, boundary: Int? = nil
    ) async throws {
        let prompt = rendered.tokens
        guard !prompt.isEmpty else { throw AetherError.unsupportedOperation("an empty prompt") }
        try parameters.sampling.validate()
        let clock = ContinuousClock()
        let t0 = clock.now
        let position = await engine.position
        liveValid = liveValid && mirror.count == position
        let processors = parameters.logitsProcessors
        let pipelined = parameters.sampling.isPlainGreedy && processors.isEmpty
        // Random sampling never picks the padding rows of a head wider than the tokenizer (Qwen3: 151,936 rows,
        // 151,669 tokens); greedy never does in practice.
        let vocab = engine.vocab
        var base = TokenConstraint.unconstrained
        if !parameters.sampling.isGreedy, let count = decoder.tokenBytes?.count, count < vocab {
            base = .allowed(TokenSet(range: 0..<count, vocabularySize: vocab))
        }
        let (logits, reused): (Logits, Int)
        var constraint = base
        do {
            async let first = Self.begin(processors, prompt, base: base)
            (logits, reused) = try await prepare(prompt, boundary: boundary)
            constraint = try await first
        } catch {
            liveValid = false
            mirror = []
            throw error
        }
        let t1 = clock.now

        let tags = Set(profile.tags.flatMap { [$0.open, $0.close] })
        var state = LoopState(
            text: TokenText(decoder: decoder, keepSpecial: tags),
            parser: TagParser(profile: profile, startsInThinking: rendered.startsInThinking,
                              valueOnly: processors.contains { $0.isStructuredOutput }),
            stopper: StopMatcher(parameters.stopSequences), stopTokens: stopTokens, maxTokens: max(1, parameters.maxTokens),
            continuation: continuation)
        var sampler = Sampler(parameters.sampling, prompt: prompt)
        func choose(_ row: ArraySlice<Float>, _ constraint: TokenConstraint) throws -> Int32 {
            if pipelined { return try Logits.argmax(row) }
            var values = Array(row)
            return try sampler.sample(&values, constraint: constraint)
        }

        var token: Int32
        var stop: Bool
        var mode = "host"
        do {
            token = try choose(logits.last, constraint)
            stop = state.accept(token)
            if pipelined {
                mode = await engine.decodeModeDescription
                if !stop {
                    try await engine.beginGreedy(first: token, limit: state.maxTokens - 1)
                    do {
                        while !stop, !Task.isCancelled, let next = try await engine.nextGreedy() { stop = state.accept(next) }
                    } catch {
                        _ = try? await engine.endGreedy()
                        throw error
                    }
                    try await engine.endGreedy()
                }
            } else {
                mode = processors.isEmpty ? "host logits + CPU sampler (depth 1)" : "host logits + CPU sampler + constraint (depth 1)"
                let capacity = engine.maxCapacity
                while !stop, !Task.isCancelled {
                    guard await engine.position < capacity else { break }
                    // The constraint on the next token is computed while the step that feeds this one runs.
                    let fed = token
                    async let next = Self.accept(processors, fed, base: base)
                    let row = try await engine.step(fed)
                    constraint = try await next
                    token = try choose(row.last, constraint)
                    stop = state.accept(token)
                }
            }
        } catch {
            liveValid = false
            mirror = prompt + state.chosen.dropLast()
            throw error
        }
        let t2 = clock.now
        // Fed: the prompt and every chosen token but the last.
        mirror = prompt + state.chosen.dropLast()
        let ranAhead = await engine.stateRanAhead
        let fedPosition = await engine.position
        liveValid = !ranAhead && fedPosition == mirror.count
        if Task.isCancelled { throw CancellationError() }
        state.finish()
        continuation.yield(.finished(GenerationSummary(
            promptTokens: prompt.count, reusedTokens: reused, tokens: state.chosen, thinkingTokens: state.thinkingTokens,
            stopReason: state.reason ?? .capacity, prefillSeconds: (t1 - t0) / .seconds(1),
            decodeSeconds: (t2 - t1) / .seconds(1), decodeMode: mode)))
    }

    /// Per-request output state: stop checks, token text, tag parser, stop sequences on the response text.
    struct LoopState {
        var text: TokenText
        var parser: TagParser
        var stopper: StopMatcher
        let stopTokens: Set<Int32>
        let maxTokens: Int
        let continuation: AsyncThrowingStream<GenerationEvent, any Error>.Continuation
        var chosen: [Int32] = []
        var thinkingTokens = 0
        var reason: GenerationSummary.StopReason?

        /// Takes one chosen token; true when generation must stop.
        mutating func accept(_ token: Int32) -> Bool {
            chosen.append(token)
            if stopTokens.contains(token) {
                reason = .stopToken(token)
                return true
            }
            let wasThinking = parser.inThinking
            let segments = parser.feed(text.append(token))
            if wasThinking || parser.inThinking { thinkingTokens += 1 }
            if let hit = route(segments) {
                reason = .stopSequence(hit)
                return true
            }
            if chosen.count >= maxTokens {
                reason = .maxTokens
                return true
            }
            return false
        }

        mutating func finish() {
            // After a stop sequence nothing more is emitted.
            if case .stopSequence? = reason { return }
            // The held last character can still complete a stop sequence.
            if let hit = route(parser.feed(text.finish())) ?? route(parser.finish()) {
                reason = .stopSequence(hit)
                return
            }
            let rest = stopper.finish()
            if !rest.isEmpty { continuation.yield(.text(rest)) }
        }

        /// Emits segments in order; returns the stop sequence if one completed in the response text.
        private mutating func route(_ segments: [TagParser.Segment]) -> String? {
            for segment in segments {
                switch segment {
                case .text(let s):
                    let (ready, hit) = stopper.feed(s)
                    if !ready.isEmpty { continuation.yield(.text(ready)) }
                    if let hit { return hit }
                case .thinking(let s):
                    flushText()
                    continuation.yield(.thinking(s))
                case .toolCall(let s):
                    flushText()
                    continuation.yield(.toolCall(s))
                }
            }
            return nil
        }

        /// Text can't continue across a tag: held text that could start a stop sequence is released.
        private mutating func flushText() {
            let held = stopper.finish()
            if !held.isEmpty { continuation.yield(.text(held)) }
        }
    }
}
