public import AetherRuntime
import Foundation

public struct GenerationParameters: Sendable {
    public var maxTokens: Int
    public var sampling: SamplingParameters
    /// Text-level stops, besides the manifest's stop tokens. The matched text is not emitted.
    public var stopSequences: [String]
    /// Thinking on or off; nil leaves the template's default.
    public var thinking: Bool?
    /// Applied in order to every next-token row on the CPU (M3's constrained decoding plugs in here).
    public var logitsProcessors: [any LogitsProcessor]

    public init(
        maxTokens: Int = 512, sampling: SamplingParameters = .greedy, stopSequences: [String] = [], thinking: Bool? = nil,
        logitsProcessors: [any LogitsProcessor] = []
    ) {
        self.maxTokens = maxTokens
        self.sampling = sampling
        self.stopSequences = stopSequences
        self.thinking = thinking
        self.logitsProcessors = logitsProcessors
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
    public var stopReason: StopReason
    public var prefillSeconds: Double
    public var decodeSeconds: Double
    public var decodeMode: String
}

/// The single decode loop (DESIGN.md §7), shared by ChatSession, the FoundationModels executor and future
/// multimodal front ends. It owns a ``DecoderEngine`` and keeps a token mirror of the engine's state:
/// - attention-only models reuse the longest common prefix by `rewind` (O(1));
/// - models with recurrent state reuse the live state when the new prompt extends it, else the snapshot taken at
///   the end of the previous prompt (with that prompt's last logits), else start over.
///
/// Plain greedy decoding with no logits processor runs on the engine's pipelined GPU path; anything else feeds one
/// token at a time and samples on the CPU. Requests are served one at a time, in arrival order.
public actor Generator {
    public nonisolated let engine: DecoderEngine
    public nonisolated let decoder: any TokenDecoding
    public nonisolated let renderer: any PromptRenderer
    public nonisolated let stopTokens: Set<Int32>
    public nonisolated let profile: OutputProfile

    private var mirror: [Int32] = []
    /// False once the engine's live state no longer matches `mirror` (an error, or recurrent state that ran ahead).
    private var liveValid = true
    /// Hybrids: the snapshots of the last prompt, at the end of its rendered history (the prefix the next turn's
    /// prompt shares) and at its end (with its last logits, for a regenerate).
    private var snapshots: [(tokens: [Int32], state: DecoderEngine.Snapshot, logits: Logits?)] = []
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init(
        engine: DecoderEngine, decoder: any TokenDecoding, renderer: any PromptRenderer, stopTokens: Set<Int32>,
        profile: OutputProfile
    ) {
        self.engine = engine
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
        stream { generator, continuation in
            let rendered = try generator.renderer.render(messages, options: RenderOptions(thinking: parameters.thinking))
            // The history alone (no generation prompt): where the next turn's prompt will diverge from this one.
            let history = try generator.renderer.render(
                messages, options: RenderOptions(thinking: parameters.thinking, addGenerationPrompt: false)).tokens
            let boundary = rendered.tokens.starts(with: history) && history.count < rendered.tokens.count ? history.count : nil
            try await generator.run(rendered, parameters, continuation, boundary: boundary)
        }
    }

    /// Generates from prompt tokens already rendered.
    public nonisolated func generate(prompt: RenderedPrompt, parameters: GenerationParameters = GenerationParameters())
        -> AsyncThrowingStream<GenerationEvent, any Error>
    {
        stream { generator, continuation in try await generator.run(prompt, parameters, continuation) }
    }

    private nonisolated func stream(
        _ body: @escaping @Sendable (isolated Generator, AsyncThrowingStream<GenerationEvent, any Error>.Continuation) async throws -> Void
    ) -> AsyncThrowingStream<GenerationEvent, any Error> {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: GenerationEvent.self)
        let task = Task {
            await self.acquire()
            do {
                try await body(self, continuation)
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
            await self.release()
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    private func acquire() async {
        if !busy {
            busy = true
            return
        }
        await withCheckedContinuation { waiters.append($0) }
    }

    private func release() {
        if waiters.isEmpty { busy = false } else { waiters.removeFirst().resume() }
    }

    // MARK: - Prefix reuse

    /// Brings the engine to `prompt`'s end, reusing what it can. Returns the prompt's last logits and the reused count.
    private func prepare(_ prompt: [Int32], boundary: Int?) async throws -> (Logits, Int) {
        let recurrent = engine.description.hasRecurrentState
        if !recurrent {
            let lcp = liveValid ? Self.commonPrefix(mirror, prompt) : 0
            let keep = min(lcp, prompt.count - 1)
            try await engine.rewind(to: keep)
            mirror = Array(prompt.prefix(keep))
            let logits = try await engine.prefill(Array(prompt[keep...]))!
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
        if live, mirror.count >= best?.tokens.count ?? 0 {
            reused = mirror.count
        } else if let best {
            try await engine.restore(best.state)
            reused = best.tokens.count
            stored = best.logits
        } else {
            try await engine.reset()
        }
        var fresh: [(tokens: [Int32], state: DecoderEngine.Snapshot, logits: Logits?)] = []
        let logits: Logits
        if reused == prompt.count, let stored {
            logits = stored
        } else {
            if let boundary, boundary > reused, boundary < prompt.count {
                _ = try await engine.prefill(Array(prompt[reused..<boundary]), logits: .none)
                fresh.append((Array(prompt[..<boundary]), try await engine.snapshot(), nil))
                logits = try await engine.prefill(Array(prompt[boundary...]))!
            } else {
                logits = try await engine.prefill(Array(prompt[reused...]))!
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

    // MARK: - The loop

    private func run(
        _ rendered: RenderedPrompt, _ parameters: GenerationParameters,
        _ continuation: AsyncThrowingStream<GenerationEvent, any Error>.Continuation, boundary: Int? = nil
    ) async throws {
        let prompt = rendered.tokens
        guard !prompt.isEmpty else { throw AetherError.unsupportedOperation("an empty prompt") }
        let clock = ContinuousClock()
        let t0 = clock.now
        let position = await engine.position
        liveValid = liveValid && mirror.count == position
        let (logits, reused): (Logits, Int)
        do {
            (logits, reused) = try await prepare(prompt, boundary: boundary)
        } catch {
            liveValid = false
            mirror = []
            throw error
        }
        let t1 = clock.now

        var state = LoopState(
            decoder: decoder, parser: TagParser(profile: profile, startsInThinking: rendered.startsInThinking),
            stopper: StopMatcher(parameters.stopSequences), stopTokens: stopTokens, maxTokens: max(1, parameters.maxTokens),
            continuation: continuation)
        var sampler = Sampler(parameters.sampling)
        let processors = parameters.logitsProcessors
        func choose(_ row: ArraySlice<Float>) -> Int32 {
            if parameters.sampling.isPlainGreedy && processors.isEmpty { return Logits.argmax(row) }
            var values = Array(row)
            Sampler.applyRepetitionPenalty(parameters.sampling.repetitionPenalty, seen: prompt + state.chosen, to: &values)
            let context = LogitsContext(prompt: prompt, generated: state.chosen)
            for processor in processors { processor.process(&values, context: context) }
            return sampler.sample(values)
        }

        var token = choose(logits.last)
        var stop = state.accept(token)
        var mode = "host"
        do {
            if parameters.sampling.isPlainGreedy && processors.isEmpty {
                mode = "\(await engine.decodeMode)"
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
                mode = "host logits + CPU sampler (depth 1)"
                let capacity = engine.description.maxCapacity
                while !stop, !Task.isCancelled {
                    guard await engine.position < capacity else { break }
                    token = choose(try await engine.step(token).last)
                    stop = state.accept(token)
                }
            }
        } catch {
            liveValid = false
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
            promptTokens: prompt.count, reusedTokens: reused, tokens: state.chosen,
            stopReason: state.reason ?? .capacity, prefillSeconds: (t1 - t0) / .seconds(1),
            decodeSeconds: (t2 - t1) / .seconds(1), decodeMode: mode)))
    }

    /// Per-request output state: stop checks, detokenizer, stop sequences, tag parser.
    struct LoopState {
        let decoder: any TokenDecoding
        var parser: TagParser
        var stopper: StopMatcher
        let stopTokens: Set<Int32>
        let maxTokens: Int
        let continuation: AsyncThrowingStream<GenerationEvent, any Error>.Continuation
        var detokenizer = IncrementalDetokenizer()
        var chosen: [Int32] = []
        var reason: GenerationSummary.StopReason?

        /// Takes one chosen token; true when generation must stop.
        mutating func accept(_ token: Int32) -> Bool {
            chosen.append(token)
            if stopTokens.contains(token) {
                reason = .stopToken(token)
                return true
            }
            let (text, hit) = stopper.feed(detokenizer.append(token, decoder))
            send(parser.feed(text))
            if let hit {
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
            if case .stopSequence? = reason {
                send(parser.finish())
                return
            }
            let (tail, _) = stopper.feed(detokenizer.finish(decoder))
            send(parser.feed(tail + stopper.finish()))
            send(parser.finish())
        }

        private func send(_ segments: [TagParser.Segment]) {
            for segment in segments {
                switch segment {
                case .text(let s): continuation.yield(.text(s))
                case .thinking(let s): continuation.yield(.thinking(s))
                case .toolCall(let s): continuation.yield(.toolCall(s))
                }
            }
        }
    }
}
