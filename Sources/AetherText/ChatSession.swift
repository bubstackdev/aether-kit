public import AetherHost
public import AetherRuntime
import Foundation

/// A conversation (DESIGN.md §7): history, configuration and a ``Generator``, and nothing more. Each reply is
/// rendered from the whole history; the Generator's prefix reuse makes a turn cost only its new tokens.
public actor ChatSession {
    public struct Configuration: Sendable {
        public var systemPrompt: String?
        public var parameters: GenerationParameters

        public init(systemPrompt: String? = nil, parameters: GenerationParameters = GenerationParameters()) {
            self.systemPrompt = systemPrompt
            self.parameters = parameters
        }
    }

    public nonisolated let generator: Generator
    public var configuration: Configuration
    /// User and assistant turns (the system prompt is added at render time).
    public private(set) var history: [ChatMessage] = []
    /// Bumped by ``clearHistory()``: a turn joins the history only if the history it was rendered from is still
    /// current.
    private var historyGeneration = 0

    public init(generator: Generator, configuration: Configuration = Configuration()) {
        self.generator = generator
        self.configuration = configuration
    }

    /// Loads a chat model from an installed bundle: the `decoder-lm@1` engine, the tokenizer, the manifest's renderer
    /// and output profile. Replies take `gpu` (by default ``GPUAdmission/shared``); a reply's priority is
    /// `configuration.parameters.job`.
    public static func load(
        bundle: InstalledBundle, manifest: Manifest, device: Device, engine options: DecoderEngine.Options = .init(),
        configuration: Configuration = Configuration(), gpu: GPUAdmissionOwner = GPUAdmission.shared.owner("text")
    ) async throws -> ChatSession {
        let generator = try await generator(bundle: bundle, manifest: manifest, device: device, engine: options, gpu: gpu)
        return ChatSession(generator: generator, configuration: configuration)
    }

    /// Generates loaded from an installed bundle, as ``load(bundle:manifest:device:engine:configuration:gpu:)``.
    public static func generator(
        bundle: InstalledBundle, manifest: Manifest, device: Device, engine options: DecoderEngine.Options = .init(),
        gpu: GPUAdmissionOwner = GPUAdmission.shared.owner("text")
    ) async throws -> Generator {
        let engine = try await DecoderEngine(bundle: bundle, manifest: manifest, device: device, options: options)
        return try await generator(engine: engine, bundle: bundle, manifest: manifest, gpu: gpu)
    }

    /// A Generator over an engine already loaded (a ``DecoderEngine`` the caller keeps for its diagnostics, or another
    /// backend's `LanguageEngine`), with the tokenizer, renderer and output profile of the bundle it was loaded from.
    public static func generator(
        engine: any LanguageEngine, bundle: InstalledBundle, manifest: Manifest,
        gpu: GPUAdmissionOwner = GPUAdmission.shared.owner("text")
    ) async throws -> Generator {
        let tokenizer = try await TokenizerAdapter(bundle: bundle, manifest: manifest)
        let renderer = try PromptRenderers.make(manifest: manifest, bundle: bundle, tokenizer: tokenizer)
        return Generator(
            engine: engine, decoder: tokenizer, renderer: renderer, stopTokens: tokenizer.stopTokenIDs,
            profile: OutputProfile(manifest: manifest.tokenizer), gpu: gpu)
    }

    public func setConfiguration(_ configuration: Configuration) { self.configuration = configuration }

    /// Empties the history. A reply already in progress is not cancelled: it streams to its end, but its turn was
    /// rendered from the old history and is not added to the new one. Turns that start after the clear (one sent
    /// earlier and still waiting for its turn included) render from the cleared history and are kept. To stop the
    /// reply too, cancel the task consuming it.
    public func clearHistory() {
        history.removeAll()
        historyGeneration += 1
    }

    /// Sends a user turn and streams the reply. Turns run one at a time in the order sent: each is rendered only
    /// when the previous one has finished, so its prompt holds the previous reply. The reply (its response text, not
    /// its thinking) joins the history only after the reply finished; an error, a cancellation or a
    /// ``clearHistory()`` while it runs drops the turn.
    public func respond(to text: String) -> AsyncThrowingStream<GenerationEvent, any Error> {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: GenerationEvent.self)
        let task = Task { await self.turn(.user(text), parameters: nil, check: nil, continuation) }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    /// Sends a user turn and returns the reply as a `T`, decoded from JSON that the JSON-schema constraint holds to
    /// `T.outputSchema` (structured output). See ``respond(to:generating:schema:includeSchemaInPrompt:)``.
    public func respond<T: StructuredOutput>(to text: String, generating type: T.Type, includeSchemaInPrompt: Bool = true) async throws -> T {
        try await respond(to: text, generating: type, schema: T.outputSchema, includeSchemaInPrompt: includeSchemaInPrompt)
    }

    /// Sends a user turn and returns the reply decoded as a `T` from JSON constrained to `schema`.
    /// - The model writes one canonical layout (`{"key": value, "key2": value}`), every property in `x-order`; an
    ///   unsupported schema (a `pattern`, say) is refused before generating (`unsupportedSchema`).
    /// - With `includeSchemaInPrompt` (the default, as in FoundationModels) the schema is described at the end of
    ///   this user turn (not in the system message, so alternating structured and plain turns keep prefix reuse); the
    ///   history keeps the turn as rendered.
    /// - Thinking: when the configuration asks for it (`thinking: true`), the reply may open a thinking block first and
    ///   is constrained from its close tag on; otherwise thinking is rendered off and the reply is constrained from its
    ///   first token. A template that opens a block regardless (it can't switch thinking off) is followed: the value is
    ///   constrained after the block.
    /// - A reply that doesn't decode (it reached `maxTokens` first) throws `decodingFailed` and is not kept in the
    ///   history; the history keeps a structured reply's JSON.
    public func respond<T: Decodable & SendableMetatype>(
        to text: String, generating type: T.Type, schema: OutputSchema, includeSchemaInPrompt: Bool = true
    ) async throws -> T {
        let thinking = configuration.parameters.thinking == true
        var parameters = configuration.parameters
        let constraint = try await generator.jsonConstraint(schema, allowThinking: thinking)
        parameters.logitsProcessors.append(constraint)
        parameters.thinking = thinking
        let message = ChatMessage.user(includeSchemaInPrompt ? Self.withSchemaInstruction(text, schema) : text)
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: GenerationEvent.self)
        let task = Task {
            await self.turn(message, parameters: parameters, check: { reply, summary in
                _ = try Self.decode(T.self, reply, summary, constraint: constraint)
            }, continuation)
        }
        var reply = ""
        var summary: GenerationSummary?
        try await withTaskCancellationHandler {
            for try await event in stream {
                if case .text(let delta) = event { reply += delta }
                if case .finished(let s) = event { summary = s }
            }
        } onCancel: {
            task.cancel()
        }
        return try Self.decode(T.self, reply, summary, constraint: constraint)
    }

    /// The text that describes a schema (0.0.x's wording).
    public static func schemaInstruction(_ schema: OutputSchema) -> String {
        "Respond with only a JSON object that matches this JSON schema:\n\(schema.json)"
    }

    /// A user turn's text with the schema described after it.
    public static func withSchemaInstruction(_ text: String, _ schema: OutputSchema) -> String {
        text + "\n\n" + schemaInstruction(schema)
    }

    /// Decodes first: a value that completes exactly at `maxTokens` is a value. Only a reply that doesn't decode is
    /// blamed on the limit it reached. With the schema's constraint, the reply must also be a complete value of the
    /// schema: a prefix cut at the limit can still decode (`100` cut to `1`) while breaking the schema's bounds.
    static func decode<T: Decodable>(
        _ type: T.Type, _ reply: String, _ summary: GenerationSummary?, constraint: JSONSchemaConstraint? = nil
    ) throws -> T {
        if let constraint, !constraint.completes(reply) {
            switch summary?.stopReason {
            case .maxTokens?: throw AetherError.generation(.decodingFailed("the reply reached maxTokens before its value ended"))
            case .capacity?: throw AetherError.generation(.decodingFailed("the reply reached the context capacity before its value ended"))
            default: throw AetherError.generation(.decodingFailed("the reply is not a complete value of the schema"))
            }
        }
        do {
            return try JSONDecoder().decode(T.self, from: Data(reply.utf8))
        } catch {
            switch summary?.stopReason {
            case .maxTokens?: throw AetherError.generation(.decodingFailed("the reply reached maxTokens before its value ended"))
            case .capacity?: throw AetherError.generation(.decodingFailed("the reply reached the context capacity before its value ended"))
            default: throw AetherError.generation(.decodingFailed("\(error)"))
            }
        }
    }

    /// One turn at a time, in the order sent.
    private let gate = SerialGate()

    /// How many turns wait for the running one (tests observe a turn reaching the queue).
    nonisolated var waitingTurns: Int { gate.waiting }

    private func turn(
        _ message: ChatMessage, parameters: GenerationParameters?,
        check: (@Sendable (String, GenerationSummary) throws -> Void)?,
        _ continuation: AsyncThrowingStream<GenerationEvent, any Error>.Continuation
    ) async {
        // A turn cancelled while it waits leaves the queue at once, and drops.
        do {
            try await gate.withTurn {
                let messages = (configuration.systemPrompt.map { [ChatMessage.system($0)] } ?? []) + history + [message]
                let generation = historyGeneration
                var reply = ""
                var summary: GenerationSummary?
                do {
                    // Cancellation ends this loop normally (the upstream iterator returns nil), so completion is the
                    // `.finished` event, not the loop's end.
                    // A consumer that stops listening cancels the turn, as `onTermination` does, without waiting for that
                    // cancellation to arrive: a reply nobody reads isn't finished, nor kept in the history.
                    var abandoned = false
                    for try await event in generator.generate(messages, parameters: parameters ?? configuration.parameters) {
                        if case .text(let delta) = event { reply += delta }
                        if case .finished(let s) = event { summary = s }
                        if case .terminated = continuation.yield(event) {
                            abandoned = true
                            break
                        }
                    }
                    if let summary, !abandoned, !Task.isCancelled {
                        try check?(reply, summary)
                        // A clear while the reply ran: its turn belongs to the history that was cleared.
                        if generation == historyGeneration {
                            history.append(message)
                            history.append(.assistant(reply))
                        }
                        continuation.finish()
                    } else {
                        continuation.finish(throwing: CancellationError())
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        } catch {
            continuation.finish(throwing: error)
        }
    }
}

extension Registry {
    /// The drivers AetherText links: `decoder-lm@1`.
    public static let text = Registry([DecoderLMDriver.self])
}
