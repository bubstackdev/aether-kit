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

    public init(generator: Generator, configuration: Configuration = Configuration()) {
        self.generator = generator
        self.configuration = configuration
    }

    /// Loads a chat model from an installed bundle: the `decoder-lm@1` engine, the tokenizer, the manifest's renderer
    /// and output profile.
    public static func load(
        bundle: InstalledBundle, manifest: Manifest, device: Device, engine options: DecoderEngine.Options = .init(),
        configuration: Configuration = Configuration()
    ) async throws -> ChatSession {
        let engine = try await DecoderEngine(bundle: bundle, manifest: manifest, device: device, options: options)
        let tokenizer = try await TokenizerAdapter(bundle: bundle, manifest: manifest)
        let renderer = try PromptRenderers.make(manifest: manifest, bundle: bundle, tokenizer: tokenizer)
        let generator = Generator(
            engine: engine, decoder: tokenizer, renderer: renderer, stopTokens: tokenizer.stopTokenIDs,
            profile: OutputProfile(manifest: manifest.tokenizer))
        return ChatSession(generator: generator, configuration: configuration)
    }

    public func setConfiguration(_ configuration: Configuration) { self.configuration = configuration }

    public func clearHistory() { history.removeAll() }

    /// Sends a user turn and streams the reply. Turns run one at a time in the order sent: each is rendered only
    /// when the previous one has finished, so its prompt holds the previous reply. The reply (its response text, not
    /// its thinking) joins the history only after the reply finished; an error or a cancellation drops the turn.
    public func respond(to text: String) -> AsyncThrowingStream<GenerationEvent, any Error> {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: GenerationEvent.self)
        let task = Task { await self.turn(.user(text), continuation) }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    private func turn(_ message: ChatMessage, _ continuation: AsyncThrowingStream<GenerationEvent, any Error>.Continuation) async {
        if busy {
            await withCheckedContinuation { waiters.append($0) }
        } else {
            busy = true
        }
        defer {
            if waiters.isEmpty { busy = false } else { waiters.removeFirst().resume() }
        }
        guard !Task.isCancelled else {
            continuation.finish(throwing: CancellationError())
            return
        }
        let messages = (configuration.systemPrompt.map { [ChatMessage.system($0)] } ?? []) + history + [message]
        var reply = ""
        var finished = false
        do {
            // Cancellation ends this loop normally (the upstream iterator returns nil), so completion is the
            // `.finished` event, not the loop's end.
            for try await event in generator.generate(messages, parameters: configuration.parameters) {
                if case .text(let delta) = event { reply += delta }
                if case .finished = event { finished = true }
                continuation.yield(event)
            }
            if finished, !Task.isCancelled {
                history.append(message)
                history.append(.assistant(reply))
                continuation.finish()
            } else {
                continuation.finish(throwing: CancellationError())
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
