import AetherLanguageRuntime
import Foundation
import os

/// A conversation with an ``AetherLanguageModel``, in FoundationModels' shape: instructions, then `respond(to:)` or
/// `streamResponse(to:)` per turn, with the transcript kept for the next. Each request renders the whole transcript
/// with the model's own template.
///
/// One request at a time: overlap throws ``AetherLanguageError/busy``. A turn joins the transcript only when it
/// completes; a failed or cancelled turn leaves it as it was. Cancelling the task that called `respond` (or abandoning a
/// stream) stops at the next decode step, after the native work already submitted has finished.
public actor AetherLanguageSession {
    public struct Response<Content: Sendable>: Sendable {
        public let content: Content
        /// The model's thinking, when the request turned reasoning on; nil otherwise.
        public let reasoning: String?
        public let stopReason: AetherStopReason
        /// The seed a random sampling mode used (the request's, or one drawn and reported here); nil when greedy.
        public let seed: UInt64?
        public let usage: Usage
        /// The entries this turn added to the transcript.
        public let transcriptEntries: [AetherTranscript.Entry]
    }

    public struct Usage: Sendable, Equatable {
        public let promptTokens: Int
        public let outputTokens: Int
    }

    public nonisolated let model: AetherLanguageModel
    public private(set) var transcript: AetherTranscript
    public private(set) var isResponding = false

    public init(model: AetherLanguageModel, instructions: String? = nil) {
        self.model = model
        transcript = AetherTranscript(entries: instructions.map { [.instructions($0)] } ?? [])
    }

    public init(model: AetherLanguageModel, transcript: AetherTranscript) {
        self.model = model
        self.transcript = transcript
    }

    /// Verifies and loads the model now rather than on the first request.
    func prewarm() async throws { try await model.prewarm() }

    public func respond(to prompt: String, options: AetherGenerationOptions = .init()) async throws -> Response<String> {
        try await run(prompt, options: options) { _ in }
    }

    /// A reply decoded into `type`: generation is constrained to JSON matching `Content.generationSchema`, so it
    /// decodes unless it reaches the output limit first (``AetherLanguageError/decodingFailure(_:)``). A schema the
    /// constraint doesn't support is refused (``AetherLanguageError/unsupportedSchema(_:)``). With
    /// `includeSchemaInPrompt` (the default, as in FoundationModels), the schema is also described to the model in the
    /// system message. The transcript keeps the reply's JSON.
    public func respond<Content: AetherGenerable>(to prompt: String, generating type: Content.Type = Content.self,
                                                  includeSchemaInPrompt: Bool = true,
                                                  options: AetherGenerationOptions = .init()) async throws -> Response<Content> {
        let structured = LanguageStructuredOutput(schema: Content.generationSchema, includeSchemaInPrompt: includeSchemaInPrompt)
        let (reply, commit) = try await generate(prompt, options: options, structured: structured) { _ in }
        let content: Content
        do { content = try JSONDecoder().decode(Content.self, from: Data(reply.content.utf8)) }
        catch { throw AetherLanguageError.decodingFailure("\(reply.stopReason == .outputLimit ? "the reply reached the output limit; " : "")\(error)") }
        return Response(content: content, reasoning: options.reasoning == .on ? reply.reasoning : nil, stopReason: reply.stopReason,
                        seed: reply.seed, usage: Usage(promptTokens: reply.promptTokens, outputTokens: reply.outputTokens),
                        transcriptEntries: commit())
    }

    /// The reply as it's generated: each snapshot carries the cumulative text, which only ever grows by whole
    /// characters, and what's new since the snapshot this iteration last saw. Snapshots coalesce when the consumer is
    /// slower than the model, without losing text. Iterate once; stopping early cancels the request.
    public nonisolated func streamResponse(to prompt: String, options: AetherGenerationOptions = .init()) -> ResponseStream {
        ResponseStream { continuation in
            Task {
                do {
                    let response = try await self.run(prompt, options: options) { snapshot in
                        continuation.yield(.snapshot(content: snapshot.content, reasoning: snapshot.reasoning))
                    }
                    continuation.yield(.finished(response))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func run(_ prompt: String, options: AetherGenerationOptions,
                     onText: @escaping @Sendable (_ cumulative: (content: String, reasoning: String)) -> Void) async throws -> Response<String> {
        let (reply, commit) = try await generate(prompt, options: options, structured: nil, onText: onText)
        return Response(content: reply.content, reasoning: options.reasoning == .on ? reply.reasoning : nil, stopReason: reply.stopReason,
                        seed: reply.seed, usage: Usage(promptTokens: reply.promptTokens, outputTokens: reply.outputTokens),
                        transcriptEntries: commit())
    }

    /// One turn: the reply, and a commit that adds the turn to the transcript (called once the reply is accepted, so a
    /// reply that fails to decode isn't kept either).
    private func generate(_ prompt: String, options: AetherGenerationOptions, structured: LanguageStructuredOutput?,
                          onText: @escaping @Sendable (_ cumulative: (content: String, reasoning: String)) -> Void)
        async throws -> (LanguageEngineReply, commit: () -> [AetherTranscript.Entry]) {
        guard !isResponding else { throw AetherLanguageError.busy }
        isResponding = true
        defer { isResponding = false }
        let messages = AetherTranscript(entries: transcript.entries + [.prompt(prompt)]).messages
        let cumulative = OSAllocatedUnfairLock(initialState: (content: "", reasoning: ""))
        let reply = try await model.generate(messages: messages, options: options, structured: structured) { delta, _ in
            let text = cumulative.withLock { text in
                text.content += delta.content
                text.reasoning += delta.reasoning
                return text
            }
            onText(text)
        }
        let entries: [AetherTranscript.Entry] = [.prompt(prompt)] + (reply.reasoning.isEmpty ? [] : [.reasoning(reply.reasoning)])
            + [.response(reply.content)]
        return (reply, { self.transcript.entries += entries; return entries })
    }
}

/// The snapshots of a streaming reply (``AetherLanguageSession/streamResponse(to:options:)``).
public struct ResponseStream: AsyncSequence, Sendable {
    public struct Snapshot: Sendable, Equatable {
        /// The reply so far.
        public let content: String
        /// What `content` gained since the snapshot this iteration last delivered.
        public let delta: String
        /// The thinking so far, when the request turned reasoning on.
        public let reasoning: String
        public let reasoningDelta: String
    }

    enum Event: Sendable {
        case snapshot(content: String, reasoning: String)
        case finished(AetherLanguageSession.Response<String>)
    }

    private let start: @Sendable (AsyncThrowingStream<Event, any Error>.Continuation) -> Task<Void, Never>
    private let started = OSAllocatedUnfairLock(initialState: false)

    init(start: @escaping @Sendable (AsyncThrowingStream<Event, any Error>.Continuation) -> Task<Void, Never>) {
        self.start = start
    }

    public func makeAsyncIterator() -> AsyncIterator {
        let first = started.withLock { started in defer { started = true }; return !started }
        // Snapshots are cumulative, so keeping only the newest pending one loses nothing.
        let events = AsyncThrowingStream<Event, any Error>(bufferingPolicy: .bufferingNewest(2)) { continuation in
            guard first else {
                continuation.finish(throwing: AetherLanguageError.invalidRequest("a response stream is iterated once"))
                return
            }
            let task = start(continuation)
            continuation.onTermination = { _ in task.cancel() }
        }
        return AsyncIterator(events: events.makeAsyncIterator())
    }

    /// Waits for the whole reply.
    public func collect() async throws -> AetherLanguageSession.Response<String> {
        var iterator = makeAsyncIterator()
        while try await iterator.next() != nil {}
        guard let response = iterator.response else { throw CancellationError() }
        return response
    }

    public struct AsyncIterator: AsyncIteratorProtocol {
        var events: AsyncThrowingStream<Event, any Error>.AsyncIterator
        private var content = "", reasoning = ""
        /// The finished reply (stop reason, usage, seed and transcript entries) once ``next()`` has returned nil; nil
        /// before then, and after a failed or cancelled request.
        public fileprivate(set) var response: AetherLanguageSession.Response<String>?

        init(events: AsyncThrowingStream<Event, any Error>.AsyncIterator) { self.events = events }

        public mutating func next() async throws -> Snapshot? {
            while let event = try await events.next() {
                switch event {
                case .snapshot(let newContent, let newReasoning):
                    if let snapshot = advance(to: newContent, newReasoning) { return snapshot }
                case .finished(let finished):
                    response = finished
                    // The final text can include what the last coalesced snapshots carried.
                    if let snapshot = advance(to: finished.content, finished.reasoning ?? reasoning) { return snapshot }
                }
            }
            return nil
        }

        private mutating func advance(to newContent: String, _ newReasoning: String) -> Snapshot? {
            guard newContent != content || newReasoning != reasoning else { return nil }
            // Emitted text only grows, so what's new is the scalars past the last delivered snapshot.
            let delta = String(newContent.unicodeScalars.dropFirst(content.unicodeScalars.count))
            let reasoningDelta = String(newReasoning.unicodeScalars.dropFirst(reasoning.unicodeScalars.count))
            content = newContent
            reasoning = newReasoning
            return Snapshot(content: newContent, delta: delta, reasoning: newReasoning, reasoningDelta: reasoningDelta)
        }
    }
}
