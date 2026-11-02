import AetherLanguage
import Foundation
import Observation

/// One streamed turn, as the chat screen consumes it.
public enum ChatUpdate: Sendable, Equatable {
    /// The reply so far.
    case text(String)
    /// The reply ended: why, and its token counts.
    case finished(stopReason: String, promptTokens: Int, outputTokens: Int)
}

/// Produces streamed replies for one conversation. ``SessionResponder`` is the facade's; tests use a scripted one.
public protocol ChatResponder: Sendable {
    func reply(to prompt: String) -> AsyncThrowingStream<ChatUpdate, any Error>
}

/// An `AetherLanguageSession` as a ``ChatResponder``: it streams snapshots, then reads the finished response's stop
/// reason and usage from the iterator.
public struct SessionResponder: ChatResponder {
    public let session: AetherLanguageSession
    public let options: AetherGenerationOptions

    public init(session: AetherLanguageSession, options: AetherGenerationOptions) {
        self.session = session
        self.options = options
    }

    public func reply(to prompt: String) -> AsyncThrowingStream<ChatUpdate, any Error> {
        let stream = session.streamResponse(to: prompt, options: options)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var iterator = stream.makeAsyncIterator()
                    while let snapshot = try await iterator.next() { continuation.yield(.text(snapshot.content)) }
                    if let response = iterator.response {
                        continuation.yield(.finished(stopReason: SessionResponder.describe(response.stopReason),
                                                     promptTokens: response.usage.promptTokens,
                                                     outputTokens: response.usage.outputTokens))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public static func describe(_ reason: AetherStopReason) -> String {
        switch reason {
        case .endOfTurn: "end of turn"
        case .outputLimit: "output limit"
        case .stopSequence: "stop sequence"
        @unknown default: "\(reason)"
        }
    }
}

public struct ChatMessage: Identifiable, Equatable, Sendable {
    public enum Role: Sendable { case user, assistant }
    public let id: UUID
    public let role: Role
    public var text: String
    public var isStreaming: Bool
    /// For a finished reply: why it stopped and how many tokens it took; or why it failed.
    public var footnote: String?

    public init(id: UUID = UUID(), role: Role, text: String, isStreaming: Bool = false, footnote: String? = nil) {
        self.id = id; self.role = role; self.text = text; self.isStreaming = isStreaming; self.footnote = footnote
    }
}

/// The chat screen: one conversation, streamed. Stop cancels the running reply; a new chat starts a new conversation.
@MainActor
@Observable
public final class ChatController {
    public static let instructions = "You are a helpful assistant in a sample app. Keep answers short."
    public static let options = AetherGenerationOptions(maximumResponseTokens: 512, reasoning: .off)

    public private(set) var messages: [ChatMessage] = []
    public private(set) var isResponding = false
    /// Whether a model is attached; the screen is disabled without one.
    public private(set) var isAvailable = false

    @ObservationIgnored private var makeResponder: (@MainActor () -> any ChatResponder)?
    @ObservationIgnored private var responder: (any ChatResponder)?
    @ObservationIgnored private var task: Task<Void, Never>?
    /// Attaches and new chats in progress; `send` refuses while any runs, so nothing starts on a responder being
    /// replaced.
    @ObservationIgnored private var switching = 0
    /// The latest attach; an earlier one that finishes waiting later doesn't overwrite it.
    @ObservationIgnored private var generation = 0
    /// Tests only: runs once an attach or new chat has drained the old reply, before it takes over.
    @ObservationIgnored var drained: (@MainActor () async -> Void)?

    public init() {}

    /// Uses `makeResponder` for each new conversation, after stopping the current reply. nil detaches.
    public func attach(_ makeResponder: (@MainActor () -> any ChatResponder)?) async {
        generation += 1
        let mine = generation
        switching += 1
        defer { switching -= 1 }
        isAvailable = false
        await stopAndWait()
        guard mine == generation else { return }
        self.makeResponder = makeResponder
        responder = nil
        messages = []
        isAvailable = makeResponder != nil
    }

    /// Attaches a fresh `AetherLanguageSession` per conversation over `model`; nil detaches (and releases the model).
    public func attach(model: AetherLanguageModel?) async {
        guard let model else { return await attach(nil) }
        await attach { SessionResponder(session: AetherLanguageSession(model: model, instructions: Self.instructions), options: Self.options) }
    }

    /// Starts a reply. Returns false, leaving the conversation unchanged, when the prompt is empty, a reply is running,
    /// no model is attached, or the conversation is being replaced.
    @discardableResult
    public func send(_ prompt: String) -> Bool {
        let prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !isResponding, switching == 0, let makeResponder else { return false }
        let responder = responder ?? makeResponder()
        self.responder = responder
        messages.append(ChatMessage(role: .user, text: prompt))
        let reply = ChatMessage(role: .assistant, text: "", isStreaming: true)
        messages.append(reply)
        isResponding = true
        task = Task { [weak self] in
            var footnote: String?
            do {
                for try await update in responder.reply(to: prompt) {
                    switch update {
                    case let .text(text): self?.update(reply.id) { $0.text = text }
                    case let .finished(reason, prompt, output): footnote = "\(reason) · \(prompt) prompt + \(output) reply tokens"
                    }
                }
                if Task.isCancelled { footnote = "stopped" }
            } catch is CancellationError {
                footnote = "stopped"
            } catch {
                footnote = "failed: \(error)"
            }
            self?.update(reply.id) { $0.isStreaming = false; $0.footnote = footnote }
            self?.isResponding = false
        }
        return true
    }

    /// Stops the running reply. Text so far stays on screen; the session's transcript keeps only completed turns.
    public func stop() { task?.cancel() }

    /// Waits for the running reply, if any, to end.
    public func waitForReply() async { await task?.value }

    public func newChat() async {
        let mine = generation
        switching += 1
        defer { switching -= 1 }
        await stopAndWait()
        guard mine == generation else { return }
        responder = nil
        messages = []
    }

    private func stopAndWait() async {
        task?.cancel()
        await task?.value
        task = nil
        await drained?()
    }

    private func update(_ id: UUID, _ change: (inout ChatMessage) -> Void) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        change(&messages[index])
    }
}
