import AetherLanguage
import AetherLanguageMacros
import Foundation
import Observation

/// The structured-output demo's type: a to-do item read from a sentence. Generation is constrained to its schema, so the
/// reply always decodes (docs/language-facade.md).
@AetherGenerable(description: "A to-do item taken from the user's sentence")
public struct TodoItem: Equatable, Encodable, Sendable {
    @AetherGenerable(description: "How urgent the task is")
    public enum Priority: String, Equatable, Encodable, Sendable { case low, medium, high }

    @AetherGuide(description: "The task, as a short imperative phrase")
    public var title: String
    public var priority: Priority
    @AetherGuide(description: "Days from today until it's due, if the sentence says", .range(0...365))
    public var dueInDays: Int?

    public init(title: String, priority: Priority, dueInDays: Int?) {
        self.title = title; self.priority = priority; self.dueInDays = dueInDays
    }
}

/// The structured screen: one sentence in, one ``TodoItem`` out, with the JSON the model produced.
@MainActor
@Observable
public final class TodoController {
    public static let example = "Remind me to renew my passport next week, it's pretty urgent."
    public static let options = AetherGenerationOptions(maximumResponseTokens: 128, reasoning: .off)

    public private(set) var item: TodoItem?
    public private(set) var json: String?
    public private(set) var error: String?
    public private(set) var isWorking = false
    public private(set) var isAvailable = false
    /// Seconds the last extraction took, request to decoded value.
    public private(set) var seconds: Double?

    @ObservationIgnored private var extract: (@Sendable (String) async throws -> TodoItem)?
    @ObservationIgnored private var task: Task<Void, Never>?
    /// Attaches in progress; `run` refuses while any runs.
    @ObservationIgnored private var switching = 0
    @ObservationIgnored private var generation = 0

    public init() {}

    /// Uses `extract` for each request, after the current one ends. nil detaches.
    public func attach(_ extract: (@Sendable (String) async throws -> TodoItem)?) async {
        generation += 1
        let mine = generation
        switching += 1
        defer { switching -= 1 }
        isAvailable = false
        task?.cancel()
        await task?.value
        guard mine == generation else { return }
        self.extract = extract
        isAvailable = extract != nil
        item = nil; json = nil; error = nil; seconds = nil
    }

    /// Each extraction is a fresh single-turn session over `model`; nil detaches (and releases the model).
    public func attach(model: AetherLanguageModel?) async {
        guard let model else { return await attach(nil) }
        await attach { sentence in
            let session = AetherLanguageSession(model: model, instructions: "Extract one to-do item from the user's sentence.")
            return try await session.respond(to: sentence, generating: TodoItem.self, options: Self.options).content
        }
    }

    /// Starts an extraction. Returns false when one is running, no model is attached, or the model is being replaced.
    @discardableResult
    public func run(_ sentence: String) -> Bool {
        guard let extract, !isWorking, switching == 0 else { return false }
        isWorking = true
        error = nil
        task = Task { [weak self] in
            let started = ContinuousClock.now
            do {
                let item = try await extract(sentence)
                let elapsed = ContinuousClock.now - started
                self?.item = item
                self?.json = Self.encode(item)
                self?.seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
            } catch {
                self?.error = "\(error)"
            }
            self?.isWorking = false
        }
        return true
    }

    public func waitForResult() async { await task?.value }

    public nonisolated static func encode(_ item: TodoItem) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return (try? encoder.encode(item)).map { String(decoding: $0, as: UTF8.self) } ?? ""
    }
}
