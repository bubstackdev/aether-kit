import AetherLanguageRuntime
import AetherLanguageTokenizers

/// A loaded model: the backend that generates tokens and the codec that renders and decodes text.
struct LoadedLanguageModel: Sendable {
    let backend: any LanguageBackend
    let codec: any LanguageTextCodec
}

/// What the engine needs from a tokenizer. `LanguageTokenizer` provides it for real bundles; CPU tests use a fake.
protocol LanguageTextCodec: Sendable {
    /// The transcript rendered by the model's chat template with the generation prompt.
    func render(_ messages: [LanguageChatMessage], thinking: Bool) throws -> [Int]
    var bytes: LanguageTokenBytes { get }
    func tokenID(_ token: String) -> Int?
}

struct TokenizerCodec: LanguageTextCodec {
    let tokenizer: LanguageTokenizer
    let bytes: LanguageTokenBytes

    func render(_ messages: [LanguageChatMessage], thinking: Bool) throws -> [Int] {
        try tokenizer.render(messages, thinking: thinking).tokenIDs
    }

    func tokenID(_ token: String) -> Int? { tokenizer.tokenID(token) }
}

/// One model's loaded backend and tokenizer, shared by every copy of its ``AetherLanguageModel`` and every session on
/// it. Loads lazily, once, and runs one request at a time: requests from several sessions wait their turn in order.
actor LanguageModelResources {
    typealias Loader = @Sendable () async throws -> LoadedLanguageModel

    private let availability: AetherLanguageModel.Availability
    private let loader: Loader
    private var loading: Task<LoadedLanguageModel, any Error>?
    private var running = false
    private var waiting: [(id: Int, continuation: CheckedContinuation<Void, any Error>)] = []
    private var nextWaiter = 0

    init(availability: AetherLanguageModel.Availability, loader: @escaping Loader) {
        self.availability = availability
        self.loader = loader
    }

    /// The loaded model, loading it first if needed. A failed load is not cached: the next call tries again.
    func loaded() async throws -> LoadedLanguageModel {
        if case .unavailable(let reason) = availability { throw AetherLanguageError.unavailable(reason) }
        if let loading { return try await loading.value }
        let task = Task { try await loader() }
        loading = task
        do {
            return try await task.value
        } catch {
            loading = nil
            throw AetherLanguageError.loading(error)
        }
    }

    /// Runs `body` with the loaded model once no other request is running. A request cancelled while it waits throws
    /// `CancellationError` without running.
    func withModel<T: Sendable>(_ body: @Sendable (LoadedLanguageModel) async throws -> T) async throws -> T {
        let model = try await loaded()
        try await acquire()
        defer { release() }
        try Task.checkCancellation()
        return try await body(model)
    }

    /// Waits for a running request, then releases the backend and tokenizer.
    func unload() async {
        // Unload must happen even when its task is cancelled, so it waits uncancellably.
        await Task { await self.acquireUncancellably() }.value
        defer { release() }
        guard let loading else { return }
        self.loading = nil
        if let model = try? await loading.value { await model.backend.unload() }
    }

    private func acquire() async throws {
        guard running else { running = true; return }
        let id = nextWaiter
        nextWaiter += 1
        try await withTaskCancellationHandler { () async throws -> Void in
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                if Task.isCancelled { continuation.resume(throwing: CancellationError()) }
                else { waiting.append((id, continuation)) }
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
    }

    private func acquireUncancellably() async {
        guard running else { running = true; return }
        let id = nextWaiter
        nextWaiter += 1
        try? await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            waiting.append((id, continuation))
        }
    }

    private func release() {
        guard !waiting.isEmpty else { running = false; return }
        waiting.removeFirst().continuation.resume()
    }

    private func cancelWaiter(_ id: Int) {
        guard let index = waiting.firstIndex(where: { $0.id == id }) else { return }
        waiting.remove(at: index).continuation.resume(throwing: CancellationError())
    }
}
