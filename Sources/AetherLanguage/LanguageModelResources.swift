import AetherKit
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
/// Its lifecycle unloads it when idle or under memory pressure (docs/model-lifecycle.md); the next request loads again.
actor LanguageModelResources {
    typealias Loader = @Sendable () async throws -> LoadedLanguageModel

    private let availability: AetherLanguageModel.Availability
    private let loader: Loader
    nonisolated let lifecycle: ModelLifecycle
    private var loading: Task<LoadedLanguageModel, any Error>?
    /// The loaded model's tokens indexed for JSON constraints, built on the first structured request.
    private var vocabulary: LanguageJSONVocabulary?
    private var running = false
    private var waiting: [(id: Int, continuation: CheckedContinuation<Void, any Error>)] = []
    private var nextWaiter = 0

    init(availability: AetherLanguageModel.Availability, lifecycle: ModelLifecycle, loader: @escaping Loader) {
        self.availability = availability
        self.loader = loader
        self.lifecycle = lifecycle
        lifecycle.setRelease { [weak self] reason in await self?.release(reason) ?? false }
    }

    /// Loads the model now, in the request slot, rather than on the first request.
    func prewarm() async throws {
        try await acquire()
        defer { release() }
        _ = try await loaded()
    }

    /// The loaded model, loading it first if needed; called in the request slot. A failed load is not cached: the
    /// next call tries again. While memory pressure is critical, a load is refused with `insufficientMemory`.
    private func loaded() async throws -> LoadedLanguageModel {
        if case .unavailable(let reason) = availability { throw AetherLanguageError.unavailable(reason) }
        if let loading { return try await loading.value }
        do { try lifecycle.admitLoad() } catch { throw AetherLanguageError.insufficientMemory }
        let task = Task { try await loader() }
        loading = task
        lifecycle.loading()
        do {
            return try await task.value
        } catch {
            if loading == task { loading = nil }
            lifecycle.notLoaded()
            throw AetherLanguageError.loading(error)
        }
    }

    /// Runs `body` with the loaded model once no other request is running. A request cancelled while it waits throws
    /// `CancellationError` without running.
    func withModel<T: Sendable>(_ body: @Sendable (LoadedLanguageModel) async throws -> T) async throws -> T {
        try await acquire()
        defer { release() }
        // Loaded inside the slot, so an idle or pressure release (which takes the slot) can't unload it in between.
        let model = try await loaded()
        try Task.checkCancellation()
        lifecycle.running()
        return try await body(model)
    }

    /// The JSON-constraint index of the loaded model's vocabulary (one per load; `excluded` is the same every time).
    func jsonVocabulary(_ model: LoadedLanguageModel, excluded: Set<Int>) -> LanguageJSONVocabulary {
        if let vocabulary { return vocabulary }
        let built = LanguageJSONVocabulary(bytes: model.codec.bytes, headRows: model.backend.vocabularySize, excluded: excluded)
        vocabulary = built
        return built
    }

    /// Waits for a running request, then releases the backend and tokenizer.
    func unload() async {
        // Unload must happen even when its task is cancelled, so it waits uncancellably.
        await Task { await self.acquireUncancellably() }.value
        defer { release() }
        guard let loading else { return }
        self.loading = nil
        vocabulary = nil
        if let model = try? await loading.value { await model.backend.unload() }
        lifecycle.unloaded(.requested)
    }

    /// The lifecycle's idle or pressure release: unloads only if no request is running or waiting. The state turns
    /// unloaded before the slot is released, so a request that was waiting loads again afterwards.
    private func release(_ reason: ModelUnloadReason) async -> Bool {
        guard !running, waiting.isEmpty, let loading else { return false }
        running = true
        defer { release() }
        self.loading = nil
        vocabulary = nil
        if let model = try? await loading.value { await model.backend.unload() }
        lifecycle.unloaded(reason)
        return true
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
        guard !waiting.isEmpty else {
            running = false
            // The slot is free and the model stays loaded: it's idle now.
            if loading != nil { lifecycle.idle() }
            return
        }
        waiting.removeFirst().continuation.resume()
    }

    private func cancelWaiter(_ id: Int) {
        guard let index = waiting.firstIndex(where: { $0.id == id }) else { return }
        waiting.remove(at: index).continuation.resume(throwing: CancellationError())
    }
}
