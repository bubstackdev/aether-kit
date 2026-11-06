/// A causal language model at the token level (DESIGN.md §6.2): the seam between the text stack (`Generator`,
/// structured output, tools, the FoundationModels bridge, the decision readout) and a backend that owns the weights,
/// the state and the position. ``DecoderEngine`` (Core AI, `decoder-lm@1`) is the implementation in this package; a
/// backend in another package conforms with its own engine (docs/BACKENDS.md §1.2).
///
/// The contract every conformer keeps, checked by the engine-invariant tests:
/// - **One request at a time.** A call that arrives during another operation (a greedy decode between
///   ``beginGreedy(first:limit:)`` and ``endGreedy()`` included) is refused with `unsupportedOperation`, not
///   interleaved.
/// - **Feeding is exact.** ``prefill(_:logits:)`` in any chunking and ``step(_:)`` one token at a time give the same
///   next-token logits (within the backend's numeric tolerance), and greedy decoding equals stepping argmax.
/// - **Rewind or snapshot.** Without recurrent state, ``rewind(to:)`` to any earlier position is exact. With it
///   (``hasRecurrentState``), only `rewind(to: 0)` is, and prefix reuse goes through ``snapshot()`` and
///   ``restore(_:)``.
/// - **Validation before mutation.** A refused call (a snapshot that doesn't fit, a feed past ``maxCapacity``)
///   leaves the state and the position as they were.
public protocol LanguageEngine: Actor {
    /// The width of a logits row: the head's rows, which may exceed the tokenizer's vocabulary.
    nonisolated var vocab: Int { get }
    /// The most tokens the state can hold.
    nonisolated var maxCapacity: Int { get }
    /// Recurrent state forces snapshot and restore instead of a partial rewind.
    nonisolated var hasRecurrentState: Bool { get }

    /// Tokens resident in the state.
    var position: Int { get }
    /// True when pipelined greedy steps ran recurrent state past ``position``: restore a snapshot or reset first.
    var stateRanAhead: Bool { get }
    /// How greedy tokens are produced, for diagnostics (`GenerationSummary.decodeMode`).
    var decodeModeDescription: String { get }

    /// Feeds `tokens` at the current position and returns the logits rows `logits` asks for (nil for `.none`).
    func prefill(_ tokens: [Int32], logits: LogitsSelection) async throws(AetherError) -> Logits?
    /// Feeds one token and returns the next-token logits (the host path for CPU sampling and constraints).
    func step(_ token: Int32) async throws(AetherError) -> Logits
    /// Moves the position back. Exact without recurrent state; with it only `rewind(to: 0)` is supported.
    func rewind(to position: Int) async throws(AetherError)
    /// Back to position 0, with recurrent state reinitialized.
    func reset() async throws(AetherError)
    /// A host copy of the state at ``position``.
    func snapshot() async throws(AetherError) -> EngineSnapshot
    /// Restores a snapshot this engine took, validated before any state is touched. Another engine's is refused.
    func restore(_ snapshot: EngineSnapshot) async throws(AetherError)

    /// Starts greedy decoding after `first` (the caller's pick from the prompt's logits, not yet fed), for at most
    /// `limit` more tokens. Then call ``nextGreedy()`` until it returns nil or the caller stops, and always
    /// ``endGreedy()``.
    func beginGreedy(first: Int32, limit: Int) async throws(AetherError)
    /// The next greedy token; nil at the limit or when the calling task is cancelled.
    func nextGreedy() async throws(AetherError) -> Int32?
    /// Ends greedy decoding: waits for in-flight work and sets the position to the tokens fed (every token returned
    /// except the last). Returns how many were fed and how many steps ran past the last returned token.
    @discardableResult
    func endGreedy() async throws(AetherError) -> (fed: Int, overshoot: Int)
}

extension LanguageEngine {
    /// Feeds `tokens` and returns the last row's logits.
    public func prefill(_ tokens: [Int32]) async throws(AetherError) -> Logits? {
        try await prefill(tokens, logits: .last)
    }
}

/// An engine's state at one position (``LanguageEngine/snapshot()``): `Sendable`, opaque outside the engine that took
/// it. Only that engine can restore it.
public struct EngineSnapshot: Sendable {
    /// The position the snapshot was taken at.
    public let position: Int
    /// The host bytes it holds.
    public let byteCount: Int
    /// The engine's own representation; other code passes it through unread.
    public let payload: any Sendable

    public init(position: Int, byteCount: Int, payload: any Sendable) {
        self.position = position
        self.byteCount = byteCount
        self.payload = payload
    }
}
