public import AetherRuntime

/// The engine invariants (``LanguageEngine``'s contract), written against the protocol so every backend's engine runs
/// the same checks: the fake and real `DecoderEngine` here, another backend's engine through its own tests. Each
/// returns what the caller compares, with the tolerance its backend allows.
public enum LanguageEngineInvariants {
    /// Greedy from a reset state: prefill, the first token from the prompt's last row, then `count - 1` more on the
    /// engine's greedy path. Also returns the seconds the greedy path took (after the first token).
    public static func greedy(
        _ engine: any LanguageEngine, prompt: [Int32], count: Int
    ) async throws(AetherError) -> (tokens: [Int32], seconds: Double) {
        try await engine.reset()
        return try await continueGreedy(engine, after: prompt, count: count)
    }

    /// Greedy from the current state: feeds `tokens` (at least one), then decodes as ``greedy(_:prompt:count:)``.
    public static func continueGreedy(
        _ engine: any LanguageEngine, after tokens: [Int32], count: Int
    ) async throws(AetherError) -> (tokens: [Int32], seconds: Double) {
        let logits = try await engine.prefill(tokens, logits: .last)!
        var out = [Logits.argmax(logits.last)]
        let clock = ContinuousClock()
        let start = clock.now
        if count > 1 {
            try await engine.beginGreedy(first: out[0], limit: count - 1)
            do {
                while let t = try await engine.nextGreedy() { out.append(t) }
            } catch {
                _ = try? await engine.endGreedy()
                throw error
            }
            try await engine.endGreedy()
        }
        return (out, (clock.now - start) / .seconds(1))
    }

    /// Greedy by ``LanguageEngine/step(_:)`` and a host argmax, one token at a time, from a reset state. Equal to
    /// ``greedy(_:prompt:count:)``'s tokens.
    public static func stepwiseGreedy(_ engine: any LanguageEngine, prompt: [Int32], count: Int) async throws(AetherError) -> [Int32] {
        try await engine.reset()
        var token = Logits.argmax(try await engine.prefill(prompt, logits: .last)!.last)
        var out = [token]
        while out.count < count {
            token = Logits.argmax(try await engine.step(token).last)
            out.append(token)
        }
        return out
    }

    /// The prompt's last row fed on the prefill schedule and fed one token at a time, from a reset state each.
    public static func prefillRows(
        _ engine: any LanguageEngine, prompt: [Int32]
    ) async throws(AetherError) -> (chunked: ArraySlice<Float>, stepwise: ArraySlice<Float>) {
        try await engine.reset()
        let chunked = try await engine.prefill(prompt, logits: .last)!.last
        try await engine.reset()
        var stepwise: Logits?
        for t in prompt { stepwise = try await engine.step(t) }
        return (chunked, stepwise!.last)
    }

    /// Two turns that share `history` and diverge after it (an edited or regenerated turn). The second turn reuses
    /// the first's state, by ``LanguageEngine/rewind(to:)`` without recurrent state and by a snapshot taken at the end
    /// of `history` with it, the way the Generator does; `fresh` is the same turn from a reset state. Equal tokens.
    public static func twoTurnReuse(
        _ engine: any LanguageEngine, history: [Int32], first: [Int32], second: [Int32], count: Int
    ) async throws(AetherError) -> (reused: [Int32], fresh: [Int32]) {
        try await engine.reset()
        _ = try await engine.prefill(history, logits: .none)
        let snapshot = try await engine.snapshot()
        guard snapshot.position == history.count else {
            throw .unsupportedOperation("snapshot at \(snapshot.position), the position was \(history.count)")
        }
        _ = try await continueGreedy(engine, after: first, count: count)
        if engine.hasRecurrentState {
            try await engine.restore(snapshot)
        } else {
            try await engine.rewind(to: history.count)
        }
        guard await engine.position == history.count else {
            throw .unsupportedOperation("reuse left the position at \(await engine.position), not \(history.count)")
        }
        let reused = try await continueGreedy(engine, after: second, count: count).tokens
        let fresh = try await greedy(engine, prompt: history + second, count: count).tokens
        return (reused, fresh)
    }
}
