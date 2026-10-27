/// Decode control flow shared by every backend and the CPU tests. It runs on the caller's actor, so `prefill` and
/// `step` may use actor-owned native state directly; token delivery and the constraint processor run in child tasks
/// beside each step (a backend's thread is busy inside its native step, so overlapping work must run elsewhere).
package enum LanguageDecodeLoop {
    package struct Output: Sendable {
        package let tokens: [Int]
        package let stopReason: LanguageStopReason
        package let seed: UInt64?
        package let timeToFirstTokenSeconds: Double
        package let totalSeconds: Double
    }

    /// One admitted request: the prefill (with the processor's `begin` beside it), the first token, then the decode
    /// steps. `prefill()` returns the prompt's last logits row and `step(token, position)` submits one token, joins its
    /// native work and returns its logits row. Sampling and the constraint follow ``LanguageSampler``. Timing starts
    /// here, so call it once the request is admitted.
    package static func generate(prompt: [Int], maximumTokens: Int, stopTokenIDs: Set<Int>, vocabularySize: Int,
                                 sampling: LanguageSampling, processor: (any LanguageLogitsProcessor)?,
                                 isolation: isolated (any Actor)? = #isolation,
                                 prefill: () async throws -> [Float],
                                 step: (_ token: Int, _ position: Int) async throws -> [Float],
                                 onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?)
        async throws -> Output {
        let start = ContinuousClock.now
        var sampler = try LanguageSampler(sampling: sampling, prompt: prompt, vocabularySize: vocabularySize)
        var logits: [Float]
        let constraint: LanguageTokenConstraint
        if let processor {
            async let begun = processor.begin(prompt: prompt)
            logits = try await prefill()
            constraint = try await begun
        } else {
            logits = try await prefill()
            constraint = .unconstrained
        }
        let first = try sampler.sample(&logits, constraint: constraint)
        let firstTokenSeconds = seconds(since: start)
        let decoded = try await run(first: first, prompt: prompt.count, maximumTokens: maximumTokens, stopTokenIDs: stopTokenIDs,
                                    processor: processor, step: step, choose: { row, constraint in
                                        try sampler.sample(&row, constraint: constraint)
                                    }, onToken: onToken)
        return Output(tokens: decoded.tokens, stopReason: decoded.reason, seed: sampler.seed,
                      timeToFirstTokenSeconds: firstTokenSeconds, totalSeconds: seconds(since: start))
    }

    /// Greedy decoding with sampling folded into `step(token, position)`, which returns the next token. Kept for
    /// callers that choose tokens themselves; the same loop as ``run(first:prompt:maximumTokens:stopTokenIDs:processor:isolation:step:choose:onToken:)``.
    package static func run(first: Int, prompt: Int, maximumTokens: Int, stopTokenIDs: Set<Int>,
                            isolation: isolated (any Actor)? = #isolation,
                            step: (_ token: Int, _ position: Int) async throws -> Int,
                            onToken: (@Sendable (Int) async throws -> Void)?) async throws -> (tokens: [Int], reason: String) {
        var delivery: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?
        if let onToken {
            delivery = { token in
                try await onToken(token.id)
                return .continue
            }
        }
        let decoded = try await run(first: first, prompt: prompt, maximumTokens: maximumTokens, stopTokenIDs: stopTokenIDs,
                                    processor: nil, step: step, choose: { (next: inout Int, _: LanguageTokenConstraint) in next },
                                    onToken: delivery)
        return (decoded.tokens, decoded.reason.rawValue)
    }

    /// `step(token, position)` submits one token, joins its native work and returns what `choose` turns into the next
    /// token under the processor's constraint. Each emitted token is delivered, and passed to the processor, while the
    /// following step runs on the GPU. Deliveries stay ordered with at most one in flight, and every delivery finishes
    /// before this returns or throws; the step after token t starts only once the processor has accepted t. A delivery
    /// or processor error surfaces after the overlapping step joins (its result discarded); a step error cancels and
    /// awaits both. A delivery returning `.stop` ends the request the same way, without an error.
    package static func run<Row>(first: Int, prompt: Int, maximumTokens: Int, stopTokenIDs: Set<Int>,
                                 processor: (any LanguageLogitsProcessor)?,
                                 isolation: isolated (any Actor)? = #isolation,
                                 step: (_ token: Int, _ position: Int) async throws -> Row,
                                 choose: (inout Row, LanguageTokenConstraint) throws -> Int,
                                 onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?)
        async throws -> (tokens: [Int], reason: LanguageStopReason) {
        var next = first
        var generated: [Int] = []
        for index in 0..<maximumTokens {
            try Task.checkCancellation()
            if stopTokenIDs.contains(next) { return (generated, .endOfSequence) }
            generated.append(next)
            let token = LanguageGeneratedToken(id: next, index: index)
            guard index + 1 < maximumTokens else {
                if try await deliver(token, to: onToken) == .stop { return (generated, .stopped) }
                break
            }
            guard onToken != nil || processor != nil else {
                var row = try await step(token.id, prompt + index)
                next = try choose(&row, .unconstrained)
                continue
            }
            async let delivered = deliver(token, to: onToken)
            async let accepted = accept(token.id, by: processor)
            var row = try await step(token.id, prompt + index)
            let disposition = try await delivered
            let constraint = try await accepted
            if disposition == .stop { return (generated, .stopped) }
            next = try choose(&row, constraint)
        }
        return (generated, .outputLimit)
    }

    private static func deliver(_ token: LanguageGeneratedToken,
                                to onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?)
        async throws -> LanguageTokenDisposition {
        guard let onToken else { return .continue }
        return try await onToken(token)
    }

    private static func accept(_ token: Int, by processor: (any LanguageLogitsProcessor)?) async throws -> LanguageTokenConstraint {
        guard let processor else { return .unconstrained }
        return try await processor.accept(token)
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}
