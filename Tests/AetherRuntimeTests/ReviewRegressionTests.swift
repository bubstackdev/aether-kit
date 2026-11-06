import Testing
@testable import AetherRuntime
import AetherTestSupport

/// Regression tests for the M2 review findings. Each failed before its fix.
@Suite("DecoderEngine review regressions")
struct ReviewRegressionTests {
    typealias T = DecoderEngineTests

    static func gated(_ config: FakeLM.Config = FakeLM.Config(), options: DecoderEngine.Options = T.options())
        async throws -> (DecoderEngine, RunGate)
    {
        let gate = RunGate()
        var c = config
        c.gate = gate
        let fake = FakeLM(c)
        let engine = try await DecoderEngine(step: fake, graph: fake.manifestGraph(), device: T.mac, options: options)
        return (engine, gate)
    }

    static func isUnsupported(_ error: AetherError) -> Bool {
        if case .unsupportedOperation = error { return true }
        return false
    }

    /// #3: a pipeline depth the token-slot ring cannot hold is refused at init (never a trap).
    @Test func pipelineDepthIsValidated() async throws {
        for depth in [0, 7, 100] {
            var options = T.options()
            options.pipelineDepth = depth
            await T.expectError({ _ = try await T.engine(options: options) }, Self.isUnsupported)
        }
        var ok = T.options()
        ok.pipelineDepth = 6
        _ = try await T.engine(options: ok)
    }

    /// #4, #5: while a greedy step is suspended inside the graph, every other call is refused, including a second
    /// nextGreedy and endGreedy; the step then completes and the position is consistent.
    @Test func greedyStepIsNotReentrant() async throws {
        let (engine, gate) = try await Self.gated()
        gate.open()
        let prompt = T.prompt(5)
        let first = try Logits.argmax(try await engine.prefill(prompt)!.last)
        let snapshot = try await engine.snapshot()
        try await engine.beginGreedy(first: first, limit: 5)
        gate.close()
        let step = Task { try await engine.nextGreedy() }
        await gate.waitForEntry()
        await T.expectError({ _ = try await engine.prefill([1]) }, Self.isUnsupported)
        await T.expectError({ try await engine.rewind(to: 0) }, Self.isUnsupported)
        await T.expectError({ try await engine.restore(snapshot) }, Self.isUnsupported)
        await T.expectError({ try await engine.beginGreedy(first: 1, limit: 2) }, Self.isUnsupported)
        await T.expectError({ _ = try await engine.nextGreedy() }, Self.isUnsupported)
        await T.expectError({ _ = try await engine.endGreedy() }, Self.isUnsupported)
        gate.open()
        let second = try await step.value
        let end = try await engine.endGreedy()
        #expect(second == T.expectedGreedy(prompt, count: 2)[1])
        #expect(end.fed == 1)
        #expect(await engine.position == 6)
    }

    /// The same for a prefill in flight.
    @Test func prefillIsNotReentrant() async throws {
        let (engine, gate) = try await Self.gated()
        let feed = Task { try await engine.prefill(T.prompt(5)) }
        await gate.waitForEntry()
        await T.expectError({ _ = try await engine.prefill([1]) }, Self.isUnsupported)
        await T.expectError({ _ = try await engine.snapshot() }, Self.isUnsupported)
        gate.open()
        _ = try await feed.value
        #expect(await engine.position == 5)
    }

    /// #7: restoring a snapshot that does not fit the engine fails without touching the engine's state.
    @Test func restoreValidatesBeforeTouchingState() async throws {
        let (engine, _, _) = try await T.engine(options: T.options(initial: 8))
        let (other, _, _) = try await T.engine(FakeLM.Config(recurrent: true), options: T.options(initial: 16))
        let prompt = T.prompt(6)
        _ = try await engine.prefill(prompt)
        _ = try await other.prefill(prompt)
        let foreign = try await other.snapshot()
        await T.expectError({ try await engine.restore(foreign) }) { _ in true }
        #expect(await engine.position == 6 && engine.description.maxCapacity > 0)
        let next = try await engine.prefill([9])!
        #expect(try Logits.argmax(next.last) == T.expectedGreedy(prompt + [9], count: 1)[0])
    }

    /// #10: a padded tail would advance recurrent state on pad tokens; the combination is refused.
    @Test func padTailWithRecurrentStateIsRefused() async throws {
        await T.expectError({
            _ = try await T.engine(FakeLM.Config(staticStep: true, recurrent: true), prefillChunk: 4, padTail: true)
        }, Self.isUnsupported)
        _ = try await T.engine(FakeLM.Config(staticStep: true), prefillChunk: 4, padTail: true)
    }

    /// #13: a snapshot copies the sequence states only up to the position, not the whole capacity.
    @Test func snapshotIsSizedByPosition() async throws {
        let (engine, _, _) = try await T.engine(FakeLM.Config(recurrent: true), options: T.options(initial: 1024))
        _ = try await engine.prefill(T.prompt(6))
        let snapshot = try await engine.snapshot()
        // keys + values: 6 floats each; rec: 1 float.
        #expect(snapshot.byteCount == (6 + 6 + 1) * 4)
        let continued = try await engine.prefill([9])!
        _ = try await engine.prefill(T.prompt(20, seed: 7))
        try await engine.restore(snapshot)
        #expect(try await engine.prefill([9])! == continued)
    }

    /// #12: a +inf logit is a forced token, not a NaN distribution.
    @Test func positiveInfinityIsForced() throws {
        var logits = [Float](repeating: 0, count: 10)
        logits[3] = .infinity
        logits[7] = .infinity
        var s = Sampler(SamplingParameters(temperature: 0.7, topK: 5, topP: 0.9, seed: 1))
        for _ in 0..<20 {
            var row = logits
            #expect(try s.sample(&row) == 3)
        }
    }
}

/// The owner's second code review: the decoder items. Each failed before its fix.
@Suite("Owner review 2: decoder")
struct DecoderOwnerReviewTests {
    typealias T = DecoderEngineTests

    static func isInvalidLogits(_ error: AetherError) -> Bool {
        if case .generation(.invalidLogits) = error { return true }
        return false
    }

    static func expectInvalid(_ row: [Float], _ comment: Comment? = nil, sourceLocation: SourceLocation = #_sourceLocation) {
        do {
            let token = try Logits.argmax(row[...])
            Issue.record("picked \(token) from \(row.prefix(4))… \(comment.map { "(\($0))" } ?? "")", sourceLocation: sourceLocation)
        } catch {
            #expect(Self.isInvalidLogits(error), "\(error)", sourceLocation: sourceLocation)
        }
    }

    /// Finding 2: one invalid-logit policy. A NaN anywhere, or nothing above −inf, is `invalidLogits`, never token 0;
    /// +inf ranks first.
    @Test func argmaxRefusesInvalidRows() throws {
        Self.expectInvalid([.nan, 1])
        Self.expectInvalid([1, .nan])
        Self.expectInvalid([.nan, .infinity])
        Self.expectInvalid([.infinity, .nan, -.infinity])
        Self.expectInvalid([-.infinity, -.infinity], "all -inf returned token 0")
        Self.expectInvalid([])
        var wide = [Float](repeating: 0.5, count: 151_936)
        wide[151_935] = .nan
        Self.expectInvalid(wide)
        #expect(try Logits.argmax([-.infinity, 2, 2]) == 1)
        #expect(try Logits.argmax([1, .infinity, .infinity]) == 1)
        #expect(try Logits.argmax([-.infinity, .infinity, 3]) == 1, "both infinities and no NaN is valid")
    }

    /// Finding 2: the sampler applies the same policy on every path; `[NaN, +inf]` picked token 1 when sampling.
    @Test func samplerRefusesInvalidRowsOnEveryPath() {
        let paths: [(SamplingParameters, TokenConstraint)] = [
            (.greedy, .unconstrained),
            (SamplingParameters(repetitionPenalty: 1.2), .unconstrained),
            (SamplingParameters(temperature: 0.7, seed: 1), .unconstrained),
            (SamplingParameters(temperature: 0.7, topK: 1, seed: 1), .unconstrained),
            (SamplingParameters(temperature: 0.7, seed: 1), .allowed(TokenSet([1], vocabularySize: 2))),
            (.greedy, .allowed(TokenSet([0, 1], vocabularySize: 2))),
        ]
        for (parameters, constraint) in paths {
            for row: [Float] in [[.nan, .infinity], [.infinity, .nan], [-.infinity, -.infinity]] {
                var sampler = Sampler(parameters)
                var values = row
                do {
                    let token = try sampler.sample(&values, constraint: constraint)
                    Issue.record("\(parameters) \(constraint) picked \(token) from \(row)")
                } catch {
                    if case .generation(.invalidLogits) = error { continue }
                    // A constraint over an all -inf row leaves no token: that is its own failure.
                    if case .generation(.constraintExhausted) = error, row.allSatisfy({ $0 == -.infinity }) { continue }
                    Issue.record("\(parameters) \(row): \(error)")
                }
            }
        }
    }

    /// Finding 2: the host greedy decode (`run` engines) takes the same policy: an invalid row mid-decode throws.
    @Test func hostGreedyRefusesAnInvalidRow() async throws {
        var config = FakeLM.Config()
        let prompt = T.prompt(6)
        config.rowOverride = { context in context.count == prompt.count + 2 ? [Float](repeating: -.infinity, count: 64) : nil }
        let (engine, _, _) = try await T.engine(config)
        let first = try Logits.argmax(try await engine.prefill(prompt)!.last)
        try await engine.beginGreedy(first: first, limit: 8)
        _ = try await engine.nextGreedy()
        await T.expectError({ _ = try await engine.nextGreedy() }, Self.isInvalidLogits)
        try await engine.endGreedy()
    }

    static func isCapacity(_ error: AetherError) -> Bool {
        if case .capacity = error { return true }
        return false
    }

    /// Finding 4: an explicit capacity below a static sequence axis is the token limit (the allocation stays the
    /// static axis); one above it is refused at load. It used to be replaced by the static axis.
    @Test func explicitCapacityLimitsAStaticSequenceAxis() async throws {
        var config = FakeLM.Config()
        config.staticSequence = 64
        let (engine, _, _) = try await T.engine(config, options: T.options(capacity: 16))
        #expect(engine.maxCapacity == 16)
        #expect(await engine.capacity == 64, "the allocation is the graph's static axis")
        let prompt = T.prompt(12)
        let logits = try await engine.prefill(prompt)!
        #expect(try Logits.argmax(logits.last) == T.expectedGreedy(prompt, count: 1)[0])
        await T.expectError({ _ = try await engine.prefill(T.prompt(5)) }, Self.isCapacity)
        #expect(await engine.position == 12, "a refused feed leaves the position")
        // A greedy decode stops at the token limit, not at the allocation.
        try await engine.beginGreedy(first: try Logits.argmax(logits.last), limit: 100)
        var count = 0
        while try await engine.nextGreedy() != nil { count += 1 }
        try await engine.endGreedy()
        #expect(count == 4)
        // A snapshot of the static allocation restores under the smaller token limit.
        try await engine.rewind(to: 0)
        _ = try await engine.prefill(prompt)
        let snapshot = try await engine.snapshot()
        try await engine.reset()
        try await engine.restore(snapshot)
        #expect(await engine.position == 12)

        let (whole, _, _) = try await T.engine(config)
        #expect(whole.maxCapacity == 64)
        await T.expectError({ _ = try await T.engine(config, options: T.options(capacity: 100)) }) { error in
            if case .capacity(let what, 100, 64) = error { return what.contains("static") }
            return false
        }
    }
}
