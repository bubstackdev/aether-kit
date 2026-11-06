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
        let first = Logits.argmax(try await engine.prefill(prompt)!.last)
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
        #expect(Logits.argmax(next.last) == T.expectedGreedy(prompt + [9], count: 1)[0])
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
