import Testing
@testable import AetherRuntime
import AetherTestSupport

/// DecoderEngine on the fake decoder (DESIGN.md §12): schedule, capacity, rewind and snapshot, logits rows, sampling.
@Suite("DecoderEngine (fake)")
struct DecoderEngineTests {
    static let mac = Device(platform: .macOS, architecture: "fake9", physicalMemory: 64 << 30)

    static func options(capacity: Int? = nil, initial: Int? = nil, budget: Int64 = 1 << 30) -> DecoderEngine.Options {
        DecoderEngine.Options(capacity: capacity, initialCapacity: initial, memoryBudget: .bytes(budget))
    }

    static func engine(
        _ config: FakeLM.Config = FakeLM.Config(), prefillChunk: Int? = nil, padTail: Bool? = nil, maxContext: Int = 4096,
        prefillLimit: Int? = nil, options: DecoderEngine.Options = options()
    ) async throws -> (DecoderEngine, CallLog, CallLog) {
        let stepLog = CallLog(), prefillLog = CallLog()
        var config = config
        config.log = stepLog
        let step = FakeLM(config)
        let prefill = prefillChunk.map { FakeLM.prefillPhase(of: config, chunk: $0, log: prefillLog) }
        var graph = step.manifestGraph(maxContext: maxContext, prefillChunk: prefillChunk, padTail: padTail)
        graph.limits?.prefillChunk = prefillLimit
        let engine = try await DecoderEngine(step: step, prefill: prefill, graph: graph, device: mac, options: options)
        return (engine, stepLog, prefillLog)
    }

    static func prompt(_ n: Int, seed: Int = 3) -> [Int32] { (0..<n).map { Int32(($0 * 13 + seed) % 61 + 1) } }

    /// Greedy reference computed directly from the fake's rule (no engine).
    static func expectedGreedy(_ prompt: [Int32], count: Int, vocab: Int = 64) -> [Int32] {
        let predict = FakeLM.mixing(vocab: vocab)
        var context = prompt
        var out: [Int32] = []
        for _ in 0..<count {
            let t = predict(context)
            out.append(t)
            context.append(t)
        }
        return out
    }

    static func greedy(_ engine: DecoderEngine, _ prompt: [Int32], count: Int) async throws -> [Int32] {
        let logits = try await engine.prefill(prompt)!
        var out = [Logits.argmax(logits.last)]
        try await engine.beginGreedy(first: out[0], limit: count - 1)
        while let t = try await engine.nextGreedy() { out.append(t) }
        try await engine.endGreedy()
        return out
    }

    static func expectError(
        _ body: () async throws -> Void, sourceLocation: SourceLocation = #_sourceLocation, _ matches: (AetherError) -> Bool
    ) async {
        do {
            try await body()
            Issue.record("expected an error", sourceLocation: sourceLocation)
        } catch let error as AetherError {
            #expect(matches(error), "\(error)", sourceLocation: sourceLocation)
        } catch {
            Issue.record("unexpected \(error)", sourceLocation: sourceLocation)
        }
    }

    // MARK: Schedule

    @Test func prefillPlanFollowsTheThreeRules() {
        func shape(_ plan: PrefillPlan) -> [String] { plan.calls.map { "\($0.phase.rawValue)\($0.count)/\($0.fed)" } }
        // 3. dynamic chunks of at most prefillChunk.
        #expect(shape(PrefillPlan(tokens: 1100, shape: .init(dynamicChunk: 512))) == ["step512/512", "step512/512", "step76/76"])
        // 2. static S=1: one token per call.
        #expect(shape(PrefillPlan(tokens: 3, shape: .init(stepStatic: 1))) == ["step1/1", "step1/1", "step1/1"])
        // 1. prefill phase: full chunks, then the remainder on step.
        #expect(shape(PrefillPlan(tokens: 130, shape: .init(prefillChunk: 64, stepStatic: 1)))
                == ["prefill64/64", "prefill64/64", "step1/1", "step1/1"])
        // ... with padTail, a padded final chunk replaces it.
        #expect(shape(PrefillPlan(tokens: 130, shape: .init(prefillChunk: 64, padTail: true, stepStatic: 1)))
                == ["prefill64/64", "prefill64/64", "prefill2/64"])
        // ... but only when the KV has room for the padding, and the prefill rows can return the last real token.
        #expect(shape(PrefillPlan(tokens: 130, shape: .init(prefillChunk: 64, padTail: true, stepStatic: 1), padRoom: 10)).last == "step1/1")
        #expect(shape(PrefillPlan(tokens: 130, shape: .init(prefillChunk: 64, padTail: true, prefillRowsAll: false, stepStatic: 1))).last == "step1/1")
        #expect(shape(PrefillPlan(tokens: 128, shape: .init(prefillChunk: 64, padTail: true))) == ["prefill64/64", "prefill64/64"])
    }

    @Test func engineFeedsOnTheSchedule() async throws {
        let (dynamic, step, _) = try await Self.engine(prefillLimit: 8)
        _ = try await dynamic.prefill(Self.prompt(20))
        #expect(step.calls == [8, 8, 4])

        let (decodeOnly, s1, _) = try await Self.engine(FakeLM.Config(staticStep: true))
        _ = try await decodeOnly.prefill(Self.prompt(5))
        #expect(s1.calls == [1, 1, 1, 1, 1])

        let (phased, main, pf) = try await Self.engine(FakeLM.Config(staticStep: true), prefillChunk: 4)
        #expect(phased.description.prefillChunk == 4)
        let logits = try await phased.prefill(Self.prompt(10))!
        #expect(pf.calls == [4, 4])
        #expect(main.calls == [1, 1])
        #expect(Logits.argmax(logits.last) == Self.expectedGreedy(Self.prompt(10), count: 1)[0])

        let (padded, main2, pf2) = try await Self.engine(FakeLM.Config(staticStep: true), prefillChunk: 4, padTail: true)
        let padLogits = try await padded.prefill(Self.prompt(10))!
        #expect(pf2.calls == [4, 4, 4])
        #expect(main2.calls.isEmpty)
        #expect(await padded.position == 10)
        #expect(Logits.argmax(padLogits.last) == Logits.argmax(logits.last))
        // The padded rows left nothing behind: decoding continues exactly.
        let continued = try await padded.step(Logits.argmax(padLogits.last))
        let reference = try await phased.step(Logits.argmax(logits.last))
        #expect(Logits.argmax(continued.last) == Logits.argmax(reference.last))
    }

    @Test func everyScheduleGivesTheSameOutput() async throws {
        let prompt = Self.prompt(37)
        let expected = Self.expectedGreedy(prompt, count: 12)
        let engines = [
            try await Self.engine(prefillLimit: 5).0,
            try await Self.engine(FakeLM.Config(staticStep: true)).0,
            try await Self.engine(FakeLM.Config(staticStep: true), prefillChunk: 8).0,
            try await Self.engine(FakeLM.Config(staticStep: true, recurrent: true), prefillChunk: 8).0,
            try await Self.engine(FakeLM.Config(staticStep: true), prefillChunk: 8, padTail: true).0,
        ]
        for engine in engines { #expect(try await Self.greedy(engine, prompt, count: 12) == expected) }
    }

    // MARK: Logits rows

    @Test func logitsRowsAreValidatedAgainstTheGraph() async throws {
        let (all, _, _) = try await Self.engine(prefillLimit: 4)
        let rows = try #require(try await all.prefill(Self.prompt(10), logits: .all))
        #expect(rows.rows == 10 && rows.firstPosition == 0)
        #expect(try await all.prefill([5], logits: .none) == nil)
        #expect(await all.position == 11)

        let (last, _, _) = try await Self.engine(FakeLM.Config(lastRowOnly: true), prefillLimit: 4)
        await Self.expectError({ _ = try await last.prefill(Self.prompt(10), logits: .all) }) { error in
            if case .manifestMismatch(let field, _, _) = error { return field == "graphs.decoder.outputs.logits.rows" }
            return false
        }
        #expect(await last.position == 0, "a refused feed changes nothing")
        let one = try #require(try await last.prefill(Self.prompt(10), logits: .last))
        #expect(Logits.argmax(one.last) == Self.expectedGreedy(Self.prompt(10), count: 1)[0])
        // One token per call can return every row even on a last-row graph.
        let (decodeOnly, _, _) = try await Self.engine(FakeLM.Config(staticStep: true))
        #expect(try await decodeOnly.prefill(Self.prompt(3), logits: .all)?.rows == 3)
    }

    // MARK: Capacity

    @Test func capacityIsDerivedFromMemory() throws {
        let sizing = KVSizing(maxContext: 8192, platformCap: nil, bytesPerToken: 1000, fixedBytes: 100_000, modelBytes: 400_000)
        #expect(try sizing.capacity(budget: 100_000_000) == 8192)  // maxContext bounds it
        #expect(try sizing.capacity(budget: 2_500_000) == 1792)  // (2.5M − 0.5M) / 1000 = 2000, rounded down to 256s
        var capped = sizing
        capped.platformCap = 1024
        #expect(try capped.capacity(budget: 100_000_000) == 1024)
        capped.maxPosition = 511
        #expect(try capped.capacity(budget: 100_000_000) == 512)
        #expect(try sizing.capacity(budget: 100_000_000, requested: 4096) == 4096)
        #expect(throws: AetherError.self) { try sizing.capacity(budget: 2_500_000, requested: 4096) }
        #expect(throws: AetherError.self) { try sizing.capacity(budget: 400_000) }  // the model alone exceeds it
        var fixed = sizing
        fixed.staticAxis = 2048
        #expect(try fixed.capacity(budget: 100_000_000) == 2048)
        #expect(throws: AetherError.self) { try fixed.capacity(budget: 1_000_000) }
        #expect(try KVSizing.grow(current: 256, needed: 300, limit: 4096) == 512)
        #expect(try KVSizing.grow(current: 256, needed: 3000, limit: 4096) == 4096)
        #expect(try KVSizing.grow(current: 3000, needed: 3500, limit: 4096) == 4096)
        #expect(throws: AetherError.self) { try KVSizing.grow(current: 256, needed: 5000, limit: 4096) }
        #expect(MemoryBudget(.bytes(1000), reserveBytes: 300).bytes(on: Self.mac) == 700)
        #expect(MemoryBudget(.fractionOfRAM(0.25)).bytes(on: Self.mac) == 16 << 30)
    }

    @Test func engineSizesKVFromTheBudgetAndGrowsGeometrically() async throws {
        // The fake's KV is 2 float32 states plus the int32 history and positions: 16 bytes per token. Fixed: 512 + 8
        // logits rows of 64 float32 = 133,120 bytes. (200,000 − 133,120) / 16 = 4,180 → 4,096.
        let (tight, _, _) = try await Self.engine(maxContext: 100_000, options: Self.options(budget: 200_000))
        #expect(tight.sizing.bytesPerToken == 16 && tight.sizing.fixedBytes == 133_120)
        #expect(tight.description.maxCapacity == 4096)
        await #expect(throws: AetherError.self) {
            _ = try await Self.engine(maxContext: 100_000, options: Self.options(capacity: 5000, budget: 200_000))
        }

        let (growing, _, _) = try await Self.engine(options: Self.options(capacity: 256, initial: 4))
        #expect(await growing.capacity == 4)
        let prompt = Self.prompt(9)
        let out = try await Self.greedy(growing, prompt, count: 30)
        #expect(out == Self.expectedGreedy(prompt, count: 30), "growth kept the cached prefix")
        #expect(await growing.capacity == 64)  // 4 → 8 → 16 → 32 → 64 for 38 slots
        await #expect(throws: AetherError.self) { _ = try await growing.prefill(Self.prompt(300)) }
        #expect(await growing.position == 38)
    }

    // MARK: Rewind and snapshot

    @Test func rewindIsExactOnAttentionModels() async throws {
        let (engine, _, _) = try await Self.engine()
        let prompt = Self.prompt(12)
        _ = try await engine.prefill(prompt + [50, 51, 52])
        try await engine.rewind(to: 12)
        #expect(await engine.position == 12)
        let after = try await engine.prefill([7])!
        #expect(Logits.argmax(after.last) == Self.expectedGreedy(prompt + [7], count: 1)[0])
        await #expect(throws: AetherError.self) { try await engine.rewind(to: 99) }
    }

    @Test func recurrentModelsRefusePartialRewindAndRestoreSnapshots() async throws {
        let (engine, _, _) = try await Self.engine(FakeLM.Config(staticStep: true, recurrent: true), options: Self.options(initial: 8))
        #expect(engine.description.hasRecurrentState)
        let prompt = Self.prompt(6)
        _ = try await engine.prefill(prompt)
        let snapshot = try await engine.snapshot()
        let first = try await Self.greedy(engine, [9], count: 20)  // grows the KV past the snapshot's capacity
        #expect(await engine.capacity > snapshot.capacity)
        await Self.expectError({ try await engine.rewind(to: 3) }) { error in
            if case .unsupportedOperation = error { return true }
            return false
        }
        try await engine.restore(snapshot)
        #expect(await engine.position == 6)
        #expect(try await Self.greedy(engine, [9], count: 20) == first)
        try await engine.rewind(to: 0)
        #expect(await engine.position == 0)
        #expect(try await Self.greedy(engine, prompt, count: 4) == Self.expectedGreedy(prompt, count: 4))
    }

    @Test func greedySessionAccountsForFedTokens() async throws {
        let (engine, _, _) = try await Self.engine()
        let logits = try await engine.prefill(Self.prompt(5))!
        try await engine.beginGreedy(first: Logits.argmax(logits.last), limit: 10)
        await #expect(throws: AetherError.self) { _ = try await engine.prefill([1]) }
        _ = try await engine.nextGreedy()
        _ = try await engine.nextGreedy()
        let end = try await engine.endGreedy()
        #expect(end.fed == 2 && end.overshoot == 0)
        #expect(await engine.position == 7, "prompt + first + one more fed; the last returned token is not")
    }

    // MARK: Sampling

    @Test func samplerIsDeterministicPerSeed() {
        let logits: [Float] = (0..<50).map { Float(($0 * 37) % 50) / 10 }
        func draw(_ p: SamplingParameters, _ n: Int = 40) -> [Int32] {
            var s = Sampler(p)
            return (0..<n).map { _ in
                var row = logits
                return try! s.sample(&row)
            }
        }
        let a = draw(SamplingParameters(temperature: 0.8, topK: 20, topP: 0.9, seed: 42))
        #expect(a == draw(SamplingParameters(temperature: 0.8, topK: 20, topP: 0.9, seed: 42)))
        #expect(a != draw(SamplingParameters(temperature: 0.8, topK: 20, topP: 0.9, seed: 43)))
        #expect(Set(a).count > 1)
        let best = Logits.argmax(logits[...])
        #expect(draw(.greedy) == Array(repeating: best, count: 40))
        #expect(draw(SamplingParameters(temperature: 1, topK: 1, seed: 7)) == Array(repeating: best, count: 40))
        #expect(draw(SamplingParameters(temperature: 1, topP: 0.0001, seed: 7)) == Array(repeating: best, count: 40))
        let top5 = Set(logits.enumerated().sorted { $0.element > $1.element }.prefix(5).map { Int32($0.offset) })
        #expect(Set(draw(SamplingParameters(temperature: 2, topK: 5, seed: 1), 200)).isSubset(of: top5))

        var penalized: [Float] = [2, -2, 1]
        try! Sampler(SamplingParameters(repetitionPenalty: 2), prompt: [0, 1, 0]).narrow(&penalized, constraint: .unconstrained)
        #expect(penalized == [1, -4, 1])
        var rng = SeededRandom(seed: 9)
        var rng2 = SeededRandom(seed: 9)
        #expect((0..<5).map { _ in rng.next() } == (0..<5).map { _ in rng2.next() })
        #expect(Logits.argmax([1, 3, 3, 2]) == 1, "ties go to the lowest index")
    }
}
