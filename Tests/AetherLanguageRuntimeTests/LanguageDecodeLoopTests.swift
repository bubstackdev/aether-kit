@testable import AetherLanguageRuntime
import Testing

private actor Trace {
    private(set) var events: [String] = []
    func add(_ event: String) { events.append(event) }
    /// Bounded, so a serial regression fails the expectation instead of hanging.
    func observed(_ event: String, within limit: Duration = .seconds(5)) async -> Bool {
        let deadline = ContinuousClock.now + limit
        while !events.contains(event) {
            guard ContinuousClock.now < deadline else { return false }
            try? await Task.sleep(for: .milliseconds(2))
        }
        return true
    }
}

private struct ProcessorFailure: Error, Equatable {}
private struct StepFailure: Error, Equatable {}

/// Allows every token except the one it was last given (so its constraint is visible in the output), and records calls.
private struct RecordingProcessor: LanguageLogitsProcessor {
    let trace: Trace
    let vocabulary: Int
    var failOn: Int?
    var slowOn: Int?

    func begin(prompt: [Int]) async throws -> LanguageTokenConstraint {
        await trace.add("begin")
        _ = await trace.observed("prefill")  // overlaps the prefill
        await trace.add("begun")
        return .unconstrained
    }

    func accept(_ token: Int) async throws -> LanguageTokenConstraint {
        await trace.add("accept:\(token)")
        if token == failOn { throw ProcessorFailure() }
        if token == slowOn {
            do { try await Task.sleep(for: .seconds(30)) } catch { await trace.add("acceptCancelled:\(token)"); throw error }
        }
        await trace.add("accepted:\(token)")
        var allowed = LanguageTokenSet(range: 0..<vocabulary, vocabularySize: vocabulary)
        allowed.remove(token)
        return .allowed(allowed)
    }
}

/// A fake backend row: token `t`'s logits peak at `(t + 1) % vocabulary`, then fall off with distance.
private func row(after token: Int, vocabulary: Int) -> [Float] {
    (0..<vocabulary).map { -Float(($0 - (token + 1) + vocabulary) % vocabulary) }
}

struct LanguageDecodeLoopTests {
    /// `begin` runs while the prefill runs; each `accept(t)` runs while step t runs; step t+1 starts only after
    /// `accept(t)` returned.
    @Test func processorOverlapsStepsAndGatesTheNext() async throws {
        let trace = Trace()
        let processor = RecordingProcessor(trace: trace, vocabulary: 8)
        let output = try await LanguageDecodeLoop.generate(prompt: [5], maximumTokens: 4, stopTokenIDs: [], vocabularySize: 8,
            sampling: .greedy, processor: processor, prefill: {
                await trace.add("prefill")
                _ = await trace.observed("begin")
                return row(after: 5, vocabulary: 8)
            }, step: { token, position in
                await trace.add("step:\(token)@\(position)")
                if await !trace.observed("accept:\(token)") { await trace.add("serial:\(token)") }
                return row(after: token, vocabulary: 8)
            }, onToken: nil)
        // Each constraint removes the token just accepted: the peak (token + 1) is allowed, so greedy follows it.
        #expect(output.tokens == [6, 7, 0, 1] && output.stopReason == .outputLimit && output.seed == nil)
        let events = await trace.events
        #expect(!events.contains { $0.hasPrefix("serial") })
        for (token, following) in [(6, "step:7@2"), (7, "step:0@3")] {
            let accepted = try #require(events.firstIndex(of: "accepted:\(token)"))
            #expect(try #require(events.firstIndex(of: following)) > accepted, "the step after \(token) waited for its accept")
        }
        #expect(!events.contains("accept:1"), "no accept for the last token")
        #expect(events.firstIndex(of: "begun")! < events.firstIndex(of: "step:6@1")!)
    }

    @Test func constraintNarrowsTheChoice() async throws {
        struct OnlyEven: LanguageLogitsProcessor {
            let set = LanguageTokenSet([0, 2, 4, 6], vocabularySize: 8)
            func begin(prompt: [Int]) async throws -> LanguageTokenConstraint { .allowed(set) }
            func accept(_ token: Int) async throws -> LanguageTokenConstraint { .allowed(set) }
        }
        let output = try await LanguageDecodeLoop.generate(prompt: [0], maximumTokens: 5, stopTokenIDs: [], vocabularySize: 8,
            sampling: .greedy, processor: OnlyEven(), prefill: { row(after: 0, vocabulary: 8) },
            step: { token, _ in row(after: token, vocabulary: 8) }, onToken: nil)
        // The peak after t is t + 1 (odd for even t): the best even token is t + 2.
        #expect(output.tokens == [2, 4, 6, 0, 2])
    }

    @Test func processorFailureSurfacesAfterTheOverlappingStepJoins() async throws {
        let trace = Trace()
        let processor = RecordingProcessor(trace: trace, vocabulary: 8, failOn: 7)
        await #expect(throws: ProcessorFailure()) {
            _ = try await LanguageDecodeLoop.generate(prompt: [5], maximumTokens: 8, stopTokenIDs: [], vocabularySize: 8,
                sampling: .greedy, processor: processor, prefill: { await trace.add("prefill"); return row(after: 5, vocabulary: 8) },
                step: { token, _ in
                    await trace.add("stepEnd:\(token)")
                    return row(after: token, vocabulary: 8)
                }, onToken: nil)
        }
        let steps = await trace.events.filter { $0.hasPrefix("stepEnd") }
        #expect(steps == ["stepEnd:6", "stepEnd:7"], "the step beside the failed accept joined; nothing followed")
    }

    @Test(.timeLimit(.minutes(1))) func stepFailureCancelsAndAwaitsTheProcessor() async throws {
        let trace = Trace()
        let processor = RecordingProcessor(trace: trace, vocabulary: 8, slowOn: 6)
        await #expect(throws: StepFailure()) {
            _ = try await LanguageDecodeLoop.generate(prompt: [5], maximumTokens: 8, stopTokenIDs: [], vocabularySize: 8,
                sampling: .greedy, processor: processor, prefill: { await trace.add("prefill"); return row(after: 5, vocabulary: 8) },
                step: { token, _ in
                    _ = await trace.observed("accept:\(token)")
                    throw StepFailure()
                }, onToken: nil)
        }
        #expect(await trace.events.contains("acceptCancelled:6"))
    }

    @Test func stopFromDeliveryEndsAfterTheOverlappingStep() async throws {
        let trace = Trace()
        let output = try await LanguageDecodeLoop.generate(prompt: [0], maximumTokens: 8, stopTokenIDs: [], vocabularySize: 8,
            sampling: .greedy, processor: nil, prefill: { row(after: 0, vocabulary: 8) }, step: { token, _ in
                await trace.add("step:\(token)")
                return row(after: token, vocabulary: 8)
            }, onToken: { token in token.id == 3 ? .stop : .continue })
        #expect(output.tokens == [1, 2, 3] && output.stopReason == .stopped)
        #expect(await trace.events == ["step:1", "step:2", "step:3"], "the step beside the stopping delivery joined; none followed")
        // A stop on the last token is still a stop.
        let last = try await LanguageDecodeLoop.generate(prompt: [0], maximumTokens: 2, stopTokenIDs: [], vocabularySize: 8,
            sampling: .greedy, processor: nil, prefill: { row(after: 0, vocabulary: 8) }, step: { token, _ in row(after: token, vocabulary: 8) },
            onToken: { token in token.index == 1 ? .stop : .continue })
        #expect(last.tokens == [1, 2] && last.stopReason == .stopped)
    }

    @Test func emptyConstraintFailsTheRequest() async throws {
        struct Nothing: LanguageLogitsProcessor {
            func begin(prompt: [Int]) async throws -> LanguageTokenConstraint { .unconstrained }
            func accept(_ token: Int) async throws -> LanguageTokenConstraint { .allowed(LanguageTokenSet(vocabularySize: 8)) }
        }
        await #expect(throws: LanguageGenerationError.invalid("The constraint allows no token")) {
            _ = try await LanguageDecodeLoop.generate(prompt: [0], maximumTokens: 4, stopTokenIDs: [], vocabularySize: 8,
                sampling: .greedy, processor: Nothing(), prefill: { row(after: 0, vocabulary: 8) },
                step: { token, _ in row(after: token, vocabulary: 8) }, onToken: nil)
        }
    }

    /// The default request (greedy, no penalty, no processor) emits exactly what the greedy loop does when its step
    /// folds in the original argmax, on seeded random rows with ties, stop tokens and deliveries.
    @Test func defaultPathMatchesTheGreedyLoop() async throws {
        for seed in UInt64(0)..<200 {
            var generator = LanguageRandomGenerator(seed: seed)
            let vocabulary = Int(generator.next() % 40) + 2
            let rows = (0..<65).map { _ in (0..<vocabulary).map { _ in Float(Int(generator.next() % 16)) / 4 } }
            let stops: Set<Int> = generator.next() % 2 == 0 ? [Int(generator.next() % UInt64(vocabulary))] : []
            let limit = Int(generator.next() % 64) + 1
            let prompt = [Int(generator.next() % UInt64(vocabulary))]
            let old = try await LanguageDecodeLoop.run(first: try LanguageSampler.greedy(rows[0]), prompt: 1, maximumTokens: limit,
                stopTokenIDs: stops, step: { _, position in try LanguageSampler.greedy(rows[position]) }, onToken: { _ in })
            let new = try await LanguageDecodeLoop.generate(prompt: prompt, maximumTokens: limit, stopTokenIDs: stops,
                vocabularySize: vocabulary, sampling: .greedy, processor: nil, prefill: { rows[0] }, step: { _, position in rows[position] },
                onToken: { _ in .continue })
            #expect(new.tokens == old.tokens && new.stopReason.rawValue == old.reason, "seed \(seed)")
            #expect(new.timeToFirstTokenSeconds >= 0 && new.totalSeconds >= new.timeToFirstTokenSeconds)
        }
    }

    @Test func randomSamplingReportsAndReplaysItsSeed() async throws {
        func run(_ sampling: LanguageSampling) async throws -> LanguageDecodeLoop.Output {
            try await LanguageDecodeLoop.generate(prompt: [0], maximumTokens: 32, stopTokenIDs: [], vocabularySize: 16,
                sampling: sampling, processor: nil, prefill: { [Float](repeating: 0, count: 16) },
                step: { _, _ in [Float](repeating: 0, count: 16) }, onToken: nil)
        }
        let first = try await run(.random())
        let seed = try #require(first.seed)
        let replay = try await run(.random(seed: seed))
        #expect(replay.tokens == first.tokens && replay.seed == seed)
    }
}
