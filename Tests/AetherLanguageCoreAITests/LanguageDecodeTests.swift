@testable import AetherLanguageCoreAI
import Testing

private actor DecodeTrace {
    private(set) var events: [String] = []
    func add(_ event: String) { events.append(event) }
    /// Bounded, so a serial regression fails the expectation instead of hanging.
    func observed(_ event: String, within limit: Duration = .seconds(5)) async -> Bool {
        let deadline = ContinuousClock.now + limit
        while !events.contains(event) {
            guard ContinuousClock.now < deadline else { return false }
            try? await Task.sleep(for: .milliseconds(5))
        }
        return true
    }
}

private struct StepFailure: Error, Equatable {}
private struct DeliveryFailure: Error, Equatable {}

struct LanguageDecodeTests {
    // Each non-final delivery waits for the following step to begin. A serial loop
    // cannot start that step until delivery returns, so every wait would miss.
    @Test func deliveryOverlapsNextStepInOrder() async throws {
        let trace = DecodeTrace()
        let result = try await LanguageDecodeLoop.run(first: 1, prompt: 10, maximumTokens: 4, stopTokenIDs: [0], step: { token, position in
            await trace.add("step:\(token)@\(position)")
            return token + 1
        }, onToken: { token in
            if token < 4, await !trace.observed("step:\(token)@\(9 + token)") { await trace.add("serial:\(token)") }
            await trace.add("delivered:\(token)")
        })
        #expect(result.tokens == [1, 2, 3, 4] && result.reason == "output-limit")
        let events = await trace.events
        #expect(events.filter { $0.hasPrefix("delivered") } == ["delivered:1", "delivered:2", "delivered:3", "delivered:4"])
        #expect(events.filter { $0.hasPrefix("step") } == ["step:1@10", "step:2@11", "step:3@12"])
        #expect(!events.contains { $0.hasPrefix("serial") })
    }

    @Test func deliveryFailureSurfacesAfterOverlappingStepJoins() async throws {
        let trace = DecodeTrace()
        await #expect(throws: DeliveryFailure()) {
            _ = try await LanguageDecodeLoop.run(first: 1, prompt: 0, maximumTokens: 8, stopTokenIDs: [], step: { token, _ in
                await trace.add("stepEnd:\(token)")
                return token + 1
            }, onToken: { token in
                if token == 2 { throw DeliveryFailure() }
            })
        }
        // The step submitted alongside the failed delivery joined; nothing followed it.
        #expect(await trace.events == ["stepEnd:1", "stepEnd:2"])
    }

    @Test(.timeLimit(.minutes(1))) func stepFailureCancelsAndAwaitsDelivery() async throws {
        let trace = DecodeTrace()
        await #expect(throws: StepFailure()) {
            _ = try await LanguageDecodeLoop.run(first: 1, prompt: 0, maximumTokens: 8, stopTokenIDs: [], step: { token, _ in
                if token == 2 { throw StepFailure() }
                return token + 1
            }, onToken: { token in
                guard token == 2 else { return }
                do { try await Task.sleep(for: .seconds(30)) } catch { await trace.add("deliveryCancelled:2") }
            })
        }
        #expect(await trace.events == ["deliveryCancelled:2"])
    }

    @Test func stopTokenAndCallbackFreeDecodingMatchSerialResults() async throws {
        let trace = DecodeTrace()
        let stopped = try await LanguageDecodeLoop.run(first: 1, prompt: 0, maximumTokens: 8, stopTokenIDs: [9], step: { _, _ in 9 },
            onToken: { await trace.add("delivered:\($0)") })
        #expect(stopped.tokens == [1] && stopped.reason == "eos")
        #expect(await trace.events == ["delivered:1"])
        let silent = try await LanguageDecodeLoop.run(first: 5, prompt: 0, maximumTokens: 3, stopTokenIDs: [], step: { token, _ in token + 1 }, onToken: nil)
        #expect(silent.tokens == [5, 6, 7] && silent.reason == "output-limit")
    }
}
