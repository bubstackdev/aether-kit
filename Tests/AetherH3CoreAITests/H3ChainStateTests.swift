import AetherKit
import Testing
@testable import AetherH3CoreAI

/// A stand-in device buffer: identity is what matters.
private final class FakeBuffer: Sendable {
    let id: Int
    init(_ id: Int) { self.id = id }
}

@Test func h3ChainStatePingPongsTwoSlotsAndBindsThePredecessor() throws {
    var chain = H3ChainState<FakeBuffer>()
    var made = 0
    func make(_ old: FakeBuffer?) -> FakeBuffer { if let old { return old }; made += 1; return FakeBuffer(made) }
    // Block 0: copied input, writes slot 0.
    #expect(try chain.begin(chainedInput: false, chaining: true) == nil)
    let first = chain.output(make)
    chain.complete()
    // Block 1 reads block 0's buffer and writes the other slot.
    #expect(try chain.begin(chainedInput: true, chaining: true) === first)
    let second = chain.output(make)
    #expect(second !== first)
    chain.complete()
    // Block 2 reads block 1's buffer and reuses slot 0: two buffers alternate.
    #expect(try chain.begin(chainedInput: true, chaining: true) === second)
    #expect(chain.output(make) === first)
    chain.complete()
    #expect(made == 2 && chain.valid)
}

@Test func h3ChainStateRefusesAChainedInputWithoutAValidPredecessor() throws {
    var chain = H3ChainState<FakeBuffer>()
    // No predecessor yet.
    #expect(throws: AetherError.self) { _ = try chain.begin(chainedInput: true, chaining: true) }
    _ = try chain.begin(chainedInput: false, chaining: true)
    _ = chain.output { _ in FakeBuffer(1) }
    chain.complete()
    // A chained input on a call that is not itself chained is refused.
    #expect(throws: AetherError.self) { _ = try chain.begin(chainedInput: true, chaining: false) }
    // The refusals left the valid predecessor in place.
    #expect(chain.valid)
    #expect(try chain.begin(chainedInput: true, chaining: true)?.id == 1)
}

@Test func h3ChainStateIsInvalidatedByACallThatFailsOrIsCancelled() throws {
    var chain = H3ChainState<FakeBuffer>()
    _ = try chain.begin(chainedInput: false, chaining: true)
    _ = chain.output { _ in FakeBuffer(1) }
    chain.complete()
    // The next chained call starts but never completes (a native error or cancellation).
    _ = try chain.begin(chainedInput: true, chaining: true)
    _ = chain.output { _ in FakeBuffer(2) }
    #expect(!chain.valid)
    #expect(throws: AetherError.self) { _ = try chain.begin(chainedInput: true, chaining: true) }
    // A plain (non-chained) call on the same graph also invalidates a valid chain.
    var other = H3ChainState<FakeBuffer>()
    _ = try other.begin(chainedInput: false, chaining: true)
    _ = other.output { _ in FakeBuffer(3) }
    other.complete()
    _ = try other.begin(chainedInput: false, chaining: false)
    #expect(throws: AetherError.self) { _ = try other.begin(chainedInput: true, chaining: true) }
    // The next step restarts from a copied input.
    _ = try chain.begin(chainedInput: false, chaining: true)
    _ = chain.output { $0 ?? FakeBuffer(4) }
    chain.complete()
    #expect(chain.valid)
}
