import AetherKit

/// Device-side chaining state for one graph: two alternating output slots, and whether the last chained call's output
/// is still valid to bind as the next call's input. Only a completed chained call validates it; starting any call
/// invalidates it, so a failed or cancelled call (which never completes) breaks the chain.
@available(macOS 27, iOS 27, *)
struct H3ChainState<Buffer> {
    private(set) var slots: [Buffer?] = [nil, nil]
    private(set) var current = 0
    private(set) var valid = false

    /// Starts a call. A `.chained` input needs a valid predecessor and a chained call; it gets the predecessor's slot.
    mutating func begin(chainedInput: Bool, chaining: Bool) throws -> Buffer? {
        guard !chainedInput || (valid && chaining) else { throw AetherError.invalidRequest("H3 chained input without a chained predecessor") }
        valid = false
        guard chainedInput else { return nil }
        guard let previous = slots[1 - current] else { throw AetherError.invalidRequest("H3 chained input without a chained predecessor") }
        return previous
    }

    /// The buffer this chained call writes: the current slot, reused or replaced by `make`.
    mutating func output(_ make: (Buffer?) throws -> Buffer) rethrows -> Buffer {
        let buffer = try make(slots[current])
        slots[current] = buffer
        return buffer
    }

    /// A chained call finished: its output becomes the predecessor and the other slot is written next.
    mutating func complete() {
        current = 1 - current
        valid = true
    }
}
