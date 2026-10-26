/// Greedy decode control flow shared by the native session and CPU tests. It runs
/// on the caller's actor, so `step` may use actor-owned native state directly.
enum LanguageDecodeLoop {
    /// `step(token, position)` submits one token, joins its native work and returns
    /// the next token. Each emitted token is delivered while the following step runs
    /// on the GPU. Deliveries stay ordered with at most one in flight, and every
    /// delivery finishes before this returns or throws. A delivery error surfaces
    /// after the overlapping step joins; a step error cancels and awaits delivery.
    static func run(first: Int, prompt: Int, maximumTokens: Int, stopTokenIDs: Set<Int>,
                    isolation: isolated (any Actor)? = #isolation,
                    step: (_ token: Int, _ position: Int) async throws -> Int,
                    onToken: (@Sendable (Int) async throws -> Void)?) async throws -> (tokens: [Int], reason: String) {
        var next = first
        var generated: [Int] = []
        for index in 0..<maximumTokens {
            try Task.checkCancellation()
            if stopTokenIDs.contains(next) { return (generated, "eos") }
            generated.append(next)
            let token = next
            guard index + 1 < maximumTokens else {
                try await onToken?(token)
                break
            }
            guard let onToken else {
                next = try await step(token, prompt + index)
                continue
            }
            async let delivered: Void = onToken(token)
            next = try await step(token, prompt + index)
            try await delivered
        }
        return (generated, "output-limit")
    }
}
