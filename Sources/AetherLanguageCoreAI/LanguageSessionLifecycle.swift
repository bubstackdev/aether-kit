/// Request admission and terminal close shared by the native session and CPU tests.
/// It is owned by, and only used on, its session's actor. One request runs at a time;
/// closing rejects new requests and waits for an active one to finish.
final class LanguageSessionLifecycle {
    private(set) var state: LanguageGenerationSession.State = .ready
    private var idle: CheckedContinuation<Void, Never>?
    private var closeWaiters: [CheckedContinuation<Void, Never>] = []

    func admit() throws {
        guard state != .closing, state != .closed else { throw LanguageGenerationError.closed }
        guard state == .ready else { throw LanguageGenerationError.busy }
        state = .running
    }

    /// Ends the admitted request, resuming a close that is waiting for it.
    func finish() {
        if state == .running { state = .ready }
        idle?.resume()
        idle = nil
    }

    /// Returns true for the one caller that must release resources and then call
    /// `didClose()`. Concurrent callers wait until that release has finished.
    func beginClose(isolation: isolated (any Actor)? = #isolation) async -> Bool {
        switch state {
        case .closed:
            return false
        case .closing:
            await withCheckedContinuation { closeWaiters.append($0) }
            return false
        case .running:
            state = .closing
            await withCheckedContinuation { idle = $0 }
            return true
        case .ready:
            state = .closing
            return true
        }
    }

    func didClose() {
        state = .closed
        closeWaiters.forEach { $0.resume() }
        closeWaiters.removeAll()
    }
}
