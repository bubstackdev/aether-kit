import os

/// `Mutex`'s shape (`init(_:)`, `withLock`) over `OSAllocatedUnfairLock`, which is available from macOS 13 and iOS 16:
/// `Synchronization.Mutex` needs macOS 15 and iOS 18, above the app-facing language facade's floor. Like `Mutex`, it
/// lets the protected state be non-Sendable; the lock is what makes access safe.
struct UnfairLock<State>: @unchecked Sendable {
    private let lock: OSAllocatedUnfairLock<State>

    init(_ initialState: State) {
        lock = OSAllocatedUnfairLock(uncheckedState: initialState)
    }

    func withLock<Result>(_ body: (inout State) throws -> Result) rethrows -> Result {
        try lock.withLockUnchecked(body)
    }
}
