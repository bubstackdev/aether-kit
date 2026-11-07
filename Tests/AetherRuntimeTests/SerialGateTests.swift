import AetherCore
import Testing
@testable import AetherRuntime

/// The one serialization primitive (second code review, batch C item 5): FIFO turns, a waiter that leaves the queue the
/// moment it is cancelled, a turn released however its holder exits, and the bound.
@Suite("Serial gate")
struct SerialGateTests {
    /// Holds a turn until released.
    actor Latch {
        private var open = false
        private var waiting: [CheckedContinuation<Void, Never>] = []
        func wait() async { if !open { await withCheckedContinuation { waiting.append($0) } } }
        func release() {
            open = true
            waiting.forEach { $0.resume() }
            waiting = []
        }
    }

    /// Polls until `condition` holds (bounded; the gate's state changes on other tasks).
    static func until(_ condition: () -> Bool) async {
        for _ in 0..<10_000 where !condition() { await Task.yield() }
    }

    @Test func turnsAreGrantedInArrivalOrder() async throws {
        let gate = SerialGate()
        let latch = Latch()
        let order = Order()
        let holder = Task { try await gate.withTurn { await latch.wait() } }
        await Self.until { gate.isBusy }
        var tasks: [Task<Void, any Error>] = []
        for i in 0..<5 {
            tasks.append(Task { try await gate.withTurn { await order.append(i) } })
            await Self.until { gate.waiting == i + 1 }
        }
        await latch.release()
        try await holder.value
        for task in tasks { try await task.value }
        #expect(await order.values == [0, 1, 2, 3, 4])
        #expect(!gate.isBusy && gate.waiting == 0)
    }

    /// Before: a cancelled waiter stayed queued until its turn came (then ran, or checked and left).
    @Test func aCancelledWaiterLeavesAtOnce() async throws {
        let gate = SerialGate()
        let latch = Latch()
        let holder = Task { try await gate.withTurn { await latch.wait() } }
        await Self.until { gate.isBusy }
        let ran = Order()
        let waiter = Task { try await gate.withTurn { await ran.append(1) } }
        await Self.until { gate.waiting == 1 }
        waiter.cancel()
        await #expect(throws: CancellationError.self) { try await waiter.value }
        #expect(gate.waiting == 0 && gate.isBusy)  // left while the holder still holds
        await latch.release()
        try await holder.value
        #expect(await ran.values.isEmpty && !gate.isBusy)
        // Already cancelled on arrival: refused even when the gate is free.
        let late = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await gate.withTurn { 1 }
        }
        await #expect(throws: CancellationError.self) { try await late.value }
        #expect(!gate.isBusy)
    }

    @Test func aCancelledHolderReleasesItsTurn() async throws {
        let gate = SerialGate()
        let holder = Task {
            try await gate.withTurn {
                while true { try Task.checkCancellation(); await Task.yield() }
            }
        }
        await Self.until { gate.isBusy }
        let next = Task { try await gate.withTurn { 7 } }
        await Self.until { gate.waiting == 1 }
        holder.cancel()
        await #expect(throws: CancellationError.self) { try await holder.value }
        #expect(try await next.value == 7)
        #expect(!gate.isBusy)
    }

    @Test func theBoundRefusesOneMoreWaiter() async throws {
        let gate = SerialGate(limit: 1)
        let latch = Latch()
        let holder = Task { try await gate.withTurn { await latch.wait() } }
        await Self.until { gate.isBusy }
        let queued = Task { try await gate.withTurn { 1 } }
        await Self.until { gate.waiting == 1 }
        await #expect { try await gate.withTurn { 2 } } throws: { error in
            if case AetherError.admission(.queueFull(limit: 1)) = error { return true }
            return false
        }
        // An unload ignores the bound and cancellation, and waits its turn.
        let unload = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return await gate.withTurnIgnoringCancellation { 3 }
        }
        await Self.until { gate.waiting == 2 }
        await latch.release()
        try await holder.value
        #expect(try await queued.value == 1)
        #expect(await unload.value == 3)
        #expect(!gate.isBusy && gate.waiting == 0)
    }

    actor Order {
        private(set) var values: [Int] = []
        func append(_ value: Int) { values.append(value) }
    }
}
