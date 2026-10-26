import AetherKit
import Foundation
import Testing

// Scheduling cases carried over with the adopted FairInferenceQueue design.
struct FairInferenceQueueTests {
    @Test func roundRobinPreservesPerOwnerFIFO() throws {
        var queue = try FairInferenceQueue<String, Int>()
        for (owner, value) in [("a", 1), ("a", 2), ("b", 3), ("b", 4), ("c", 5)] {
            try queue.enqueue(id: UUID(), owner: owner, work: value)
        }
        var order: [Int] = []
        while let ticket = queue.next() { order.append(ticket.work); try queue.release(id: ticket.id) }
        #expect(order == [1, 3, 5, 2, 4])
        #expect(queue.outstandingCount == 0)
    }

    @Test func newOwnerGetsTurnAfterRunningBurstOwner() throws {
        var queue = try FairInferenceQueue<String, Int>()
        try queue.enqueue(id: UUID(), owner: "a", work: 1)
        let started = queue.next()
        let first = try #require(started)
        try queue.enqueue(id: UUID(), owner: "a", work: 2)
        try queue.enqueue(id: UUID(), owner: "b", work: 3)
        try queue.release(id: first.id)
        #expect(queue.next()?.work == 3)
    }

    @Test func cancelledRunningWorkRetainsSlotUntilDrain() throws {
        var queue = try FairInferenceQueue<String, Int>(maximumOutstanding: 3, maximumOutstandingPerOwner: 2)
        let active = UUID()
        try queue.enqueue(id: active, owner: "a", work: 1)
        try queue.enqueue(id: UUID(), owner: "b", work: 2)
        #expect(queue.next()?.id == active)
        #expect(queue.cancel(id: active) == .drainRunning)
        #expect(queue.runningCount == 1 && queue.next() == nil)
        try queue.release(id: active)
        #expect(queue.next()?.work == 2)
    }

    @Test func limitsIncludeRunningWorkAndCancellationReturnsAdmission() throws {
        var queue = try FairInferenceQueue<String, Int>(maximumOutstanding: 2, maximumOutstandingPerOwner: 1)
        let a = UUID(), b = UUID()
        try queue.enqueue(id: a, owner: "a", work: 1)
        #expect(queue.next()?.id == a)
        #expect(throws: InferenceAdmissionError.ownerLimit) { try queue.enqueue(id: UUID(), owner: "a", work: 2) }
        try queue.enqueue(id: b, owner: "b", work: 2)
        #expect(throws: InferenceAdmissionError.globalLimit) { try queue.enqueue(id: UUID(), owner: "c", work: 3) }
        #expect(queue.cancel(id: b) == .removedQueued)
        try queue.enqueue(id: UUID(), owner: "c", work: 3)
        #expect(queue.queuedCount == 1)
        #expect(throws: InferenceAdmissionError.notRunning) { try queue.release(id: b) }
    }

    @Test func duplicateAndInvalidConfigurationFailWithoutMutation() throws {
        #expect(throws: InferenceAdmissionError.invalidLimits) { try FairInferenceQueue<String, Int>(maximumRunning: 0) }
        #expect(throws: InferenceAdmissionError.invalidLimits) { try FairInferenceQueue<String, Int>(maximumOutstanding: 1, maximumRunning: 2) }
        var queue = try FairInferenceQueue<String, Int>()
        let id = UUID()
        try queue.enqueue(id: id, owner: "a", work: 1)
        #expect(throws: InferenceAdmissionError.duplicateID) { try queue.enqueue(id: id, owner: "b", work: 9) }
        #expect(queue.outstandingCount == 1 && queue.next()?.work == 1)
    }

    @Test func severalRunningSlotsStartInFairOrder() throws {
        var queue = try FairInferenceQueue<String, Int>(maximumRunning: 2)
        for (owner, value) in [("a", 1), ("a", 2), ("b", 3)] { try queue.enqueue(id: UUID(), owner: owner, work: value) }
        let started = [queue.next(), queue.next()]
        let first = try #require(started[0]), second = try #require(started[1])
        #expect(first.work == 1 && second.work == 3)
        #expect(queue.next() == nil && queue.runningCount == 2 && queue.queuedCount == 1)
        try queue.release(id: first.id)
        #expect(queue.next()?.work == 2)
    }
}
