import AetherHost
import Foundation
import Testing

// Scheduling cases carried over from 0.0.7's fair queue, re-expressed for priority order.
struct PriorityInferenceQueueTests {
    @Test func equalPrioritiesStartInArrivalOrderWhateverTheOwner() throws {
        var queue = try PriorityInferenceQueue<String, Int>()
        for (owner, value) in [("a", 1), ("a", 2), ("b", 3), ("b", 4), ("c", 5)] {
            try queue.enqueue(id: UUID(), owner: owner, work: value)
        }
        var order: [Int] = []
        while let ticket = queue.next() { order.append(ticket.work); try queue.release(id: ticket.id) }
        #expect(order == [1, 2, 3, 4, 5])
        #expect(queue.outstandingCount == 0)
    }

    @Test func highJumpsTheLineAndLowWaitsForEverythingElse() throws {
        var queue = try PriorityInferenceQueue<String, Int>()
        let running = UUID()
        try queue.enqueue(id: running, owner: "a", priority: .low, work: 0)
        #expect(queue.next()?.id == running)
        for (priority, value) in [(GPUPriority.low, 1), (.medium, 2), (.low, 3), (.high, 4), (.medium, 5), (.high, 6)] {
            try queue.enqueue(id: UUID(), owner: "a", priority: priority, work: value)
        }
        // A high request never preempts the running one.
        #expect(queue.next() == nil && queue.runningCount == 1)
        try queue.release(id: running)
        var order: [Int] = []
        while let ticket = queue.next() { order.append(ticket.work); try queue.release(id: ticket.id) }
        #expect(order == [4, 6, 2, 5, 1, 3])
    }

    @Test func aQueuedTicketCanBeReprioritizedButARunningOneCannot() throws {
        var queue = try PriorityInferenceQueue<String, Int>()
        let running = UUID(), late = UUID()
        try queue.enqueue(id: running, owner: "a", work: 0)
        #expect(queue.next()?.id == running)
        let changedRunning = queue.setPriority(.high, of: running)
        #expect(!changedRunning)
        try queue.enqueue(id: UUID(), owner: "a", work: 1)
        try queue.enqueue(id: late, owner: "a", priority: .low, work: 2)
        try queue.enqueue(id: UUID(), owner: "a", priority: .high, work: 3)
        let changedQueued = queue.setPriority(.high, of: late), changedUnknown = queue.setPriority(.high, of: UUID())
        #expect(changedQueued && !changedUnknown)
        try queue.release(id: running)
        var order: [Int] = []
        while let ticket = queue.next() { order.append(ticket.work); try queue.release(id: ticket.id) }
        // Raised to high, it keeps its arrival order there: ahead of the later high, behind none.
        #expect(order == [2, 3, 1])
    }

    @Test func cancelledRunningWorkRetainsSlotUntilDrain() throws {
        var queue = try PriorityInferenceQueue<String, Int>(maximumOutstanding: 3, maximumOutstandingPerOwner: 2)
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
        var queue = try PriorityInferenceQueue<String, Int>(maximumOutstanding: 2, maximumOutstandingPerOwner: 1)
        let a = UUID(), b = UUID()
        try queue.enqueue(id: a, owner: "a", work: 1)
        #expect(queue.next()?.id == a)
        #expect { try queue.enqueue(id: UUID(), owner: "a", work: 2) } throws: { admissionFailure($0) == .ownerLimit }
        try queue.enqueue(id: b, owner: "b", work: 2)
        #expect { try queue.enqueue(id: UUID(), owner: "c", work: 3) } throws: { admissionFailure($0) == .globalLimit }
        #expect(queue.cancel(id: b) == .removedQueued)
        try queue.enqueue(id: UUID(), owner: "c", work: 3)
        #expect(queue.queuedCount == 1)
        #expect { try queue.release(id: b) } throws: { admissionFailure($0) == .notRunning }
    }

    @Test func duplicateAndInvalidConfigurationFailWithoutMutation() throws {
        #expect { try PriorityInferenceQueue<String, Int>(maximumRunning: 0) } throws: { admissionFailure($0) == .invalidLimits }
        #expect { try PriorityInferenceQueue<String, Int>(maximumOutstanding: 1, maximumRunning: 2) } throws: { admissionFailure($0) == .invalidLimits }
        var queue = try PriorityInferenceQueue<String, Int>()
        let id = UUID()
        try queue.enqueue(id: id, owner: "a", work: 1)
        #expect { try queue.enqueue(id: id, owner: "b", work: 9) } throws: { admissionFailure($0) == .duplicateID }
        #expect(queue.outstandingCount == 1 && queue.next()?.work == 1)
    }

    @Test func severalRunningSlotsStartInPriorityOrder() throws {
        var queue = try PriorityInferenceQueue<String, Int>(maximumRunning: 2)
        for (priority, value) in [(GPUPriority.low, 1), (.medium, 2), (.high, 3)] {
            try queue.enqueue(id: UUID(), owner: "a", priority: priority, work: value)
        }
        let started = [queue.next(), queue.next()]
        let first = try #require(started[0]), second = try #require(started[1])
        #expect(first.work == 3 && second.work == 2)
        #expect(queue.next() == nil && queue.runningCount == 2 && queue.queuedCount == 1)
        try queue.release(id: first.id)
        #expect(queue.next()?.work == 1)
    }
}
