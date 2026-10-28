import AetherH3CoreAI
import AetherKit
import Foundation
import Synchronization
import Testing
@testable import AetherProbe

private func call(_ graph: String, _ index: Int, _ phase: H3NativeCall.Phase) -> H3NativeJournal.Entry {
    .native(H3NativeCall(graph: graph, call: index, phase: phase))
}

@Test func h3CancellationChecksRequireTheTargetCallToJoinBeforeTheErrorAndNothingAfter() {
    let good: [H3NativeJournal.Entry] = [call("body", 59, .begin), call("body", 59, .end), call("body", 60, .begin), .marker("cancelRequested"),
                                         call("body", 60, .end), .marker("cancellationReturned")]
    #expect(H3NativeJournal.cancellationChecks(good, target: "body").values.allSatisfy { $0 })
    // A later submission, an unjoined call, or a call of another stage each fail their check.
    let later = good.dropLast() + [call("body", 61, .begin), call("body", 61, .end), .marker("cancellationReturned")]
    #expect(H3NativeJournal.cancellationChecks(Array(later), target: "body")["noNativeCallAfterCancellation"] == false)
    let unjoined: [H3NativeJournal.Entry] = [call("body", 60, .begin), .marker("cancelRequested"), .marker("cancellationReturned"), call("body", 60, .end)]
    #expect(H3NativeJournal.cancellationChecks(unjoined, target: "body")["inFlightCallJoinsBeforeTheError"] == false)
    #expect(H3NativeJournal.cancellationChecks(good, target: "audio")["inFlightCallIsTheTargetStage"] == false)
    // Nothing in flight at the request is not a qualifying cancellation.
    let idle: [H3NativeJournal.Entry] = [call("body", 60, .begin), call("body", 60, .end), .marker("cancelRequested"), .marker("cancellationReturned")]
    #expect(H3NativeJournal.cancellationChecks(idle, target: "body")["inFlightCallJoinsBeforeTheError"] == false)
}

@Test func h3UnloadChecksRequireTheActiveRenderToFinishBeforeUnloadReturns() {
    let good: [H3NativeJournal.Entry] = [call("body", 60, .begin), .marker("unloadRequested"), call("body", 60, .end), call("body", 61, .begin),
                                         call("body", 61, .end), call("audio", 0, .begin), call("audio", 0, .end), .marker("unloadReturned")]
    #expect(H3NativeJournal.unloadChecks(good).values.allSatisfy { $0 })
    let late = good + [call("body", 0, .begin)]
    #expect(H3NativeJournal.unloadChecks(late)["noNativeCallAfterUnload"] == false)
    let stopped: [H3NativeJournal.Entry] = [call("body", 60, .begin), .marker("unloadRequested"), call("body", 60, .end), .marker("unloadReturned")]
    #expect(H3NativeJournal.unloadChecks(stopped)["activeRenderContinuesAfterUnloadRequest"] == false)
}

@Test func h3NativeTriggerFiresOnceAtTheArmedCallsBegin() {
    let fired = Mutex(0)
    let trigger = H3NativeTrigger()
    trigger.arm(graph: "videoTile", call: 1) { fired.withLock { $0 += 1 } }
    for (graph, index, phase) in [("videoTile", 0, H3NativeCall.Phase.begin), ("videoTile", 1, .end), ("audio", 1, .begin),
                                  ("videoTile", 1, .begin), ("videoTile", 1, .begin)] {
        trigger.observe(H3NativeCall(graph: graph, call: index, phase: phase))
    }
    #expect(fired.withLock { $0 } == 1)
}

@Test func h3SessionLifecycleCommandIsAGPUCommandAndValidatesArgumentsFirst() async throws {
    #expect(ProbeGPULease.usesGPU(["qualify-h3-session-lifecycle", "p", "d", "o"]))
    #expect(ProbeGPULease.usesGPU(["h3-memory-render", "p", "d", "o"]) && ProbeGPULease.usesGPU(["h3-memory-render", "p", "d", "o", "--spike-capture=12"]))
    #expect(ProbeGPULease.usesGPU(["h3-memory-encoder", "p", "288", "288", "k.png", "o"]))
    #expect(!ProbeGPULease.usesGPU(["qualify-h3-session-lifecycle", "p", "d"]) && !ProbeGPULease.usesGPU(["h3-memory-render", "p", "d", "o", "--x"]))
    await #expect(throws: AetherError.self) { try await H3SessionLifecycleCommand.run(["qualify-h3-session-lifecycle", "p", "d"]) }
    #expect(try await H3SessionLifecycleCommand.run(["other"]) == false)
    #expect(H3SessionLifecycleCommand.cancellations(keyframe: false).map(\.graph) == ["conditioner", "body", "videoTile", "audio"])
    #expect(H3SessionLifecycleCommand.cancellations(keyframe: true).map(\.graph) == ["keyframe"])
    #expect(try H3SessionLifecycleCommand.scenarios(["qualify-h3-session-lifecycle", "p", "d", "o"]) == nil)
    #expect(try H3SessionLifecycleCommand.scenarios(["qualify-h3-session-lifecycle", "p", "d", "o", "--scenarios=cancel-audio,unload"]) == ["cancel-audio", "unload"])
    #expect(ProbeGPULease.usesGPU(["qualify-h3-session-lifecycle", "p", "d", "o", "--scenarios=reopen"]))
    for bad in ["--scenarios=", "--scenarios=cancel-everything", "--other"] {
        #expect(throws: AetherError.self) { _ = try H3SessionLifecycleCommand.scenarios(["qualify-h3-session-lifecycle", "p", "d", "o", bad]) }
        #expect(!ProbeGPULease.usesGPU(["qualify-h3-session-lifecycle", "p", "d", "o", bad]))
    }
}
