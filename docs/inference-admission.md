# Shared inference admission

`AetherKit.FairInferenceQueue<Owner, Work>` provides bounded, transport-independent
admission for native inference consumers. It is a Sendable value owned by the
consumer's coordination actor. Aether Kit does not interpret Owner as a credential
or grant permissions; consumers authorize work before admitting it.

The queue counts queued and running tickets against global/per-owner outstanding
limits, and separately bounds active work. It serves each owner's tickets in FIFO
order and owners in round-robin order. A new owner arriving while a burst owner
runs gets the next turn. For Shilo's host consumer, Owner is the user rather than
the device, so adding devices cannot multiply a user's compute allowance.

```swift
var queue = try FairInferenceQueue<UUID, MyRequest>(
    maximumOutstanding: 32,
    maximumOutstandingPerOwner: 8,
    maximumRunning: 1
)
try queue.enqueue(id: requestID, owner: authenticatedUserID, work: request)
if let ticket = queue.next() {
    // Execute through a Kit generation session; join native submissions.
    // Persist the consumer's result before releasing the runtime slot.
    try queue.release(id: ticket.id)
}
```

`cancel(id:)` removes queued work immediately. For running work it returns
`drainRunning` and deliberately retains the ticket and slot. The consumer cancels
its runtime task, awaits native completion/error cleanup, then calls `release`.
Repeated cancellation does not release that slot. The queue itself does not own,
start, cancel or unload native models.

`InferenceAdmissionError` distinguishes invalid configuration, duplicate IDs,
global/per-owner limits and attempts to release work that is not running. CPU
checks cover fairness/FIFO, new-owner admission, limits including running work,
queued cancellation and running cancellation/drain. These are infrastructure
checks, not model-quality or sustained-performance qualification.
