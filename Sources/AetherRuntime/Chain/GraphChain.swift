import Foundation
import CoreAI
import Metal

/// A sequence of graph calls where each call's output is the next call's input, without host copies (DESIGN.md §6.6,
/// lab/h3-spike §1). The 0.0.x prototype's H3 chain, made an explicit value.
///
/// The chain owns two **slots**, buffers of the linked tensor, and binds them alternately: call *i* reads the
/// ``head`` (slot *i* mod 2 before the call) as its `link.input` and writes the other slot as its `link.output`; the
/// output becomes the head. Two slots, never one: an output written over its own input gave the right bytes in the
/// spike, but nothing documents it as legal, and ``Bindings`` refuses binding one storage twice anyway.
///
/// Two modes, picked by the arena:
/// - **Device** (a Metal ``TensorArena``): every call is *encoded* on one `ComputeStream`, which orders consecutive
///   calls, so the chain waits once, when ``run(_:)`` ends, or at a ``synchronize()`` the caller places. The arena is
///   lent for the whole run.
/// - **Host** (a host arena): every call is `run()` and completes before the next. This is the unchained reference
///   path (the bytes go through host memory each call), and the path fakes take in tests.
///
/// Both modes keep the same bookkeeping: which buffers the calls since the last ``synchronize()`` bound. A weight bank
/// (``WeightBanks``) refuses to refill a buffer in that set, the sync rule the spike found: wait before overwriting
/// a buffer the GPU may still read.
///
/// Write the first input into ``head`` before ``run(_:)``; read the result from ``head`` after it. Non-`Sendable`:
/// the chain belongs to the actor that owns its graphs.
public final class GraphChain {
    public let arena: TensorArena
    /// The input and output names linked from call to call.
    public let link: (input: String, output: String)
    /// The two buffers the link alternates between.
    public let slots: (TensorBuffer, TensorBuffer)
    /// The gate runs take their turns through. Chains that share graphs share a gate.
    public let gate: SerialGate
    /// When set, every call's input shapes must be in this profile (``call(_:as:bind:)``).
    public let profile: ShapeProfile?

    /// The buffer the next call reads: the last call's output, or the first input.
    public private(set) var head: TensorBuffer
    /// Calls made in the current (or last) run.
    public private(set) var calls = 0

    private let gpu: (queue: any MTLCommandQueue, stream: ComputeStream)?
    /// The arena slots of the storage bound by calls since the last synchronization.
    private var inFlight: Set<Int> = []
    private var running = false

    /// A chain over two slots of the linked tensor (distinct buffers of `arena`, same shape and scalar type).
    public init(
        arena: TensorArena, link: (input: String, output: String), slots: (TensorBuffer, TensorBuffer),
        gate: SerialGate = SerialGate(), profile: ShapeProfile? = nil
    ) throws(AetherError) {
        try arena.checkOwned(slots.0)
        try arena.checkOwned(slots.1)
        guard !slots.0.isSameStorage(as: slots.1) else {
            throw .tensor(name: link.output, reason: .unsupported("a chain needs two distinct slots"))
        }
        guard slots.0.shape == slots.1.shape, slots.0.scalarType == slots.1.scalarType else {
            throw .tensor(name: link.output, reason: .shapeMismatch(expected: slots.0.shape, found: slots.1.shape))
        }
        self.arena = arena
        self.link = link
        self.slots = slots
        self.gate = gate
        self.profile = profile
        head = slots.0
        if let device = arena.device {
            guard let queue = device.makeCommandQueue() else {
                throw .runtime(underlying: CocoaError(.featureUnsupported, userInfo: [NSLocalizedDescriptionKey: "cannot make a command queue"]))
            }
            gpu = (queue, ComputeStream(commandQueue: queue))
        } else {
            gpu = nil
        }
    }

    /// A chain whose slots are allocated for `graph`'s `link.output`, with its dynamic dimensions from `dimensions`.
    public convenience init(
        graph: any GraphBackend, link: (input: String, output: String), dimensions: [Int] = [], arena: TensorArena,
        gate: SerialGate = SerialGate(), profile: ShapeProfile? = nil
    ) throws(AetherError) {
        let a = try graph.makeBuffer(for: link.output, role: .output, dimensions: dimensions, in: arena)
        let b = try graph.makeBuffer(for: link.output, role: .output, dimensions: dimensions, in: arena)
        try self.init(arena: arena, link: link, slots: (a, b), gate: gate, profile: profile)
    }

    /// Whether calls are encoded on the GPU (a Metal arena) rather than run one by one.
    public var isDeviceChained: Bool { gpu != nil }

    /// Runs `body`, which makes the calls, in the chain's turn. The head is not reset: a run continues from the
    /// current head, so set it up (write into ``head``, or ``reset()``) before.
    ///
    /// On every exit, a cancellation or an error included, the run waits for all encoded work before it returns and
    /// the arena is lent back: no buffer is ever handed back while the GPU may still use it. Cancelling the calling
    /// task stops before the next call with `CancellationError`.
    public func run<R>(_ body: (GraphChain) async throws -> R) async throws -> R {
        try await gate.withTurn {
            try Task.checkCancellation()
            calls = 0
            inFlight.removeAll()
            if gpu != nil { try arena.beginRun("chain") }
            running = true
            defer { running = false }
            do {
                let result = try await body(self)
                await drain()
                // Drained: nothing the run bound is still read, so a bank may be refilled before the next run.
                inFlight.removeAll()
                if gpu != nil { arena.endRun() }
                return result
            } catch {
                await drain()
                inFlight.removeAll()
                if gpu != nil { arena.endRun() }
                throw error
            }
        }
    }

    /// Makes the head slot 0 again (between runs).
    public func reset() throws(AetherError) {
        guard !running else { throw .tensor(name: link.input, reason: .unavailable("the chain is running")) }
        head = slots.0
    }

    /// Makes one call: binds the head as `link.input` and the other slot as `link.output`, lets `bind` add the
    /// call's other inputs (its weights, say), and encodes (device) or runs (host) it. The output becomes the head.
    ///
    /// - `role` names the graph in ``profile``; with a profile, input shapes outside it are refused before anything
    ///   runs (a measured footprint holds only for the shapes it was measured at).
    /// - `bind` must not rebind the link names.
    public func call(
        _ graph: any GraphBackend, as role: String = "", bind: (inout Bindings) throws(AetherError) -> Void = { _ in }
    ) async throws {
        guard running else { throw AetherError.tensor(name: link.input, reason: .unavailable("call a chain inside run(_:)")) }
        try Task.checkCancellation()
        let target = head.isSameStorage(as: slots.0) ? slots.1 : slots.0
        var bindings = gpu != nil ? Bindings(lentArena: arena) : Bindings(arena: arena)
        try bindings.bindInput(link.input, to: head)
        try bindings.bindOutput(link.output, to: target)
        try bind(&bindings)
        guard let input = bindings.input(link.input), input.isSameStorage(as: head),
              let output = bindings.output(link.output), output.isSameStorage(as: target)
        else {
            throw AetherError.tensor(name: link.input, reason: .unsupported("the chain binds '\(link.input)' and '\(link.output)'"))
        }
        if let profile { try profile.check(role, bindings) }
        if let gpu {
            guard let graph = graph as? Graph else {
                throw AetherError.tensor(name: link.input, reason: .unsupported("device chaining encodes Core AI graphs"))
            }
            try graph.encode(bindings, to: gpu.stream)
        } else {
            try await graph.run(&bindings)
        }
        for bound in bindings.inputs + bindings.states + bindings.outputs { inFlight.insert(bound.buffer.slot) }
        head = target
        calls += 1
    }

    /// Waits until every call made so far has completed (a sync point). Needed before a weight bank refills a buffer
    /// an earlier call bound; otherwise the stream orders calls by itself.
    public func synchronize() async {
        await drain()
        inFlight.removeAll()
    }

    /// Whether a call since the last ``synchronize()`` bound `buffer` (so the GPU may still read it).
    public func isInFlight(_ buffer: TensorBuffer) -> Bool { inFlight.contains(buffer.slot) }

    private func drain() async {
        guard let gpu else { return }
        // Core AI forces this: `ComputeStream` is not Sendable and `currentWorkCompleted()` is a concurrent async
        // method. Sound for the reason GreedyKernels.drain gives: only this chain encodes to its stream, inside the
        // owning actor, and every path that waits here holds the chain's turn, so nothing encodes meanwhile.
        nonisolated(unsafe) let stream = gpu.stream
        await stream.currentWorkCompleted()
        guard let sentinel = gpu.queue.makeCommandBuffer() else { return }
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            sentinel.addCompletedHandler { _ in done.resume() }
            sentinel.commit()
        }
    }
}
