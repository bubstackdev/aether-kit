import CoreAI
import Foundation
import Metal

/// The `decoder-lm@1` engine's implementation (DESIGN.md §6.2). Non-`Sendable`: exactly one ``DecoderEngine``
/// actor owns it, and its graphs, arena and buffers never leave that actor (§11). With
/// `NonisolatedNonsendingByDefault` its async methods run on the actor's executor.
final class DecoderCore {
    /// One function of the decoder graph (`step`, optionally `prefill`) and the buffers bound to its logits.
    final class Phase {
        let kind: PrefillPlan.Phase
        /// Replaced only by a reload (``DecoderCore/unloadGraphs()``, ``DecoderCore/installGraphs(step:prefill:)``).
        fileprivate(set) var graph: any GraphBackend
        /// The static token axis, or nil when dynamic.
        let staticTokens: Int?
        /// The static logits row axis, or nil when dynamic (one row per fed token).
        let staticRows: Int?
        /// Logits for the largest call this phase takes.
        let logits: TensorBuffer

        init(kind: PrefillPlan.Phase, graph: any GraphBackend, staticTokens: Int?, staticRows: Int?, logits: TensorBuffer) {
            self.kind = kind
            self.graph = graph
            self.staticTokens = staticTokens
            self.staticRows = staticRows
            self.logits = logits
        }

        /// Whether a call feeding `fed` tokens returns one row per token.
        func rowsAll(fed: Int) -> Bool { staticRows == nil || staticRows == fed }
    }

    struct Roles {
        var tokens: String
        var positions: String
        var logits: String
    }

    struct SequenceState {
        var name: String
        var seqAxis: Int
    }

    let roles: Roles
    /// `binding: static` inputs without a `source` (a VLM's image rows and rotary inputs): role → (input name, its one
    /// buffer at the descriptor's static shape), bound whole on every call. Their owner writes them while idle.
    let statics: [String: (name: String, buffer: TensorBuffer)]
    /// Token ids past the head's rows that the graph takes (`inputEncodings.tokens.extensionIds`: image tokens).
    let extensionIds: Range<Int>
    let step: Phase
    let prefillPhase: Phase?
    let vocab: Int
    let logitsType: ScalarType
    let sequenceStates: [SequenceState]
    let recurrentNames: [String]
    /// The token limit (an explicit capacity, or what the budget fits); a static sequence axis may be larger.
    let maxCapacity: Int
    let staticCapacity: Bool
    let shape: PrefillPlan.Shape
    let padToken: Int32
    let padPosition: Int32
    let arena: TensorArena
    /// Present in encode mode: the stream, its queue and the argmax kernel.
    let kernels: GreedyKernels?
    /// Token history (prompt tokens and fed tokens) and the constant positions `0..<maxCapacity`.
    let history: TensorBuffer
    let positions: TensorBuffer
    /// Encode mode: one small token slot and one `[1,1,V]` logits buffer per in-flight decode step.
    let slots: [TensorBuffer]
    let ringLogits: [TensorBuffer]

    /// Diagnostics: hand tokens through views into the (page-aligned) token history, the lab's racing hand-off.
    let racingHandoff: Bool
    /// Diagnostics: where the argmax writes instead of the next slot (a stale hand-off), when set.
    let scratchSlot: TensorBuffer?

    private(set) var states: StateSet
    private(set) var capacity: Int
    private(set) var position = 0
    /// True when recurrent states were advanced past `position` (steps encoded past a stop); only `restore` or
    /// `reset` clears it.
    private(set) var stateAhead = false
    private var session: GreedySession?
    /// The async operation in progress (a feed, a greedy step, the end of a decode). While set, every other entry
    /// point is refused: the actor is re-entrant at each `await`, and the core's state and buffers belong to the
    /// operation until it returns.
    private var operation: String?
    /// Set by a failed image bind while the state holds tokens from this position on that were fed with images the
    /// static inputs no longer hold (their rotary inputs are text-only again). Feeding, decoding and snapshots are
    /// refused until a rewind to at most this position, a reset or a restore.
    private(set) var imageStateStaleFrom: Int?
    /// Diagnostics: host seconds per greedy phase (bindings, encode, argmax commit, waiting for a token).
    var timings = DecoderEngine.GreedyTimings()
    private var tokenIterator: AsyncStream<(Int, Result<Int32, AetherError>)>.Iterator?

    var hasRecurrentState: Bool { !recurrentNames.isEmpty }
    var isEncoding: Bool { kernels != nil }

    static let ring = 8

    init(
        step stepGraph: any GraphBackend, prefill prefillGraph: (any GraphBackend)?, graph: Manifest.Graph,
        arena: TensorArena, kernels: GreedyKernels?, capacityLimit: Int, initialCapacity: Int, prefillWidth: Int? = nil,
        racingHandoff: Bool = false, staleHandoff: Bool = false
    ) throws(AetherError) {
        let prefix = "graphs.decoder"
        let inputs = graph.inputs(forPhase: "step")
        guard let tokens = inputs["tokens"]?.name, let positionsName = inputs["positions"]?.name,
              let logitsName = graph.outputs["logits"]?.name
        else {
            throw .manifestInvalid(field: "\(prefix).inputs", reason: "decoder-lm@1 needs tokens, positions and logits roles")
        }
        let extras = inputs.filter { $0.key != "tokens" && $0.key != "positions" }
        if let sourced = extras.keys.sorted().first(where: { !extras[$0]!.isStatic || extras[$0]!.source != nil }) {
            throw .unsupportedOperation("input role '\(sourced)' (static tables and fills) is not implemented by the DecoderEngine")
        }
        roles = Roles(tokens: tokens, positions: positionsName, logits: logitsName)
        let signature = stepGraph.signature
        var statics: [String: (name: String, buffer: TensorBuffer)] = [:]
        for (role, binding) in extras.sorted(by: { $0.key < $1.key }) {
            guard let spec = signature.input(binding.name), !spec.isDynamic else {
                throw .manifestMismatch(field: "\(prefix).inputs.\(role)", expected: "a static input '\(binding.name)' with a fully static shape",
                                        found: "\(signature.input(binding.name)?.summary ?? "no such input")")
            }
            statics[role] = (binding.name, try stepGraph.makeBuffer(for: binding.name, role: .input, dimensions: [], in: arena))
        }
        self.statics = statics
        let ext = graph.inputEncodings?["tokens"]?.extensionIds
        extensionIds = ext.map { $0.base..<($0.base + max(0, $0.count)) } ?? 0..<0
        guard let tokenSpec = signature.input(tokens), let logitsSpec = signature.output(logitsName),
              let positionSpec = signature.input(positionsName)
        else { throw .manifestMismatch(field: "\(prefix).inputs", expected: "the roles' tensors", found: "\(signature.inputs.map(\.name))") }
        guard tokenSpec.scalarType == .int32, positionSpec.scalarType == .int32 else {
            throw .unsupportedOperation("tokens/positions of type \(tokenSpec.scalarType)/\(positionSpec.scalarType); only int32 is implemented")
        }
        guard logitsSpec.rank == 3, let v = logitsSpec.shape.last, v > 0 else {
            throw .manifestMismatch(field: "\(prefix).outputs.logits", expected: "[1, S, V] with a static V", found: logitsSpec.summary)
        }
        if graph.outputs["logits"]?.rows == Manifest.OutputBinding.Rows.none {
            throw .unsupportedOperation("rows: none logits on decoder-lm@1")
        }
        vocab = v
        logitsType = logitsSpec.scalarType
        let stepStatic = tokenSpec.shape[1] == TensorSpec.dynamic ? nil : tokenSpec.shape[1]
        if let stepStatic, stepStatic != 1 {
            throw .unsupportedOperation("a step phase with a static token axis of \(stepStatic); decoder-lm@1 steps take 1 token")
        }

        // States: manifest roles, checked against the descriptor (the manifest never guesses a shape).
        let manifestStates = graph.states ?? Manifest.States()
        var sequence: [SequenceState] = []
        var staticAxis: Int?
        for entry in manifestStates.sequence {
            for name in entry.tensors {
                guard let spec = signature.state(name), entry.seqAxis < spec.rank else {
                    throw .manifestMismatch(field: "\(prefix).states.sequence", expected: "state '\(name)'", found: "\(signature.states.map(\.name))")
                }
                if spec.shape[entry.seqAxis] == TensorSpec.dynamic {
                    guard spec.dynamicAxes == [entry.seqAxis] else {
                        throw .manifestMismatch(field: "\(prefix).states.sequence.seqAxis", expected: "the only dynamic axis of '\(name)'", found: spec.summary)
                    }
                } else {
                    guard !spec.isDynamic else {
                        throw .manifestMismatch(field: "\(prefix).states.sequence.seqAxis", expected: "the dynamic axis of '\(name)'", found: spec.summary)
                    }
                    staticAxis = min(staticAxis ?? .max, spec.shape[entry.seqAxis])
                }
                sequence.append(SequenceState(name: name, seqAxis: entry.seqAxis))
            }
        }
        for entry in manifestStates.recurrent {
            guard let spec = signature.state(entry.name), !spec.isDynamic else {
                throw .manifestMismatch(field: "\(prefix).states.recurrent", expected: "a static state '\(entry.name)'", found: "\(signature.states.map(\.summary))")
            }
        }
        sequenceStates = sequence
        recurrentNames = manifestStates.recurrent.map(\.name)
        // The token limit and the allocation are separate: a static axis fixes the allocation, and a smaller limit
        // (an app's explicit capacity) still caps the tokens it holds.
        if let staticAxis, capacityLimit > staticAxis {
            throw .capacity(what: "KV capacity (static sequence axis)", needed: Int64(capacityLimit), available: Int64(staticAxis))
        }
        staticCapacity = staticAxis != nil
        maxCapacity = capacityLimit
        capacity = staticAxis ?? min(max(initialCapacity, 1), capacityLimit)

        let limits = graph.limits ?? Manifest.Limits()
        // A dynamic call feeds at most the manifest's chunk, narrowed by the engine's prefill width (Core AI's reshape
        // retention, ``DecoderEngine/reshapeSafePrefillWidth``).
        let dynamicChunk = max(1, min(limits.prefillChunk ?? 512, prefillWidth ?? .max))
        guard let padToken = Int32(exactly: limits.padTokenId ?? 0),
              let padPosition = Int32(exactly: graph.inputEncodings?["positions"]?.pad ?? -1) else {
            throw AetherError.manifestInvalid(field: "limits.padTokenId", reason: "the pad token and position must fit Int32")
        }
        self.padToken = padToken
        self.padPosition = padPosition
        self.arena = arena
        self.kernels = kernels
        self.racingHandoff = racingHandoff
        scratchSlot = staleHandoff && kernels != nil ? try arena.allocateTokenSlot(name: "staleTokenSlot") : nil

        // Step logits sized for the largest call: a dynamic chunk, or the static shape.
        let stepRows = logitsSpec.shape[1] == TensorSpec.dynamic ? nil : logitsSpec.shape[1]
        let stepLogits = try stepGraph.makeBuffer(
            for: logitsName, role: .output, dimensions: stepRows == nil ? [stepStatic ?? dynamicChunk] : [], in: arena)
        step = Phase(kind: .step, graph: stepGraph, staticTokens: stepStatic, staticRows: stepRows, logits: stepLogits)

        if let prefillGraph {
            let p = prefillGraph.signature
            guard p.inputs.map(\.name).sorted() == signature.inputs.map(\.name).sorted(),
                  p.states == signature.states, let pTokens = p.input(tokens), let pLogits = p.output(logitsName),
                  pTokens.shape.count == 2, pTokens.shape[1] != TensorSpec.dynamic, pTokens.shape[1] > 1,
                  pLogits.shape.last == v, pLogits.scalarType == logitsSpec.scalarType
            else {
                throw .manifestMismatch(
                    field: "\(prefix).functions.prefill", expected: "the step's inputs, states and logits with a static chunk > 1",
                    found: "inputs \(p.inputs.map(\.summary)), states \(p.states.map(\.summary))")
            }
            let c = pTokens.shape[1]
            if limits.padTail == true, !manifestStates.recurrent.isEmpty {
                // Pad rows would advance conv/SSM state; the schema has no field yet for "this graph masks pads".
                throw .unsupportedOperation(
                    "limits.padTail with recurrent states \(manifestStates.recurrent.map(\.name)): pad tokens would advance them")
            }
            let rows = pLogits.shape[1] == TensorSpec.dynamic ? nil : pLogits.shape[1]
            let buffer = try prefillGraph.makeBuffer(for: logitsName, role: .output, dimensions: rows == nil ? [c] : [], in: arena)
            prefillPhase = Phase(kind: .prefill, graph: prefillGraph, staticTokens: c, staticRows: rows, logits: buffer)
            shape = PrefillPlan.Shape(
                prefillChunk: c, padTail: limits.padTail ?? false, prefillRowsAll: rows == nil || rows == c,
                stepStatic: stepStatic, dynamicChunk: min(dynamicChunk, c))
        } else {
            prefillPhase = nil
            shape = PrefillPlan.Shape(stepStatic: stepStatic, dynamicChunk: dynamicChunk)
        }

        history = try arena.allocate(name: "tokenHistory", shape: [1, maxCapacity], scalarType: .int32)
        positions = try arena.allocate(name: "positions", shape: [1, maxCapacity], scalarType: .int32)
        try positions.write(Int32.self, count: maxCapacity) { Int32($0) }
        states = try Self.makeStates(stepGraph, sequence: sequence, recurrent: recurrentNames, capacity: capacity, arena: arena)

        if kernels != nil {
            var slots: [TensorBuffer] = []
            var ring: [TensorBuffer] = []
            for i in 0..<Self.ring {
                let slot = try arena.allocateTokenSlot(name: "tokenSlot\(i)")
                // DESIGN.md §6.1: a page-aligned token buffer races; the allocator retries, and this asserts it.
                let slotBuffer = try arena.metalBuffer(of: slot)
                precondition(!TensorArena.isPageAligned(slotBuffer), "token slot \(i) is page-aligned")
                slots.append(slot)
                ring.append(try stepGraph.makeBuffer(
                    for: logitsName, role: .output, dimensions: stepRows == nil ? [1] : [], in: arena))
            }
            self.slots = slots
            ringLogits = ring
        } else {
            slots = []
            ringLogits = []
        }
    }

    private static func makeStates(
        _ graph: any GraphBackend, sequence: [SequenceState], recurrent: [String], capacity: Int, arena: TensorArena
    ) throws(AetherError) -> StateSet {
        let dynamic = Set(sequence.map(\.name))
        return try graph.makeStateSet(sequence.map(\.name) + recurrent, in: arena) { spec in
            dynamic.contains(spec.name) && spec.isDynamic ? [capacity] : []
        }
    }

    // MARK: - Capacity

    /// Grows the sequence axis geometrically until `needed` slots fit (DESIGN.md §6.2), copying the cached prefix.
    ///
    /// During the copy both the old and the new states are live: the budget charges that pair at `maxCapacity`
    /// (``KVSizing/peakBytes(capacity:)``, the previous allocation beside the last one). Snapshots are sized by the
    /// position, not the capacity (``snapshot()``).
    func ensureCapacity(_ needed: Int) throws(AetherError) {
        guard needed > capacity else { return }
        guard !staticCapacity else {
            throw .capacity(what: "KV capacity (static sequence axis)", needed: Int64(needed), available: Int64(capacity))
        }
        let next = try KVSizing.grow(current: capacity, needed: needed, limit: maxCapacity)
        let old = states
        // Only the sequence axis grows. Keep the recurrent buffers in place: on the 27B those
        // fixed states hold ~147 MiB, which would otherwise be allocated and copied on every growth.
        let grownSequence = try Self.makeStates(step.graph, sequence: sequenceStates, recurrent: [], capacity: next, arena: arena)
        for (name, buffer) in grownSequence.entries {
            guard let source = old[name] else { continue }
            if source.shape == buffer.shape { try buffer.copy(from: source) } else { try buffer.copyRegion(from: source) }
        }
        let sequenceNames = Set(sequenceStates.map(\.name))
        let grown = try StateSet(arena: arena, entries: old.entries.map { entry in
            (name: entry.name, buffer: grownSequence[entry.name] ?? entry.buffer)
        })
        for (name, buffer) in old.entries where sequenceNames.contains(name) { try arena.release(buffer) }
        states = grown
        capacity = next
    }

    /// Replaces the states with zeroed ones at `newCapacity` (no copy). A static allocation stays as it is.
    private func reallocateStates(capacity newCapacity: Int) throws(AetherError) {
        guard newCapacity != capacity, !staticCapacity else { return }
        let fresh = try Self.makeStates(step.graph, sequence: sequenceStates, recurrent: recurrentNames, capacity: newCapacity, arena: arena)
        for (_, buffer) in states.entries { try arena.release(buffer) }
        states = fresh
        capacity = newCapacity
    }

    // MARK: - Feeding

    private func requireIdle(_ request: String) throws(AetherError) {
        if let operation {
            throw .unsupportedOperation("\(request) while \(operation) is in progress")
        }
        guard session == nil else {
            throw .unsupportedOperation("\(request) during a greedy decode; end it first")
        }
    }

    /// Feeds `tokens` at the current position on the prefill schedule and returns the selected logits rows.
    func feed(_ tokens: [Int32], select: LogitsSelection) async throws(AetherError) -> Logits? {
        try requireIdle("feeding tokens")
        try requireLoaded("feeding tokens")
        operation = "a prefill"
        defer { operation = nil }
        return try await feedTokens(tokens, select: select)
    }

    /// The feed itself; the caller holds `operation`.
    private func feedTokens(_ tokens: [Int32], select: LogitsSelection) async throws(AetherError) -> Logits? {
        try requireImageStateConsistent("feeding tokens")
        guard !stateAhead else {
            throw .unsupportedOperation("the recurrent state ran past position \(position) (a pipelined stop); restore a snapshot or reset")
        }
        guard !tokens.isEmpty else { return select == .none ? nil : nil }
        if let bad = tokens.first(where: { $0 < 0 || (Int($0) >= vocab && !extensionIds.contains(Int($0))) }) {
            throw .tensor(name: roles.tokens, reason: .outOfBounds(index: Int(bad), count: vocab))
        }
        let n = tokens.count
        guard position + n <= maxCapacity else {
            throw .capacity(what: "KV capacity (tokens)", needed: Int64(position + n), available: Int64(maxCapacity))
        }
        let plan = PrefillPlan(tokens: n, shape: shape, padRoom: maxCapacity - position - n)
        try check(plan, select: select)
        let end = plan.calls.map { position + $0.start + $0.fed }.max() ?? position + n
        try ensureCapacity(end)

        try history.write(tokens, startingAt: position)
        var padded: [Range<Int>] = []
        for call in plan.calls where call.padding > 0 {
            let pads = (position + call.start + call.count)..<(position + call.start + call.fed)
            try history.write(Int32.self, startingAt: pads.lowerBound, count: pads.count) { _ in padToken }
            try positions.write(Int32.self, startingAt: pads.lowerBound, count: pads.count) { _ in padPosition }
            padded.append(pads)
        }
        defer {
            for pads in padded {
                try? positions.write(Int32.self, startingAt: pads.lowerBound, count: pads.count) { Int32(pads.lowerBound + $0) }
            }
        }

        var rows: [Float] = []
        if select == .all { rows.reserveCapacity(n * vocab) }
        var scratch: [Float] = []
        let start = position
        // Encode mode: the arena is lent once, before the first chunk, and the loan ends only after the stream has
        // drained, on every exit path. Rows are read between drains (the loan is suspended while nothing is in flight).
        var lent = false
        func lend() throws(AetherError) {
            try arena.beginRun("decoder.prefill")
            lent = true
        }
        /// Drains and ends the loan; returns the GPU's failure, if the drain saw one.
        func settle() async -> AetherError? {
            guard lent, let kernels else { return nil }
            let failure = await kernels.drain()
            arena.endRun()
            lent = false
            return failure
        }
        do throws(AetherError) {
            if isEncoding { try lend() }
            for (i, call) in plan.calls.enumerated() {
                // A long prompt is many chunks: stop between them when the caller is cancelled. The position stays at
                // `start` (it advances only once every chunk has run), so the engine stays consistent.
                if i > 0, Task.isCancelled { throw AetherError.runtime(underlying: CancellationError()) }
                let phase = call.phase == .prefill ? prefillPhase! : step
                let at = start + call.start
                var b = try bindings(phase, at: at, fed: call.fed, tokens: nil, logits: phase.logits)
                let isLast = i == plan.calls.count - 1
                let needsRead = select == .all || (select == .last && isLast)
                if let kernels {
                    // Chunks chain on the stream with no host sync; wait only where rows are read.
                    try (phase.graph as! Graph).encode(b, to: kernels.stream)
                    if needsRead, let failure = await settle() { throw failure }
                } else {
                    try await phase.graph.run(&b)
                }
                if needsRead {
                    if select == .all {
                        for r in 0..<call.count {
                            try readRow(phase, fed: call.fed, row: r, into: &scratch)
                            rows += scratch
                        }
                    } else {
                        try readRow(phase, fed: call.fed, row: call.count - 1, into: &scratch)
                        rows = scratch
                    }
                    if isEncoding, !isLast { try lend() }
                }
            }
            if let failure = await settle() { throw failure }
        } catch {
            _ = await settle()
            throw error
        }
        position = start + n
        switch select {
        case .none: return nil
        case .last: return Logits(firstPosition: position - 1, rows: 1, vocab: vocab, values: rows)
        case .all: return Logits(firstPosition: start, rows: n, vocab: vocab, values: rows)
        }
    }

    /// The `logits rows` contract: a schedule must be able to return the rows asked for.
    private func check(_ plan: PrefillPlan, select: LogitsSelection) throws(AetherError) {
        func phase(_ call: PrefillPlan.Call) -> Phase { call.phase == .prefill ? prefillPhase! : step }
        switch select {
        case .none:
            return
        case .last:
            guard let last = plan.calls.last else { return }
            let p = phase(last)
            // A last-row graph's one row is the last *fed* row: right unless the call is padded.
            if last.padding > 0, !p.rowsAll(fed: last.fed) {
                throw .manifestMismatch(
                    field: "graphs.decoder.outputs.logits.rows", expected: "row \(last.count - 1) of a \(last.fed)-token call",
                    found: "\(p.staticRows ?? last.fed) row(s) per call (rows: last)")
            }
        case .all:
            for call in plan.calls where !phase(call).rowsAll(fed: call.fed) {
                throw .manifestMismatch(
                    field: "graphs.decoder.outputs.logits.rows", expected: "logits at every fed position",
                    found: "\(phase(call).staticRows ?? call.fed) row(s) per \(call.fed)-token call (rows: last)")
            }
        }
    }

    private func readRow(_ phase: Phase, fed: Int, row: Int, into out: inout [Float]) throws(AetherError) {
        let r = phase.rowsAll(fed: fed) ? row : 0
        try phase.logits.readFloats((r * vocab)..<((r + 1) * vocab), into: &out)
    }

    private func bindings(
        _ phase: Phase, at position: Int, fed: Int, tokens slot: TensorBuffer?, logits: TensorBuffer
    ) throws(AetherError) -> Bindings {
        var b = arena.runningGraph == nil ? Bindings(arena: arena) : Bindings(lentArena: arena)
        try b.bindStates(states)
        if let slot {
            try b.bindInput(roles.tokens, to: slot)
        } else {
            try b.bindInput(roles.tokens, to: history, origin: [0, position], extent: [1, fed])
        }
        try b.bindInput(roles.positions, to: positions, extent: [1, position + fed])
        for (_, input) in statics { try b.bindInput(input.name, to: input.buffer) }
        try b.bindOutput(roles.logits, to: logits, extent: phase.staticRows == nil ? [1, fed, vocab] : nil)
        return b
    }

    /// The static input buffers by role, for their owner to write while no operation runs.
    func withStatics<R>(_ body: ([String: TensorBuffer]) throws(AetherError) -> R) throws(AetherError) -> R {
        try requireIdle("writing static inputs")
        return try body(statics.mapValues(\.buffer))
    }

    /// The static input buffers for an asynchronous write (an image bind and its tower calls): holds `operation` as
    /// `name` until `body` returns, so no feed, decode, rewind or other write is accepted while it is suspended.
    func withStatics<R>(
        operation name: String, _ body: ([String: TensorBuffer]) async throws(AetherError) -> R
    ) async throws(AetherError) -> R {
        try requireIdle(name)
        operation = name
        defer { operation = nil }
        return try await body(statics.mapValues(\.buffer))
    }

    /// After a failed image bind: the state from `start` on was fed with images the statics no longer hold. Nothing
    /// to refuse when the position is already at or before it.
    func markImageStateStale(from start: Int) {
        if position > start { imageStateStaleFrom = min(start, imageStateStaleFrom ?? start) }
    }

    private func requireImageStateConsistent(_ request: String) throws(AetherError) {
        if let start = imageStateStaleFrom {
            throw .unsupportedOperation(
                "\(request) after a failed image bind: positions \(start)..<\(position) were fed with images the static inputs no longer hold; rewind to \(start) or less, reset, or restore a snapshot")
        }
    }

    // MARK: - Rewind, snapshot

    func rewind(to target: Int) throws(AetherError) {
        try requireIdle("rewind")
        guard target >= 0, target <= position else {
            throw .unsupportedOperation("rewind(to: \(target)) outside 0...\(position)")
        }
        if target == position, !stateAhead { return }
        if target == 0 { return try reset() }
        guard !hasRecurrentState else {
            throw .unsupportedOperation(
                "partial rewind of recurrent states \(recurrentNames) (to \(target) from \(position)); restore a snapshot instead")
        }
        position = target
        if let start = imageStateStaleFrom, target <= start { imageStateStaleFrom = nil }
    }

    func reset() throws(AetherError) {
        try requireIdle("reset")
        try states.zero()
        position = 0
        stateAhead = false
        imageStateStaleFrom = nil
    }

    func snapshot() throws(AetherError) -> DecoderEngine.Snapshot {
        try requireIdle("snapshot")
        try requireImageStateConsistent("snapshot")
        guard !stateAhead else { throw .unsupportedOperation("snapshot of a state that ran past its position") }
        // Sequence states only up to the position: a snapshot costs what the prompt holds, not the capacity.
        var regions: [String: [Int]] = [:]
        for entry in sequenceStates where !staticCapacity {
            guard var shape = states[entry.name]?.shape else { continue }
            shape[entry.seqAxis] = position
            regions[entry.name] = shape
        }
        return DecoderEngine.Snapshot(position: position, capacity: capacity, states: try states.snapshot(regions: regions))
    }

    /// Restores a snapshot. It is validated before any state is touched: a snapshot that does not fit leaves the
    /// engine as it was.
    func restore(_ snapshot: DecoderEngine.Snapshot) throws(AetherError) {
        try requireIdle("restore")
        // The allocation it carries: up to the token limit, or the static axis itself.
        let allocation = staticCapacity ? capacity : maxCapacity
        guard snapshot.position <= maxCapacity, snapshot.capacity <= allocation else {
            throw .capacity(what: "KV capacity (tokens)", needed: Int64(max(snapshot.position, snapshot.capacity)), available: Int64(allocation))
        }
        if snapshot.capacity == capacity {
            try states.restore(snapshot.states)  // validates every entry before copying
        } else {
            let fresh = try Self.makeStates(
                step.graph, sequence: sequenceStates, recurrent: recurrentNames, capacity: snapshot.capacity, arena: arena)
            do {
                try fresh.restore(snapshot.states)
            } catch {
                for (_, buffer) in fresh.entries { try? arena.release(buffer) }
                throw error
            }
            for (_, buffer) in states.entries { try arena.release(buffer) }
            states = fresh
            capacity = snapshot.capacity
        }
        position = snapshot.position
        stateAhead = false
        imageStateStaleFrom = nil
    }

    // MARK: - Greedy decode

    struct GreedySession {
        let start: Int
        let limit: Int
        let depth: Int
        let delay: UInt32
        var issued = 0
        var received = 0
        var last: Int32
        /// Argmax outcomes that arrived ahead of their turn, by token index.
        var pending: [Int: Result<Int32, AetherError>] = [:]
        var continuation: AsyncStream<(Int, Result<Int32, AetherError>)>.Continuation?
        /// Canary only: the expected token per step; a wrong one is written into each slot before its argmax runs.
        var sentinels: [Int32]?
    }

    /// Starts greedy decoding from `first` (chosen by the caller, not yet fed) for at most `limit` more tokens.
    /// `depth` steps stay in flight on the GPU (encode mode); `0` decodes on the host (`run` + CPU argmax).
    func beginGreedy(first: Int32, limit: Int, depth: Int, delay: UInt32 = 0, sentinels: [Int32]? = nil) throws(AetherError) {
        try requireIdle("greedy decode")
        try requireLoaded("greedy decode")
        try requireImageStateConsistent("greedy decode")
        guard !stateAhead else { throw .unsupportedOperation("the recurrent state ran past its position; restore or reset") }
        let limit = max(0, min(limit, maxCapacity - position))
        var s = GreedySession(start: position, limit: limit, depth: depth, delay: delay, last: first)
        s.sentinels = sentinels
        if kernels != nil, depth > 0 {
            guard Self.ring > depth + 1 else {
                throw .unsupportedOperation("a pipeline depth of \(depth); at most \(Self.ring - 2) with \(Self.ring) token slots")
            }
            try slots[0].write([first])
            if racingHandoff { try history.write([first], startingAt: position) }
            let (stream, continuation) = AsyncStream<(Int, Result<Int32, AetherError>)>.makeStream()
            s.continuation = continuation
            tokenIterator = stream.makeAsyncIterator()
            try arena.beginRun("decoder.greedy")
        }
        session = s
    }

    /// The next greedy token, or nil at the limit or when the calling task is cancelled. Holds `operation` for its
    /// whole duration, so nothing else (a second nextGreedy, endGreedy, a feed) can run at its awaits; the session is
    /// re-read after every await all the same. Throws the step's failure (a failed command buffer, an invalid logits
    /// row) in token order; the token is not counted, and ``endGreedy()`` still ends the decode.
    func nextGreedy() async throws(AetherError) -> Int32? {
        if let operation { throw .unsupportedOperation("nextGreedy while \(operation) is in progress") }
        guard let s = session else { throw .unsupportedOperation("nextGreedy without beginGreedy") }
        guard s.received < s.limit else { return nil }
        operation = "a greedy step"
        defer { operation = nil }
        guard kernels != nil, s.depth > 0 else {
            // Host path: run the step and take the argmax of its one row. The session stays in place.
            guard !Task.isCancelled else { return nil }
            let logits = try await feedTokens([s.last], select: .last)!
            let token = try Logits.argmax(logits.last)
            guard session != nil else { throw .unsupportedOperation("the greedy decode ended during a step") }
            session!.last = token
            session!.received += 1
            return token
        }
        try await refill()
        while true {
            guard let current = session else { throw .unsupportedOperation("the greedy decode ended during a step") }
            if let outcome = current.pending[current.received + 1] {
                let t = try outcome.get()
                session!.pending[current.received + 1] = nil
                session!.received += 1
                try await refill()
                return t
            }
            guard var iterator = tokenIterator else { return nil }
            let w0 = ContinuousClock.now
            let next = await iterator.next(isolation: #isolation)
            timings.wait += (ContinuousClock.now - w0) / .seconds(1)
            tokenIterator = iterator
            guard let (index, outcome) = next else { return nil }  // cancelled
            guard session != nil else { throw .unsupportedOperation("the greedy decode ended during a step") }
            session!.pending[index] = outcome
        }
    }

    /// Keeps `depth` steps in flight. Step k feeds the token in slot k (t_k at position start + k); its argmax
    /// writes slot k+1 on the same queue, and the completion handler hands t_{k+1} to the host.
    private func refill() async throws(AetherError) {
        guard let kernels else { return }
        while let s = session, s.issued - s.received < s.depth, s.issued < s.limit {
            let k = s.issued
            let at = s.start + k
            if at + 1 > capacity {
                guard s.issued == s.received else { return }  // grow once the pipeline is empty
                let failure = await kernels.drain()
                arena.endRun()
                do throws(AetherError) {
                    if let failure { throw failure }
                    try ensureCapacity(at + 1)
                } catch {
                    try arena.beginRun("decoder.greedy")  // endGreedy ends the loan
                    throw error
                }
                try arena.beginRun("decoder.greedy")
                guard session != nil else { return }
            }
            let ring = ringLogits[k % Self.ring]
            // Step k reads token k from its slot; its argmax writes token k+1 into the next slot (or, for the racing
            // diagnostic, into the token history at at + 1, read back through a view at that offset).
            let input: TensorBuffer? = racingHandoff ? nil : slots[k % Self.ring]
            let next = racingHandoff ? try arena.metalBuffer(of: history) : try arena.metalBuffer(of: slots[(k + 1) % Self.ring])
            let nextOffset = racingHandoff ? 4 * (at + 1) : 0
            if let expected = s.sentinels, k + 1 < expected.count {
                // The slot's last reader (step k+1-ring) and writer (argmax k-ring) completed: ring > depth + 1.
                (next.contents() + nextOffset).storeBytes(of: (expected[k + 1] + 1) % Int32(vocab), as: Int32.self)
            }
            var target = next
            if let scratchSlot { target = try arena.metalBuffer(of: scratchSlot) }
            let targetOffset = scratchSlot == nil ? nextOffset : 0
            let t0 = ContinuousClock.now
            let b = try bindings(step, at: at, fed: 1, tokens: input, logits: ring)
            let t1 = ContinuousClock.now
            try (step.graph as! Graph).encode(b, to: kernels.stream)
            let t2 = ContinuousClock.now
            timings.bindings += (t1 - t0) / .seconds(1)
            timings.encode += (t2 - t1) / .seconds(1)
            timings.steps += 1
            defer { timings.commit += (ContinuousClock.now - t2) / .seconds(1) }
            // The handler reads the slot through its address: the command buffer retains the slot's MTLBuffer (it is
            // bound as the kernel's output) until the handler has run, and the arena keeps it for the engine's life.
            let address = UInt(bitPattern: target.contents() + targetOffset)
            let continuation = s.continuation
            let index = k + 1
            try kernels.commitArgmax(
                logits: try arena.metalBuffer(of: ring), scalarType: logitsType, vocab: vocab, slot: target,
                slotOffset: targetOffset, status: index % Self.ring, delay: s.delay
            ) { outcome in
                continuation?.yield((index, outcome.map { UnsafePointer<Int32>(bitPattern: address)!.pointee }))
            }
            guard session != nil else { return }
            session!.issued += 1
        }
    }

    struct GreedyEnd {
        var fed: Int
        var overshoot: Int
    }

    /// Ends the decode: drains the GPU, sets the position to the tokens the caller accepted (the last one not fed),
    /// and flags recurrent state that ran ahead. Refused while a step is in progress. The decode ends either way; when
    /// the drain saw the GPU fail, it then throws that failure.
    @discardableResult
    func endGreedy() async throws(AetherError) -> GreedyEnd {
        if let operation { throw .unsupportedOperation("endGreedy while \(operation) is in progress") }
        guard let s = session else { return GreedyEnd(fed: 0, overshoot: 0) }
        operation = "the end of a greedy decode"
        defer { operation = nil }
        if let kernels, s.depth > 0 {
            let failure = await kernels.drain()
            arena.endRun()
            s.continuation?.finish()
            tokenIterator = nil
            let final = session ?? s
            position = final.start + final.received
            let overshoot = final.issued - final.received
            if overshoot > 0, hasRecurrentState { stateAhead = true }
            session = nil
            if let failure { throw failure }
            return GreedyEnd(fed: final.received, overshoot: overshoot)
        }
        session = nil
        return GreedyEnd(fed: s.received, overshoot: 0)
    }

    // MARK: - Reload

    /// False between ``unloadGraphs()`` and ``installGraphs(step:prefill:)``: nothing runs on the placeholders.
    private(set) var graphsLoaded = true

    private func requireLoaded(_ request: String) throws(AetherError) {
        guard graphsLoaded else { throw .unsupportedOperation("\(request) while the decoder's graphs are reloading") }
    }

    /// Drops the phases' graphs, so their model (and the memory Core AI retained for it) can be released before a
    /// reload loads it again. The states, buffers, statics, stream and position stay. Refused during an operation.
    func unloadGraphs() throws(AetherError) {
        try requireIdle("a reload")
        guard graphsLoaded else { return }
        step.graph = UnloadedGraph(signature: step.graph.signature)
        if let prefillPhase { prefillPhase.graph = UnloadedGraph(signature: prefillPhase.graph.signature) }
        graphsLoaded = false
    }

    /// Puts reloaded graphs in place of the dropped ones. Their signatures must equal the originals', so every buffer
    /// still binds; on a mismatch nothing is installed.
    func installGraphs(step newStep: any GraphBackend, prefill newPrefill: (any GraphBackend)?) throws(AetherError) {
        try requireIdle("a reload")
        guard newStep.signature == step.graph.signature,
              (newPrefill == nil) == (prefillPhase == nil),
              newPrefill.map({ $0.signature == prefillPhase!.graph.signature }) ?? true else {
            throw .manifestMismatch(field: "graphs.decoder", expected: "the reloaded graphs' signatures to equal the loaded ones'",
                                    found: "\(newStep.signature.inputs.map(\.summary))")
        }
        step.graph = newStep
        prefillPhase?.graph = newPrefill!
        graphsLoaded = true
    }

    /// One step at the current position after a reload, so the model's memory arrives now and not in the next
    /// request's prefill. The position, and every row below it, stay as they were: the row it writes is past the
    /// position and the next feed overwrites it. Attention-only models (a recurrent state would advance); skipped at
    /// capacity or while the image state is stale.
    func warmAfterReload() async throws(AetherError) {
        guard !hasRecurrentState, imageStateStaleFrom == nil, position < maxCapacity else { return }
        let at = position
        _ = try await feed([padToken >= 0 && Int(padToken) < vocab ? padToken : 0], select: .none)
        position = at
    }

    // MARK: - Canary

    /// The load-time delayed-argmax canary (DESIGN.md §6.1). Decodes `steps` tokens at depth 1, then again at
    /// `depth` with each argmax delayed past the next `encode` and a wrong token pre-written into every slot, so a
    /// step that reads its token before the argmax wrote it diverges for certain. Leaves the engine reset.
    func runCanary(depth: Int, steps: Int, delay: UInt32, capacity canaryCapacity: Int) async throws(AetherError) -> DecoderEngine.CanaryReport {
        let clock = ContinuousClock()
        let began = clock.now
        let restoreCapacity = capacity
        try reset()
        try reallocateStates(capacity: min(max(canaryCapacity, capacity), maxCapacity))
        let tested = capacity
        let prompt = (0..<16).map { Int32(($0 * 7919 + 101) % vocab) }
        let steps = min(steps, tested - prompt.count - 1)

        func decode(depth: Int, delay: UInt32, sentinels: [Int32]?) async throws(AetherError) -> [Int32] {
            try reset()
            let first = try Logits.argmax(try await feed(prompt, select: .last)!.last)
            try beginGreedy(first: first, limit: steps, depth: depth, delay: delay, sentinels: sentinels)
            var out = [first]
            do {
                while let t = try await nextGreedy() { out.append(t) }
            } catch {
                _ = try? await endGreedy()
                throw error
            }
            try await endGreedy()
            return out
        }
        let reference = try await decode(depth: 1, delay: 0, sentinels: nil)
        let pipelined = try await decode(depth: depth, delay: delay, sentinels: reference)
        let mismatch = zip(reference, pipelined).enumerated().first { $0.element.0 != $0.element.1 }?.offset
            ?? (reference.count == pipelined.count ? nil : min(reference.count, pipelined.count))
        try reset()
        try reallocateStates(capacity: restoreCapacity)
        let seconds = (clock.now - began) / .seconds(1)
        return DecoderEngine.CanaryReport(
            passed: mismatch == nil, depth: depth, capacity: tested, steps: reference.count - 1,
            firstMismatch: mismatch, seconds: seconds)
    }
}

/// What a reloading engine's phases hold between dropping their graphs and installing the reloaded ones: the
/// signature, for the checks, and a refusal to run.
final class UnloadedGraph: GraphBackend {
    let signature: GraphSignature
    init(signature: GraphSignature) { self.signature = signature }
    func run(_ bindings: inout Bindings) async throws(AetherError) {
        throw .unsupportedOperation("running a decoder graph that is being reloaded")
    }
}
