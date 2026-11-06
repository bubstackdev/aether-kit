public import AetherRuntime
import Synchronization

/// Holds a fake's `run` at a suspension point until the test opens it, so tests can interleave other engine calls.
public final class RunGate: Sendable {
    private let state = Mutex<(entered: Int, open: Bool, waiters: [CheckedContinuation<Void, Never>])>((0, false, []))
    public init() {}
    public var entered: Int { state.withLock { $0.entered } }

    public func open() {
        let waiters = state.withLock { s in
            s.open = true
            defer { s.waiters = [] }
            return s.waiters
        }
        for w in waiters { w.resume() }
    }

    func pass() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            let resume = state.withLock { s in
                s.entered += 1
                if s.open { return true }
                s.waiters.append(c)
                return false
            }
            if resume { c.resume() }
        }
    }

    /// Closes the gate again and resets the entry count.
    public func close() {
        state.withLock { s in
            s.open = false
            s.entered = 0
        }
    }

    /// Waits until a run has reached the gate.
    public func waitForEntry() async {
        while entered == 0 { await Task.yield() }
    }
}

/// The token-axis length of every call a fake ran, shared with the test (the fake itself moves into the engine).
public final class CallLog: Sendable {
    private let storage = Mutex<[Int]>([])
    public init() {}
    public var calls: [Int] { storage.withLock { $0 } }
    func append(_ s: Int) { storage.withLock { $0.append(s) } }
}

/// A deterministic toy causal LM with the `decoder-lm@1` contract, configurable into each bundle class the engine
/// must handle (DESIGN.md §12):
/// - attention only (dynamic token axis, logits for every row), like qwen3-0.6b;
/// - `staticStep`: a decode-only graph, `input_ids [1×1]` and logits `[1×1×V]`, like the hybrid zoo bundles;
/// - `recurrent`: an extra fully static state (a running hash of every fed token), like convState/recState;
/// - ``prefillPhase(chunk:)``: a second function with a static chunk sharing the states, like LFM3's pf64.
///
/// Per fed token `t` at position `p` (positions `-1` are padding and are skipped entirely):
/// `keys[p] = t`; `rec = hash(rec, t)`. The row's logits peak at `predict(keys[0...p])`; with recurrent state the
/// peak moves by one if `rec` disagrees with the hash of `keys[0...p]`. So stale KV, lost recurrent state or a
/// recurrent state that ran ahead all change the output.
public final class FakeLM: GraphBackend {
    public struct Config: Sendable {
        public var vocab: Int
        public var staticStep: Bool
        public var recurrent: Bool
        /// Static chunk of a prefill function (nil: the step function).
        public var chunk: Int?
        /// Logits `[1×1×V]` (`rows: last`) even with a dynamic token axis.
        public var lastRowOnly: Bool
        public var predict: @Sendable ([Int32]) -> Int32
        public var log: CallLog?
        /// Every run waits here first (a suspension point, as a real graph's run has).
        public var gate: RunGate?

        public init(vocab: Int = 64, staticStep: Bool = false, recurrent: Bool = false, chunk: Int? = nil,
                    lastRowOnly: Bool = false, log: CallLog? = nil,
                    predict: @escaping @Sendable ([Int32]) -> Int32 = FakeLM.mixing(vocab: 64)) {
            self.vocab = vocab
            self.staticStep = staticStep
            self.recurrent = recurrent
            self.chunk = chunk
            self.lastRowOnly = lastRowOnly
            self.log = log
            self.predict = predict
        }
    }

    public let config: Config
    public let signature: GraphSignature

    public init(_ config: Config = Config()) {
        self.config = config
        let s = config.chunk ?? (config.staticStep ? 1 : -1)
        let rows = config.lastRowOnly ? 1 : config.chunk ?? (config.staticStep ? 1 : -1)
        var states = [
            TensorSpec(name: "keys", scalarType: .float32, shape: [1, -1]),
            TensorSpec(name: "values", scalarType: .float32, shape: [1, -1]),
        ]
        if config.recurrent { states.append(TensorSpec(name: "rec", scalarType: .float32, shape: [1, 1])) }
        signature = GraphSignature(
            function: config.chunk == nil ? "main" : "prefill",
            inputs: [
                TensorSpec(name: "input_ids", scalarType: .int32, shape: [1, s]),
                TensorSpec(name: "position_ids", scalarType: .int32, shape: [1, -1]),
            ],
            states: states,
            outputs: [TensorSpec(name: "logits", scalarType: .float32, shape: [1, rows, config.vocab])])
    }

    /// The prefill phase of a model: same states, a static chunk, its own call log.
    public static func prefillPhase(of config: Config, chunk: Int, log: CallLog? = nil) -> FakeLM {
        var c = config
        c.chunk = chunk
        c.log = log
        return FakeLM(c)
    }

    /// A predictor that mixes the whole context into the next token.
    public static func mixing(vocab: Int) -> @Sendable ([Int32]) -> Int32 {
        { context in
            var h: UInt64 = 1469598103934665603
            for t in context { h = (h ^ UInt64(UInt32(bitPattern: t))) &* 1099511628211 }
            return Int32(h % UInt64(vocab))
        }
    }

    public static func hash(_ context: some Sequence<Int32>) -> Float {
        var h: Int = 0
        for t in context { h = (h * 31 + Int(t) + 1) % 65521 }
        return Float(h)
    }

    /// The manifest graph for this fake (roles, states, limits).
    public func manifestGraph(maxContext: Int = 4096, prefillChunk: Int? = nil, padTail: Bool? = nil) -> Manifest.Graph {
        var functions: [String: String]? = nil
        var function: String? = "main"
        var staticSequence: Manifest.StaticSequence? = config.staticStep ? .one(1) : nil
        if let p = prefillChunk {
            functions = ["step": "main", "prefill": "prefill"]
            function = nil
            staticSequence = .perPhase(config.staticStep ? ["step": .one(1), "prefill": .one(p)] : ["prefill": .one(p)])
        }
        return Manifest.Graph(
            file: .single(AssetPath(asset: "main", path: "fake.aimodel")), function: function, functions: functions,
            inputs: ["tokens": "input_ids", "positions": "position_ids"],
            outputs: ["logits": "logits"],
            states: Manifest.States(
                sequence: [Manifest.SequenceState(tensors: ["keys", "values"], seqAxis: 1)],
                recurrent: config.recurrent ? [Manifest.RecurrentState(name: "rec")] : []),
            limits: Manifest.Limits(
                maxContext: maxContext, staticSequence: staticSequence, padTail: padTail),
            compute: .gpu, expectFrequentReshapes: true)
    }

    public func run(_ bindings: inout Bindings) async throws(AetherError) {
        try bindings.validate(against: signature)
        guard let tokens = bindings.input("input_ids"), let positions = bindings.input("position_ids"),
              let keys = bindings.state("keys"), let values = bindings.state("values")
        else { throw .tensor(name: "input_ids", reason: .unbound) }
        let s = bindings.boundShape(of: "input_ids")![1]
        let p = bindings.boundShape(of: "position_ids")![1]
        config.log?.append(s)
        if let gate = config.gate { await gate.pass() }
        let origin = bindings.inputOrigin("input_ids")![1]
        let ids = try tokens.read(Int32.self, origin..<(origin + s))
        let pos = try positions.read(Int32.self, 0..<p)
        var cache = try keys.read(Float.self)
        var rec = config.recurrent ? try bindings.state("rec")!.read(Float.self)[0] : 0
        let logits = bindings.output("logits")
        let rowsAll = signature.outputs[0].shape[1] != 1 || s == 1
        var lastRow: [Float]? = nil
        for i in 0..<s {
            let position = Int(pos[p - s + i])
            guard position >= 0 else {
                if rowsAll { try logits?.write([Float](repeating: 0, count: config.vocab), startingAt: i * config.vocab) }
                continue
            }
            guard position < cache.count else {
                throw .capacity(what: "fake kv", needed: Int64(position + 1), available: Int64(cache.count))
            }
            cache[position] = Float(ids[i])
            try values.write([Float(position)], startingAt: position)
            let context = cache[0...position].map { Int32($0) }
            rec = config.recurrent ? Self.hashStep(rec, ids[i]) : 0
            var target = Int(config.predict(context)) % config.vocab
            if config.recurrent, rec != Self.hash(context) { target = (target + 1) % config.vocab }
            let row = (0..<config.vocab).map { $0 == target ? Float(10) : -Float(abs($0 - target)) * 0.01 }
            if rowsAll { try logits?.write(row, startingAt: i * config.vocab) }
            lastRow = row
        }
        if !rowsAll, let lastRow { try logits?.write(lastRow) }
        try keys.write(cache)
        if config.recurrent { try bindings.state("rec")!.write([rec]) }
    }

    static func hashStep(_ h: Float, _ t: Int32) -> Float {
        Float((Int(h) * 31 + Int(t) + 1) % 65521)
    }
}
