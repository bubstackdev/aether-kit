public import AetherRuntime

/// A deterministic toy causal LM behind ``GraphBackend``, with the `decoder-lm@1` graph contract
/// (tokens `[1,S]`, positions `[1,P]`, one KV pair with the sequence on axis 1, logits `[1,S,V]`).
///
/// Step semantics, for each new token `t` at position `p` (the last `S` entries of `positions`):
/// `keys[0, p] = t`, `values[0, p] = p`, and `logits[s, v] = Σ keys[0, 0...p] + v`. The logits depend on the whole
/// cached prefix, so stale or lost state shows up in the output.
public final class FakeDecoder: GraphBackend {
    public static let vocab = 4
    public let signature = GraphSignature(
        function: "main",
        inputs: [
            TensorSpec(name: "input_ids", scalarType: .int32, shape: [1, -1]),
            TensorSpec(name: "position_ids", scalarType: .int32, shape: [1, -1]),
        ],
        states: [
            TensorSpec(name: "keys", scalarType: .float32, shape: [1, -1]),
            TensorSpec(name: "values", scalarType: .float32, shape: [1, -1]),
        ],
        outputs: [TensorSpec(name: "logits", scalarType: .float32, shape: [1, -1, vocab])])
    public private(set) var runs = 0

    public init() {}

    public func run(_ bindings: inout Bindings) async throws(AetherError) {
        try bindings.validate(against: signature)
        runs += 1
        guard let tokens = bindings.input("input_ids"), let positions = bindings.input("position_ids"),
              let keys = bindings.state("keys"), let values = bindings.state("values")
        else { throw .tensor(name: "input_ids", reason: .unbound) }
        let s = bindings.boundShape(of: "input_ids")![1]
        let p = bindings.boundShape(of: "position_ids")![1]
        let ids = try tokens.read(Int32.self, 0..<s)
        let pos = try positions.read(Int32.self, 0..<p)
        var cache = try keys.read(Float.self)
        for i in 0..<s {
            let position = Int(pos[p - s + i])
            guard position < cache.count else {
                throw .capacity(what: "fake kv", needed: Int64(position + 1), available: Int64(cache.count))
            }
            cache[position] = Float(ids[i])
            try values.write([Float(position)], startingAt: position)
            let sum = cache[0...position].reduce(0, +)
            if let logits = bindings.output("logits") {
                try logits.write((0..<Self.vocab).map { sum + Float($0) }, startingAt: i * Self.vocab)
            }
        }
        try keys.write(cache)
    }
}
