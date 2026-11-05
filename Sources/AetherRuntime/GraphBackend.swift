/// Which side of a graph a tensor is on.
public enum TensorRole: String, Sendable, Hashable {
    case input
    case state
    case output
}

/// What higher layers drive: a ``Graph`` in production, a deterministic fake in tests (DESIGN.md §12).
///
/// Non-`Sendable`, like ``Graph``: the owning actor holds it. Fakes implement `run` by reading the bound inputs and
/// states from `bindings` and writing states and outputs, on the CPU.
public protocol GraphBackend: AnyObject {
    var signature: GraphSignature { get }

    /// Allocates a buffer for `name`, resolving its dynamic dimensions in order from `dimensions`.
    func makeBuffer(for name: String, role: TensorRole, dimensions: [Int], in arena: TensorArena) throws(AetherError) -> TensorBuffer

    func run(_ bindings: inout Bindings) async throws(AetherError)
}

extension GraphBackend {
    /// Default allocation from the signature: contiguous row-major.
    public func makeBuffer(
        for name: String, role: TensorRole, dimensions: [Int], in arena: TensorArena
    ) throws(AetherError) -> TensorBuffer {
        let spec: TensorSpec? = switch role {
        case .input: signature.input(name)
        case .state: signature.state(name)
        case .output: signature.output(name)
        }
        guard let spec else { throw .tensor(name: name, reason: .unknownName) }
        return try arena.allocate(name: name, shape: spec.resolving(dimensions), scalarType: spec.scalarType)
    }

    /// Allocates a zeroed ``StateSet`` for `names`, resolving each state's dynamic dimensions with `dimensions`
    /// (for a KV cache: the capacity along the sequence axis).
    public func makeStateSet(
        _ names: [String], in arena: TensorArena, dimensions: (TensorSpec) -> [Int] = { _ in [] }
    ) throws(AetherError) -> StateSet {
        var entries: [(name: String, buffer: TensorBuffer)] = []
        for name in names {
            guard let spec = signature.state(name) else { throw .tensor(name: name, reason: .unknownName) }
            entries.append((name, try makeBuffer(for: name, role: .state, dimensions: dimensions(spec), in: arena)))
        }
        return try StateSet(arena: arena, entries: entries)
    }
}
