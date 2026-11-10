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

/// A graph whose model file declares more than one function (``Manifest/Graph/phases``), specialized once so the
/// functions share their weights: the graph itself is its `step` function, and ``function(forPhase:)`` gives the others
/// (Krea 2's text encoder with its `references` function, say).
package protocol PhasedGraphBackend: GraphBackend {
    /// The function of `phase`, or nil when the graph has none.
    func function(forPhase phase: String) -> (any GraphBackend)?
}

/// The functions of one specialized model file, by phase; `step` is the graph's own (signature, buffers, run).
package final class GraphFunctionSet: PhasedGraphBackend {
    package let functions: [String: any GraphBackend]
    private let step: any GraphBackend

    /// Fails unless `functions` has a `step` phase.
    package init(functions: [String: any GraphBackend]) throws(AetherError) {
        guard let step = functions["step"] else {
            throw .manifestInvalid(field: "functions.step", reason: "a graph with several functions needs a step function")
        }
        self.functions = functions
        self.step = step
    }

    public var signature: GraphSignature { step.signature }

    public func makeBuffer(
        for name: String, role: TensorRole, dimensions: [Int], in arena: TensorArena
    ) throws(AetherError) -> TensorBuffer {
        try step.makeBuffer(for: name, role: role, dimensions: dimensions, in: arena)
    }

    public func run(_ bindings: inout Bindings) async throws(AetherError) { try await step.run(&bindings) }

    package func function(forPhase phase: String) -> (any GraphBackend)? { functions[phase] }
}
