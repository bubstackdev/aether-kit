/// Maps graph names to ``TensorBuffer``s for one run shape: inputs, states and outputs (DESIGN.md §6.1).
///
/// Build it once per shape and reuse it; rebuild (or rebind single names) only when a dynamic dimension changes.
/// An input or output may be bound to the leading `extent` of a larger buffer, so a buffer sized for the largest
/// shape can serve smaller ones without reallocation; the graph then sees `[0..<extent[d]]` along each axis.
/// An input may also start at an `origin` (`[origin[d] ..< origin[d] + extent[d]]`): the decoder feeds each prefill
/// chunk from its own region of one token-history buffer, so encoded chunks never share bytes the host rewrites.
public struct Bindings {
    public let arena: TensorArena

    struct Bound {
        var name: String
        var buffer: TensorBuffer
        /// The region bound, or nil for the whole buffer.
        var extent: [Int]?
        /// Where the region starts (nil: at the origin). Only inputs have one.
        var origin: [Int]? = nil

        var shape: [Int] { extent ?? buffer.shape }

        /// Strides to hand Core AI for the region. A region whose elements are contiguous in memory (a leading-extent
        /// view of a row-major buffer, a `[1, S]` window of the token history) gets plain row-major strides for its
        /// own shape: the earlier AetherKit measured Core AI copying inputs whose strides are not its preferred ones
        /// on every step (_reference/aether-kit-remote docs/language-generation.md:206-208, 295-296).
        var regionStrides: [Int] {
            let shape = self.shape
            let packed = StridedLayout.rowMajorStrides(shape)
            let contiguous = shape.indices.allSatisfy { shape[$0] == 1 || buffer.strides[$0] == packed[$0] }
            return contiguous ? packed : buffer.strides
        }

        /// The element offset of the region's first element.
        var elementOffset: Int {
            guard let origin else { return 0 }
            return zip(origin, buffer.strides).reduce(0) { $0 + $1.0 * $1.1 }
        }
    }

    private(set) var inputs: [Bound] = []
    private(set) var states: [Bound] = []
    private(set) var outputs: [Bound] = []

    /// Encode mode binds buffers while the arena is lent to in-flight GPU work (binding only records handles).
    let whileLent: Bool

    public init(arena: TensorArena) {
        self.arena = arena
        whileLent = false
    }

    /// Bindings that may be built while the arena is lent (the pipelined decoder, between encodes).
    init(lentArena arena: TensorArena) {
        self.arena = arena
        whileLent = true
    }

    public mutating func bindInput(_ name: String, to buffer: TensorBuffer, extent: [Int]? = nil) throws(AetherError) {
        try Self.upsert(&inputs, Bound(name: name, buffer: try owned(buffer, extent: extent), extent: extent))
    }

    /// Binds the region `[origin[d] ..< origin[d] + extent[d]]` of `buffer` as input `name`.
    public mutating func bindInput(_ name: String, to buffer: TensorBuffer, origin: [Int], extent: [Int]) throws(AetherError) {
        if whileLent { try arena.checkOwned(buffer) } else { try arena.check(buffer) }
        guard origin.count == buffer.shape.count, extent.count == buffer.shape.count,
              zip(zip(origin, extent), buffer.shape).allSatisfy({ $0.0 >= 0 && $0.1 > 0 && $0.0 + $0.1 <= $1 })
        else {
            throw .tensor(name: buffer.name, reason: .outOfBounds(index: zip(origin, extent).map { $0 + $1 }.max() ?? 0,
                                                                 count: buffer.shape.max() ?? 0))
        }
        try Self.upsert(&inputs, Bound(name: name, buffer: buffer, extent: extent, origin: origin))
    }

    /// The region origin bound for input `name` (zeros when bound at the start).
    public func inputOrigin(_ name: String) -> [Int]? {
        guard let bound = inputs.first(where: { $0.name == name }) else { return nil }
        return bound.origin ?? [Int](repeating: 0, count: bound.buffer.shape.count)
    }

    /// States are always bound whole: the graph reads and updates the full state in place.
    public mutating func bindState(_ name: String, to buffer: TensorBuffer) throws(AetherError) {
        try Self.upsert(&states, Bound(name: name, buffer: try owned(buffer, extent: nil), extent: nil))
    }

    public mutating func bindStates(_ set: StateSet) throws(AetherError) {
        for (name, buffer) in set.entries { try bindState(name, to: buffer) }
    }

    public mutating func bindOutput(_ name: String, to buffer: TensorBuffer, extent: [Int]? = nil) throws(AetherError) {
        try Self.upsert(&outputs, Bound(name: name, buffer: try owned(buffer, extent: extent), extent: extent))
    }

    public func input(_ name: String) -> TensorBuffer? { inputs.first { $0.name == name }?.buffer }
    public func state(_ name: String) -> TensorBuffer? { states.first { $0.name == name }?.buffer }
    public func output(_ name: String) -> TensorBuffer? { outputs.first { $0.name == name }?.buffer }

    /// The shape the graph sees for a bound name (the extent if one is set).
    public func boundShape(of name: String) -> [Int]? {
        (inputs + states + outputs).first { $0.name == name }?.shape
    }

    /// Checks the bindings against a signature: every bound buffer is still the arena's (not released, its slot not
    /// reused since it was bound), every input and state is bound, every bound name exists, and each bound shape and
    /// scalar type fits its declaration. Outputs may be left unbound. Does not allocate on success.
    public func validate(against signature: GraphSignature) throws(AetherError) {
        // A buffer released after binding may have had its slot reused by storage of another shape and type: running
        // on it would read or write the wrong bytes.
        for bound in inputs { try stillOwned(bound.buffer) }
        for bound in states { try stillOwned(bound.buffer) }
        for bound in outputs { try stillOwned(bound.buffer) }
        for spec in signature.inputs {
            guard let bound = inputs.first(where: { $0.name == spec.name }) else {
                throw .tensor(name: spec.name, reason: .unbound)
            }
            try Self.check(bound, spec)
        }
        for spec in signature.states {
            guard let bound = states.first(where: { $0.name == spec.name }) else {
                throw .tensor(name: spec.name, reason: .unbound)
            }
            try Self.check(bound, spec)
        }
        for bound in inputs where signature.input(bound.name) == nil {
            throw .tensor(name: bound.name, reason: .unknownName)
        }
        for bound in states where signature.state(bound.name) == nil {
            throw .tensor(name: bound.name, reason: .unknownName)
        }
        for bound in outputs {
            guard let spec = signature.output(bound.name) else { throw .tensor(name: bound.name, reason: .unknownName) }
            try Self.check(bound, spec)
        }
        // One storage, one binding: Core AI must never see two views of the same bytes.
        try Self.checkDistinct(inputs, inputs, sameList: true)
        try Self.checkDistinct(inputs, states, sameList: false)
        try Self.checkDistinct(inputs, outputs, sameList: false)
        try Self.checkDistinct(states, states, sameList: true)
        try Self.checkDistinct(states, outputs, sameList: false)
        try Self.checkDistinct(outputs, outputs, sameList: true)
    }

    // MARK: -

    private func stillOwned(_ buffer: TensorBuffer) throws(AetherError) {
        if whileLent { try arena.checkOwned(buffer) } else { try arena.check(buffer) }
    }

    private func owned(_ buffer: TensorBuffer, extent: [Int]?) throws(AetherError) -> TensorBuffer {
        if whileLent { try arena.checkOwned(buffer) } else { try arena.check(buffer) }
        if let extent {
            guard extent.count == buffer.shape.count,
                  zip(extent, buffer.shape).allSatisfy({ $0 > 0 && $0 <= $1 })
            else {
                throw .tensor(name: buffer.name, reason: .shapeMismatch(expected: buffer.shape, found: extent))
            }
        }
        return buffer
    }

    private static func upsert(_ list: inout [Bound], _ bound: Bound) throws(AetherError) {
        if let i = list.firstIndex(where: { $0.name == bound.name }) {
            list[i] = bound
        } else {
            list.append(bound)
        }
    }

    private static func checkDistinct(_ a: [Bound], _ b: [Bound], sameList: Bool) throws(AetherError) {
        for (i, x) in a.enumerated() {
            for (j, y) in b.enumerated() where !sameList || j > i {
                if x.buffer.isSameStorage(as: y.buffer) {
                    throw .tensor(
                        name: y.name, reason: .unsupported("shares its buffer with '\(x.name)'; bind distinct buffers"))
                }
            }
        }
    }

    private static func check(_ bound: Bound, _ spec: TensorSpec) throws(AetherError) {
        guard spec.kind == .ndArray else {
            throw .tensor(name: spec.name, reason: .unsupported("\(spec.kind) values are not bindable yet"))
        }
        guard bound.buffer.scalarType == spec.scalarType else {
            throw .tensor(
                name: spec.name,
                reason: .typeMismatch(expected: spec.scalarType.rawValue, found: bound.buffer.scalarType.rawValue))
        }
        guard spec.accepts(bound.shape) else {
            throw .tensor(name: spec.name, reason: .shapeMismatch(expected: spec.shape, found: bound.shape))
        }
    }
}
