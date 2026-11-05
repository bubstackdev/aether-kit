/// The closed set of roles a driver understands (DESIGN.md §4: "Roles are a closed set per driver").
///
/// A contract is pure data: AetherCore checks a manifest's roles against it, and AetherRuntime checks the same
/// roles against a loaded graph's signature (rank and scalar kind). Drivers own their contract; this file only
/// holds the ones whose drivers are part of the plan, so the adapter and the tests can refer to them.
public struct DriverContract: Sendable, Hashable {
    public enum ScalarKind: String, Sendable, Hashable {
        case integer
        case floatingPoint

        public func admits(_ type: ScalarType) -> Bool {
            switch self {
            case .integer: type.isInteger
            case .floatingPoint: type.isFloatingPoint
            }
        }
    }

    /// What a role requires of the graph tensor it names.
    public struct RoleSpec: Sendable, Hashable {
        public var required: Bool
        /// The exact rank, or nil for any rank.
        public var rank: Int?
        public var scalar: ScalarKind

        public init(required: Bool, rank: Int?, scalar: ScalarKind) {
            self.required = required
            self.rank = rank
            self.scalar = scalar
        }
    }

    /// The roles of one graph of the bundle.
    public struct GraphContract: Sendable, Hashable {
        public var inputs: [String: RoleSpec]
        public var outputs: [String: RoleSpec]
        /// Minimum number of kv state names (0 when kv is not allowed), and whether they must come in pairs.
        public var minimumKV: Int
        public var kvInPairs: Bool
        public var kvScalar: ScalarKind
        public var allowsRecurrent: Bool
        public var recurrentScalar: ScalarKind

        public init(
            inputs: [String: RoleSpec], outputs: [String: RoleSpec], minimumKV: Int = 0, kvInPairs: Bool = true,
            kvScalar: ScalarKind = .floatingPoint, allowsRecurrent: Bool = false,
            recurrentScalar: ScalarKind = .floatingPoint
        ) {
            self.inputs = inputs
            self.outputs = outputs
            self.minimumKV = minimumKV
            self.kvInPairs = kvInPairs
            self.kvScalar = kvScalar
            self.allowsRecurrent = allowsRecurrent
            self.recurrentScalar = recurrentScalar
        }
    }

    public var driver: DriverID
    /// Graph role → its contract. Every key is required; no other graph role is accepted.
    public var graphs: [String: GraphContract]
    public var requiresTokenizer: Bool

    public init(driver: DriverID, graphs: [String: GraphContract], requiresTokenizer: Bool) {
        self.driver = driver
        self.graphs = graphs
        self.requiresTokenizer = requiresTokenizer
    }

    /// Checks the manifest's roles against the contract. Errors name the offending field.
    public func validate(_ manifest: Manifest) throws(AetherError) {
        guard manifest.driver == driver else {
            throw .manifestMismatch(field: "driver", expected: driver.description, found: manifest.driver.description)
        }
        if requiresTokenizer, manifest.tokenizer == nil {
            throw .manifestInvalid(field: "tokenizer", reason: "\(driver) requires a tokenizer")
        }
        for key in manifest.graphs.keys.sorted() where graphs[key] == nil {
            throw .manifestInvalid(
                field: "graphs.\(key)",
                reason: "\(driver) has no graph role '\(key)'; known: \(graphs.keys.sorted().joined(separator: ", "))")
        }
        for (key, contract) in graphs.sorted(by: { $0.key < $1.key }) {
            guard let graph = manifest.graphs[key] else {
                throw .manifestInvalid(field: "graphs.\(key)", reason: "required by \(driver)")
            }
            try contract.validate(graph, at: "graphs.\(key)", driver: driver)
        }
    }
}

extension DriverContract.GraphContract {
    func validate(_ graph: Manifest.Graph, at prefix: String, driver: DriverID) throws(AetherError) {
        try Self.checkRoles(graph.inputs, against: inputs, at: "\(prefix).inputs", driver: driver)
        try Self.checkRoles(graph.outputs, against: outputs, at: "\(prefix).outputs", driver: driver)
        let kv = graph.states.kv
        if minimumKV == 0 {
            try Manifest.require(kv.isEmpty, "\(prefix).states.kv", "\(driver) does not take kv states")
        } else {
            try Manifest.require(kv.count >= minimumKV, "\(prefix).states.kv",
                                 "\(driver) requires at least \(minimumKV) kv state names")
        }
        if kvInPairs {
            try Manifest.require(kv.count % 2 == 0, "\(prefix).states.kv", "kv states come in [keys, values] pairs")
        }
        if !kv.isEmpty {
            try Manifest.require(graph.states.sequenceAxis != nil, "\(prefix).states.sequenceAxis",
                                 "required when kv states are present")
        }
        if !allowsRecurrent {
            try Manifest.require(graph.states.recurrent.isEmpty, "\(prefix).states.recurrent",
                                 "\(driver) does not take recurrent states")
        }
    }

    static func checkRoles(
        _ roles: [String: String], against specs: [String: DriverContract.RoleSpec], at prefix: String,
        driver: DriverID
    ) throws(AetherError) {
        for role in roles.keys.sorted() where specs[role] == nil {
            throw .manifestInvalid(
                field: "\(prefix).\(role)",
                reason: "\(driver) has no role '\(role)'; known: \(specs.keys.sorted().joined(separator: ", "))")
        }
        for (role, spec) in specs.sorted(by: { $0.key < $1.key }) where spec.required && roles[role] == nil {
            throw .manifestInvalid(field: "\(prefix).\(role)", reason: "required by \(driver)")
        }
    }
}

extension DriverContract {
    /// `decoder-lm@1` (§4, §6.2): inputs `tokens [1,S]` and `positions [1,P]`, output `logits [1,S,V]` and an optional
    /// `hidden`, at least one kv pair with a declared sequence axis, and optional recurrent states.
    public static let decoderLM = DriverContract(
        driver: .decoderLM,
        graphs: [
            "decoder": GraphContract(
                inputs: [
                    "tokens": RoleSpec(required: true, rank: 2, scalar: .integer),
                    "positions": RoleSpec(required: true, rank: 2, scalar: .integer),
                ],
                outputs: [
                    "logits": RoleSpec(required: true, rank: 3, scalar: .floatingPoint),
                    "hidden": RoleSpec(required: false, rank: 3, scalar: .floatingPoint),
                ],
                minimumKV: 2, kvInPairs: true, kvScalar: .floatingPoint,
                allowsRecurrent: true, recurrentScalar: .floatingPoint),
        ],
        requiresTokenizer: true)

    /// The contracts this build knows by driver id.
    public static let known: [DriverID: DriverContract] = [DriverContract.decoderLM.driver: .decoderLM]
}
