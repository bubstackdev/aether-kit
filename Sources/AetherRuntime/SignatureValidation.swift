/// Checks a manifest against the graphs it describes (DESIGN.md §4): every role names a tensor that exists, ranks
/// match the driver's contract, scalar types are supported, and nothing the graph needs is left unbound.
/// A mismatch names the offending manifest field.
public enum SignatureValidator {
    /// Validates `manifest` (structure, driver roles, and every graph) against `signatures`, keyed by graph role.
    public static func validate(
        _ manifest: Manifest, signatures: [String: GraphSignature], contract: DriverContract? = nil
    ) throws(AetherError) {
        try manifest.validate()
        guard let contract = contract ?? DriverContract.known[manifest.driver] else {
            throw .unsupported(driver: manifest.driver)
        }
        try contract.validate(manifest)
        for (role, graphContract) in contract.graphs.sorted(by: { $0.key < $1.key }) {
            guard let graph = manifest.graphs[role] else { continue }  // contract.validate already required it
            guard let signature = signatures[role] else {
                throw .manifestInvalid(field: "graphs.\(role)", reason: "no graph signature to validate against")
            }
            try validate(graph, at: "graphs.\(role)", against: signature, contract: graphContract)
        }
    }

    /// Validates one manifest graph against its signature.
    public static func validate(
        _ graph: Manifest.Graph, at prefix: String, against signature: GraphSignature,
        contract: DriverContract.GraphContract
    ) throws(AetherError) {
        guard signature.function == graph.function else {
            throw .manifestMismatch(field: "\(prefix).function", expected: graph.function, found: signature.function)
        }

        // Inputs: each role exists with the right rank and scalar kind, and every graph input has a role.
        for (role, name) in graph.inputs.sorted(by: { $0.key < $1.key }) {
            let field = "\(prefix).inputs.\(role)"
            guard let spec = signature.input(name) else {
                throw .manifestMismatch(
                    field: field, expected: "a graph input named '\(name)'",
                    found: "inputs \(signature.inputs.map(\.name))")
            }
            try check(spec, field: field, rank: contract.inputs[role]?.rank, scalar: contract.inputs[role]?.scalar)
        }
        let boundInputs = Set(graph.inputs.values)
        if let unbound = signature.inputs.first(where: { !boundInputs.contains($0.name) }) {
            throw .manifestMismatch(
                field: "\(prefix).inputs", expected: "a role for every graph input",
                found: "graph input '\(unbound.name)' has no role")
        }

        for (role, name) in graph.outputs.sorted(by: { $0.key < $1.key }) {
            let field = "\(prefix).outputs.\(role)"
            guard let spec = signature.output(name) else {
                throw .manifestMismatch(
                    field: field, expected: "a graph output named '\(name)'",
                    found: "outputs \(signature.outputs.map(\.name))")
            }
            try check(spec, field: field, rank: contract.outputs[role]?.rank, scalar: contract.outputs[role]?.scalar)
        }

        // States: each listed state exists, and every graph state is listed (the engine must own all of them).
        for (kind, names, scalar) in [("kv", graph.states.kv, contract.kvScalar),
                                      ("recurrent", graph.states.recurrent, contract.recurrentScalar)] {
            for (i, name) in names.enumerated() {
                let field = "\(prefix).states.\(kind).\(i)"
                guard let spec = signature.state(name) else {
                    throw .manifestMismatch(
                        field: field, expected: "a graph state named '\(name)'",
                        found: "states \(signature.states.map(\.name))")
                }
                try check(spec, field: field, rank: nil, scalar: scalar)
                if kind == "kv", let axis = graph.states.sequenceAxis, axis >= spec.rank {
                    throw .manifestMismatch(
                        field: "\(prefix).states.sequenceAxis", expected: "an axis of '\(name)' (rank \(spec.rank))",
                        found: "\(axis)")
                }
            }
        }
        let listed = Set(graph.states.all)
        if let unlisted = signature.states.first(where: { !listed.contains($0.name) }) {
            throw .manifestMismatch(
                field: "\(prefix).states", expected: "every graph state listed as kv or recurrent",
                found: "graph state '\(unlisted.name)' is not listed")
        }

        // The token axis: static in the graph exactly when the manifest says `staticSequence`.
        if let tokensName = graph.inputs["tokens"], let tokens = signature.input(tokensName), tokens.rank >= 2 {
            let axis = tokens.shape[1]
            let field = "\(prefix).limits.staticSequence"
            switch (axis == TensorSpec.dynamic, graph.limits.staticSequence) {
            case (true, let s?):
                throw .manifestMismatch(field: field, expected: "\(s)", found: "a dynamic token axis")
            case (false, nil):
                throw .manifestMismatch(field: field, expected: "unset (dynamic token axis)", found: "a static token axis of \(axis)")
            case (false, let s?) where s != axis:
                throw .manifestMismatch(field: field, expected: "\(s)", found: "\(axis)")
            default:
                break
            }
        }
    }

    private static func check(
        _ spec: TensorSpec, field: String, rank: Int?, scalar: DriverContract.ScalarKind?
    ) throws(AetherError) {
        guard spec.kind == .ndArray else {
            throw .manifestMismatch(field: field, expected: "an NDArray", found: "\(spec.kind) '\(spec.name)'")
        }
        if let rank, spec.rank != rank {
            throw .manifestMismatch(field: field, expected: "rank \(rank)", found: "'\(spec.name)' \(spec.summary)")
        }
        if let scalar {
            guard scalar.admits(spec.scalarType), spec.scalarType.byteWidth != nil else {
                throw .manifestMismatch(
                    field: field, expected: "a supported \(scalar.rawValue) type", found: "'\(spec.name)' \(spec.summary)")
            }
        }
    }
}
