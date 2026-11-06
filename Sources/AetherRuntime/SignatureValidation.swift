/// Checks a manifest against the graphs it describes (MANIFEST.md §2.1.1: a shape in a manifest is an assertion
/// the driver checks). Every role names a tensor that exists, ranks and scalar kinds fit the driver's contract,
/// tables match the inputs they feed, and nothing the graph needs is unbound. Errors name the manifest field.
public enum SignatureValidator {
    /// Signatures keyed by graph role, then function name.
    public typealias Signatures = [String: [String: GraphSignature]]

    public static func validate(_ manifest: Manifest, signatures: Signatures) throws(AetherError) {
        try manifest.validate()
        try DriverCatalog.requireKnown(manifest.driver)
        guard let contract = DriverCatalog.contracts[manifest.driver] else {
            throw .unsupported(driver: manifest.driver)
        }
        try contract.validate(manifest)
        for (role, graph) in manifest.graphs.sorted(by: { $0.key < $1.key }) {
            guard let graphContract = contract.graph(role) else { continue }
            for (phase, function) in graph.phases.sorted(by: { $0.key < $1.key }) {
                guard let signature = signatures[role]?[function] else {
                    throw .manifestInvalid(
                        field: "graphs.\(role).functions.\(phase)", reason: "no signature for function '\(function)' to validate against")
                }
                try validate(graph, phase: phase, at: "graphs.\(role)", against: signature, contract: graphContract,
                             tables: manifest.tables ?? [:])
            }
        }
    }

    /// Validates one phase of a manifest graph against that function's signature.
    public static func validate(
        _ graph: Manifest.Graph, phase: String, at prefix: String, against signature: GraphSignature,
        contract: DriverContract.GraphContract, tables: [String: Manifest.Table]
    ) throws(AetherError) {
        let function = graph.phases[phase] ?? ""
        guard signature.function == function else {
            throw .manifestMismatch(field: "\(prefix).functions.\(phase)", expected: function, found: signature.function)
        }

        let inputs = graph.inputs(forPhase: phase)
        let inputsField = graph.inputsByFunction?[phase] != nil ? "\(prefix).inputsByFunction.\(phase)" : "\(prefix).inputs"
        for (role, binding) in inputs.sorted(by: { $0.key < $1.key }) {
            let field = "\(inputsField).\(role)"
            guard let spec = signature.input(binding.name) else {
                throw .manifestMismatch(
                    field: field, expected: "a graph input named '\(binding.name)'", found: "inputs \(signature.inputs.map(\.name))")
            }
            try check(spec, field: field, role: contract.inputs[role])
            if case .table(let name, _)? = binding.source, let table = tables[name] {
                guard table.dtype == spec.scalarType, table.shape == spec.shape else {
                    throw .manifestMismatch(
                        field: "tables.\(name)", expected: "'\(spec.name)' \(spec.summary)",
                        found: "\(table.dtype) [\(table.shape.map(String.init).joined(separator: "×"))]")
                }
            }
        }
        // A weights table (§5.8) binds its inputs per block; they need no role.
        let bound = Set(inputs.values.map(\.name)).union(graph.weights?.inputs ?? [])
        if let unbound = signature.inputs.first(where: { !bound.contains($0.name) }) {
            throw .manifestMismatch(
                field: inputsField, expected: "a role for every graph input", found: "graph input '\(unbound.name)' has no role")
        }
        for role in (graph.inputEncodings ?? [:]).keys.sorted() where inputs[role] == nil {
            throw .manifestInvalid(field: "\(prefix).inputEncodings.\(role)", reason: "names a role the graph does not bind")
        }

        for (role, output) in graph.outputs.sorted(by: { $0.key < $1.key }) {
            let field = "\(prefix).outputs.\(role)"
            guard let spec = signature.output(output.name) else {
                throw .manifestMismatch(
                    field: field, expected: "a graph output named '\(output.name)'", found: "outputs \(signature.outputs.map(\.name))")
            }
            try check(spec, field: field, role: output.rows == Manifest.OutputBinding.Rows.none ? nil : contract.outputs[role])
            switch output.rows {
            case .last?:
                guard spec.rank >= 2, spec.shape[1] == 1 else {
                    throw .manifestMismatch(field: "\(field).rows", expected: "last (a static row axis of 1)", found: spec.summary)
                }
            case .none?:
                guard spec.rank == 0 else {
                    throw .manifestMismatch(field: "\(field).rows", expected: "none (a rank-0 output)", found: spec.summary)
                }
            case .all?, nil:
                break
            }
        }
        for (i, name) in (graph.unusedOutputs ?? []).enumerated() where signature.output(name) == nil {
            throw .manifestMismatch(field: "\(prefix).unusedOutputs.\(i)", expected: "a graph output named '\(name)'", found: "none")
        }

        let states = graph.states ?? Manifest.States()
        for (i, entry) in states.sequence.enumerated() {
            for name in entry.tensors {
                let field = "\(prefix).states.sequence.\(i).tensors"
                guard let spec = signature.state(name) else {
                    throw .manifestMismatch(
                        field: field, expected: "a graph state named '\(name)'", found: "states \(signature.states.map(\.name))")
                }
                try check(spec, field: field, role: .init(required: true, rank: nil, scalar: contract.sequenceScalar))
                guard entry.seqAxis < spec.rank else {
                    throw .manifestMismatch(
                        field: "\(prefix).states.sequence.\(i).seqAxis", expected: "an axis of '\(name)' (rank \(spec.rank))",
                        found: "\(entry.seqAxis)")
                }
            }
        }
        for (i, entry) in states.recurrent.enumerated() {
            let field = "\(prefix).states.recurrent.\(i).name"
            guard let spec = signature.state(entry.name) else {
                throw .manifestMismatch(
                    field: field, expected: "a graph state named '\(entry.name)'", found: "states \(signature.states.map(\.name))")
            }
            try check(spec, field: field, role: .init(required: true, rank: nil, scalar: contract.recurrentScalar))
        }
        let listed = Set(states.allNames)
        if let unlisted = signature.states.first(where: { !listed.contains($0.name) }) {
            throw .manifestMismatch(
                field: "\(prefix).states", expected: "every graph state listed as sequence or recurrent",
                found: "graph state '\(unlisted.name)' is not listed")
        }

        // A static token axis must be declared, and a declared one must match.
        if let tokens = inputs["tokens"], let spec = signature.input(tokens.name), spec.rank >= 2 {
            let axis = spec.shape[1]
            let declared = graph.limits?.staticSequence?.values(forPhase: phase)
            let field = "\(prefix).limits.staticSequence"
            switch (axis == TensorSpec.dynamic, declared) {
            case (true, let values?):
                throw .manifestMismatch(field: field, expected: "\(values) for phase \(phase)", found: "a dynamic token axis")
            case (false, nil):
                throw .manifestMismatch(field: field, expected: "unset (dynamic token axis)", found: "a static token axis of \(axis) in phase \(phase)")
            case (false, let values?) where !values.contains(axis):
                throw .manifestMismatch(field: field, expected: "\(values) for phase \(phase)", found: "\(axis)")
            default:
                break
            }
        }
    }

    private static func check(_ spec: TensorSpec, field: String, role: DriverContract.RoleSpec?) throws(AetherError) {
        guard spec.kind == .ndArray else {
            throw .manifestMismatch(field: field, expected: "an NDArray", found: "\(spec.kind) '\(spec.name)'")
        }
        guard let role else { return }
        if let rank = role.rank, spec.rank != rank {
            throw .manifestMismatch(field: field, expected: "rank \(rank)", found: "'\(spec.name)' \(spec.summary)")
        }
        if let scalar = role.scalar {
            guard scalar.admits(spec.scalarType), spec.scalarType.byteWidth != nil else {
                throw .manifestMismatch(
                    field: field, expected: "a supported \(scalar.rawValue) type", found: "'\(spec.name)' \(spec.summary)")
            }
        }
    }
}
