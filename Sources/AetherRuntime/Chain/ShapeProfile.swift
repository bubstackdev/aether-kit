public import AetherCore

/// The fixed set of input shapes a driver runs a profile at (DESIGN.md §6.6).
///
/// Core AI keeps a specialization's memory for every distinct input shape until the model is released, even with
/// `expectFrequentReshapes`, and returning to an earlier shape can grow it again (lab/h3-spike §3). So a footprint
/// measured for a profile (a canvas and frame count, say) holds only while every call stays inside the profile's
/// shapes. The profile names those shapes; ``GraphChain`` refuses a call outside them before anything runs, and the
/// profile's ``identity`` keys its ``FootprintLedger`` entries. (0.0.x kept this invariant implicitly, by padding every
/// H3 step to one shape.)
public struct ShapeProfile: Sendable, Hashable {
    /// What the profile is for, as a driver names it (`"960x544x124"`).
    public var name: String
    /// Per graph role, the shapes each input may take. An input not listed is unconstrained (a weight, whose shape
    /// the graph fixes anyway); a role not listed is refused.
    public var shapes: [String: [String: Set<[Int]>]]

    public init(name: String, shapes: [String: [String: Set<[Int]>]]) {
        self.name = name
        self.shapes = shapes
    }

    /// A stable string for the profile: its name and its shapes in a canonical order. Two profiles with the same name
    /// but different shapes get different identities, so their measurements never mix.
    public var identity: String {
        let body = shapes.keys.sorted().map { role in
            let inputs = shapes[role]!.keys.sorted().map { input in
                let list = shapes[role]![input]!.map { $0.map(String.init).joined(separator: "x") }.sorted()
                return "\(input)=\(list.joined(separator: "|"))"
            }
            return "\(role){\(inputs.joined(separator: ","))}"
        }
        return "\(name):\(body.joined(separator: ";"))"
    }

    /// The ledger key of a phase measured under this profile.
    public func ledgerKey(model: String, phase: Manifest.Residency) -> FootprintLedger.Key {
        FootprintLedger.Key(model: model, profile: identity, phase: phase)
    }

    /// Whether `shape` is one of input `name`'s shapes for `role` (unconstrained inputs always are).
    public func allows(_ role: String, input name: String, shape: [Int]) -> Bool {
        guard let inputs = shapes[role] else { return false }
        guard let allowed = inputs[name] else { return true }
        return allowed.contains(shape)
    }

    /// Refuses bindings whose input shapes are outside the profile for `role`.
    public func check(_ role: String, _ bindings: Bindings) throws(AetherError) {
        guard shapes[role] != nil else {
            throw .tensor(name: role, reason: .unsupported("graph role '\(role)' is not in shape profile '\(name)'"))
        }
        for bound in bindings.inputs where !allows(role, input: bound.name, shape: bound.shape) {
            throw .tensor(
                name: bound.name,
                reason: .unsupported("shape \(bound.shape) is outside shape profile '\(name)' for '\(role)', "
                                     + "so its measured footprint would not hold"))
        }
    }
}
