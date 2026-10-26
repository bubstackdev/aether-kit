import AetherKit
import Foundation

/// Corrected production chains: three resident blocks or fifty streamed blocks.
/// Numerical checks use captured native contexts, not a CPU chain trajectory.
public struct H3ChainProbePlan: Sendable {
    public struct Step: Sendable {
        public let block: Int
        public let weights: [H3TensorFile]
        public let table: H3TensorFile
        public let reference: H3TensorFile?
        public let expectedOutputSHA256: String?
    }
    public struct Case: Sendable {
        public let base: H3BlockProbePlan.Case
        public let steps: [Step]
        public let streamsWeights: Bool
        public let referenceBlocks: [Int]
        public var boundBytes: Int {
            let fixed = base.inputs.filter { !$0.name.hasPrefix("w") && $0.name != "temb" }
            return fixed.reduce(2 * base.output.byteCount) { $0 + $1.spec.byteCount }
                + steps.reduce(0) { $0 + $1.table.spec.byteCount }
                + (streamsWeights ? (steps.map { $0.weights.reduce(0) { $0 + $1.spec.byteCount } }.max() ?? 0)
                    : steps.reduce(0) { $0 + $1.weights.reduce(0) { $0 + $1.spec.byteCount } })
        }
        public func verify() throws {
            try base.model.verify(); try base.verifyInputs()
            for step in steps {
                _ = try step.table.read()
                for weight in step.weights { _ = try weight.read() }
                if let reference = step.reference { _ = try reference.read() }
            }
        }
    }
    public let blockPlan: H3BlockProbePlan
    public let manifestSHA256: String
    public let cases: [Case]
    public let referenceScope = "Independent CPU block comparisons using complete captured native input contexts; not independent CPU trajectory parity"

    public init(bundle: URL, phases: URL, chain: URL, requireReferences: Bool) throws {
        blockPlan = try H3BlockProbePlan(bundle: bundle, phases: phases)
        let data = try Data(contentsOf: chain)
        manifestSHA256 = h3SHA256(data)
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        let bundleData = try Data(contentsOf: bundle)
        guard h3SHA256(bundleData) == blockPlan.header.manifestSHA256 else { throw AetherError.invalidModel("H3 bundle changed during chain inspection") }
        let source = try JSONDecoder().decode(Weights.self, from: bundleData)
        let streams = manifest.schema == 2
        let blocks = streams ? Array(0..<50) : [0,1,2]
        let references = streams ? [0,1,2,24,49] : [0,1,2]
        guard (manifest.schema == 1 || manifest.schema == 2), manifest.block_indices == blocks,
              !streams || (manifest.weight_residency == "one_block" && manifest.reference_blocks == references),
              manifest.bundle_sha256 == blockPlan.header.manifestSHA256,
              manifest.phase_manifest_sha256 == blockPlan.phaseManifestSHA256,
              manifest.predefined_relative_l2_limit == blockPlan.relativeL2Limit,
              h3IsSHA256(manifest.oracle_recipe_sha256), manifest.cases.count == 2,
              Set(manifest.cases.map(\.name)) == Set(blockPlan.cases.map(\.name)) else {
            throw AetherError.invalidModel("H3 chain scope, source identity or numerical contract differs")
        }
        var cases: [Case] = []
        for base in blockPlan.cases {
            let item = manifest.cases.first { $0.name == base.name }!
            guard item.steps.map(\.block) == blocks,
                  let baseTable = base.inputs.first(where: { $0.name == "temb" }) else {
                throw AetherError.invalidModel("H3 chain block order differs from its admitted profile")
            }
            var steps: [Step] = []
            for step in item.steps {
                let weights = try source.parameters[step.block].map { try $0.file(relativeTo: bundle) }
                let table = try step.temb.file(relativeTo: chain)
                guard table.name == "temb", table.spec == baseTable.spec,
                      step.block != 0 || table.sha256 == baseTable.sha256 else {
                    throw AetherError.invalidModel("H3 chain modulation type/cardinality or block-0 identity differs")
                }
                let reference = try step.reference?.file(relativeTo: chain)
                guard step.expected_output_sha256.map(h3IsSHA256) ?? true,
                      !requireReferences || (step.expected_output_sha256 != nil && (!references.contains(step.block) || reference != nil)) else {
                    throw AetherError.invalidModel("H3 chain lacks pinned outputs or required CPU checkpoint references")
                }
                if let reference {
                    guard reference.name == "hidden_out", reference.spec == base.reference.spec,
                          step.reference?.row_indices == base.referenceRows,
                          step.reference?.native_shape == base.output.shape,
                          step.block != 0 || reference.sha256 == base.reference.sha256 else {
                        throw AetherError.invalidModel("H3 chain reference rows, extent or block-0 identity differs")
                    }
                }
                steps.append(Step(block: step.block, weights: weights, table: table, reference: reference,
                                  expectedOutputSHA256: step.expected_output_sha256))
            }
            let value = Case(base: base, steps: steps, streamsWeights: streams, referenceBlocks: references)
            guard value.boundBytes <= 4 << 30 else { throw AetherError.invalidModel("H3 chain exceeds the 4 GiB buffer admission bound") }
            cases.append(value)
        }
        self.cases = cases
    }
    public func verify() throws { for item in cases { try item.verify() } }
}

private struct Weights: Decodable { let parameters: [[ChainTensor]] }
private struct Manifest: Decodable {
    let schema: Int; let block_indices: [Int]
    let weight_residency: String?; let reference_blocks: [Int]?
    let bundle_sha256: String; let phase_manifest_sha256: String
    let predefined_relative_l2_limit: Double; let oracle_recipe_sha256: String
    let cases: [Item]
    struct Item: Decodable {
        let name: String; let steps: [Entry]
        struct Entry: Decodable {
            let block: Int; let temb: ChainTensor; let reference: ChainTensor?
            let expected_output_sha256: String?
        }
    }
}
private struct ChainTensor: Decodable {
    let name: String; let path: String; let dtype: H3ScalarType; let shape: [Int]; let sha256: String
    let row_indices: [Int]?; let native_shape: [Int]?
    func file(relativeTo base: URL) throws -> H3TensorFile {
        guard !path.isEmpty, !path.contains("://") else { throw AetherError.invalidModel("H3 chain requires local assets") }
        return try H3TensorFile(name: name, url: URL(fileURLWithPath: path, relativeTo: base.deletingLastPathComponent()),
            spec: H3TensorSpec(scalarType: dtype, shape: shape), sha256: sha256)
    }
}
