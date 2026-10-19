import AetherKit
import Foundation

/// Fixed-input outer-module gate. Manifests are trusted local files; they may
/// refer to external assets, which must stay immutable throughout use.
public struct H3PhaseProbePlan: Sendable {
    public struct Case: Sendable {
        public let name: String
        public let model: H3ModelAsset
        public let inputs: [H3TensorFile]
        public let references: [H3TensorFile]
        private let timestepPlan: H3Layout.RowTimesteps?
        init(name: String, model: H3ModelAsset, inputs: [H3TensorFile], references: [H3TensorFile], timestepPlan: H3Layout.RowTimesteps? = nil) {
            self.name = name; self.model = model; self.inputs = inputs; self.references = references; self.timestepPlan = timestepPlan
        }
        public var boundBytes: Int { (inputs + references).reduce(0) { $0 + $1.spec.byteCount } }
        public func verifyInputs() throws {
            for input in inputs { try validate(input.read(), name: input.name) }
        }
        fileprivate func verifyTimestepInputs() throws {
            for input in inputs where input.name == "timestep" || input.name == "indices" {
                try validate(input.read(), name: input.name)
            }
        }
        private func validate(_ tensor: H3TensorData, name: String) throws {
            guard let plan = timestepPlan else { return }
            if name == "timestep" {
                guard try tensor.floatValues().map(\.bitPattern) == plan.values.map(\.bitPattern) else {
                    throw AetherError.invalidTensor("H3 timestep values differ from captured Turbo8 schedule")
                }
            } else if name == "indices" {
                let valid = tensor.bytes.withUnsafeBytes { raw in
                    plan.indices.enumerated().allSatisfy { i, expected in
                        Int32(bitPattern: raw.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self).littleEndian) == expected
                    }
                }
                guard valid else { throw AetherError.invalidTensor("H3 head indices differ from actual row timesteps") }
            }
        }
    }
    public let header: H3ExportHeader
    public let phaseManifestSHA256: String
    public let cpuRecordSHA256: String
    public let cases: [Case]
    public let relativeL2Limit = 0.02

    public init(bundle: URL, phases: URL, cpuRecord: URL) throws {
        guard bundle.isFileURL, phases.isFileURL, cpuRecord.isFileURL else {
            throw AetherError.invalidModel("H3 phase manifests must be local")
        }
        // Reuse corrected header, all-block declarations, canonical schedule,
        // geometry and body-oracle linkage validation without acquiring native resources.
        let backbone = try H3BlockProbePlan(bundle: bundle, phases: phases)
        let data = try Data(contentsOf: bundle), phaseData = try Data(contentsOf: phases)
        let cpuData = try Data(contentsOf: cpuRecord), decoder = JSONDecoder()
        let source = try decoder.decode(Source.self, from: data)
        let manifest = try decoder.decode(Manifest.self, from: phaseData)
        let cpu = try decoder.decode(CPURecord.self, from: cpuData)
        guard h3SHA256(data) == backbone.header.manifestSHA256,
              h3SHA256(phaseData) == backbone.phaseManifestSHA256,
              cpu.schema == 1, cpu.passed, !cpu.gpu_work,
              cpu.bundle_sha256 == backbone.header.manifestSHA256,
              h3IsSHA256(cpu.source_recipe_sha256) else {
            throw AetherError.invalidModel("H3 phase CPU provenance differs from corrected bundle")
        }
        let layout = try H3Layout(geometry: backbone.geometry, textTags: Array(repeating: 1, count: source.text_rows))
        let schedule = try H3Schedule()
        let plans = schedule.steps.prefix(2).map { layout.timesteps(for: $0) }
        let video = try H3TensorSpec(scalarType: .float32, shape: [1, backbone.geometry.targetVideoRows, 96])
        let audio = try H3TensorSpec(scalarType: .float32, shape: [1, layout.audioIndices.count, 32])
        let text = try H3TensorSpec(scalarType: .bfloat16, shape: [1, source.text_rows, 5120])
        let hidden = try H3TensorSpec(scalarType: .bfloat16, shape: [1, layout.rows, 5376])
        var result: [Case] = []
        let names = ["prepare", "time-0", "heads-0", "time-1", "heads-1"]
        for name in names {
            let matches = manifest.cases.filter { $0.name == name }
            let originals = cpu.cases.filter { $0.name == name }
            guard matches.count == 1, originals.count == 1 else { throw AetherError.invalidModel("missing or duplicate H3 outer phase: \(name)") }
            let item = matches[0], original = originals[0]
            let model = try item.model.asset(base: phases)
            let expectedModel = try (name == "prepare" ? source.prepare : name.hasPrefix("time-") ? source.time : source.heads).asset(base: bundle)
            guard Self.same(model, expectedModel), Self.same(model, try original.model.asset(base: cpuRecord)) else {
                throw AetherError.invalidModel("H3 outer graph identity differs")
            }
            let ordinal = name.hasSuffix("1") ? 1 : 0
            let n = plans[ordinal].values.count
            let temb = try H3TensorSpec(scalarType: .float32, shape: [n, 2688])
            let inputSpecs: [String: H3TensorSpec]
            let outputSpecs: [String: H3TensorSpec]
            if name == "prepare" {
                inputSpecs = ["video": video, "audio": audio, "text": text]; outputSpecs = ["hidden": hidden]
            } else if name.hasPrefix("time-") {
                inputSpecs = ["timestep": try H3TensorSpec(scalarType: .float32, shape: [n])]; outputSpecs = ["temb": temb]
            } else {
                inputSpecs = ["hidden": hidden, "temb": temb, "indices": try H3TensorSpec(scalarType: .int32, shape: [layout.rows])]
                outputSpecs = ["video": video, "audio": audio]
            }
            let inputs = try item.inputs.map { try $0.file(base: phases) }
            let references = try item.outputs.map { try $0.file(base: phases) }
            try Self.validate(inputs, specs: inputSpecs); try Self.validate(references, specs: outputSpecs)
            // CPU record pins every selected input and complete reference, not just
            // a passed flag. Absolute paths can differ only through symlinks.
            try Self.match(inputs, try original.inputs.map { try $0.file(base: cpuRecord) })
            try Self.match(references, try original.outputs.map { try $0.file(base: cpuRecord) })
            guard item.outputs.allSatisfy({ $0.row_indices == nil && $0.native_shape == nil }) else {
                throw AetherError.invalidModel("H3 outer gate requires complete CPU references")
            }
            let value = Case(name: name, model: model, inputs: inputs, references: references, timestepPlan: name == "prepare" ? nil : plans[ordinal])
            guard value.boundBytes <= 2 << 30 else { throw AetherError.invalidModel("H3 outer buffers exceed 2 GiB admission") }
            result.append(value)
        }
        // Outer preparation is the already qualified backbone's initial boundary.
        let prepared = result[0].references[0]
        let initial = backbone.cases[0].inputs.first { $0.name == "hidden" }!
        guard prepared.spec == initial.spec, prepared.sha256 == initial.sha256 else {
            throw AetherError.invalidModel("H3 prepare output differs from the backbone input context")
        }
        for ordinal in 0..<2 {
            let time = result[1 + ordinal * 2], heads = result[2 + ordinal * 2]
            guard let embedding = heads.inputs.first(where: { $0.name == "temb" }),
                  embedding.spec == time.references[0].spec, embedding.sha256 == time.references[0].sha256,
                  heads.inputs.first(where: { $0.name == "hidden" })?.sha256 == prepared.sha256 else {
                throw AetherError.invalidModel("H3 outer phase contexts do not link")
            }
        }
        header = backbone.header; phaseManifestSHA256 = backbone.phaseManifestSHA256
        cpuRecordSHA256 = h3SHA256(cpuData); cases = result
    }

    /// CPU verification only: no Core AI load, Metal device or graph execution.
    public func verify() throws {
        for item in cases {
            try item.model.verify(); try item.verifyInputs()
            for reference in item.references { _ = try reference.read() }
        }
        try verifyTimestepInputs()
    }

    /// Checks the two captured row plans without reading models or references.
    public func verifyTimestepInputs() throws {
        for item in cases { try item.verifyTimestepInputs() }
    }

    private static func same(_ a: H3ModelAsset, _ b: H3ModelAsset) -> Bool {
        a.url.resolvingSymlinksInPath() == b.url.resolvingSymlinksInPath() && a.sha256 == b.sha256 && a.function == b.function
    }
    private static func validate(_ files: [H3TensorFile], specs: [String: H3TensorSpec]) throws {
        guard files.count == specs.count, Set(files.map(\.name)) == Set(specs.keys),
              files.allSatisfy({ specs[$0.name] == $0.spec }) else { throw AetherError.invalidModel("H3 outer tensor names, shape or precision differ") }
    }
    private static func match(_ files: [H3TensorFile], _ originals: [H3TensorFile]) throws {
        guard files.count == originals.count, Set(originals.map(\.name)).count == originals.count else {
            throw AetherError.invalidModel("duplicate or missing H3 CPU identity")
        }
        for file in files {
            guard let original = originals.first(where: { $0.name == file.name }),
                  file.spec == original.spec, file.sha256 == original.sha256,
                  file.url.resolvingSymlinksInPath() == original.url.resolvingSymlinksInPath() else {
                throw AetherError.invalidModel("H3 outer tensor differs from original CPU record")
            }
        }
    }
}

private struct Model: Decodable {
    let path: String; let function: String; let sha256: String
    func asset(base: URL) throws -> H3ModelAsset { try H3ModelAsset(url: resolve(path, base), function: function, sha256: sha256) }
}
private struct Tensor: Decodable {
    let name: String; let path: String; let dtype: H3ScalarType; let shape: [Int]; let sha256: String
    let row_indices: [Int]?; let native_shape: [Int]?
    func file(base: URL) throws -> H3TensorFile {
        try H3TensorFile(name: name, url: resolve(path, base), spec: H3TensorSpec(scalarType: dtype, shape: shape), sha256: sha256)
    }
}
private struct Item: Decodable { let name: String; let model: Model; let inputs: [Tensor]; let outputs: [Tensor] }
private struct Source: Decodable { let text_rows: Int; let prepare: Model; let time: Model; let heads: Model }
private struct Manifest: Decodable { let cases: [Item] }
private struct CPURecord: Decodable {
    let schema: Int; let passed: Bool; let gpu_work: Bool
    let source_recipe_sha256: String; let bundle_sha256: String; let cases: [Item]
}
private func resolve(_ path: String, _ base: URL) throws -> URL {
    guard !path.isEmpty, !path.contains("://") else { throw AetherError.invalidModel("H3 outer assets must be local") }
    return URL(fileURLWithPath: path, relativeTo: base.deletingLastPathComponent()).standardizedFileURL
}
