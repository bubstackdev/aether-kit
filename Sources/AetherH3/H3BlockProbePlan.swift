import AetherKit
import CryptoKit
import Foundation

/// A declared model stream. Verification is explicit and performs no model load.
public struct H3ModelAsset: Sendable {
    public let url: URL
    public let function: String
    public let sha256: String
    public init(url: URL, function: String, sha256: String) throws {
        guard url.isFileURL, !function.isEmpty, h3IsSHA256(sha256) else { throw AetherError.invalidModel("invalid H3 model identity") }
        self.url = url.standardizedFileURL; self.function = function; self.sha256 = sha256
    }
    public func verify() throws {
        guard try Data(contentsOf: url.appendingPathComponent("main.hash")) == Data((0..<32).map {
            let start = sha256.index(sha256.startIndex, offsetBy: $0 * 2)
            return UInt8(sha256[start..<sha256.index(start, offsetBy: 2)], radix: 16)!
        }) else { throw AetherError.invalidModel("H3 model declaration differs from manifest") }
        let file = try FileHandle(forReadingFrom: url.appendingPathComponent("main.mlirb"))
        defer { try? file.close() }
        var hasher = SHA256()
        // FileHandle bridges through Foundation autoreleased storage. A long
        // synchronous verification pass must drain each chunk, rather than keep
        // tens of GB alive until the next asynchronous suspension/autorelease pool.
        while try autoreleasepool(invoking: { () throws -> Bool in
            guard let bytes = try file.read(upToCount: 8 << 20), !bytes.isEmpty else { return false }
            try Task.checkCancellation(); hasher.update(data: bytes); return true
        }) {}
        guard hasher.finalize().map({ String(format: "%02x", $0) }).joined() == sha256 else {
            throw AetherError.invalidModel("H3 model stream hash mismatch")
        }
    }
}

/// CPU-inspected, corrected block-0 probe inputs for the two actual T2VA timestep
/// cardinalities. This validates the selected block route, not the whole pipeline.
/// Supplied manifests are trusted local files and may reference external files.
public struct H3BlockProbePlan: Sendable {
    public struct Case: Sendable {
        public let name: String
        public let model: H3ModelAsset
        public let inputs: [H3TensorFile]
        public let output: H3TensorSpec
        public let reference: H3TensorFile
        public let referenceRows: [Int]
        public var boundBytes: Int { inputs.reduce(output.byteCount) { $0 + $1.spec.byteCount } }
        public func verifyInputs() throws {
            guard let table = inputs.first(where: { $0.name == "temb" }), table.spec.shape.count == 3 else {
                throw AetherError.invalidModel("H3 block lacks its modulation table")
            }
            for input in inputs {
                let data = try input.read()
                if input.name == "tags" {
                    let valid = data.bytes.withUnsafeBytes { raw in
                        stride(from: 0, to: raw.count, by: 4).allSatisfy {
                            let index = Int(Int32(bitPattern: raw.loadUnaligned(fromByteOffset: $0, as: UInt32.self).littleEndian))
                            return (0..<table.spec.shape[1]).contains(index)
                        }
                    }
                    guard valid else { throw AetherError.invalidTensor("H3 modulation index is outside the table") }
                }
            }
        }
    }
    public let header: H3ExportHeader
    public let phaseManifestSHA256: String
    public let geometry: H3Geometry
    public let cases: [Case]
    public let relativeL2Limit: Double = 0.02

    public init(bundle: URL, phases: URL) throws {
        guard bundle.isFileURL, phases.isFileURL else { throw AetherError.invalidModel("H3 manifests must be local") }
        let data = try Data(contentsOf: bundle), phaseData = try Data(contentsOf: phases)
        header = try H3ExportHeader(data: data); phaseManifestSHA256 = h3SHA256(phaseData)
        let decoder = JSONDecoder()
        let source = try decoder.decode(Backbone.self, from: data)
        let phase = try decoder.decode(Phase.self, from: phaseData)
        let geometry = try H3Geometry(width: source.width, height: source.height, requestedFrames: source.frames, diagnostic: true)
        guard geometry.frames == source.frames, source.frames >= 22, (1...4096).contains(source.text_rows),
              source.bodies.count == 2, source.parameters.count == 50,
              phase.schema == 1, phase.predefined_relative_l2_limit == 0.02 else {
            throw AetherError.invalidModel("H3 backbone geometry, cardinalities or oracle contract differs")
        }
        let rows = try H3Layout(geometry: geometry, textTags: Array(repeating: 1, count: source.text_rows)).rows
        for block in source.parameters {
            guard block.count == 22 else { throw AetherError.invalidModel("H3 Turbo8 requires all 22 dynamic bindings") }
            for (i, tensor) in block.enumerated() {
                guard tensor.name == "w\(i)", tensor.dtype == .bfloat16, tensor.shape == Self.weightShapes[i] else {
                    throw AetherError.invalidModel("H3 Turbo8 weight binding layout differs")
                }
                _ = try tensor.file(relativeTo: bundle)
            }
        }
        let scheduleData = try Data(contentsOf: resolve(source.schedule, relativeTo: bundle))
        guard let canonical = Bundle.module.url(forResource: "turbo8-schedule", withExtension: "json", subdirectory: "Resources"),
              try JSONSerialization.jsonObject(with: scheduleData) as? NSDictionary == JSONSerialization.jsonObject(with: Data(contentsOf: canonical)) as? NSDictionary else {
            throw AetherError.invalidModel("H3 bundle schedule differs from pinned Turbo8 grid")
        }
        var result: [Case] = []
        for ordinal in 0..<2 {
            let name = "body-\(ordinal)", matches = phase.cases.filter { $0.name == name }
            guard matches.count == 1 else { throw AetherError.invalidModel("missing or duplicate H3 oracle case: \(name)") }
            let item = matches[0]
            let model = try item.model.asset(relativeTo: phases), declared = try source.bodies[ordinal].asset(relativeTo: bundle)
            guard model.url.resolvingSymlinksInPath() == declared.url.resolvingSymlinksInPath(),
                  model.sha256 == declared.sha256, model.function == declared.function,
                  item.outputs.count == 1, item.inputs.count == 27, Set(item.inputs.map(\.name)).count == 27 else {
                throw AetherError.invalidModel("H3 probe model or input identities differ")
            }
            let roots: [String: (H3ScalarType, [Int])] = [
                "hidden": (.bfloat16, [1, rows, 5376]), "temb": (.bfloat16, [6, (ordinal + 1) * 3, 5376]),
                "tags": (.int32, [rows]), "cos": (.float32, [rows, 96]), "sin": (.float32, [rows, 96]),
            ]
            guard Set(item.inputs.map(\.name)) == Set(roots.keys).union(source.parameters[0].map(\.name)) else {
                throw AetherError.invalidModel("H3 block root/weight names differ")
            }
            for input in item.inputs {
                if let root = roots[input.name] {
                    guard input.dtype == root.0, input.shape == root.1 else { throw AetherError.invalidModel("H3 root binding shape/type differs") }
                } else {
                    guard let i = source.parameters[0].firstIndex(where: { $0.name == input.name }) else { throw AetherError.invalidModel("unknown H3 weight binding") }
                    let expected = try source.parameters[0][i].file(relativeTo: bundle), actual = try input.file(relativeTo: phases)
                    guard expected.url.resolvingSymlinksInPath() == actual.url.resolvingSymlinksInPath(), expected.sha256 == actual.sha256,
                          expected.spec == actual.spec else { throw AetherError.invalidModel("H3 probe weight differs from corrected block 0") }
                }
            }
            for (name, tensor) in [("cos", source.rotary_cos), ("sin", source.rotary_sin)] {
                let actual = try item.inputs.first { $0.name == name }!.file(relativeTo: phases), expected = try tensor.file(relativeTo: bundle)
                guard expected.spec == actual.spec, expected.sha256 == actual.sha256 else { throw AetherError.invalidModel("H3 rotary identity differs") }
            }
            let expected = item.outputs[0], output = try H3TensorSpec(scalarType: .bfloat16, shape: [1, rows, 5376])
            guard expected.name == "hidden_out", expected.dtype == .bfloat16, expected.native_shape == output.shape,
                  let selectedRows = expected.row_indices, !selectedRows.isEmpty, Set(selectedRows).count == selectedRows.count,
                  selectedRows.allSatisfy({ (0..<rows).contains($0) }), expected.shape == [1, selectedRows.count, 5376] else {
                throw AetherError.invalidModel("H3 oracle row selection differs")
            }
            let value = Case(name: name, model: model, inputs: try item.inputs.map { try $0.file(relativeTo: phases) },
                             output: output, reference: try expected.file(relativeTo: phases), referenceRows: selectedRows)
            guard value.boundBytes <= 2 << 30 else { throw AetherError.invalidModel("H3 probe exceeds the 2 GiB buffer admission limit") }
            result.append(value)
        }
        self.geometry = geometry; self.cases = result
    }

    /// Hashes/validates only the selected models, input tensors and CPU references.
    /// No native models or Metal buffers are acquired. Does not verify all 50 blocks.
    public func verify() throws {
        for item in cases {
            try item.model.verify()
            try item.verifyInputs()
            _ = try item.reference.read()
        }
    }
    private static let weightShapes = [[5376],[7168,5376],[128,5376],[7168,128],[7168,5376],[128,5376],[7168,128],
        [7168,5376],[128,5376],[7168,128],[128],[128],[5376,7168],[128,7168],[5376,128],[5376],
        [28672,5376],[128,5376],[28672,128],[5376,14336],[128,14336],[5376,128]]
}

private struct Model: Decodable {
    let path: String; let function: String; let sha256: String
    func asset(relativeTo base: URL) throws -> H3ModelAsset { try H3ModelAsset(url: resolve(path, relativeTo: base), function: function, sha256: sha256) }
}
private struct TensorFile: Decodable {
    let name: String; let path: String; let dtype: H3ScalarType; let shape: [Int]; let sha256: String
    var native_shape: [Int]?; var row_indices: [Int]?
    func file(relativeTo base: URL) throws -> H3TensorFile {
        try H3TensorFile(name: name, url: resolve(path, relativeTo: base), spec: H3TensorSpec(scalarType: dtype, shape: shape), sha256: sha256)
    }
}
private struct Backbone: Decodable {
    let width: Int; let height: Int; let frames: Int; let text_rows: Int; let schedule: String
    let bodies: [Model]; let parameters: [[TensorFile]]; let rotary_cos: TensorFile; let rotary_sin: TensorFile
}
private struct Phase: Decodable {
    let schema: Int; let predefined_relative_l2_limit: Double; let cases: [Item]
    struct Item: Decodable { let name: String; let model: Model; let inputs: [TensorFile]; let outputs: [TensorFile] }
}
private func resolve(_ path: String, relativeTo base: URL) throws -> URL {
    guard !path.isEmpty, !path.contains("://") else { throw AetherError.invalidModel("H3 assets must be local paths") }
    return URL(fileURLWithPath: path, relativeTo: base.deletingLastPathComponent()).standardizedFileURL
}
