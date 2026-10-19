// Derived from the Apache-2.0 Aether prototype. See NOTICE.
import AetherKit
import Foundation

/// A complete exported H3 profile (the prototype forge's schema 1 `bundle.json`), validated
/// structurally on open with no model load. A profile pins canvas, frame grid and text-row
/// count. `verifyModels()` hashes every model stream once; `verifyTensors()` hashes every bound
/// weight and table file. Manifests are trusted local data and may name external absolute paths;
/// all referenced files must stay immutable from verification through use.
public struct H3Bundle: Sendable {
    public struct Conditioner: Sendable { public let embedding: H3ModelAsset; public let layers: [H3ModelAsset] }
    public let header: H3ExportHeader
    public let geometry: H3Geometry
    public let textRows: Int
    public let tokenizer: URL
    public let schedule: URL
    public let prepare: H3ModelAsset, time: H3ModelAsset, heads: H3ModelAsset
    public let bodies: [H3ModelAsset]
    public let modulations: [H3ModelAsset]
    public let parameters: [[H3TensorFile]]
    public let rotaryCos: H3TensorFile, rotarySin: H3TensorFile
    public let videoDecoder: H3ModelAsset
    public let videoParameters: [H3TensorFile]
    public let videoTiling: H3SpatialTiling?
    public let audioDecoder: H3ModelAsset
    public let conditioner: Conditioner?

    /// Every distinct model stream, in a stable order.
    public var models: [H3ModelAsset] {
        var seen = Set<String>(), result: [H3ModelAsset] = []
        let all = [prepare, time, heads, videoDecoder, audioDecoder] + bodies + modulations
            + (conditioner.map { [$0.embedding] + $0.layers } ?? [])
        for model in all where seen.insert("\(model.url.path)\u{0}\(model.sha256)").inserted { result.append(model) }
        return result
    }
    /// Every distinct bound tensor file.
    public var tensors: [H3TensorFile] {
        var seen = Set<String>(), result: [H3TensorFile] = []
        for file in parameters.flatMap({ $0 }) + [rotaryCos, rotarySin] + videoParameters
        where seen.insert("\(file.url.path)\u{0}\(file.sha256)").inserted { result.append(file) }
        return result
    }
    /// Declared bytes of bound tensors (model streams are counted by `diskBytes()`).
    public var tensorBytes: Int { tensors.reduce(0) { $0 + $1.spec.byteCount } }

    public init(url: URL) throws {
        guard url.isFileURL else { throw AetherError.invalidModel("H3 bundles must be local") }
        let data = try Data(contentsOf: url)
        let m: Manifest
        do {
            header = try H3ExportHeader(data: data)
            m = try JSONDecoder().decode(Manifest.self, from: data)
        } catch let error as DecodingError {
            throw AetherError.invalidModel("H3 bundle manifest is malformed or incomplete: \(error)")
        }
        geometry = try H3Geometry(width: m.width, height: m.height, requestedFrames: m.frames, diagnostic: true)
        guard geometry.frames == m.frames else { throw AetherError.invalidModel("H3 bundle frame grid is not aligned") }
        guard (1...4096).contains(m.text_rows), m.bodies.count == 2, m.modulations.count == 50, m.parameters.count == 50,
              m.parameters.allSatisfy({ $0.count == 22 && $0.enumerated().allSatisfy { $1.name == "w\($0)" } }) else {
            throw AetherError.invalidModel("H3 bundle must contain both actual T2VA cardinalities and all 50 blocks")
        }
        guard m.conditioner.map({ $0.layers.count == 50 }) ?? true else {
            throw AetherError.invalidModel("H3 conditioner must preserve the unnormalized 50-layer tap")
        }
        let video = m.video_parameters ?? []
        guard video.isEmpty || (video.count == 703 && video.enumerated().allSatisfy { $1.name == "w\($0)" && $1.dtype == "float32" }) else {
            throw AetherError.invalidModel("H3 functional video decoder bindings differ")
        }
        if let tiling = m.video_spatial_tiles {
            try tiling.validate(width: m.width, height: m.height)
            guard !video.isEmpty else { throw AetherError.invalidModel("H3 spatial export requires explicit decoder weights") }
        }
        let windowed = m.video_window_latents == 7 && m.video_window_frames == 22 && m.video_window_stride == 5
        let single = m.frames == 22 && m.video_window_latents == nil
        guard geometry.latentFrames >= 7, windowed || single else {
            throw AetherError.invalidModel("H3 bundle lacks the source temporal decoder-window contract")
        }
        func path(_ value: String) throws -> URL {
            guard value.hasPrefix("/"), !value.utf8.contains(0) else { throw AetherError.invalidModel("H3 bundle paths must be absolute") }
            return URL(fileURLWithPath: value)
        }
        textRows = m.text_rows; tokenizer = try path(m.tokenizer); schedule = try path(m.schedule)
        prepare = try m.prepare.asset(path); time = try m.time.asset(path); heads = try m.heads.asset(path)
        bodies = try m.bodies.map { try $0.asset(path) }; modulations = try m.modulations.map { try $0.asset(path) }
        parameters = try m.parameters.map { try $0.map { try $0.file(path) } }
        rotaryCos = try m.rotary_cos.file(path); rotarySin = try m.rotary_sin.file(path)
        videoDecoder = try m.video_decoder.asset(path); videoParameters = try video.map { try $0.file(path) }
        videoTiling = m.video_spatial_tiles; audioDecoder = try m.audio_decoder.asset(path)
        conditioner = try m.conditioner.map { Conditioner(embedding: try $0.embedding.asset(path), layers: try $0.layers.map { try $0.asset(path) }) }
    }

    /// Hashes every distinct model stream (`main.mlirb` against the manifest and `main.hash`).
    public func verifyModels(progress: (Int, Int) -> Void = { _, _ in }) throws {
        let all = models
        for (index, model) in all.enumerated() { try model.verify(); progress(index + 1, all.count) }
    }

    /// Hashes every bound tensor file, one at a time, without retaining their contents.
    public func verifyTensors(progress: (Int, Int) -> Void = { _, _ in }) throws {
        let all = tensors
        for (index, file) in all.enumerated() { try autoreleasepool { _ = try file.read() }; progress(index + 1, all.count) }
    }

    /// Bytes on disk of every referenced model directory and tensor file (each counted once).
    /// Core AI's derived compiled copies live elsewhere and are not included.
    public func diskBytes() throws -> Int64 {
        var total: Int64 = 0
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .totalFileAllocatedSizeKey, .fileAllocatedSizeKey]
        for model in models {
            guard let walker = FileManager.default.enumerator(at: model.url, includingPropertiesForKeys: Array(keys)) else {
                throw AetherError.invalidModel("H3 model directory is unreadable")
            }
            for case let file as URL in walker {
                let values = try file.resourceValues(forKeys: keys)
                if values.isRegularFile == true { total += Int64(values.totalFileAllocatedSize ?? values.fileAllocatedSize ?? 0) }
            }
        }
        for file in tensors {
            let values = try file.url.resourceValues(forKeys: keys)
            total += Int64(values.totalFileAllocatedSize ?? values.fileAllocatedSize ?? 0)
        }
        return total
    }

    private struct Asset: Decodable {
        let path: String; let function: String; let sha256: String
        func asset(_ resolve: (String) throws -> URL) throws -> H3ModelAsset { try H3ModelAsset(url: resolve(path), function: function, sha256: sha256) }
    }
    private struct Tensor: Decodable {
        let name: String; let path: String; let dtype: String; let shape: [Int]; let sha256: String
        func file(_ resolve: (String) throws -> URL) throws -> H3TensorFile {
            guard let type = H3ScalarType(rawValue: dtype), type != .int32 else {
                throw AetherError.invalidModel("unsupported H3 tensor dtype \(dtype)")
            }
            return try H3TensorFile(name: name, url: resolve(path), spec: H3TensorSpec(scalarType: type, shape: shape), sha256: sha256)
        }
    }
    private struct ConditionerAssets: Decodable { let embedding: Asset; let layers: [Asset] }
    private struct Manifest: Decodable {
        let width: Int; let height: Int; let frames: Int; let text_rows: Int
        let tokenizer: String; let schedule: String
        let prepare: Asset; let time: Asset; let heads: Asset
        let bodies: [Asset]; let modulations: [Asset]; let parameters: [[Tensor]]
        let rotary_cos: Tensor; let rotary_sin: Tensor
        let video_decoder: Asset; let video_parameters: [Tensor]?; let video_spatial_tiles: H3SpatialTiling?
        let audio_decoder: Asset; let conditioner: ConditionerAssets?
        let video_window_latents: Int?; let video_window_frames: Int?; let video_window_stride: Int?
    }
}
