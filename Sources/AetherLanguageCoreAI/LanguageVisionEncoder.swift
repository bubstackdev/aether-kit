import CoreAI
import AetherCoreAI
import AetherKit
import AetherVision
import Foundation
import Metal

/// How the vision encoder asset is specialized. `expectFrequentReshapes` is a correctness hazard
/// for it, not a tuning choice: specialized with the flag, the dynamic-shape Qwen3-VL encoder
/// computes wrong outputs on the GPU (cosine 0.16–0.67 against the reference; engine F-P3-26,
/// reproduced in `docs/evidence/qwen3-vl-vision-encoder-2026-10-26.json`). A registration that
/// requests it is refused until a qualification shows it correct.
public enum LanguageVisionSpecialization {
    /// Whether any qualification has shown reshape-optimized specialization correct for the encoder
    /// (the registration's rule, ``LanguageVisionRegistration/reshapesQualified``).
    public static var reshapesQualified: Bool { LanguageVisionRegistration.reshapesQualified }

    /// The flag to specialize the encoder with, from the registration's request (nil: default).
    public static func expectFrequentReshapes(requested: Bool?) throws -> Bool {
        guard requested == true else { return false }
        guard reshapesQualified else {
            throw LanguageGenerationError.invalid("The vision encoder must not be specialized with expectFrequentReshapes (F-P3-26)")
        }
        return true
    }
}

/// Sequential owner of a Qwen3-VL vision encoder asset with one static-shape function per fixed
/// patch grid (`encode_<h>x<w>`). One request at a time; every submission is joined before outputs
/// are read; `unload()` releases the model.
///
/// Opening refuses a grid without a static function, and a function with any dynamic dimension: a
/// dynamic-shape export intermittently computed wrong outputs on the GPU even without
/// `expectFrequentReshapes` (2.5 % of encodes, always the same wrong bits; docs/qwen3-vl.md).
public actor LanguageVisionEncoder {
    public struct Configuration: Sendable, Equatable {
        public let mergeSize: Int
        public let positionTableSide: Int
        /// The registration's fixed patch grids; each needs its static-shape function.
        public let imageGrids: [Qwen3VLImagePreprocessor.Grid]
        public init(mergeSize: Int = 2, positionTableSide: Int = 48, imageGrids: [Qwen3VLImagePreprocessor.Grid]) {
            self.mergeSize = mergeSize
            self.positionTableSide = positionTableSide
            self.imageGrids = imageGrids
        }
    }

    /// The function name for a grid, `encode_<height>x<width>` (``LanguageImageGrid/functionName``).
    public static func functionName(for grid: Qwen3VLImagePreprocessor.Grid) -> String {
        LanguageImageGrid(height: grid.height, width: grid.width).functionName
    }

    private var model: AIModel?
    private var functions: [Qwen3VLImagePreprocessor.Grid: InferenceFunction] = [:]
    private var stream: SerialComputeStream?
    private let configuration: Configuration
    private let rowLength: Int
    private let hiddenSize: Int
    private let deepstackLevels: Int
    private let lifecycle = LanguageSessionLifecycle()

    /// Opens the encoder GPU-preferred. `expectFrequentReshapes` is the registration's request and
    /// is refused when true (see ``LanguageVisionSpecialization``).
    public static func open(at url: URL, configuration: Configuration, expectFrequentReshapes: Bool? = nil,
                            compute: ModelComputePolicy = .preferGPU) async throws -> LanguageVisionEncoder {
        var options: SpecializationOptions
        switch compute {
        case .automatic: options = .default
        case .cpuOnly: options = .cpuOnly
        case .preferGPU: options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        }
        options.expectFrequentReshapes = try LanguageVisionSpecialization.expectFrequentReshapes(requested: expectFrequentReshapes)
        let model = try await AIModel(contentsOf: url, options: options)
        return try LanguageVisionEncoder(model: model, configuration: configuration)
    }

    init(model: AIModel, configuration: Configuration) throws {
        var functions: [Qwen3VLImagePreprocessor.Grid: InferenceFunction] = [:]
        var layout: (row: Int, hidden: Int, levels: Int)?
        try LanguageVisionContract.requireGrids(configuration.imageGrids, mergeSize: configuration.mergeSize)
        for grid in configuration.imageGrids {
            let name = Self.functionName(for: grid)
            guard let d = model.functionDescriptor(for: name) else {
                throw LanguageGenerationError.invalid("The vision encoder has no static function for grid \(grid.height)x\(grid.width)")
            }
            guard Set(d.inputNames) == ["pixels", "pos_index", "pos_weight", "patch_rc"],
                  Set(d.outputNames) == ["image_embeds", "deepstack"], d.stateNames.isEmpty,
                  case .ndArray(let pixels)? = d.inputDescriptor(of: "pixels"), pixels.scalarType == .float16,
                  case .ndArray(let index)? = d.inputDescriptor(of: "pos_index"),
                  case .ndArray(let weight)? = d.inputDescriptor(of: "pos_weight"),
                  case .ndArray(let rc)? = d.inputDescriptor(of: "patch_rc"),
                  case .ndArray(let embeds)? = d.outputDescriptor(of: "image_embeds"),
                  case .ndArray(let deep)? = d.outputDescriptor(of: "deepstack"),
                  let found = LanguageVisionContract.staticLayout(grid: grid, mergeSize: configuration.mergeSize,
                      inputs: [pixels.shape, index.shape, weight.shape, rc.shape], outputs: [embeds.shape, deep.shape]),
                  layout == nil || layout! == found,
                  let function = try model.loadFunction(named: name)
            else { throw LanguageGenerationError.invalid("Unsupported static vision encoder contract for \(name)") }
            layout = found
            functions[grid] = function
        }
        guard let layout, let queue = MTLCreateSystemDefaultDevice()?.makeCommandQueue() else {
            throw LanguageGenerationError.invalid("Unsupported vision encoder contract")
        }
        self.model = model
        self.functions = functions
        self.stream = SerialComputeStream(commandQueue: queue)
        self.configuration = configuration
        self.rowLength = layout.row
        self.hiddenSize = layout.hidden
        self.deepstackLevels = layout.levels
    }

    /// Encodes one image's patch rows ([patches, rowLength] fp16, merge-window order) for a patch
    /// grid (1, h, w) that is one of the fixed grids; any other grid is refused. The per-patch
    /// tables are computed here.
    public func encode(pixels: [Float16], grid: (t: Int, h: Int, w: Int)) async throws -> LanguageImageFeatures {
        try Task.checkCancellation()
        try lifecycle.admit()
        defer { lifecycle.finish() }
        guard let stream else { throw LanguageGenerationError.closed }
        guard grid.t == 1, let function = functions[Qwen3VLImagePreprocessor.Grid(height: grid.h, width: grid.w)] else {
            throw LanguageGenerationError.invalid("Grid \(grid.t)x\(grid.h)x\(grid.w) is not one of the encoder's fixed grids")
        }
        let tables = try Qwen3VLVisionTables(t: grid.t, h: grid.h, w: grid.w, merge: configuration.mergeSize,
                                             tableSide: configuration.positionTableSide)
        let n = tables.patchCount
        guard pixels.count == n * rowLength else { throw LanguageGenerationError.invalid("Pixel rows do not match the grid") }
        let inputs: [String: InferenceFunction.AsyncValue] = [
            "pixels": .init(NDArray(scalars: pixels, shape: [n, rowLength])),
            "pos_index": .init(NDArray(scalars: tables.positionIndex, shape: [n, 4])),
            "pos_weight": .init(NDArray(scalars: tables.positionWeight, shape: [n, 4])),
            "patch_rc": .init(NDArray(scalars: tables.patchCoordinates, shape: [n, 2])),
        ]
        let outputs: [String: InferenceFunction.AsyncValue]
        do {
            outputs = try stream.encode(function, inputs: inputs)
        } catch {
            await stream.currentWorkCompleted()
            throw error
        }
        await stream.currentWorkCompleted()
        try Task.checkCancellation()
        let tokens = n / (configuration.mergeSize * configuration.mergeSize)
        let embeddings = try await Self.read(outputs["image_embeds"], shape: [tokens, hiddenSize])
        let deepstack = try await Self.read(outputs["deepstack"], shape: [deepstackLevels, tokens, hiddenSize])
        return try LanguageImageFeatures(grid: grid, mergeSize: configuration.mergeSize, hiddenSize: hiddenSize,
                                         deepstackLevels: deepstackLevels, embeddings: embeddings, deepstack: deepstack)
    }

    /// Releases the model; later requests fail with `closed`.
    public func unload() async {
        guard await lifecycle.beginClose() else { return }
        functions = [:]
        stream = nil
        model = nil
        lifecycle.didClose()
    }

    private static func read(_ value: InferenceFunction.AsyncValue?, shape: [Int]) async throws -> [Float16] {
        guard let value, let array = try await value.ndArray, array.scalarType == .float16, array.shape == shape else {
            throw LanguageGenerationError.invalid("Unexpected vision encoder output")
        }
        return array.view(as: Float16.self).withUnsafePointer { pointer, dims, strides in
            let count = shape.reduce(1, *)
            var out = [Float16](repeating: 0, count: count)
            var index = [Int](repeating: 0, count: shape.count)
            for linear in 0..<count {
                var offset = 0
                for d in 0..<shape.count { offset += index[d] * strides[d] }
                out[linear] = pointer[offset]
                var d = shape.count - 1
                while d >= 0 { index[d] += 1; if index[d] < dims[d] { break }; index[d] = 0; d -= 1 }
            }
            return out
        }
    }
}

/// The static per-grid encoder contract, checked before any function loads (pure; CPU-tested).
enum LanguageVisionContract {
    /// Grids must be present, distinct, positive and multiples of the merge size.
    static func requireGrids(_ grids: [Qwen3VLImagePreprocessor.Grid], mergeSize: Int) throws {
        guard mergeSize > 0, !grids.isEmpty, Set(grids).count == grids.count,
              grids.allSatisfy({ $0.height > 0 && $0.width > 0 && $0.height % mergeSize == 0 && $0.width % mergeSize == 0 })
        else { throw LanguageGenerationError.invalid("The vision encoder needs distinct fixed image grids (multiples of the merge size)") }
    }

    /// (row length, hidden size, DeepStack levels) of a grid's function when every shape is static
    /// and matches the grid: pixels [n, row], pos_index/pos_weight [n, 4], patch_rc [n, 2],
    /// image_embeds [n/merge², hidden], deepstack [levels, n/merge², hidden]. Nil for any dynamic
    /// (-1) dimension or mismatch.
    static func staticLayout(grid: Qwen3VLImagePreprocessor.Grid, mergeSize: Int, inputs: [[Int]], outputs: [[Int]])
        -> (row: Int, hidden: Int, levels: Int)? {
        guard inputs.count == 4, outputs.count == 2, (inputs + outputs).allSatisfy({ $0.allSatisfy { $0 > 0 } }) else { return nil }
        let n = grid.height * grid.width, tokens = n / (mergeSize * mergeSize)
        let (pixels, index, weight, rc) = (inputs[0], inputs[1], inputs[2], inputs[3])
        let (embeds, deep) = (outputs[0], outputs[1])
        guard pixels.count == 2, pixels[0] == n, index == [n, 4], weight == [n, 4], rc == [n, 2],
              embeds.count == 2, embeds[0] == tokens, deep.count == 3, deep[1] == tokens, deep[2] == embeds[1]
        else { return nil }
        return (pixels[1], embeds[1], deep[0])
    }
}
