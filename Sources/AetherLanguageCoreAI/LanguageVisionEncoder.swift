@preconcurrency import CoreAI
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

/// Sequential owner of a Qwen3-VL vision encoder asset (`encode`). One request at a time; every
/// submission is joined before outputs are read; `unload()` releases the model.
public actor LanguageVisionEncoder {
    public struct Configuration: Sendable, Equatable {
        public let mergeSize: Int
        public let positionTableSide: Int
        public init(mergeSize: Int = 2, positionTableSide: Int = 48) {
            self.mergeSize = mergeSize
            self.positionTableSide = positionTableSide
        }
    }

    private var model: AIModel?
    private var function: InferenceFunction?
    private var stream: ComputeStream?
    private let configuration: Configuration
    private let rowLength: Int
    private let hiddenSize: Int
    private let deepstackLevels: Int
    private let lifecycle = LanguageSessionLifecycle()

    /// Opens the encoder GPU-preferred. `expectFrequentReshapes` is the registration's request and
    /// is refused when true (see ``LanguageVisionSpecialization``).
    public static func open(at url: URL, configuration: Configuration = .init(), expectFrequentReshapes: Bool? = nil,
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
        guard let d = model.functionDescriptor(for: "encode"),
              Set(d.inputNames) == ["pixels", "pos_index", "pos_weight", "patch_rc"],
              Set(d.outputNames) == ["image_embeds", "deepstack"], d.stateNames.isEmpty,
              case .ndArray(let pixels)? = d.inputDescriptor(of: "pixels"), pixels.shape.count == 2, pixels.shape[1] > 0,
              pixels.scalarType == .float16,
              case .ndArray(let embeds)? = d.outputDescriptor(of: "image_embeds"), embeds.shape.count == 2, embeds.shape[1] > 0,
              case .ndArray(let deep)? = d.outputDescriptor(of: "deepstack"), deep.shape.count == 3, deep.shape[0] >= 0,
              let function = try model.loadFunction(named: "encode"),
              let queue = MTLCreateSystemDefaultDevice()?.makeCommandQueue()
        else { throw LanguageGenerationError.invalid("Unsupported vision encoder contract") }
        self.model = model
        self.function = function
        self.stream = ComputeStream(commandQueue: queue)
        self.configuration = configuration
        self.rowLength = pixels.shape[1]
        self.hiddenSize = embeds.shape[1]
        self.deepstackLevels = deep.shape[0]
    }

    /// Encodes one image's patch rows ([patches, rowLength] fp16, merge-window order) for a patch
    /// grid (t, h, w). The per-patch tables are computed here.
    public func encode(pixels: [Float16], grid: (t: Int, h: Int, w: Int)) async throws -> LanguageImageFeatures {
        try Task.checkCancellation()
        try lifecycle.admit()
        defer { lifecycle.finish() }
        guard let function, let stream else { throw LanguageGenerationError.closed }
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
            outputs = try function.encode(inputs: inputs, states: InferenceFunction.AsyncMutableViews(), to: stream)
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
        function = nil
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
