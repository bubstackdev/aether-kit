import AetherCoreAI
import AetherDiffusion
import AetherKit
import Foundation

/// Core AI binding for the shared float32 NCHW contracts. Model adapters supply geometry and URLs.
/// Retains the denoiser across steps; releases the encoder before it and denoiser before decoding.
@available(macOS 27, iOS 27, *)
public actor CoreAIEpsilonBackend: EpsilonDiffusionBackend {
    private let textURL: URL, denoiserURL: URL, decoderURL: URL
    private let latentShape: [Int], textShape: [Int], imageShape: [Int], vocabularySize: Int
    private let compute: ModelComputePolicy
    private var denoiser: ModelSession?
    private var busy = false

    public init(textEncoder: URL, denoiser: URL, decoder: URL, width: Int, height: Int,
                latentChannels: Int, latentScale: Int, textLength: Int, textWidth: Int, vocabularySize: Int,
                compute: ModelComputePolicy = .preferGPU) throws {
        guard width > 0, height > 0, latentChannels > 0, latentScale > 0,
              width.isMultiple(of: latentScale), height.isMultiple(of: latentScale),
              textLength > 0, textWidth > 0, vocabularySize > 0 else {
            throw AetherError.invalidRequest("invalid latent diffusion geometry")
        }
        latentShape = [1, latentChannels, height / latentScale, width / latentScale]
        textShape = [1, textLength, textWidth]; imageShape = [3, height, width]
        _ = try FloatTensor.elementCount(for: latentShape); _ = try FloatTensor.elementCount(for: textShape)
        _ = try FloatTensor.elementCount(for: imageShape)
        self.textURL = textEncoder; self.denoiserURL = denoiser; self.decoderURL = decoder
        self.compute = compute; self.vocabularySize = vocabularySize
    }

    public func encode(_ tokenIDs: Int32Tensor) async throws -> FloatTensor {
        guard !busy else { throw AetherError.sessionBusy }; busy = true; defer { busy = false }
        guard tokenIDs.shape == [1, textShape[1]], tokenIDs.values.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw AetherError.invalidTensor("invalid text encoder IDs")
        }
        await releaseDenoiser()
        return try await withModel(textURL, entry: "encode", inputs: ["input_ids": .int32(tokenIDs)],
                                   inputTypes: ["input_ids": "int32"], output: "text_embeds", shape: textShape)
    }

    public func denoise(sample: FloatTensor, timestep: Float, conditioning: FloatTensor) async throws -> FloatTensor {
        guard !busy else { throw AetherError.sessionBusy }; busy = true; defer { busy = false }
        guard sample.shape == latentShape, conditioning.shape == textShape, timestep.isFinite, timestep >= 0 else {
            throw AetherError.invalidTensor("invalid denoiser input")
        }
        let inputs: [String: TensorValue] = ["sample": .float32(sample), "timestep": .float32(try FloatTensor(shape: [1], values: [timestep])),
                                            "text_embeds": .float32(conditioning)]
        if denoiser == nil { denoiser = try await ModelSession.open(at: denoiserURL, compute: compute) }
        guard let model = denoiser else { throw AetherError.sessionClosed }
        do {
            try validate(model, entry: "denoise", inputs: inputs, inputTypes: [:], output: "epsilon", shape: latentShape)
            let out = try await model.runTensors(function: "denoise", inputs: inputs)
            return try result(out, name: "epsilon", shape: latentShape)
        } catch { await releaseDenoiser(); throw error }
    }

    public func decode(_ latents: FloatTensor) async throws -> GeneratedImage {
        guard !busy else { throw AetherError.sessionBusy }; busy = true; defer { busy = false }
        guard latents.shape == latentShape else { throw AetherError.invalidTensor("invalid decoder latents") }
        await releaseDenoiser()
        return try await GeneratedImage(pixelsCHW: withModel(decoderURL, entry: "decode", inputs: ["latents": .float32(latents)],
                                                            inputTypes: [:], output: "image", shape: imageShape))
    }

    private func withModel(_ url: URL, entry: String, inputs: [String: TensorValue], inputTypes: [String: String],
                           output: String, shape: [Int]) async throws -> FloatTensor {
        let model = try await ModelSession.open(at: url, compute: compute)
        do {
            try validate(model, entry: entry, inputs: inputs, inputTypes: inputTypes, output: output, shape: shape)
            let out = try await model.runTensors(function: entry, inputs: inputs)
            let tensor = try result(out, name: output, shape: shape)
            await model.unload(); return tensor
        } catch { await model.unload(); throw error }
    }

    private func validate(_ model: ModelSession, entry: String, inputs: [String: TensorValue], inputTypes: [String: String],
                          output: String, shape: [Int]) throws {
        let f = try signature(model, named: entry)
        guard f.states.isEmpty, Set(f.inputs.keys) == Set(inputs.keys), Set(f.outputs.keys) == [output] else {
            throw AetherError.invalidModel("invalid diffusion function signature")
        }
        func check(_ value: ModelValueDescription?, type: String, shape: [Int]) throws {
            guard let value, value.kind == "tensor", value.scalarType == type, value.shape == shape else {
                throw AetherError.invalidModel("diffusion declaration must be \(type) \(shape)")
            }
        }
        for (key, value) in inputs { try check(f.inputs[key], type: inputTypes[key] ?? "float32", shape: value.shape) }
        try check(f.outputs[output], type: "float32", shape: shape)
    }
    private func result(_ out: [String: TensorValue], name: String, shape: [Int]) throws -> FloatTensor {
        guard Set(out.keys) == [name], case .float32(let tensor) = out[name], tensor.shape == shape else {
            throw AetherError.invalidTensor("invalid diffusion output \(name)")
        }
        return tensor
    }
    private func releaseDenoiser() async {
        let model = denoiser; denoiser = nil; await model?.unload()
    }
    /// Call after joined inference. DiffusionSession enforces that ordering.
    public func unload() async { await releaseDenoiser() }
}
