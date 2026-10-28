// Qwen3 dense decoder on MLX. Structure adapted from mlx-swift-lm's Libraries/MLXLLM/Models/Qwen3.swift (3.32.3,
// itself a port of mlx-lm's qwen3.py), reduced to what the spike needs: no quantization, a plain concatenating KV cache,
// and MLX core only (no MLXLMCommon, which would bring a swift-syntax pin that conflicts with Kit's).
//
// Portions Copyright (c) 2024 ml-explore, used under the MIT License; see ../../LICENSE-mlx-swift-lm.

import Foundation
import MLX
import MLXNN

public struct Qwen3Configuration: Decodable, Sendable {
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let ropeTheta: Float
    let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", hiddenLayers = "num_hidden_layers", intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads", kvHeads = "num_key_value_heads", headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps", vocabularySize = "vocab_size", ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }
}

/// One layer's keys and values so far, [batch, kvHeads, tokens, headDim].
final class LayerCache {
    var keys: MLXArray?
    var values: MLXArray?
    var offset: Int { keys?.dim(2) ?? 0 }

    func append(_ newKeys: MLXArray, _ newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let keys, let values {
            self.keys = concatenated([keys, newKeys], axis: 2)
            self.values = concatenated([values, newValues], axis: 2)
        } else {
            keys = newKeys
            values = newValues
        }
        return (keys!, values!)
    }
}

final class Attention: Module {
    let configuration: Qwen3Configuration
    let scale: Float
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    init(_ c: Qwen3Configuration) {
        configuration = c
        scale = pow(Float(c.headDim), -0.5)
        _queryProjection.wrappedValue = Linear(c.hiddenSize, c.attentionHeads * c.headDim, bias: false)
        _keyProjection.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: false)
        _valueProjection.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: false)
        _outputProjection.wrappedValue = Linear(c.attentionHeads * c.headDim, c.hiddenSize, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: LayerCache) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var queries = queryNorm(queryProjection(x).reshaped(batch, length, configuration.attentionHeads, -1)).transposed(0, 2, 1, 3)
        var keys = keyNorm(keyProjection(x).reshaped(batch, length, configuration.kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = valueProjection(x).reshaped(batch, length, configuration.kvHeads, -1).transposed(0, 2, 1, 3)
        let offset = cache.offset
        queries = MLXFast.RoPE(queries, dimensions: configuration.headDim, traditional: false, base: configuration.ropeTheta,
                               scale: 1, offset: offset)
        keys = MLXFast.RoPE(keys, dimensions: configuration.headDim, traditional: false, base: configuration.ropeTheta,
                            scale: 1, offset: offset)
        let (allKeys, allValues) = cache.append(keys, values)
        let output = MLXFast.scaledDotProductAttention(queries: queries, keys: allKeys, values: allValues, scale: scale,
                                                       mask: length > 1 ? .causal : .none)
        return outputProjection(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ c: Qwen3Configuration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class Block: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: FeedForward
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(_ c: Qwen3Configuration) {
        _attention.wrappedValue = Attention(c)
        mlp = FeedForward(c)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: LayerCache) -> MLXArray {
        let h = x + attention(inputNorm(x), cache: cache)
        return h + mlp(postAttentionNorm(h))
    }
}

final class Inner: Module {
    @ModuleInfo(key: "embed_tokens") var embedding: Embedding
    let layers: [Block]
    let norm: RMSNorm

    init(_ c: Qwen3Configuration) {
        _embedding.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        layers = (0..<c.hiddenLayers).map { _ in Block(c) }
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
}

/// Qwen3 for causal LM. Loads a Hugging Face snapshot's safetensors directly.
public final class Qwen3: Module {
    public let configuration: Qwen3Configuration
    let model: Inner
    @ModuleInfo(key: "lm_head") var head: Linear?

    init(_ c: Qwen3Configuration) {
        configuration = c
        model = Inner(c)
        if !c.tieWordEmbeddings { _head.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false) }
    }

    /// Reads `config.json` and every `*.safetensors` in `directory`, cast to `dtype`.
    public static func load(directory: URL, dtype: DType) throws -> Qwen3 {
        let configuration = try JSONDecoder().decode(Qwen3Configuration.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let model = Qwen3(configuration)
        var weights: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        where file.pathExtension == "safetensors" {
            for (key, value) in try loadArrays(url: file) { weights[key] = value.asType(dtype) }
        }
        if configuration.tieWordEmbeddings { weights.removeValue(forKey: "lm_head.weight") }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    /// Fresh per-layer caches for one request.
    func caches() -> [LayerCache] { (0..<configuration.hiddenLayers).map { _ in LayerCache() } }

    /// Logits for every position of `tokens` [1, n], continuing `caches`: [1, n, vocab].
    func logits(_ tokens: MLXArray, caches: [LayerCache], lastOnly: Bool) -> MLXArray {
        var h = model.embedding(tokens)
        for (layer, cache) in zip(model.layers, caches) { h = layer(h, cache: cache) }
        if lastOnly { h = h[0..., (h.dim(1) - 1)..., 0...] }
        h = model.norm(h)
        if let head { return head(h) }
        return matmul(h, model.embedding.weight.T)
    }
}
