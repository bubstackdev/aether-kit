import AetherKit
import CryptoKit
import Foundation

/// Supported recipe identity only; native H3 execution is not implemented here.
public enum H3Recipe: String, Sendable { case turbo8 }

/// Inspects the existing prototype manifest's identity fields without reading
/// referenced assets. This is not a full bundle validator or an integrity check.
/// A declared QKV marker cannot prove the actual weight conversion is correct.
public struct H3ExportHeader: Sendable {
    public static let snapshot = "42ed227ee7df40d41602854ae760620d6eb651fe"
    public static let diffusersCommit = "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941"
    public static let transformerWeightLayout = "raw_per_head_qkv_to_contiguous_qkv_v1"
    public let recipe: H3Recipe
    public let manifestSHA256: String

    public init(data: Data) throws {
        struct Adapter: Decodable {
            let repository: String; let revision: String; let sha256: String
            let execution: String; let rank: Int; let alpha: Float
        }
        struct Header: Decodable {
            let schema: Int; let complete: Bool; let snapshot: String; let diffusers_commit: String
            let recipe: String; let transformer_weight_layout: String; let adapter: Adapter?
        }
        let header = try JSONDecoder().decode(Header.self, from: data)
        guard header.schema == 1, header.complete, header.snapshot == Self.snapshot,
              header.diffusers_commit == Self.diffusersCommit,
              header.transformer_weight_layout == Self.transformerWeightLayout else {
            throw AetherError.invalidModel("H3 requires complete pinned provenance and corrected per-head QKV layout")
        }
        guard header.recipe == H3Recipe.turbo8.rawValue, let adapter = header.adapter,
              adapter.repository == "lightx2v/Minimax-h3-Turbo",
              adapter.revision == "3ec17a324ced54151364f24f8b5fb6bf7e26414f",
              adapter.sha256 == "e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f",
              adapter.execution == "dynamic_lora", adapter.rank == 128, adapter.alpha == 8 else {
            throw AetherError.unsupported("H3 integration currently accepts only the pinned dynamic-LoRA Turbo8 recipe")
        }
        recipe = .turbo8
        manifestSHA256 = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
