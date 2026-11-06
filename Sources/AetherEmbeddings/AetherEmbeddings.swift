// AetherEmbeddings: text embedder, reranker, classifier (DESIGN.md §8). So far `text-embedder@1` (``TextEmbedder``,
// M4c), on AetherHost's `StatelessPipeline` (re-exported here, with AetherRuntime's `Embedding` and `ImageInput`).
// Text goes through AetherText's tokenizer adapter, which stays an implementation detail.
@_exported public import AetherHost
