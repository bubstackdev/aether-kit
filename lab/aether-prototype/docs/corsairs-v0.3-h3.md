# Corsairs ownership for H3

BF16 and transitive `Runtime::drain()` landed in v0.2. Aether now pins the published crates.io `0.3.0` release. It provides checked storage/CPU execution up to 4 GiB and a qualified 2 GiB GPU inference boundary, native symbol caching and repaired queue-completion ownership. These runtime checks and qualification belong to Corsairs. Aether consumes the public API without modifying it or maintaining a second cap.

The earlier [v0.2 boundary record](corsairs-v0.2-h3.md) is historical. The conditioner, processor/packing, model export recipe, joint schedule and latent updates, heads, VAE decode, residency and media output belong to Aether. Current implementation and qualification are tracked in [H3 integration](h3.md) and [validation](h3-validation.md).
