Copied from aether-kit 0.0.7 (`Tests/AetherKreaTests/Fixtures/reference/`) on 2026-11-08. Synthetic images only, no
third-party pictures.
- `cases.jsonl`, `fixture.json`, `<case>/ids.i32`, `<case>/rope.i32`, `<case>/image*.png`: the 0.0.x Krea 2
  reference-prompt fixture `c4f6229db52d11b9` (oracle Qwen3VLModel float32 on CPU, krea/Krea-2-Turbo @ 98e0fe11). The
  oracle's tensors (`text.f32`, `image.f32`, `deepstack.f32`, `pixels.f32`) are not copied; `fixture.json` keeps their
  sha256s, and `Forge/krea2/reference_krea2.py --references` regenerates them.
- `grids.json`: the Qwen-VL smart-resize grids (Qwen2VLImageProcessorFast) for source sizes, with its refusals.
