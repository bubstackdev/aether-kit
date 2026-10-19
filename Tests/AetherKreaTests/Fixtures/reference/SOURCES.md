Copied read-only on 2026-10-19 (synthetic images only, no third-party pictures):
- grids.json, patch-tables.json: ~/Code/aether-engine/Tests/AetherVisionTests/Fixtures (forge p3_s3_dump.py / p3_s1_tables_dump.py; Qwen2VLImageProcessorFast, qwen3_vl_vision.host_inputs).
- krea2-c4f6229d/: ~/Code/aether-engine/forge/out/krea-2-turbo-realism/fixtures/c4f6229db52d11b9 (kind krea2-reference, oracle Qwen3VLModel float32 on CPU, krea/Krea-2-Turbo @ 98e0fe11). Only ids.i32, rope.i32, image*.png, cases.jsonl and fixture.json are copied; pixels.f32 is checked by its sha256 in fixture.json.
