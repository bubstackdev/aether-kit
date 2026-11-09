"""Records Diffusers' own FL2VA keyframe conditioning for the Swift driver's hermetic tests
(Tests/AetherVideoTests/Fixtures/h3-keyframes.json). No weights are read: only the pinned checkpoint's tokenizer,
processor and configs.

    Forge/.venv/bin/python -I Forge/h3/keyframe_fixture_h3.py [--snapshot DIR] > Tests/AetherVideoTests/Fixtures/h3-keyframes.json

What it records, each from upstream code:

  presentation  `MiniMaxH3FL2VATextEncoderStep.__call__` run verbatim with its conditioner call replaced by a capture:
                the token ids, the row tags, the processor's pixels (cast to bf16 as the conditioner call casts them)
                and grids, for one keyframe (first, or last alone) and for two (first + last); the M-RoPE positions from
                `Qwen3VLModel.get_rope_index` (built on the meta device) and `create_mm_token_type_ids`. A 256x256
                canvas in full; 960x544 and 544x960 by SHA-256.
  vision        the tower's host tables as the exporter feeds its vision graphs (`get_vision_interpolation_indices_and_
                weights`, `get_vision_position_ids`), by SHA-256.
  rotary        the text rotary module's bf16 cos table at positions 0..N-1, and its value at the small presentation's
                M-RoPE positions (the gather the driver does), by SHA-256.
  layouts       `build_packed_sequence` with no keyframe, first, last and first + last on a tiny geometry of 37 latent
                frames (where the last anchor's pairwise sum matters), and `build_row_timesteps` with the conditioning
                rows held at max(t, keyframe_noise_aug).
  noising       `MiniMaxH3Scheduler.scale_noise` at keyframe_noise_aug on fixed values.

The synthetic keyframes are byte patterns (byte i = (7 i + 3) mod 256, the second keyframe (5 i + 11) mod 256).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3_graphs as hg  # noqa: E402

DEFAULT_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots" / hg.REVISION
PROMPT = "A paper boat drifts across a still pond as ripples spread. Soft water lapping."


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def keyframe(width: int, height: int, multiplier: int, offset: int) -> np.ndarray:
    return ((np.arange(width * height * 3, dtype=np.int64) * multiplier + offset) % 256).astype(np.uint8).reshape(height, width, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    from PIL import Image
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextRotaryEmbedding
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
    from diffusers.modular_pipelines.minimax_h3 import encoders
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG,
                                                                         MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG)
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    fl2va = args.snapshot / "FL2VA"
    tokenizer = AutoTokenizer.from_pretrained(fl2va / "tokenizer")
    processor = AutoProcessor.from_pretrained(fl2va / "processor")
    config = Qwen3VLConfig.from_pretrained(fl2va / "text_encoder")
    with torch.device("meta"):
        model = Qwen3VLModel(config)
    merge = config.vision_config.spatial_merge_size
    side = int(config.vision_config.num_position_embeddings ** 0.5)

    captured = {}

    def capture(text_encoder, processor_, token_ids, vision_inputs=None, text_encoder_layer=50, device=None, dtype=None):
        captured["ids"] = list(token_ids)
        captured["vision"] = {k: v.to(text_encoder.dtype) if k.startswith("pixel_") else v for k, v in (vision_inputs or {}).items()}
        captured["mm"] = processor_.create_mm_token_type_ids([token_ids])
        return torch.zeros(1, len(token_ids), 1)

    encoders.get_qwen3vl_prompt_embeds = capture
    step = encoders.MiniMaxH3FL2VATextEncoderStep()
    components = SimpleNamespace(tokenizer=tokenizer, processor=processor, text_encoder=SimpleNamespace(dtype=torch.bfloat16),
                                 text_encoder_layer=50, text_tag=MINIMAX_H3_TEXT_TAG, video_tag=MINIMAX_H3_VIDEO_TAG,
                                 _execution_device=torch.device("cpu"))

    def present(images: list[np.ndarray]) -> dict:
        state = PipelineState()
        state.set("prompt", PROMPT)
        state.set("keyframes", [Image.fromarray(image) for image in images])
        step(components, state)
        ids, tags = captured["ids"], state.get("text_token_tags").tolist()
        grid = captured["vision"]["image_grid_thw"]
        positions, _ = model.get_rope_index(torch.tensor([ids]), torch.tensor(captured["mm"]), image_grid_thw=grid)
        pixels = captured["vision"]["pixel_values"]
        return {"ids": ids, "tags": tags, "positions": positions[:, 0].tolist(), "grids": grid.tolist(),
                "pixels": pixels.contiguous().view(torch.int16).numpy().tobytes()}

    prompt_ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    labels = {f"<Picture {i}>: ": tokenizer(f"<Picture {i}>: ", add_special_tokens=False)["input_ids"] for i in (1, 2)}
    special = {name: tokenizer.convert_tokens_to_ids(token) for name, token in
               (("visionStart", "<|vision_start|>"), ("imagePad", "<|image_pad|>"), ("visionEnd", "<|vision_end|>"))}

    presentations, embedded = [], None
    for width, height in ((256, 256), (960, 544), (544, 960)):
        first, last = keyframe(width, height, 7, 3), keyframe(width, height, 5, 11)
        for anchors, images in ((["first"], [first]), (["last"], [last]), (["first", "last"], [first, last])):
            p = present(images)
            case = {"canvas": f"{width}x{height}", "anchors": anchors, "count": len(p["ids"]), "grids": p["grids"],
                    "ids_sha256": sha(np.array(p["ids"], dtype=np.int32).tobytes()),
                    "tags_sha256": sha(np.array(p["tags"], dtype=np.int32).tobytes()),
                    "positions_sha256": sha(np.array(p["positions"], dtype=np.int32).tobytes()),
                    "pixels_sha256": sha(p["pixels"])}
            presentations.append(case)
            if (width, height) == (256, 256) and anchors == ["first", "last"]:
                embedded = {"canvas": "256x256", "anchors": anchors, "ids": p["ids"], "tags": p["tags"], "positions": p["positions"]}

    vision = []
    for width, height in ((256, 256), (960, 544), (544, 960)):
        grid = torch.tensor([[1, height // 16, width // 16]])
        index, weight = get_vision_interpolation_indices_and_weights(grid, num_grid_per_side=side, mode="bilinear",
                                                                     align_corners=True, spatial_merge_size=merge)
        rc = get_vision_position_ids(grid, merge)
        vision.append({"canvas": f"{width}x{height}", "grid": [height // 16, width // 16],
                       "pos_index_sha256": sha(index.to(torch.int32).numpy().tobytes()),
                       "pos_weight_sha256": sha(weight.to(torch.float32).numpy().tobytes()),
                       "patch_rc_sha256": sha(rc.to(torch.int32).numpy().tobytes())})

    rotary = Qwen3VLTextRotaryEmbedding(config.text_config)
    positions = torch.tensor(embedded["positions"])
    rows = int(positions.max()) + 1
    x = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
    with torch.inference_mode():
        table, _ = rotary(x, torch.arange(rows).view(1, 1, -1).expand(3, 1, -1))
        cos, _ = rotary(x, positions.view(3, 1, -1))
    rotary_record = {"rows": rows, "mropeSection": list(rotary.mrope_section),
                     "cosTableBase64": base64.b64encode(table[0].contiguous().view(torch.int16).numpy().tobytes()).decode(),
                     "cosSha256": sha(cos[0].contiguous().view(torch.int16).numpy().tobytes())}

    T, FRAMES, LH, LW, NA = 7, 37, 4, 6, 9
    text_tags = torch.tensor([1, 1, 0, 0, 0, 1, 1])
    video_t, audio_t = 0.0476190447807312, 0.1666666865348816
    layouts = []
    for anchors in ((), ("first",), ("last",), ("first", "last")):
        position_ids, tags, video, audio, text, cv, ca = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_tags, FRAMES, LH, LW, NA, (1, 2, 2), MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG,
            MINIMAX_H3_VIDEO_TAG, anchors)
        unique, indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video, audio, cv, ca, T, video_t, audio_t, max(video_t, 0.999), 1.0)
        layouts.append({"anchors": list(anchors), "conditionRows": cv,
                        "positions": position_ids.to(torch.float32).flatten().tolist(),
                        "anchorTimes": [float(position_ids[T + i * (LH // 2) * (LW // 2), 0]) for i in range(len(anchors))],
                        "tags": tags.tolist(), "videoIndices": video.tolist(), "audioIndices": audio.tolist(),
                        "timesteps": [float(v) for v in unique.to(torch.float32)], "timestepIndices": indices.tolist()})

    g = torch.Generator().manual_seed(20261108)
    condition, noise = torch.randn(64, generator=g), torch.randn(64, generator=g)
    noised = MiniMaxH3Scheduler().scale_noise(condition, 0.999, noise)

    json.dump({"schema": 1, "generator": "Forge/h3/keyframe_fixture_h3.py", "revision": hg.REVISION,
               "diffusersCommit": hg.DIFFUSERS_COMMIT, "keyframes": "byte i = (7 i + 3) mod 256; last (5 i + 11) mod 256",
               "prompt": PROMPT, "promptIds": prompt_ids, "labels": labels, "special": special,
               "presentations": presentations, "embedded": embedded, "vision": vision, "rotary": rotary_record,
               "layout": {"textTags": text_tags.tolist(), "latentFrames": FRAMES, "latentHeight": LH, "latentWidth": LW,
                          "audioLatents": NA, "videoTimestep": video_t, "audioTimestep": audio_t, "cases": layouts},
               "noising": {"timestep": 0.999, "condition": condition.tolist(), "noise": noise.tolist(),
                           "noised": noised.tolist()}}, sys.stdout)


if __name__ == "__main__":
    main()
