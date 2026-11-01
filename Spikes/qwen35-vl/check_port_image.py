#!/usr/bin/env python3
"""CPU check of the eager Qwen3.5 ports on an image prompt (float32, real weights), against the upstream image
reference (reference_qwen35_image.py), in Forge's environment:

    Forge/.venv/bin/python -I Spikes/qwen35-vl/check_port_image.py --checkpoint <snapshot> --reference <dir> --out <json>

Per case, on the reference's own pixels: the vision port (`qwen3_vl_vision`, zero DeepStack levels, host position
tables) against upstream's image embeddings; Kit's host M-RoPE positions (`mrope_positions`) against upstream's rope
index; then the text port fed as the host feeds the asset (64-wide padded prefill with image rows aligned to the
tokens, then one token per padded call), with the port's own image embeddings, against the forced rows.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Forge" / "language"))
from qwen35_hybrid import build, load_weights, read_config  # noqa: E402
from qwen3_vl_text import mrope_positions  # noqa: E402
from qwen3_vl_vision import Qwen3VLVision, host_inputs, load_weights as load_vision  # noqa: E402


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def aligned_rows(ids, start, count, embeds, image_token, consumed):
    """Image rows for tokens ids[start:start+count] (zero for text), taking embeds in order; returns (rows, consumed)."""
    rows = np.zeros((count, embeds.shape[1]), np.float32)
    for i, t in enumerate(ids[start:start + count]):
        if t == image_token:
            rows[i] = embeds[consumed]
            consumed += 1
    return rows, consumed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--width", type=int, default=64)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    ref = json.loads((a.reference / "reference.json").read_text())
    config = json.loads((a.checkpoint / "config.json").read_text())
    cfg, image_token = read_config(a.checkpoint)
    vision = Qwen3VLVision(dict(config["vision_config"]))
    load_vision(vision, a.checkpoint)
    vision.eval()
    port = build(cfg, image_token, torch.float32, max_capacity=8192, max_width=a.width)
    load_weights(port, a.checkpoint)
    port.eval()
    vocab, w, results, t0 = cfg["vocab_size"], a.width, {}, time.time()
    for c in ref["cases"]:
        _, gh, gw = c["grid"]
        pixels = np.fromfile(a.reference / c["pixels"]["file"], dtype="<f4").reshape(c["pixels"]["shape"])
        want_embeds = np.fromfile(a.reference / c["imageEmbeds"]["file"], dtype="<f4").reshape(c["imageEmbeds"]["shape"])
        tables = host_inputs(1, gh, gw)
        embeds = vision(torch.from_numpy(pixels), torch.from_numpy(tables["pos_index"]), torch.from_numpy(tables["pos_weight"]),
                        torch.from_numpy(tables["patch_rc"])).double().numpy()
        ids, forced = c["promptTokens"], c["forcedTokens"]
        rope_host = np.array(mrope_positions(ids, [tuple(c["grid"])], image_token, 2)).T          # [3, len]
        rope_equal = bool(np.array_equal(rope_host, np.array(c["ropePositions"])))
        want = np.fromfile(a.reference / c["forcedLogits"]["file"], dtype="<f4").reshape(len(forced), vocab)
        state, rows, start, consumed = port.empty_state(len(ids) + len(forced) + w), [], 0, 0
        seq = ids + forced
        rope_all = np.array(mrope_positions(seq, [tuple(c["grid"])], image_token, 2)).T
        plan = [(s, min(w, len(ids) - s)) for s in range(0, len(ids), w)] + [(len(ids) + i, 1) for i in range(len(forced) - 1)]
        for s, n in plan:
            img, consumed = aligned_rows(seq, s, n, embeds, image_token, consumed)
            logits = port.step(seq[s:s + n], s, w, state, rope=rope_all[:, s:s + n], image_rows=torch.from_numpy(img))
            if s + n >= len(ids):
                rows.append(logits.double().numpy())
        cos = [cosine(x, y) for x, y in zip(rows, want)]
        results[c["name"]] = {
            "visionEmbedsCosine": round(cosine(embeds, want_embeds), 8),
            "visionEmbedsRowMinCosine": round(min(cosine(x, y) for x, y in zip(embeds, want_embeds)), 8),
            "ropeEqual": rope_equal, "imageRowsUsed": consumed, "forcedRows": len(rows),
            "minCosine": round(min(cos), 8),
            "argmaxEqualRows": int(sum(int(np.argmax(x)) == int(np.argmax(y)) for x, y in zip(rows, want)))}
        print(c["name"], results[c["name"]], f"{time.time() - t0:.0f} s", flush=True)
    a.out.write_text(json.dumps({"width": w, "dtype": "float32", "results": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
