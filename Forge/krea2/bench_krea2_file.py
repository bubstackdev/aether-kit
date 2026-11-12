#!/usr/bin/env python3
"""One exported denoiser segment file's run time on the GPU (development tooling only).

    Forge/.venv/bin/python -I Forge/krea2/bench_krea2_file.py --bundle <bundle> --role denoiser1 --report <new .json> \
        [--function segment] [--rows 27]

Loads only `--role`'s file (one function), runs it `--runs` times at 1024² (4096 image tokens) after a warm-up, and
records the median, so a bundle's file (one function or two) can be compared with a fresh export of the same graph.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import krea2_graphs as kg  # noqa: E402
from check_krea2 import Graph  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--role", default="denoiser1")
    parser.add_argument("--function", default="segment")
    parser.add_argument("--rows", type=int, default=27)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("report already exists")
    manifest = json.loads((args.bundle / ".aether/aether.json").read_text())
    spec = manifest["graphs"][args.role]
    masked = "textMask" in spec["inputs"]
    index = int(args.role.removeprefix("denoiser"))
    length = kg.text_length(manifest) if masked else args.rows
    rng = np.random.default_rng(0)
    n = 4096
    rope = kg.rotary_table(64, 64).numpy()
    if index == 0:
        embeds = np.zeros((length, kg.CONDITIONING), dtype=np.float16)
        embeds[:args.rows] = (rng.standard_normal((args.rows, kg.CONDITIONING)) * 0.1).astype(np.float16)
        feeds = {"latents": rng.standard_normal((n, kg.LATENT_CHANNELS)).astype(np.float16), "text_embeds": embeds,
                 "sigma": np.array([0.75], dtype=np.float32), "rope_image": rope}
    else:
        text = np.zeros((length, kg.HIDDEN), dtype=np.float32)
        text[:args.rows] = (rng.standard_normal((args.rows, kg.HIDDEN)) * 0.5).astype(np.float32)
        feeds = {f"image_{index}": (rng.standard_normal((n, kg.HIDDEN)) * 0.5).astype(np.float32),
                 f"text_{index}": text, "cond": (rng.standard_normal((kg.CONDITION_ROWS, kg.HIDDEN)) * 0.1).astype(np.float32),
                 "rope_image": rope}
    if masked:
        feeds["text_mask"] = kg.text_mask(args.rows, length).numpy()
    file = args.bundle / spec["file"].split(":", 1)[1]
    graph = Graph(file, args.function)
    times = []
    for _ in range(args.runs + 1):
        t = time.monotonic()
        graph(**feeds)
        times.append(time.monotonic() - t)
    graph.close()
    record = {"bundle": str(args.bundle), "role": args.role, "function": args.function, "masked": masked,
              "functions": sorted((spec.get("functions") or {"step": spec.get("function")}).values()),
              "firstSeconds": round(times[0], 2), "medianSeconds": round(statistics.median(times[1:]), 3)}
    args.report.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
