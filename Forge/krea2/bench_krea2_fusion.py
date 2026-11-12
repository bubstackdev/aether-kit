#!/usr/bin/env python3
"""Development probe: move invariant text fusion out of segment zero without changing the shipping contract.

    Forge/.venv/bin/python -I Forge/krea2/bench_krea2_fusion.py --export --workdir <new dir>
    scripts/gpu-lease -- Forge/.venv/bin/python -I Forge/krea2/bench_krea2_fusion.py \
        --mode baseline --bundle <256-row bundle> --workdir <dir>
    scripts/gpu-lease -- Forge/.venv/bin/python -I Forge/krea2/bench_krea2_fusion.py \
        --mode split --bundle <same bundle> --workdir <dir>

Export uses CPU and the local pinned checkpoint/adapter. Benchmark modes run in separate fresh processes on GPU.
The split graph takes fused float32 text rather than the normal conditioning; it is a probe, not a kit bundle.
Measures segment-zero calls only, not a whole denoising loop. Split mode releases the fusion graph before loading
the segment, and compares every segment output with baseline mode. Retained Metal is sampled at call boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import types

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import krea2_graphs as kg
from export_krea2 import DEFAULT_SNAPSHOT, FORGE_INPUTS, ADAPTER_FILE, convert
from check_krea2 import Graph, cosine
from bench_krea2_vae_tiles import metal_counter


class Fusion(nn.Module):
    def __init__(self, segment):
        super().__init__()
        self.segment = segment

    def forward(self, text, mask):
        return self.segment.fuse_text(text, (mask > 0.5)[None, None, None, :])[0]


def use_fused(self, text, mask=None):
    return text[None]


def export(args):
    args.workdir.mkdir(parents=True)
    torch.set_grad_enabled(False)
    checkpoint = kg.Checkpoint(args.snapshot)
    segment = kg.DenoiserSegment.load(checkpoint, 0, kg.LoKr(args.adapter), text_rows=args.text_rows)
    example, dynamic = segment.example()
    fusion = Fusion(segment)
    record = {'textRows': args.text_rows}
    record['fusion'] = convert(fusion, (example[1], example[2]), (None, None),
                               ['text_embeds', 'text_mask'], ['fused_text'], 'fuse', args.workdir / 'fusion.aimodel')
    expected = segment(*example)
    fused = fusion(example[1], example[2])
    segment.fuse_text = types.MethodType(use_fused, segment)
    inputs = (example[0], fused, *example[2:])
    actual = segment(*inputs)
    record['cpuMaxAbs'] = [float((a - b).abs().max()) for a, b in zip(actual, expected)]
    if any(record['cpuMaxAbs']):
        raise RuntimeError('splitting fusion changed CPU segment outputs')
    record['segment'] = convert(segment, inputs, dynamic, segment.input_names, segment.output_names,
                                'segment', args.workdir / 'denoiser0.aimodel')
    (args.workdir / 'export.json').write_text(json.dumps(record, indent=2) + '\n')


def benchmark(args):
    report = args.workdir / f'{args.mode}.json'
    if report.exists():
        raise SystemExit('report exists; use a fresh directory or mode')
    manifest = json.loads((args.bundle / '.aether/aether.json').read_text())
    length = kg.text_length(manifest)
    if length != args.text_rows or args.rows > length:
        raise SystemExit('bundle text length must match --text-rows and fit --rows')
    memory = metal_counter()
    record = {'mode': args.mode, 'metalBefore': memory(), 'textRows': length, 'rows': args.rows, 'size': args.size}
    rng = np.random.default_rng(0)
    side = args.size // kg.PIXELS_PER_TOKEN
    text = np.zeros((length, kg.CONDITIONING), np.float16)
    text[:args.rows] = (rng.standard_normal((args.rows, kg.CONDITIONING)) * .1).astype(np.float16)
    feeds = {'latents': (rng.standard_normal((side * side, 64)) * .1).astype(np.float16), 'text_embeds': text,
             'text_mask': kg.text_mask(args.rows, length).numpy(), 'sigma': np.array([.75], np.float32),
             'rope_image': kg.rotary_table(side, side).numpy()}
    if args.mode == 'split':
        fusion = Graph(args.workdir / 'fusion.aimodel', 'fuse')
        fusion_times = []
        for _ in range(3):
            start = time.monotonic()
            fused = fusion(text_embeds=text, text_mask=feeds['text_mask'])['fused_text']
            fusion_times.append(time.monotonic() - start)
        record.update(fusionSeconds=fusion_times, metalWithFusion=memory())
        feeds['text_embeds'] = fused
        fusion.close()
        path = args.workdir / 'denoiser0.aimodel'
    else:
        path = args.bundle / manifest['graphs']['denoiser0']['file'].split(':', 1)[1]
    graph = Graph(path, 'segment')
    record['loadSeconds'] = graph.load_seconds
    times = []
    for _ in range(args.runs):
        start = time.monotonic()
        result = graph(**feeds)
        times.append(time.monotonic() - start)
        print(args.mode, len(times), times[-1], flush=True)
    record.update(seconds=times, warmMedianSeconds=statistics.median(times[2:]), metalAfter=memory(),
                  finite=all(bool(np.isfinite(x).all()) for x in result.values()))
    if args.mode == 'split':
        baseline = np.load(args.workdir / 'baseline.npz')
        record['cosines'] = {k: cosine(result[k], baseline[k]) for k in result}
        record['maxAbs'] = {k: float(np.max(np.abs(result[k] - baseline[k]))) for k in result}
    np.savez(report.with_suffix('.npz'), **result)
    report.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)
    graph.close()
    if not record['finite'] or any(v < 0.999 for v in record.get('cosines', {}).values()):
        raise SystemExit('nonfinite outputs or segment parity failure')


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--export', action='store_true')
    modes.add_argument('--mode', choices=['baseline', 'split'])
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--snapshot', type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument('--adapter', type=Path, default=FORGE_INPUTS / ADAPTER_FILE)
    parser.add_argument('--text-rows', type=int, default=256)
    parser.add_argument('--rows', type=int, default=27)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--runs', type=int, default=7)
    args = parser.parse_args()
    if args.rows < 1 or args.runs < 3 or args.size < 32 or args.size % 16:
        parser.error('positive rows, at least three runs, and a size divisible by 16 (at least 32) are required')
    if args.export:
        export(args)
    else:
        if args.bundle is None:
            parser.error('--mode needs --bundle')
        benchmark(args)


if __name__ == '__main__':
    main()
