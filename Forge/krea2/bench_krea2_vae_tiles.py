#!/usr/bin/env python3
"""Compare an overlapping tiled Krea 2 VAE decode with a full decode (development tooling).

Run each configuration in a fresh process under scripts/gpu-lease. --tile 0 is the full baseline; all tiles
have the same shape, including those shifted back from the right/bottom edges, to avoid accumulating Core AI
specializations. Sizes are output pixels, multiples of 16. Example:

    scripts/gpu-lease -- Forge/.venv/bin/python -I Forge/krea2/bench_krea2_vae_tiles.py \
        --asset <vae.aimodel> --reference <reference directory> --tile 512 --overlap 128 \
        --compare <full baseline .npy> --output <new output prefix>

Reports allocations retained after calls, rather than a continuously sampled memory high-water mark. No model
is exported or changed. Output: .json metrics, .npy planar RGB, .png image and (with --compare) an error heatmap.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def origins(extent: int, tile: int, overlap: int) -> list[int]:
    if extent <= tile:
        return [0]
    starts = list(range(0, extent - tile + 1, tile - overlap))
    if starts[-1] != extent - tile:
        starts.append(extent - tile)
    return starts


def axis_weights(starts: list[int], index: int, size: int) -> np.ndarray:
    """Cosine fades in the actual overlaps, including an edge tile shifted back to keep its shape fixed."""
    weights = np.ones(size, dtype=np.float32)
    if index:
        overlap = starts[index - 1] + size - starts[index]
        if overlap:
            phase = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
            weights[:overlap] *= np.sin(phase * (np.pi / 2)) ** 2
    if index + 1 < len(starts):
        overlap = starts[index] + size - starts[index + 1]
        if overlap:
            phase = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
            weights[-overlap:] *= np.cos(phase * (np.pi / 2)) ** 2
    return weights


def tiled_decode(latents, decode, tile: int, overlap: int):
    height, width = latents.shape[0] * 16, latents.shape[1] * 16
    th, tw = min(tile, height), min(tile, width)
    ys, xs = origins(height, th, min(overlap, th - 16)), origins(width, tw, min(overlap, tw - 16))
    total = np.zeros((3, height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    for iy, y in enumerate(ys):
        wy = axis_weights(ys, iy, th)
        for ix, x in enumerate(xs):
            weight = wy[:, None] * axis_weights(xs, ix, tw)[None, :]
            patch = decode(latents[y // 16:(y + th) // 16, x // 16:(x + tw) // 16])
            if patch.shape != (3, th, tw) or not np.isfinite(patch).all():
                raise ValueError(f"invalid decoder tile at {y}, {x}: {patch.shape}")
            total[:, y:y + th, x:x + tw] += patch * weight
            weights[y:y + th, x:x + tw] += weight
    if not np.all(weights > 0):
        raise ValueError("tile weights leave uncovered pixels")
    return total / weights[None], ys, xs


def metal_counter():
    metal = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    device = metal.MTLCreateSystemDefaultDevice()
    if not device:
        raise RuntimeError("no Metal device")
    objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    allocated = ctypes.CFUNCTYPE(ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
    selector = objc.sel_registerName(b'currentAllocatedSize')
    return lambda: allocated(device, selector)


def selftest():
    # Identity nearest-neighbor decoder: catch coverage, edge shifts, channel layout and normalization errors.
    rng = np.random.default_rng(7)
    for h, w, tile, overlap in [(64, 64, 512, 128), (48, 80, 512, 256), (16, 40, 512, 128), (64, 64, 768, 256),
                                (64, 64, 512, 0)]:
        latents = rng.standard_normal((h, w, 3)).astype(np.float32)
        decode = lambda z: z.transpose(2, 0, 1).repeat(16, axis=1).repeat(16, axis=2)
        actual, _, _ = tiled_decode(latents, decode, tile, overlap)
        np.testing.assert_allclose(actual, decode(latents), atol=1e-6, rtol=1e-6)
    print("tile coverage/layout selftest passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--tile', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.asset or not args.reference or not args.output:
        parser.error('--asset, --reference and --output are required')
    if args.tile < 0 or args.tile % 16 or args.overlap < 0 or args.overlap % 16 or args.runs < 1:
        parser.error('tile and overlap must be nonnegative multiples of 16; runs must be positive')
    if args.tile and args.overlap >= args.tile:
        parser.error('overlap must be smaller than tile')
    for suffix in ('.json', '.npy', '.png', '-error.png'):
        if Path(str(args.output) + suffix).exists():
            parser.error(f'output already exists: {args.output}{suffix}')
    from check_krea2 import Graph, psnr, ssim
    from PIL import Image

    config = json.loads((args.reference / 'reference.json').read_text())
    height, width = config['height'], config['width']
    with np.load(args.reference / f"step{config['steps'] - 1}.npz") as step:
        latents = step['latents_out'].reshape(height // 16, width // 16, 64).astype(np.float16)
    expected = np.load(args.reference / 'image.npy')
    memory = metal_counter()
    record = {'asset': str(args.asset), 'reference': str(args.reference), 'height': height, 'width': width,
              'tile': args.tile, 'overlap': args.overlap, 'metalBefore': memory(), 'runs': args.runs}
    graph = Graph(args.asset, 'decode')
    allocations, call_seconds = [], []

    def decode(z):
        started = time.monotonic()
        patch = graph(latents=z)['image'].astype(np.float32)
        call_seconds.append(time.monotonic() - started)
        allocations.append(memory())
        return patch

    seconds, repeat_identical = [], True
    earlier = None
    for _ in range(args.runs):
        started = time.monotonic()
        if args.tile:
            output, ys, xs = tiled_decode(latents, decode, args.tile, args.overlap)
        else:
            output, ys, xs = decode(latents), [0], [0]
        seconds.append(time.monotonic() - started)
        if earlier is not None:
            repeat_identical &= np.array_equal(output, earlier)
        earlier = output
    record.update(seconds=seconds, tileCallSeconds=call_seconds, originsY=ys, originsX=xs,
                  tilesPerRun=len(ys) * len(xs), repeatIdentical=bool(repeat_identical),
                  finite=bool(np.isfinite(output).all()), metalAfterRun=memory(), sampledMetalMax=max(allocations),
                  psnrToReferenceDB=psnr(output, expected), ssimToReference=ssim(output, expected))
    if args.compare:
        full = np.load(args.compare)
        if full.shape != output.shape:
            raise ValueError('comparison image shape mismatch')
        error = np.abs(output - full)
        band = np.zeros((height, width), dtype=bool)
        tile_height = min(args.tile, height) if args.tile else height
        tile_width = min(args.tile, width) if args.tile else width
        for y in set(ys + [v + tile_height for v in ys]) - {0, height}:
            band[max(0, y - 8):min(height, y + 8), :] = True
        for x in set(xs + [v + tile_width for v in xs]) - {0, width}:
            band[:, max(0, x - 8):min(width, x + 8)] = True
        record.update(psnrToFullDB=psnr(output, full), ssimToFull=ssim(output, full), maxAbsToFull=float(error.max()),
                      meanAbsToFull=float(error.mean()),
                      seamBandMeanAbs=float(error[:, band].mean()) if band.any() else 0,
                      outsideSeamMeanAbs=float(error[:, ~band].mean()) if (~band).any() else 0)
        # Grayscale error: white = at least 0.05 on the decoder's [-1, 1] scale. Fixed scale across comparisons.
        Image.fromarray((np.clip(error.mean(axis=0) / 0.05, 0, 1) * 255).astype(np.uint8)).save(str(args.output) + '-error.png')
    np.save(str(args.output) + '.npy', output)
    rgb = ((output.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).round().astype(np.uint8)
    Image.fromarray(rgb).save(str(args.output) + '.png')
    Path(str(args.output) + '.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)
    if not record['finite'] or not repeat_identical:
        raise SystemExit('nonfinite or nondeterministic decode')


if __name__ == '__main__':
    main()
