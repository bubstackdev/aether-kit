"""Rotary oracle from pinned Diffusers 7ecb1e1b: build_packed_sequence + MiniMaxH3RotaryPosEmbed for a small
first-keyframe layout (CPU, no weights). Usage: python -I -B prepare-h3-rotary-oracle.py OUTPUT.json"""
import json, sys, hashlib
import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
text = torch.ones(7, dtype=torch.int64)
pos, tags, vi, ai, ti, cond, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(text, 7, 4, 6, 37, (1, 2, 2), 2, 2, 0, ("first",))
cos, sin = MiniMaxH3RotaryPosEmbed(16, 10000.0)(pos)
digest = lambda t: hashlib.sha256(t.contiguous().numpy().astype("<f4").tobytes()).hexdigest()
json.dump(dict(width=96, height=64, requested_frames=22, text_rows=7, anchors=["first"], rows=int(pos.shape[0]),
               positions=pos.to(torch.float32).flatten().tolist(), cos=cos.flatten().tolist(), sin=sin.flatten().tolist(),
               cos_sha256=digest(cos), sin_sha256=digest(sin), scope="pinned Diffusers layout + rotary; no weights"),
          open(sys.argv[1], "w"))
print(pos.shape, cos.shape)
