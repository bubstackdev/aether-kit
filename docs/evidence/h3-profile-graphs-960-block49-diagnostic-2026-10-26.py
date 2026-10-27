"""Diagnostic (CPU only): block 49 on the gate's probe input in float32, to place the native/CPU-bf16 gap against bf16 noise."""
import importlib.util, json, sys, time
from pathlib import Path
import torch

W = Path('/Users/zakkeown/Code/bubstackdev/aether/aether-kit/.claude/worktrees/agent-a493f969f1fbbf36e')
G = Path('/Volumes/Scratch/aether/aether-kit-h3-2026-10-26/graphs-960')
OUT = Path('/Volumes/Scratch/aether/aether-kit-h3-2026-10-26/diagnostics/block49-fp32')
sys.path.insert(0, str(W / 'Scripts'))
spec = importlib.util.spec_from_file_location('oracle', W / 'Scripts/prepare-h3-profile-oracle.py')
o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)
h = o.h
torch.set_num_threads(8); torch.set_grad_enabled(False)
block_index = int(sys.argv[1]) if len(sys.argv) > 1 else 49
dump = o.Raw(G / 'dump-t2va')
checkpoint = h.Checkpoint(o.SNAPSHOT_ROOT, o.converter())
adapter = h.adapter_path()
model = o.denoiser_model(checkpoint, adapter)
shape, checks = o.layout(dump, model.config)
assert all(checks.values())
unique, inverse = shape['plan'][0]
temb = model.time_embedder(model.time_proj(unique).to(torch.float32))
rotary = model.rope(shape['position_ids'])
adaln = inverse * 3 + shape['token_tags']
probe = dump.get('probe_hidden')
del model
block = o.real_block(checkpoint, adapter, block_index).float()
t = time.monotonic()
out32 = block(probe.float(), temb.float(), adaln, tuple(r.float() for r in rotary))
seconds = time.monotonic() - t


def get(d, name):
    return o.Raw(G / d).get(name)


native = get('native-t2va', f'block{block_index:02d}')[0].double()
cpu16 = get('oracle-graphs-t2va', f'block{block_index:02d}')[0].double()
ref = out32[0].double()
rl = lambda a, b: float((a - b).norm() / b.norm())
video = dump.get('video_indices').long(); audio = dump.get('audio_indices').long()
result = dict(block=block_index, fp32_seconds=round(seconds, 1),
              native_vs_cpu_bf16=rl(native, cpu16), native_vs_cpu_fp32=rl(native, ref), cpu_bf16_vs_cpu_fp32=rl(cpu16, ref),
              video=dict(native_vs_fp32=rl(native[video], ref[video]), bf16_vs_fp32=rl(cpu16[video], ref[video])),
              audio=dict(native_vs_fp32=rl(native[audio], ref[audio]), bf16_vs_fp32=rl(cpu16[audio], ref[audio])),
              fp32_output_sha=h.sha256_bytes(out32.contiguous().view(torch.uint8).numpy().tobytes()))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f'block{block_index:02d}.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
