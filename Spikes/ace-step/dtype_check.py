"""Is the Core AI fp16 DiT error intrinsic to half precision? Eager bf16 and fp16 (CPU) vs the eager fp32 reference.

usage: dtype_check.py <export-dir> [frames] [enc]   (imports export_dit's model setup; CPU only)
"""
import sys
from pathlib import Path

import numpy as np
import torch

E = Path(sys.argv[1])
T = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
L = int(sys.argv[3]) if len(sys.argv) > 3 else 512
sys.argv = [sys.argv[0], str(E), "--frames", str(T), "--enc", str(L), "--skip-export", "--skip-ref"]
sys.path.insert(0, str(Path(__file__).parent))
import export_dit as X  # noqa: E402  (builds the decoder; --skip-export skips conversion)

ref = np.load(E / f"dit-ref-v-{T}x{L}.npy").astype(np.float64).ravel()
feeds = [np.load(E / f"dit-ref-{n}-{T}x{L}.npy") for n in ("xt", "t", "context", "encoder")]
for dt in (torch.bfloat16, torch.float16):
    step = X.Step(X.dit).to(dt)
    with torch.no_grad():
        v = step(*[torch.from_numpy(x).to(dt) for x in feeds]).float().numpy().astype(np.float64).ravel()
    print(dt, "relL2 %.4f cosine %.6f finite %s" % (np.linalg.norm(v - ref) / np.linalg.norm(ref),
                                                   v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref)),
                                                   np.isfinite(v).all()))
