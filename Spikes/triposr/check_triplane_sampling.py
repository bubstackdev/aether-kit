"""CPU check, no weights: GridField and Field against F.grid_sample on random planes and a random decoder.

usage: check_triplane_sampling.py [res] [slab]
grid_sample (align_corners=False, zero padding) is upstream's renderer.query_triplane sampling; it is the reference.
"""
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from tsr_graphs import Field, GridField  # noqa: E402

res = int(sys.argv[1]) if len(sys.argv) > 1 else 64
slab = int(sys.argv[2]) if len(sys.argv) > 2 else 8
torch.manual_seed(0)
radius, bias, C = 0.87, -1.0, 40
layers = torch.nn.Sequential(torch.nn.Linear(3 * C, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(),
                             torch.nn.Linear(64, 4))
m = types.SimpleNamespace(decoder=types.SimpleNamespace(layers=layers),
                          renderer=types.SimpleNamespace(cfg=types.SimpleNamespace(radius=radius, density_bias=bias)))
planes = torch.randn(3, C, 64, 64, dtype=torch.float64)
layers.double()


def reference(points):                                   # upstream: planes sampled at (x, y), (x, z), (y, z)
    x = points / radius
    coords = torch.stack([x[:, [0, 1]], x[:, [0, 2]], x[:, [1, 2]]])[:, None]          # [3, 1, N, 2]
    feats = F.grid_sample(planes, coords, align_corners=False, mode="bilinear")        # [3, C, 1, N]
    h = feats[:, :, 0].permute(2, 0, 1).reshape(points.shape[0], 3 * C)                # plane-major, like upstream
    return torch.exp(layers(h)[:, 0] + bias)


lin = torch.linspace(0, 1, res, dtype=torch.float64) * 2 * radius - radius
zs = lin[:slab]
gx, gy, gz = torch.meshgrid(lin, lin, zs, indexing="ij")
points = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], -1)
with torch.no_grad():
    ref = reference(points)
    field = Field(m)(planes, points)[0][:, 0]
    grid = GridField(m)(planes, lin, zs).reshape(-1)
for name, value in [("Field", field), ("GridField", grid)]:
    err = float((value - ref).abs().max() / ref.abs().max())
    print(f"{name} vs grid_sample over {res}x{res}x{slab}: max relative error {err:.2e}")
    assert err < 1e-10, name
print("both samplers match grid_sample")
