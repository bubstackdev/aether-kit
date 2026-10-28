"""Diagnose the gather-free grid function: the einsum GridField runs correctly in eager (fp32 and fp16) but its Core AI
export returns unrelated densities. Exports the pieces and a matmul formulation, then (with --run, on the GPU under the
lock) compares each against eager fp32 on one z-slab of a real image's scene codes.

usage: diag_grid.py <tsr-source-dir> <codes.npy> <out-dir> --export
       USE_OS_COREAI=1 diag_grid.py <tsr-source-dir> <codes.npy> <out-dir> --run
"""
import argparse
import asyncio
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("codes")
ap.add_argument("out")
ap.add_argument("--export", action="store_true")
ap.add_argument("--run", action="store_true")
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--slab", type=int, default=16)
ap.add_argument("--k", type=int, default=128)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
R, S, K = a.res, a.slab, a.k
lin = torch.linspace(0, 1, R) * 2 * 0.87 - 0.87
codes = torch.from_numpy(np.load(a.codes))

sys.modules["torchmcubes"] = types.SimpleNamespace(marching_cubes=None)
sys.modules.setdefault("rembg", types.ModuleType("rembg"))
sys.path.insert(0, a.src)
sys.path.insert(0, str(Path(__file__).parent))
from tsr_graphs import GridField  # noqa: E402


class Planes(GridField):
    """Only the three planes' interpolated features (einsum), concatenated: [R, R, S, 3C] flattened to [R·R·S, 3C]."""

    def forward(self, planes, xs, zs):
        P, C, H, W = planes.shape
        padded = F.pad(planes, (1, 1, 1, 1))
        x_w, y_h, y_w, z_h = self.weights(xs, W), self.weights(xs, H), self.weights(xs, W), self.weights(zs, H)
        f0 = torch.einsum("jv,cvu,iu->ijc", y_h, padded[0], x_w)
        return f0.reshape(-1, C)


class MatmulGridField(GridField):
    """GridField with each plane's interpolation as two batched matmuls (no einsum) and explicit repeats."""

    def forward(self, planes, xs, zs):
        P, C, H, W = planes.shape
        padded = F.pad(planes, (1, 1, 1, 1))                               # [3, C, H+2, W+2]
        x_w, y_h, y_w, z_h = self.weights(xs, W), self.weights(xs, H), self.weights(xs, W), self.weights(zs, H)

        def sample(plane, rows, cols):                                     # plane [C, H+2, W+2] → [len(cols), len(rows), C]
            # Two linear layers with the hat weights as the weight matrix: a broadcast [N, H+2] @ [C, H+2, W+2], or bmm
            # over an expanded batch, fails Core AI's optimizer (broadcast_in_dims rank mismatch).
            t = F.linear(plane.transpose(1, 2), rows)                      # [C, W+2, Nrows]
            t = F.linear(t.transpose(1, 2), cols)                          # [C, Nrows, Ncols]
            return t.permute(2, 1, 0)                                      # [Ncols, Nrows, C]

        f0 = sample(padded[0], y_h, x_w)                                   # [R(x), R(y), C]
        f1 = sample(padded[1], z_h, x_w)                                   # [R(x), S(z), C]
        f2 = sample(padded[2], z_h, y_w)                                   # [R(y), S(z), C]
        h = torch.cat([f0.unsqueeze(2).repeat(1, 1, S, 1), f1.unsqueeze(1).repeat(1, R, 1, 1),
                       f2.unsqueeze(0).repeat(R, 1, 1, 1)], dim=-1)         # [R, R, S, 3C]
        o = self.decoder.layers(h.reshape(R * R * S, 3 * C))
        return torch.exp(o[:, 0] + self.density_bias).reshape(R, R, S)


variants = {"einsum": GridField, "planes0": Planes, "matmul": MatmulGridField}

if a.export:
    import coreai_torch as ct
    from tsr.system import TSR

    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snap = next((hub / "models--stabilityai--TripoSR/snapshots").iterdir())
    model = TSR.from_pretrained(str(snap), config_name="config.yaml", weight_name="model.ckpt").eval()
    xs, zs = lin, lin[K:K + S]
    refs = {}
    with torch.no_grad():
        for name, cls in variants.items():
            refs[name] = cls(model)(codes, xs, zs).numpy()
            np.save(out / f"eager32-{name}.npy", refs[name])
        print("eager fp32 matmul vs einsum max rel %.2e" % (np.abs(refs["matmul"] - refs["einsum"]).max()
                                                           / np.abs(refs["einsum"]).max()))
    model = model.to(torch.float16)
    for name, cls in variants.items():
        with torch.no_grad():
            ep = torch.export.export(cls(model), (codes.half(), xs.half(), zs.half())).run_decompositions(
                ct.get_decomp_table())
        c = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
        c.add_exported_program(ep, input_names=["scene_codes", "xs", "zs"], output_names=["density"])
        prog = c.to_coreai()
        prog.optimize()
        prog.save_asset(out / f"{name}-fp16.aimodel")
        print("exported", name, flush=True)

if a.run:
    import coreai.runtime as rt

    async def main():
        opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
        report = {}
        inputs = {"scene_codes": rt.NDArray(codes.numpy().astype(np.float16)),
                  "xs": rt.NDArray(lin.numpy().astype(np.float16)),
                  "zs": rt.NDArray(lin[K:K + S].numpy().astype(np.float16))}
        for name in variants:
            model = await rt.AIModel.load(out / f"{name}-fp16.aimodel", opts)
            got = (await model.load_function("main")(inputs))["density"].numpy().astype(np.float32)
            ref = np.load(out / f"eager32-{name}.npy")
            got = got.reshape(ref.shape)
            finite = np.isfinite(got)
            entry = {"shape": list(got.shape), "nonfinite": int((~finite).sum()),
                     "maxRelErr": float(np.abs(np.where(finite, got, 0) - ref).max() / np.abs(ref).max()),
                     "cosine": float(np.dot(np.where(finite, got, 0).ravel(), ref.ravel())
                                     / (np.linalg.norm(np.where(finite, got, 0)) * np.linalg.norm(ref)))}
            if name != "planes0":
                entry["occupancyIoU"] = float(((got > 25) & (ref > 25)).sum() / max(1, ((got > 25) | (ref > 25)).sum()))
            report[name] = entry
            print(name, json.dumps(entry), flush=True)
        (out / "diag-run.json").write_text(json.dumps(report, indent=1))

    asyncio.run(main())
