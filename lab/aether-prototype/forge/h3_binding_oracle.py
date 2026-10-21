"""CPU-only changing-weight oracle for the H3 phase probe.

Checks the original residual block is exactly the identity when all learned
projections are zero. The full-row native sequence then alternates real, zero
and real bindings through one function, separately for each execution API.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from h3_export import digest, save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--adapter-probe", type=Path, required=True)
    ap.add_argument("--phase-probe", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    lab, out = args.lab.resolve(), args.output.resolve()
    if out.is_relative_to(lab):
        raise ValueError("outputs must stay outside the read-only lab")
    out.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(lab / "forge"), str(lab / "forge/vendor/python"), str(lab / "forge/vendor/diffusers/src")]
    import qualify_block as q
    import torch
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    bundle = json.loads(args.bundle.read_text())
    pilot = json.loads(args.adapter_probe.read_text())
    phases = json.loads(args.phase_probe.read_text())
    with torch.device("meta"):
        block = q.MiniMaxH3Transformer3DModel().transformer_blocks[0]
    block.adaln_proj = q.CachedModulation()
    block = block.to(dtype=torch.bfloat16).to_empty(device="cpu").eval()
    for p in block.parameters():
        p.zero_()
    def read(item):
        raw = Path(item["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        return torch.frombuffer(bytearray(raw), dtype=getattr(torch, item["dtype"])).reshape(item["shape"])
    h, table, tags, cos, sin = (read(item) for item in pilot["inputs"])
    assert torch.equal(block(h, table, tags, (cos, sin)), h)
    selected = next(c for c in phases["cases"] if c["name"] == "body-1")
    zero = copy.deepcopy(selected)
    zero["name"] = "body-zero-bindings"
    zero["outputs"] = [dict(bundle["oracle"]["hidden"], name="hidden_out")]
    zero["inputs"] = zero["inputs"][:5]
    for item in bundle["parameters"][0]:
        count = 2
        assert item["dtype"] == "bfloat16"
        for d in item["shape"]:
            count *= d
        path = out / (item["name"] + ".raw")
        raw = bytes(count)
        path.write_bytes(raw)
        zero["inputs"].append(dict(item, path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
    phases["cases"].append(zero)
    phases["cases"].append(dict(selected, name="body-original-bindings-restored"))
    phases["scope"] += "; zero-binding case uses the source residual identity, independently checked on 128 CPU rows"
    save(out / "probe.json", phases)
    save(out / "cpu-checks.json", dict(schema=1, passed=True, gpu_work=False,
         source_recipe_sha256=digest(__file__), input_phase_probe_sha256=digest(args.phase_probe),
         source_zero_projection_identity_byte_identical=True, cpu_identity_rows=128,
         scope="Original CPU zero-projection residual identity; native changing-binding behavior remains untested",
         case_order=[c["name"] for c in phases["cases"]], probe_sha256=digest(out / "probe.json")))
    print("CPU changing-binding oracle ready", flush=True)


if __name__ == "__main__":
    main()
