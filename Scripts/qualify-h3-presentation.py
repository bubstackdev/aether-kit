"""Gate a native FL2VA presentation run (`aether-probe h3-profile-presentation ... --vision|--conditioner`) against the CPU
presentation oracle (`prepare-h3-profile-oracle.py --part presentation`), with the plan's bars (D5, and the user's
2026-10-29 decision for the tower):

  exact      keyframe bytes, token ids, tags, M-RoPE positions and the bf16 pixels the tower reads;
  <= 0.03    relative L2 of the merger rows and each DeepStack level against the tower's fp32 reference
             (`prepare-h3-vision-reference.py`: upstream's bf16 weights, fp32 compute). Upstream's own bf16 tower is
             0.033 from that reference on the merger at 288x288, so no bf16 tower meets 0.02 against upstream bf16;
  <= 0.02    (--conditioner) relative L2 of the tap against the full oracle, over all rows and separately over the
             image rows and the text rows.

The tower's distance to upstream's bf16 output and upstream's own bf16-vs-fp32 spread are recorded as diagnostics.

Prints and writes a JSON verdict; exits 1 when any bar fails. CPU only; reads both directories, writes only --output.
Usage (Forge's H3 oracle environment):
       Forge/h3-oracle/.venv/bin/python -I -B Scripts/qualify-h3-presentation.py --reference ORACLE_DIR --vision-reference FP32_DIR \
           --native RUN_DIR --output NEW.json
"""
import argparse, hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch

TAP_LIMIT = 0.02
TOWER_LIMIT = 0.03
EXACT = ("keyframe0", "token_ids", "text_token_tags", "position_ids", "pixel_values")


def load(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())

    def get(name):
        entry = manifest["tensors"][name]
        data = (directory / entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], f"{directory}/{name} changed on disk"
        return torch.frombuffer(bytearray(data), dtype=getattr(torch, entry["dtype"])).view(entry["shape"]).clone()
    return manifest, get


def distance(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return dict(relative_l2=float((a - b).norm() / b.norm().clamp_min(1e-300)), cosine=float(a @ b / (a.norm() * b.norm()).clamp_min(1e-300)),
                max_abs=float((a - b).abs().max()), reference_max_abs=float(b.abs().max()), finite=bool(torch.isfinite(a).all()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--vision-reference", type=Path, required=True)
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("output exists; choose a new file")
    ref_manifest, ref = load(args.reference)
    run_manifest, run = load(args.native)
    fp32_manifest, fp32 = load(args.vision_reference)
    assert fp32_manifest["fields"]["oracle_manifest_sha256"] == hashlib.sha256((args.reference / "manifest.json").read_bytes()).hexdigest(), \
        "the fp32 reference was made from another oracle run"
    checks, failures = {}, []
    for name in EXACT:
        a, b = run(name), ref(name)
        ok = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        checks[name] = dict(exact=ok, shape=list(a.shape))
        if not ok:
            failures.append(f"{name} differs")
    levels = range(ref("deepstack").shape[0])
    tower = [("image_embeds", run("image_embeds"), fp32("image_embeds"))] + [(f"deepstack[{k}]", run("deepstack")[k], fp32("deepstack")[k]) for k in levels]
    near = [(name, a, b, TOWER_LIMIT) for name, a, b in tower]
    if "tap" in run_manifest["tensors"]:
        tap, reference = run("tap")[0], ref("tap")[0]
        image = ref("token_ids") == 151655
        near += [("tap", tap, reference, TAP_LIMIT), ("tap[image rows]", tap[image], reference[image], TAP_LIMIT),
                 ("tap[text rows]", tap[~image], reference[~image], TAP_LIMIT)]
    for name, a, b, limit in near:
        row = distance(a, b) if a.shape == b.shape else dict(error=f"shape {list(a.shape)} vs {list(b.shape)}")
        row["limit"] = limit
        row["passed"] = "error" not in row and row["finite"] and row["relative_l2"] <= limit
        checks[name] = row
        if not row["passed"]:
            failures.append(f"{name}: {row}")
    # Diagnostics, not bars: the tower against upstream's bf16 output, and upstream's own bf16-vs-fp32 spread.
    diagnostics = dict(native_vs_upstream_bf16={name: distance(a, b) for name, a, b in
                                                [("image_embeds", run("image_embeds"), ref("image_embeds"))]
                                                + [(f"deepstack[{k}]", run("deepstack")[k], ref("deepstack")[k]) for k in levels]},
                       upstream_bf16_vs_fp32=fp32_manifest["report"]["bf16_vs_fp32"])
    verdict = dict(schema=2, gate="h3-presentation", passed=not failures, failures=failures,
                   limits=dict(tower=TOWER_LIMIT, towerReference="upstream bf16 weights, fp32 compute", tap=TAP_LIMIT, tapReference="full presentation oracle"),
                   mode=run_manifest["fields"].get("mode"), canvas=run_manifest["fields"].get("canvas"),
                   profileSHA256=run_manifest["fields"].get("profile_sha256"),
                   referenceFields={k: ref_manifest["fields"].get(k) for k in ("canvas", "layers", "prompt_sha256", "oracle_sha256", "diffusers_commit", "torch")},
                   referenceManifestSHA256=hashlib.sha256((args.reference / "manifest.json").read_bytes()).hexdigest(),
                   nativeManifestSHA256=hashlib.sha256((args.native / "manifest.json").read_bytes()).hexdigest(),
                   visionReferenceManifestSHA256=hashlib.sha256((args.vision_reference / "manifest.json").read_bytes()).hexdigest(),
                   nativeReport=run_manifest.get("report"), checks=checks, diagnostics=diagnostics)
    args.output.write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps(dict(passed=verdict["passed"], failures=failures,
                          near={k: round(v["relative_l2"], 6) for k, v in checks.items() if "relative_l2" in v}), indent=1))
    sys.exit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main()
