"""Gate a native FL2VA presentation run (`aether-probe h3-profile-presentation ... --vision|--conditioner`) against the CPU
presentation oracle (`prepare-h3-profile-oracle.py --part presentation`), with the bars the user decided on 2026-11-02
(plan D5, revised):

  exact      keyframe bytes, token ids, tags, M-RoPE positions and the bf16 pixels the tower reads;
  tower      per tensor (merger rows and each DeepStack level), relative L2 against the tower's fp32 reference
             (`prepare-h3-vision-reference.py`: upstream's bf16 weights, fp32 compute) at most TOWER_RATIO times upstream's
             own bf16 tower's distance to that reference at the same canvas. A fixed bar cannot hold: upstream's own bf16
             merger is 0.033-0.075 from fp32 across the canvases;
  layers     (--conditioner, with --substituted: upstream's step fed this run's tower outputs, `--part presentation
             --substitute-vision`) relative L2 <= 0.02 of decoder layers 0-3 and of the tap's text rows. The whole tap and
             its image rows are diagnostics: a few image rows carry very large activations whose placement follows the
             tower's bf16 rounding (upstream's own bf16 and fp32 towers give taps 0.55 apart), so no bf16 tower can
             match upstream's image rows.

The tower's distance to upstream's bf16 output, and the tap against the unsubstituted oracle, are recorded as
diagnostics. Prints and writes a JSON verdict; exits 1 when any bar fails. CPU only; reads the directories, writes only
--output. Usage (Forge's H3 oracle environment):
       Forge/h3-oracle/.venv/bin/python -I -B Scripts/qualify-h3-presentation.py --reference ORACLE_DIR --vision-reference FP32_DIR \
           --native RUN_DIR [--substituted SUBSTITUTED_ORACLE_DIR] --output NEW.json
"""
import argparse, hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch

LAYER_LIMIT = 0.02
TOWER_RATIO = 1.05
LAYERS = ("layer00", "layer01", "layer02", "layer03")
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
    ap.add_argument("--substituted", type=Path, help="--conditioner runs: upstream's step fed this run's tower outputs")
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
    spread = fp32_manifest["report"]["bf16_vs_fp32"]
    tower = [("image_embeds", run("image_embeds"), fp32("image_embeds"))] + [(f"deepstack[{k}]", run("deepstack")[k], fp32("deepstack")[k]) for k in levels]
    near = [(name, a, b, TOWER_RATIO * spread[name]["relative_l2"]) for name, a, b in tower]
    image = ref("token_ids") == 151655
    tap_diagnostics = {}
    if "tap" in run_manifest["tensors"]:
        tap, reference = run("tap")[0], ref("tap")[0]
        tap_diagnostics["against_oracle"] = {n: distance(a, b) for n, a, b in
                                             (("tap", tap, reference), ("tap[image rows]", tap[image], reference[image]),
                                              ("tap[text rows]", tap[~image], reference[~image]))}
        if args.substituted is None:
            failures.append("a --conditioner run needs --substituted (upstream's step on this run's tower outputs)")
        else:
            sub_manifest, sub = load(args.substituted)
            for name in ("image_embeds", "deepstack"):
                if not torch.equal(sub(name), run(name)):
                    failures.append(f"--substituted was not fed this run's {name}")
            reference = sub("tap")[0]
            near += [(name, run(name)[0], sub(name)[0], LAYER_LIMIT) for name in LAYERS if name in run_manifest["tensors"]]
            if not all(name in run_manifest["tensors"] for name in LAYERS):
                failures.append("the native run did not record decoder layers 0-3")
            near += [("tap[text rows]", tap[~image], reference[~image], LAYER_LIMIT)]
            tap_diagnostics["against_substituted"] = {n: distance(a, b) for n, a, b in
                                                      (("tap", tap, reference), ("tap[image rows]", tap[image], reference[image]))}
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
                       upstream_bf16_vs_fp32=spread, tower_ratio_to_upstream={name: checks[name]["relative_l2"] / spread[name]["relative_l2"]
                                                                               for name, _, _ in tower if "relative_l2" in checks[name]},
                       tap=tap_diagnostics)
    verdict = dict(schema=2, gate="h3-presentation", passed=not failures, failures=failures,
                   limits=dict(towerRatio=TOWER_RATIO, towerReference="upstream bf16 weights, fp32 compute; bar = ratio x upstream bf16's distance",
                               layers=LAYER_LIMIT, layersReference="upstream's step fed the native tower outputs (--substituted)",
                               decided="2026-11-02, the user (plan D5, revised)"),
                   mode=run_manifest["fields"].get("mode"), canvas=run_manifest["fields"].get("canvas"),
                   profileSHA256=run_manifest["fields"].get("profile_sha256"),
                   referenceFields={k: ref_manifest["fields"].get(k) for k in ("canvas", "layers", "prompt_sha256", "oracle_sha256", "diffusers_commit", "torch")},
                   referenceManifestSHA256=hashlib.sha256((args.reference / "manifest.json").read_bytes()).hexdigest(),
                   nativeManifestSHA256=hashlib.sha256((args.native / "manifest.json").read_bytes()).hexdigest(),
                   visionReferenceManifestSHA256=hashlib.sha256((args.vision_reference / "manifest.json").read_bytes()).hexdigest(),
                   substitutedManifestSHA256=hashlib.sha256((args.substituted / "manifest.json").read_bytes()).hexdigest() if args.substituted else None,
                   nativeReport=run_manifest.get("report"), checks=checks, diagnostics=diagnostics)
    args.output.write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps(dict(passed=verdict["passed"], failures=failures,
                          near={k: round(v["relative_l2"], 6) for k, v in checks.items() if "relative_l2" in v}), indent=1))
    sys.exit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main()
