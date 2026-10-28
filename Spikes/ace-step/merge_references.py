"""Assemble the four corrected references the gate reads, and record where each half came from.

usage: merge_references.py <out-dir> --first <dir> <id> [<dir> <id> ...] --redo <bf16-dir> <fp32-dir> <id> [...]

reference_trajectory.py used to cast the DiT to bf16 in place for the bf16 trajectory and back with .float(), so every
later prompt in a process had bf16-rounded weights behind its "fp32" trajectory. A process's first prompt is correct
throughout, and so is every bf16 trajectory (re-casting rounded weights to bf16 changes nothing).
  --first  copies a reference whose fp32 and bf16 are both valid (the first prompt of its process);
  --redo   takes the bf16 half from the original run and every fp32_* array from a later --skip-bf16 rerun with the
           fixed script, after checking that the conditions, noise and token IDs are identical in both.
Writes <out-dir>/<id>.npz and <out-dir>/provenance.json (sources, the identity checks, the old fp32 final latents'
relative L2 from the corrected ones, and every file's SHA-256).
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SHARED = ["encoder", "encoder_mask", "context", "noise", "text_ids", "lyric_ids", "text_hidden", "lyric_hidden"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(argv):
    out, first, redo, mode, i = Path(argv[0]), [], [], None, 1
    while i < len(argv):
        if argv[i] in ("--first", "--redo"):
            mode = argv[i]
            i += 1
            continue
        if mode == "--first":
            first.append((Path(argv[i]), argv[i + 1]))
            i += 2
        elif mode == "--redo":
            redo.append((Path(argv[i]), Path(argv[i + 1]), argv[i + 2]))
            i += 3
        else:
            raise SystemExit(__doc__)
    return out, first, redo


def main():
    out, first, redo = parse(sys.argv[1:])
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"rule": "fp32 and bf16 valid for each process's first prompt; later prompts take fp32 from the "
                          "--skip-bf16 rerun", "references": {}}
    for directory, stem in first:
        source = directory / f"{stem}.npz"
        shutil.copyfile(source, out / f"{stem}.npz")
        provenance["references"][stem] = {"fp32": source.parent.name, "bf16": source.parent.name,
                                          "sourceSHA256": sha256(source)}
    for bf16_dir, fp32_dir, stem in redo:
        bf16_src, fp32_src = bf16_dir / f"{stem}.npz", fp32_dir / f"{stem}.npz"
        b, f = np.load(bf16_src), np.load(fp32_src)
        identical = {k: bool(np.array_equal(b[k], f[k])) for k in SHARED}
        if not all(identical.values()):
            raise SystemExit(f"{stem}: conditions differ between runs: {identical}")
        merged = {k: b[k] for k in b.files}
        merged.update({k: f[k] for k in f.files if k.startswith("fp32_")})
        np.savez(out / f"{stem}.npz", **merged)
        old_fp32_final = float(np.linalg.norm(b["fp32_x0"] - f["fp32_x0"]) / np.linalg.norm(f["fp32_x0"]))
        provenance["references"][stem] = {"fp32": fp32_src.parent.name, "bf16": bf16_src.parent.name,
                                          "sharedArraysIdentical": identical,
                                          "oldFp32FinalRelL2FromCorrected": old_fp32_final,
                                          "bf16SourceSHA256": sha256(bf16_src), "fp32SourceSHA256": sha256(fp32_src)}
    for stem, entry in provenance["references"].items():
        entry["mergedSHA256"] = sha256(out / f"{stem}.npz")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(json.dumps(provenance, indent=1))


if __name__ == "__main__":
    main()
