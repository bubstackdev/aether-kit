"""Extend an H3 export to profiles it lacks, in a NEW folder that hard-links what it reuses (standard library only).

    python3 -I Forge/h3/extend_h3.py --source <export> --output <new export> --manifest-only
    python3 -I Forge/h3/extend_h3.py --source <export> --output <new export> --python Forge/.venv/bin/python

An export is `<dir>/bundle` (the installed-revision layout) and `<dir>/work` (build.json, the converted VAEs). The new
folder gets every bundle file and the converted VAEs as hard links to the source (no extra disk; both must be on one
volume), and fresh copies of the three files that change: `.aether/aether.json`, `SHA256SUMS` and
`export-report.json`, and `work/build.json`. The source is never written.

- `--manifest-only`: no graph is made. The manifest names every profile of `profiles_h3.PROFILES` whose graphs the
  export already has: a profile whose audio decoder is missing (another frame count) is left out, and one whose
  canvas has no keyframe encoder tail or vision tower is text-to-video only (no `keyframeEncoder`).
- Otherwise the exporter's `audio-decoder`, `video-encoder` and `vision` stages run with `--only-missing` under
  `--python` (the Forge environment): only the latent counts and canvases the source lacks are converted, on the CPU
  (torch.export and the Core AI converter; nothing is loaded or specialized on the GPU). Every file a stage writes is
  new; the exporter refuses to write a hard-linked one.

Then the manifest is rebuilt (`profiles_h3.manifest`) with the reused files' hashes taken from the source manifest
(same inode, so the same bytes) and new files hashed. Measured envelopes (`h3.envelopes`) an earlier run recorded are
kept only when nothing else in the manifest changed. Rerunning is safe: links and fresh files already there are kept.
Stops when free disk would fall below --minimum-free-gb (default 100).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import profiles_h3 as ph  # noqa: E402

REWRITTEN = (".aether/aether.json", "SHA256SUMS", "export-report.json")
STAGES = ("audio-decoder", "video-encoder", "vision")
GB = 1e9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / GB


def room(path: Path, needed_gb: float, floor_gb: float) -> None:
    free = free_gb(path)
    if free - needed_gb < floor_gb:
        raise SystemExit(f"stopping: {free:.1f} GB free, this needs ~{needed_gb:.1f} GB, floor {floor_gb} GB")


def write_fresh(path: Path, text: str) -> None:
    """Writes `path` as a new file (never through a hard link)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def mirror(source: Path, output: Path) -> dict[str, int]:
    """Links the source export's files into `output` (copies the rewritten ones); returns counts."""
    counts = {"linked": 0, "copied": 0, "kept": 0}
    pairs = [(source / "bundle", output / "bundle", REWRITTEN), (source / "work/converted", output / "work/converted", ())]
    for top, target, rewritten in pairs:
        if not top.exists():
            continue
        for p in sorted(q for q in top.rglob("*") if q.is_file() and not q.is_symlink()):
            rel = p.relative_to(top).as_posix()
            dest = target / rel
            if dest.exists():
                counts["kept"] += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if rel in rewritten:
                shutil.copyfile(p, dest)
                counts["copied"] += 1
            else:
                os.link(p, dest)
                counts["linked"] += 1
    build = output / "work/build.json"
    if not build.exists():
        build.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "work/build.json", build)
        counts["copied"] += 1
    return counts


def files_of(bundle: Path, source_bundle: Path, source_manifest: dict) -> tuple[dict[str, dict], int]:
    """The bundle's files for the manifest, hashes reused for files still linked to the source; and how many were
    hashed."""
    known = {f["path"]: f for f in source_manifest["assets"]["main"]["files"]}
    files, hashed = {}, 0
    for p in sorted(q for q in bundle.rglob("*") if q.is_file()):
        rel = p.relative_to(bundle).as_posix()
        if rel in REWRITTEN or rel.endswith(".tmp"):
            continue
        stat, src = p.stat(), source_bundle / rel
        if rel in known and src.exists() and os.path.samestat(stat, src.stat()) and known[rel]["bytes"] == stat.st_size:
            files[rel] = {"bytes": stat.st_size, "sha256": known[rel]["sha256"]}
        else:
            files[rel] = {"bytes": stat.st_size, "sha256": sha256_file(p)}
            hashed += 1
    return files, hashed


def refresh(source: Path, output: Path, note: str) -> dict:
    """Rebuilds the output bundle's manifest, SHA256SUMS and export report as fresh files; returns the manifest."""
    bundle, source_bundle = output / "bundle", source / "bundle"
    source_manifest = json.loads((source_bundle / ".aether/aether.json").read_text())
    meta = json.loads((output / "work/build.json").read_text())
    files, hashed = files_of(bundle, source_bundle, source_manifest)
    document = ph.manifest(meta, files)
    target = bundle / ".aether/aether.json"
    if target.exists():
        current = json.loads(target.read_text())
        envelopes = current.get("h3", {}).pop("envelopes", None)
        if envelopes is not None:
            if current == document:
                document["h3"]["envelopes"] = envelopes
            else:
                print(json.dumps({"warning": f"dropped {len(envelopes)} measured envelope(s): the manifest changed, so they "
                                             "no longer describe this export; measure it again"}), flush=True)
    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    write_fresh(target, text)
    report = json.loads((source_bundle / "export-report.json").read_text()) if (source_bundle / "export-report.json").exists() else {}
    report["extended"] = {"from": sha256_file(source_bundle / ".aether/aether.json"), "note": note,
                          "profiles": [p["key"] for p in document["h3"]["profiles"]], "hashedFiles": hashed}
    report["bundleBytes"] = sum(m["bytes"] for m in files.values())
    write_fresh(bundle / "export-report.json", json.dumps(report, indent=1, sort_keys=True))
    sums = [f"{m['sha256']}  {p}" for p, m in sorted(files.items())]
    sums += [f"{sha256_file(bundle / p)}  {p}" for p in (".aether/aether.json", "export-report.json")]
    write_fresh(bundle / "SHA256SUMS", "\n".join(sums) + "\n")
    with open(output / "manifest-refresh.log", "a") as log:
        log.write(json.dumps({"t": time.time(), "change": note, "newSHA256": sha256_file(target),
                              "profiles": report["extended"]["profiles"]}) + "\n")
    return document


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="the export to extend (bundle/ and work/); never written")
    parser.add_argument("--output", type=Path, required=True, help="the new export folder (created; rerunning resumes)")
    parser.add_argument("--manifest-only", action="store_true", help="make no graph: name only the profiles the graphs allow")
    parser.add_argument("--python", type=Path, help="the Forge environment's python (required without --manifest-only)")
    parser.add_argument("--snapshot", type=Path, help="passed to the exporter (its default otherwise)")
    parser.add_argument("--minimum-free-gb", type=float, default=100.0)
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    if not (source / "bundle/.aether/aether.json").exists() or not (source / "work/build.json").exists():
        raise SystemExit(f"{source} is not an H3 export (bundle/.aether/aether.json and work/build.json)")
    if output == source or source in output.parents or output in source.parents:
        raise SystemExit("--output must be a new folder outside the source export")
    output.mkdir(parents=True, exist_ok=True)
    if os.stat(output).st_dev != os.stat(source).st_dev:
        raise SystemExit("--output must be on the source's volume (its files are hard-linked)")
    if not args.manifest_only and not args.python:
        raise SystemExit("--python (the Forge environment) is required unless --manifest-only")
    room(output, 0.1 if args.manifest_only else 1.5, args.minimum_free_gb)
    print(json.dumps({"mirror": mirror(source, output)}), flush=True)
    note = "extend_h3.py --manifest-only" if args.manifest_only else "extend_h3.py: " + ", ".join(STAGES) + " --only-missing"
    if not args.manifest_only:
        exporter = Path(__file__).resolve().parent / "export_h3.py"
        for stage in STAGES:
            room(output, 1.0, args.minimum_free_gb)
            command = [str(args.python), "-I", str(exporter), "--output", str(output), "--stage", stage, "--only-missing",
                       "--minimum-free-gb", str(args.minimum_free_gb)]
            if args.snapshot:
                command += ["--snapshot", str(args.snapshot)]
            print(json.dumps({"run": command}), flush=True)
            subprocess.run(command, check=True)
    document = refresh(source, output, note)
    profiles = document["h3"]["profiles"]
    print(json.dumps({"profiles": [{"key": p["key"], "fl2va": "keyframeEncoder" in p, "recipes": p["recipes"]} for p in profiles],
                      "freeGB": round(free_gb(output), 1)}), flush=True)


if __name__ == "__main__":
    main()
