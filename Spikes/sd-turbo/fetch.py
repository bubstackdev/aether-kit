"""Fetch only pinned, checksum-verified source files. No remote code execution."""

import argparse, hashlib, json
from pathlib import Path
from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
args = p.parse_args()
files = json.loads(Path(__file__).with_name("checkpoint-sha256.json").read_text())
snapshot_download(
    "stabilityai/sd-turbo",
    revision="b261bac6fd2cf515557d5d0707481eafa0485ec2",
    allow_patterns=list(files),
    local_dir=args.output,
)
for name, expected in files.items():
    with (args.output / name).open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise SystemExit(f"Checksum mismatch: {name}")
print("All pinned SD-Turbo files verified")
