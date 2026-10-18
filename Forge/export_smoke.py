#!/usr/bin/env python3
"""Export a tiny, weight-free Core AI fixture; never required by SDK consumers.

Requires an environment containing torch and Apple's coreai-torch.
Writes only under the explicitly supplied output directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model_path = args.output / "affine.aimodel"
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")

    import torch
    import coreai_torch as ct

    class Affine(torch.nn.Module):
        def forward(self, x):
            return x * 2.0 + 1.0

    args.output.mkdir(parents=True)
    with torch.inference_mode():
        exported = torch.export.export(Affine().eval(), (torch.zeros(1, 4),))
        exported = exported.run_decompositions(ct.get_decomp_table())
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(
        exported, input_names=["x"], output_names=["y"],
        state_names=[], entrypoint_name="main",
    )
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(model_path)
    report = {
        "schema": 1,
        "recipe": "float32-affine-v1",
        "torch": torch.__version__,
        "coreai-torch": importlib.metadata.version("coreai-torch"),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_files": {
            str(path.relative_to(model_path)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(model_path.rglob("*")) if path.is_file()
        },
        "input": {"name": "x", "shape": [1, 4], "values": [-2, 0, 1, 3]},
        "expected": {"name": "y", "shape": [1, 4], "values": [-3, 1, 3, 7]},
    }
    (args.output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(model_path)


if __name__ == "__main__":
    main()
