"""The device-chain gate's fixture (Tests/AetherRuntimeTests/Fixtures/chain/): one tiny H3-shaped block graph with its
weights lifted to runtime inputs, so a single graph serves every block (the 0.0.x `Lifted` pattern, lab/h3-spike).

A block is `x + mod[1] * SwiGLU(rmsnorm(x) * (1 + mod[0]))` in float16 with dynamic rows. The graph carries no
weights (it is a few KB): the Swift gate draws the weights and inputs from the seeded noise stream and compares the
chained (encoded, device-bound) run against the unchained one (`run()` per call through host memory).

    Forge/.venv/bin/python -I Forge/chain/export_chain_fixture.py --out Tests/AetherRuntimeTests/Fixtures/chain
"""
import argparse
import json
from pathlib import Path

import torch

DIM, FFN = 64, 128
ROWS_MIN, ROWS_MAX = 8, 1024


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(DIM, FFN, bias=False)
        self.gate = torch.nn.Linear(DIM, FFN, bias=False)
        self.down = torch.nn.Linear(FFN, DIM, bias=False)

    def forward(self, hidden, mod):
        h = hidden.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        h = (h * (1 + mod[0].float())).to(hidden.dtype)
        y = self.down(torch.nn.functional.silu(self.gate(h)) * self.up(h))
        return hidden + mod[1] * y


class Lifted(torch.nn.Module):
    """`module` with every parameter lifted to a positional input."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.names = [n for n, _ in module.named_parameters()]

    def forward(self, hidden, mod, *weights):
        return torch.func.functional_call(self.module, dict(zip(self.names, weights)), (hidden, mod))


def main() -> None:
    import coreai_torch as ct
    from torch.export import Dim

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    block = Block().to(torch.float16).eval()
    lifted = Lifted(block).eval()
    params = [p.detach() for _, p in block.named_parameters()]
    example = (torch.zeros(1, 64, DIM, dtype=torch.float16), torch.zeros(2, DIM, dtype=torch.float16), *params)
    rows = Dim("rows", min=ROWS_MIN, max=ROWS_MAX)
    dynamic = ({1: rows}, None, tuple([None] * len(params)))  # forward(hidden, mod, *weights)
    weights = [f"w{k}" for k in range(len(params))]
    with torch.no_grad():
        ep = torch.export.export(lifted, example, dynamic_shapes=dynamic, strict=False)
        ep = ep.run_decompositions(ct.get_decomp_table())
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=["hidden", "mod", *weights], output_names=["hidden_out"],
                                   state_names=[], entrypoint_name="block")
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(out / "lifted-block.aimodel")
    spec = {"dim": DIM, "ffn": FFN, "rows": [ROWS_MIN, ROWS_MAX], "function": "block",
            "inputs": ["hidden", "mod", *weights], "outputs": ["hidden_out"],
            "weights": [{"input": w, "param": n, "shape": list(p.shape)}
                        for w, n, p in zip(weights, lifted.names, params)]}
    (out / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps(spec))


if __name__ == "__main__":
    main()
