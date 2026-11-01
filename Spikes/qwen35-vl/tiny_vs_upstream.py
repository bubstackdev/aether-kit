#!/usr/bin/env python3
"""CPU proof of the Qwen3.5 port on a random tiny model, against upstream (run in Forge/reference-qwen35):

    Forge/reference-qwen35/.venv/bin/python -I Spikes/qwen35-vl/tiny_vs_upstream.py

Upstream `Qwen3_5ForCausalLM` (random weights, float32) over a whole sequence vs the port (`delta_rule` loop, or chunk as argv[1]) fed in
padded fixed-width submissions (W = 8: the last one padded) and then one-token steps through the same W-wide padded
call. Prints the worst cosine and argmax agreement of every row, and checks padded == unpadded on the port.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Forge" / "language"))
from qwen35_hybrid import PREFIX, Qwen35HybridDecoder, name_map  # noqa: E402

from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

RULE = sys.argv[1] if len(sys.argv) > 1 else "loop"   # the port's delta rule: loop or chunk
torch.manual_seed(0)
cfg = dict(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=8, num_attention_heads=4,
           num_key_value_heads=2, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
           linear_value_head_dim=32, linear_conv_kernel_dim=4, full_attention_interval=4, rms_norm_eps=1e-6,
           tie_word_embeddings=True, attn_output_gate=True, attention_bias=False, hidden_act="silu",
           rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                            "mrope_section": [3, 3, 2], "mrope_interleaved": True})
cfg["layer_types"] = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(8)]
up = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**cfg)).eval()
with torch.no_grad():
    for n, p in up.named_parameters():
        p.normal_(0, 0.3 if "norm" in n else 0.08)
port = Qwen35HybridDecoder(cfg, image_token=511, max_capacity=64, delta_rule=RULE).eval()
sd = up.state_dict()
names = name_map(port)
params = dict(port.named_parameters())
with torch.no_grad():
    for key, t in sd.items():
        if key == "lm_head.weight":
            continue
        dst, transform = names[PREFIX + key.removeprefix("model.")]
        t = t.reshape(params[dst].shape) if key.endswith("conv1d.weight") else t
        params[dst].copy_(1.0 + t if transform == "gain" else t)

tokens = torch.randint(0, 500, (37,)).tolist()
with torch.no_grad():
    ref = up(torch.tensor([tokens])).logits[0].double().numpy()


def run(width, prompt):
    state = port.empty_state(48)
    rows, start = [], 0
    with torch.no_grad():
        while start < prompt:
            chunk = tokens[start:min(start + width, prompt)]
            rows.append(port.step(chunk, start, width, state).double().numpy())
            start += len(chunk)
        for i in range(prompt, len(tokens)):
            rows.append(port.step([tokens[i]], i, width, state).double().numpy())
    return rows, state


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


prompt = 21
rows, state = run(8, prompt)
want = [ref[min(prompt, s + 8) - 1] for s in range(0, prompt, 8)] + [ref[i] for i in range(prompt, len(tokens))]
cs = [cos(a, b) for a, b in zip(rows, want)]
am = sum(int(np.argmax(a)) == int(np.argmax(b)) for a, b in zip(rows, want))
print(f"port (W=8, padded) vs upstream: min cosine {min(cs):.8f}, argmax {am}/{len(rows)}")
rows1, state1 = run(1, prompt)
print("padded vs unpadded recurrent state max abs:", float((state[3] - state1[3]).abs().max()),
      "conv:", float((state[2] - state1[2]).abs().max()))
assert min(cs) > 0.99999 and am == len(rows), "port disagrees with upstream"
