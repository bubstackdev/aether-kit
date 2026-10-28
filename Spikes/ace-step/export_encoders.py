"""Throwaway spike: export ACE-Step 1.5's text-to-music encoders to Core AI (the 0.0.2 report's open item 2).

usage: export_encoders.py <out-dir> [--text 256] [--lyric 512] [--dtype float16] [--only text,condition]

  text-encoder-<text>-<tag>.aimodel  `main`: input_ids [1, text] int32 → hidden [1, text, 1024]. Qwen3-Embedding-0.6B
      is causal, so a prompt right-padded to the fixed length gives the same first rows as the unpadded prompt.
  condition-<text>x<lyric>-<tag>.aimodel  `main`: text_hidden [1, text, 1024], lyric_embeds [1, lyric, 1024],
      lyric_mask [1, lyric] → text_projected [1, text, 2048], lyric_encoded [1, lyric, 2048]: the condition encoder's
      text projector and its 8-layer bidirectional lyric encoder.

Left on the host, so they need no export:
  - lyric token embeddings: a row lookup in Qwen3-Embedding's embedding table (saved here as lyric-embedding-table.npy);
  - the timbre embedding: with no reference audio the timbre encoder sees the fixed silence reference, so its single
    output row is a constant (saved as timbre-silence.npy, computed in fp32);
  - packing (pack_sequences): valid lyric rows, then the timbre row, then valid caption rows, padded to the DiT's L.
Heavy CPU: measured 5.0 GB (text) and 8.4 GB (condition) RSS; run one at a time.

The packing order follows pack_sequences and the condition encoder's forward in the pinned model file
(ACE-Step/Ace-Step1.5 @ 19671f4, modeling_acestep_v15_turbo.py), Copyright (c) 2026 ACEStep, used under the MIT
License; see LICENSE-ACE-Step-1.5.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
import acestep_port as port  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--text", type=int, default=256)
ap.add_argument("--lyric", type=int, default=512)
ap.add_argument("--dtype", default="float16")
ap.add_argument("--only", default="text,condition")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
snap = next((hub / "models--ACE-Step--Ace-Step1.5/snapshots").iterdir())
mdir = snap / "acestep-v15-turbo"
DT = torch.float16 if a.dtype == "float16" else torch.float32
tag = "fp16" if DT == torch.float16 else "fp32"
torch.set_grad_enabled(False)

import coreai_torch as ct  # noqa: E402


def convert(module, example, inputs, outputs, path):
    t0 = time.time()
    ep = torch.export.export(module, example).run_decompositions(ct.get_decomp_table())
    c = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    c.add_exported_program(ep, input_names=inputs, output_names=outputs)
    prog = c.to_coreai()
    prog.optimize()
    if path.exists():
        import shutil
        shutil.rmtree(path)
    prog.save_asset(path)
    print(path.name, "exported in %.1fs" % (time.time() - t0), flush=True)


if "text" in a.only:
    from transformers import AutoModel  # noqa: E402

    encoder = AutoModel.from_pretrained(snap / "Qwen3-Embedding-0.6B", torch_dtype=torch.float32).eval()
    np.save(out / "lyric-embedding-table.npy", encoder.embed_tokens.weight.to(DT).numpy())

    class TextEncoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids):
            return self.m(input_ids=input_ids).last_hidden_state

    ids = torch.full((1, a.text), 151643, dtype=torch.int32)     # <|endoftext|>, Qwen's pad, as filler
    convert(TextEncoder(encoder.to(DT)), (ids,), ["input_ids"], ["hidden"], out / f"text-encoder-{a.text}-{tag}.aimodel")
    del encoder

if "condition" in a.only:
    def load_module(name, path):
        s = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(s)
        sys.modules[name] = m
        s.loader.exec_module(m)
        return m

    cfg_mod = load_module("configuration_acestep_v15", mdir / "configuration_acestep_v15.py")
    modeling = load_module("modeling_acestep_v15_turbo", mdir / "modeling_acestep_v15_turbo.py")
    config = cfg_mod.AceStepConfig(**json.loads((mdir / "config.json").read_text()))
    config._attn_implementation = "sdpa"
    cond = modeling.AceStepConditionEncoder(config).eval()
    with safe_open(mdir / "model.safetensors", framework="pt") as f:
        state = {k.removeprefix("encoder."): f.get_tensor(k).float() for k in f.keys() if k.startswith("encoder.")}
    missing, unexpected = cond.load_state_dict(state, strict=False)
    print("condition encoder: missing", missing[:5], "unexpected", unexpected[:5], flush=True)
    silence = port.load_silence_latent(mdir / "silence_latent.pt").float()
    timbre, timbre_mask = cond.timbre_encoder(silence[:, :port.REFERENCE_FRAMES, :], torch.tensor([0]))
    np.save(out / "timbre-silence.npy", timbre.numpy())
    print("timbre rows", tuple(timbre.shape), "mask", timbre_mask.tolist(), flush=True)

    class ConditionParts(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c

        def forward(self, text_hidden, lyric_embeds, lyric_mask):
            text = self.c.text_projector(text_hidden)
            lyric = self.c.lyric_encoder(inputs_embeds=lyric_embeds, attention_mask=lyric_mask > 0.5).last_hidden_state
            return text, lyric

    example = (torch.zeros(1, a.text, 1024, dtype=DT), torch.zeros(1, a.lyric, 1024, dtype=DT),
               torch.ones(1, a.lyric, dtype=DT))
    convert(ConditionParts(cond.to(DT)), example, ["text_hidden", "lyric_embeds", "lyric_mask"],
            ["text_projected", "lyric_encoded"], out / f"condition-{a.text}x{a.lyric}-{tag}.aimodel")
