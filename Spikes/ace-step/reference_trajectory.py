"""Eager CPU references for the ACE-Step 1.5 turbo trajectory gate: fp32 (the reference) and bf16 (the authors' own
inference dtype, the yardstick a fp16 Core AI run has to meet).

usage: reference_trajectory.py <out-dir> [--prompts prompts.json] [--enc 512] [--only id,id] [--skip-bf16]
For each prompt it builds the text-to-music conditions with acestep_port (Qwen3-Embedding caption states, lyric token
embeddings, silence reference and source), runs the model's condition encoder, then 8 turbo steps (shift 3, Euler ODE)
through the DiT with use_cache=False, as the exported velocity step does, once in fp32 and once in bf16 from the same
noise. Both final latents are decoded by the fp32 VAE (weight norm folded), so audio differences come from the DiT.
For the first prompt it also runs the pinned model's own generate_audio in fp32 and records how far it is from the
ported loop.

Writes per prompt <id>.npz:
  conditions  encoder [1, enc, 2048] (padded), encoder_mask [1, enc], context [1, T, 128], noise [1, T, 64]
  per dtype   <dtype>_xt [8, 1, T, 64] (the state entering each step), <dtype>_vt [8, 1, T, 64], <dtype>_x0 [1, T, 64]
  audio       <dtype>_audio [2, samples] at 48 kHz (fp32 VAE)
  encoders    text_ids, lyric_ids, text_hidden [1, Lt, 1024], lyric_hidden [1, Ll, 1024] (for the encoder exports)
plus <id>-<dtype>.wav and references.json. Heavy CPU: about 13–14 GB RSS (fp32 DiT and condition encoder, text
encoder, VAE) and several minutes per prompt; run one at a time.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
import acestep_port as port  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--prompts", default=str(Path(__file__).parent / "prompts.json"))
ap.add_argument("--enc", type=int, default=512, help="padded condition length (the DiT export's L)")
ap.add_argument("--only", default="")
ap.add_argument("--skip-bf16", action="store_true")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
spec = json.loads(Path(a.prompts).read_text())
frames = spec["duration_seconds"] * port.LATENT_RATE
hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
snap = next((hub / "models--ACE-Step--Ace-Step1.5/snapshots").iterdir())
mdir = snap / "acestep-v15-turbo"


def load_module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


torch.set_grad_enabled(False)
cfg_mod = load_module("configuration_acestep_v15", mdir / "configuration_acestep_v15.py")
modeling = load_module("modeling_acestep_v15_turbo", mdir / "modeling_acestep_v15_turbo.py")
config = cfg_mod.AceStepConfig(**json.loads((mdir / "config.json").read_text()))
config._attn_implementation = "sdpa"
model = modeling.AceStepConditionGenerationModel(config).eval()
with safe_open(mdir / "model.safetensors", framework="pt") as f:
    state = {k: f.get_tensor(k).float() for k in f.keys()}
missing, unexpected = model.load_state_dict(state, strict=False)
del state
print("model: missing", missing[:5], "unexpected", unexpected[:5], flush=True)
silence_latent = port.load_silence_latent(mdir / "silence_latent.pt").float()

from transformers import AutoModel, AutoTokenizer  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained(snap / "Qwen3-Embedding-0.6B")
text_encoder = AutoModel.from_pretrained(snap / "Qwen3-Embedding-0.6B", torch_dtype=torch.float32).eval()

from diffusers import AutoencoderOobleck  # noqa: E402

vae = AutoencoderOobleck.from_pretrained(snap / "vae", torch_dtype=torch.float32).eval()
for module in vae.modules():
    try:
        torch.nn.utils.remove_weight_norm(module)
    except ValueError:
        pass


def write_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").T.copy()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(pcm.tobytes())


def trajectory(encoder, encoder_mask, context, noise, attention_mask, dtype):
    decoder = model.decoder.to(dtype)
    xt = noise.to(dtype)
    schedule = port.turbo_schedule(spec["shift"])
    xs, vs = [], []
    for i, t in enumerate(schedule):
        tt = torch.full((1,), t, dtype=dtype)
        vt = decoder(hidden_states=xt, timestep=tt, timestep_r=tt, attention_mask=attention_mask.to(dtype),
                     encoder_hidden_states=encoder.to(dtype), encoder_attention_mask=encoder_mask,
                     context_latents=context.to(dtype), use_cache=False)[0]
        xs.append(xt.float().numpy())
        vs.append(vt.float().numpy())
        xt = port.final_x0(xt, vt, t) if i == len(schedule) - 1 else port.euler_step(xt, vt, t, schedule[i + 1])
    return np.stack(xs), np.stack(vs), xt.float()


report = {"snapshot": snap.name, "frames": frames, "shift": spec["shift"], "encoderLength": a.enc, "prompts": {}}
only = set(filter(None, a.only.split(",")))
for index, prompt in enumerate(spec["prompts"]):
    if only and prompt["id"] not in only:
        continue
    rep, saved = {}, {}
    t0 = time.time()
    cond = port.build_conditions(tokenizer, text_encoder, silence_latent, prompt["caption"], prompt["lyrics"],
                                 prompt["language"], port.metas(spec["duration_seconds"], bpm=prompt["bpm"]), frames)
    enc, enc_mask, context = model.prepare_condition(
        text_hidden_states=cond.text_hidden_states, text_attention_mask=cond.text_attention_mask,
        lyric_hidden_states=cond.lyric_hidden_states, lyric_attention_mask=cond.lyric_attention_mask,
        refer_audio_acoustic_hidden_states_packed=cond.refer_audio_acoustic_hidden_states_packed,
        refer_audio_order_mask=cond.refer_audio_order_mask, hidden_states=cond.src_latents,
        attention_mask=cond.attention_mask, silence_latent=cond.silence_latent, src_latents=cond.src_latents,
        chunk_masks=cond.chunk_masks, is_covers=cond.is_covers)
    rep["conditionTokens"] = int(enc_mask.sum())
    rep["conditionSeconds"] = time.time() - t0
    enc_padded, mask_padded = port.pad_encoder(enc, enc_mask, a.enc)
    noise = model.prepare_noise(context, prompt["seed"])
    saved.update(encoder=enc_padded.numpy(), encoder_mask=mask_padded.numpy(), context=context.numpy(),
                 noise=noise.numpy(), text_ids=cond.text_ids.numpy(), lyric_ids=cond.lyric_ids.numpy(),
                 text_hidden=cond.text_hidden_states.numpy(), lyric_hidden=cond.lyric_hidden_states.numpy())
    for dtype in [torch.float32] + ([] if a.skip_bf16 else [torch.bfloat16]):
        name = "fp32" if dtype == torch.float32 else "bf16"
        t0 = time.time()
        xs, vs, x0 = trajectory(enc, enc_mask, context, noise, cond.attention_mask, dtype)
        rep[f"{name}TrajectorySeconds"] = time.time() - t0
        audio = vae.decode(x0.transpose(1, 2)).sample[0].numpy()
        saved.update({f"{name}_xt": xs, f"{name}_vt": vs, f"{name}_x0": x0.numpy(), f"{name}_audio": audio})
        write_wav(out / f"{prompt['id']}-{name}.wav", audio)
        if dtype == torch.float32 and index == 0:
            model.decoder.float()
            upstream = model.generate_audio(**cond.as_kwargs(), seed=prompt["seed"], shift=spec["shift"])
            ref = upstream["target_latents"].float()
            rep["upstreamGenerateAudioRelL2"] = float((x0 - ref).norm() / ref.norm())
        print(prompt["id"], name, json.dumps({k: v for k, v in rep.items()}), flush=True)
    model.decoder.float()
    np.savez(out / f"{prompt['id']}.npz", **saved)
    report["prompts"][prompt["id"]] = rep
    (out / "references.json").write_text(json.dumps(report, indent=1))
