"""SD-Turbo (stabilityai/sd-turbo) as `latent-diffusion@1` export graphs, plus the host recipe the manifest pins.

The three graphs are thin, stateless float32 wrappers around the pinned upstream modules (no re-implemented
architecture): CLIP text encoder, epsilon-prediction U-Net, VAE decoder (which owns the latent scaling factor).
Geometry is fixed per export (512x512 by default); the graphs' static shapes are the supported size.

The host recipe (tokenizer, seeded noise, schedule tables, Euler epsilon update) is what the Swift driver replays:
- tokens: CLIPTokenizer ids padded to 77 with the pad id ("!", 0) and truncated (BOS + 75 + EOS);
- noise: Kit's seeded SplitMix64 + Box-Muller stream (`seeded_noise`), read in NCHW order, times `initNoiseSigma`;
- schedule: Diffusers' EulerDiscreteScheduler tables for each supported step count, captured verbatim;
- step: `scale_model_input` (x / sqrt(sigma^2 + 1)) and `step` (epsilon, gamma 0), float32, in Diffusers' order.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

REPO = "stabilityai/sd-turbo"
REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
DEFAULT_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--stabilityai--sd-turbo/snapshots" / REVISION
# Every checkpoint file the export reads, at REVISION (the 0.0.x spike pinned the same hashes).
CHECKPOINT_SHA256 = {
    "LICENSE.md": "d6f6b1a4dce5c852bd6d7d9482d002baf0ccdb71e662250b73be9eec8764ee8d",
    "model_index.json": "679fe7c2a03a419a046f4330bb2e32570d96e7a6da51b7df147d4a87bc7f775c",
    "scheduler/scheduler_config.json": "5599066c725fc2747a201fa3165ad7c8c3de3be629ce325451752e8f94c0d1ae",
    "text_encoder/config.json": "9f6935d3d41bf8ce49b1d7ed912b29e770649d1aa3f3972215284f9397cd97d2",
    "text_encoder/model.fp16.safetensors": "bc1827c465450322616f06dea41596eac7d493f4e95904dcb51f0fc745c4e13f",
    "tokenizer/merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    "tokenizer/special_tokens_map.json": "c2d0fb8b86ad86b1f46134d4a5f93fd1e688c932a78efc8d149087c33a53ad06",
    "tokenizer/tokenizer_config.json": "516314ccab66918d7cb647480ac27abd3520d14e1e5825aac99f50c851c16071",
    "tokenizer/vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
    "unet/config.json": "ca09d494527ebe13343006e22b4468d45bf12aa34e76cf03639b6dbba1fb5c6d",
    "unet/diffusion_pytorch_model.fp16.safetensors": "40ec400881e27d1376c7c95c5bd495f407b33756e80eb6365e301c33a07af6e5",
    "vae/config.json": "699a58ef7ecf3bffa201a507a3eaa3eed7efabab457ac547bf82df798b52a9be",
    "vae/diffusion_pytorch_model.fp16.safetensors": "3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a",
}
TOKENIZER_FILES = ("merges.txt", "special_tokens_map.json", "tokenizer_config.json", "vocab.json")

TEXT_LENGTH = 77
TEXT_WIDTH = 1024
VOCABULARY = 49408
LATENT_CHANNELS = 4
PIXELS_PER_LATENT = 8          # the VAE's downsampling; sizes must also be multiples of 64 for the U-Net
SIZE = 512
STEPS = (1, 2, 3, 4)
DEFAULT_STEPS = 1
TRAIN_TIMESTEPS = 1000
NOTICE_LINE = "This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved"


class TextEncoder(nn.Module):
    """input_ids int32 [1, 77] -> text_embeds float32 [1, 77, 1024] (CLIP's last hidden state). The graph's inputs and
    outputs are float32 whatever the module's compute dtype: an fp16 export casts at its boundary."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, return_dict=False)[0].float()


class EpsilonDenoiser(nn.Module):
    """sample [1, 4, h, w], timestep [1], text_embeds [1, 77, 1024] -> epsilon [1, 4, h, w], all float32."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample, timestep, text_embeds):
        dtype = self.model.dtype
        return self.model(sample.to(dtype), timestep, encoder_hidden_states=text_embeds.to(dtype),
                          return_dict=False)[0].float()


class LatentDecoder(nn.Module):
    """latents [1, 4, h, w] (unscaled, as the scheduler leaves them) -> image float32 [3, H, W] on [-1, 1]."""

    def __init__(self, model):
        super().__init__()
        self.post_quant_conv = model.post_quant_conv
        self.decoder = model.decoder
        self.scale = float(model.config.scaling_factor)

    def forward(self, latents):
        latents = (latents / self.scale).to(self.post_quant_conv.weight.dtype)
        return self.decoder(self.post_quant_conv(latents))[0].float()


def load_text_encoder(root: Path, dtype: torch.dtype = torch.float32):
    from transformers import CLIPTextModel
    return CLIPTextModel.from_pretrained(root / "text_encoder", variant="fp16", torch_dtype=dtype,
                                         local_files_only=True, attn_implementation="eager").eval()


def load_unet(root: Path, dtype: torch.dtype = torch.float32):
    from diffusers import UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor2_0
    unet = UNet2DConditionModel.from_pretrained(root / "unet", variant="fp16", torch_dtype=dtype,
                                                local_files_only=True).eval()
    unet.set_attn_processor(AttnProcessor2_0())
    return unet


def load_vae(root: Path, dtype: torch.dtype = torch.float32):
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(root / "vae", variant="fp16", torch_dtype=dtype,
                                         local_files_only=True).eval()


def scheduler(root: Path):
    from diffusers import EulerDiscreteScheduler
    s = EulerDiscreteScheduler.from_pretrained(root / "scheduler", local_files_only=True)
    c = s.config
    if (c.prediction_type, c.timestep_spacing, c.interpolation_type, c.use_karras_sigmas) != (
            "epsilon", "trailing", "linear", False):
        raise ValueError("unsupported SD-Turbo scheduler config")
    return s


def schedule_tables(root: Path) -> dict[str, dict]:
    """Diffusers' tables for every supported step count, as float32 values widened to Python floats (exact)."""
    tables = {}
    for steps in STEPS:
        s = scheduler(root)
        s.set_timesteps(steps)
        tables[str(steps)] = {"timesteps": s.timesteps.tolist(), "sigmas": s.sigmas.tolist(),
                              "initNoiseSigma": float(s.init_noise_sigma)}
    return tables


def tokenize(tokenizer, prompt: str) -> list[int]:
    return tokenizer(prompt, padding="max_length", max_length=TEXT_LENGTH, truncation=True).input_ids


def seeded_noise(seed: int, count: int):
    """Kit's SplitMix64 + double-precision Box-Muller stream (Swift `SeededGaussianNoise`), float32."""
    import numpy as np
    mask = (1 << 64) - 1
    state = seed & mask
    out = np.empty(count, dtype=np.float32)
    i = 0
    while i < count:
        values = []
        for _ in range(2):
            state = (state + 0x9E3779B97F4A7C15) & mask
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            values.append(z ^ (z >> 31))
        u1 = ((values[0] >> 11) + 1) / 9007199254740992.0
        u2 = (values[1] >> 11) / 9007199254740992.0
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        out[i] = radius * math.cos(angle)
        if i + 1 < count:
            out[i + 1] = radius * math.sin(angle)
        i += 2
    return out


def latent_shape(size: int) -> tuple[int, int, int, int]:
    side = size // PIXELS_PER_LATENT
    return (1, LATENT_CHANNELS, side, side)
