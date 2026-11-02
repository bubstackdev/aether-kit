"""Stateless, float32 latent diffusion contracts; upstream modules retain their architecture.

New U-Net families can reuse these boundaries without inheriting FLUX token packing,
rotary tables or flow-matching arithmetic. Fixed geometry is deliberate for this spike.
"""

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, return_dict=False)[0].float()


class EpsilonDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample, timestep, text_embeds):
        return self.model(
            sample, timestep, encoder_hidden_states=text_embeds, return_dict=False
        )[0].float()


class LatentDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.post_quant_conv = model.post_quant_conv
        self.decoder = model.decoder
        self.scale = float(model.config.scaling_factor)

    def forward(self, latents):
        return self.decoder(self.post_quant_conv(latents / self.scale))[0].float()
