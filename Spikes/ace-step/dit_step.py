"""One ACE-Step 1.5 turbo DiT velocity step with a fixed-length, padded condition sequence, for export.

The upstream DiT discards its `encoder_attention_mask` argument: its forward rebuilds the cross-attention mask from the
self-attention mask, which is all ones for text-to-music. Upstream never pads a single request's conditions, so it
doesn't matter there, but a static-shape export pads them to L, and unmasked padding rows take softmax weight (eager:
step-0 velocity cosine 0.660 against the unpadded step on the lo-fi prompt, diag_dit_mask.py). Step therefore hands every layer an additive
cross-attention mask built from `encoder_mask` (0 for condition rows, the dtype's most negative value for padding),
through a forward pre-hook on each layer, so a padded step equals the unpadded one.
"""
import torch

ENCODER_MASK_ARG = 10           # AceStepDiTLayer's positional encoder_attention_mask, as AceStepDiTModel.forward calls it


class Step(torch.nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.d = dit
        self._additive = None
        for layer in dit.layers:
            layer.register_forward_pre_hook(self._mask_hook)

    def _mask_hook(self, module, args):
        if self._additive is None:
            return None
        args = list(args)
        args[ENCODER_MASK_ARG] = self._additive
        return tuple(args)

    def forward(self, xt, t, context, encoder, encoder_mask):
        ones = torch.ones(xt.shape[:2], dtype=xt.dtype)
        keep = (encoder_mask > 0.5).to(xt.dtype)
        self._additive = ((1 - keep) * torch.finfo(xt.dtype).min)[:, None, None, :]      # [B, 1, 1, L]
        try:
            return self.d(hidden_states=xt, timestep=t, timestep_r=t, attention_mask=ones,
                          encoder_hidden_states=encoder, encoder_attention_mask=encoder_mask > 0.5,
                          context_latents=context, use_cache=False)[0]
        finally:
            self._additive = None
