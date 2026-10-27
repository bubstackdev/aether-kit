"""Runs the unmodified upstream Kokoro KModel with seeded SineGen noise.

Only the istftnet module's view of ``torch`` is wrapped: ``randn_like`` for the
[1,L,9] harmonic noise returns AetherSpeech's SplitMix64/Box-Muller sequence, and
the unused [1,L,1] noise-branch draw returns zeros. ``torch.rand`` (initial phase)
is left upstream; it cannot affect the output (see kokoro_graphs docstring), and
repeated runs check that.
"""
from __future__ import annotations

import torch

from kokoro_graphs import HARMONICS, load_upstream, splitmix_normal


class _SeededTorch:
    def __init__(self, seed: int):
        self._seed = seed
        self.harmonic_draws = 0

    def __getattr__(self, name):
        return getattr(torch, name)

    def randn_like(self, t, *args, **kwargs):
        if t.shape[-1] == HARMONICS:
            self.harmonic_draws += 1
            values = splitmix_normal(self._seed, t.numel())
            return torch.from_numpy(values).reshape(t.shape).to(t.dtype)
        return torch.zeros_like(t)


def run_upstream(kmodel, upstream_root, input_ids: list[int], ref_s: torch.Tensor, speed: float, seed: int):
    """Returns (audio [N], pred_dur [T]) from KModel.forward_with_tokens."""
    _, istftnet, _ = load_upstream(upstream_root)
    original = istftnet.torch
    seeded = _SeededTorch(seed)
    istftnet.torch = seeded
    try:
        with torch.no_grad():
            audio, pred_dur = kmodel.forward_with_tokens(torch.LongTensor([input_ids]), ref_s, speed)
    finally:
        istftnet.torch = original
    if seeded.harmonic_draws != 1:
        raise RuntimeError(f"expected one harmonic noise draw, saw {seeded.harmonic_draws}")
    return audio.reshape(-1), pred_dur.reshape(-1)
