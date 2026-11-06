"""Records Diffusers' own H3 packed layout for a tiny geometry, the Swift host recipe's hermetic fixture
(Tests/AetherVideoTests/Fixtures/h3-layout.json): `build_packed_sequence` (positions, token tags, the video/audio/text
indices), `build_row_timesteps` for one step, and `MiniMaxH3RotaryPosEmbed` on the positions. No weights are read.

    Forge/.venv/bin/python -I Forge/h3/layout_fixture_h3.py > Tests/AetherVideoTests/Fixtures/h3-layout.json
"""
import json
import sys

sys.dont_write_bytecode = True
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG,
                                                                     MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG)

T, FRAMES, LH, LW, NA = 5, 7, 4, 6, 9          # text tokens, latent frames, latent height and width, audio latents
VIDEO_T, AUDIO_T = 0.0476190447807312, 0.1666666865348816
positions, tags, video, audio, text, cv, ca = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
    torch.full((T,), MINIMAX_H3_TEXT_TAG, dtype=torch.long), FRAMES, LH, LW, NA, (1, 2, 2), MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG, MINIMAX_H3_VIDEO_TAG)
unique, indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(video, audio, cv, ca, T, VIDEO_T, AUDIO_T, 0.0, 0.0)
cos, sin = MiniMaxH3RotaryPosEmbed()(positions.to(torch.float32))
json.dump({"textTokens": T, "latentFrames": FRAMES, "latentHeight": LH, "latentWidth": LW, "audioLatents": NA,
           "videoTimestep": VIDEO_T, "audioTimestep": AUDIO_T,
           "positions": positions.to(torch.float32).flatten().tolist(), "tags": tags.tolist(),
           "videoIndices": video.tolist(), "audioIndices": audio.tolist(), "textIndices": text.tolist(),
           "timesteps": [float(v) for v in unique.to(torch.float32)], "timestepIndices": indices.tolist(),
           "cos": cos.flatten().tolist(), "sin": sin.flatten().tolist()}, sys.stdout)
