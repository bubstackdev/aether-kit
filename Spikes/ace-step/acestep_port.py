"""ACE-Step 1.5 text-to-music conditioning and the turbo sampler, ported so the spike needn't install the ACE-Step package
(its pipeline pulls gradio, torchaudio, torchao, mlx, lightning and more).

Sources:
  - ace-step/ACE-Step-1.5 @ ca1e85fe9430 (the pipeline):
      acestep/constants.py                          SFT_GEN_PROMPT, DEFAULT_DIT_INSTRUCTION
      core/generation/handler/prompt_utils.py       _format_instruction, _format_lyrics
      core/generation/handler/metadata_utils.py     the metas block
      core/generation/handler/conditioning_text.py  tokenization (caption ≤ 256 tokens, lyrics ≤ 2048)
      core/generation/handler/conditioning_embed.py caption hidden states from Qwen3-Embedding, lyric embeddings from
                                                    its token table, the silence reference when there's no timbre audio
      core/generation/handler/conditioning_masks.py text-to-music: chunk mask of ones, silence source latents
      core/generation/handler/conditioning_target.py silence-latent tiling
  - ACE-Step/Ace-Step1.5 @ 19671f4 (the weights' own modeling file): generate_audio's shift schedules and Euler ODE
    update. The GitHub pipeline has since added a wavelet correction (DCW) that the pinned model file doesn't have; the
    spike follows the pinned model.

Portions Copyright (c) 2026 ACEStep, used under the MIT License; see LICENSE-ACE-Step-1.5.
"""
from dataclasses import dataclass, field
from pathlib import Path

import torch

SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""
DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
INSTRUMENTAL_LYRICS = "[Instrumental]"
LATENT_RATE = 25            # latent frames per second (48 kHz audio, 1920× VAE upsampling)
REFERENCE_FRAMES = 750      # the silence reference stands in for timbre audio: silence_latent[:, :750]

# generate_audio's turbo schedules: 8 steps at shift 1, 2 or 3 (shift 3 is the default).
SHIFT_TIMESTEPS = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693, 0.6666666666666666, 0.5454545454545454,
          0.4, 0.2222222222222222],
    3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3],
}


def metas(duration_seconds: int, bpm="N/A", timesignature="N/A", keyscale="N/A") -> str:
    """The `# Metas` block (MetadataMixin._dict_to_meta_string)."""
    return (f"- bpm: {bpm}\n"
            f"- timesignature: {timesignature}\n"
            f"- keyscale: {keyscale}\n"
            f"- duration: {int(duration_seconds)} seconds\n")


def caption_prompt(caption: str, metas_block: str, instruction: str = DEFAULT_DIT_INSTRUCTION) -> str:
    if not instruction.endswith(":"):
        instruction += ":"
    return SFT_GEN_PROMPT.format(instruction, caption, metas_block)


def lyric_prompt(lyrics: str, language: str) -> str:
    return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"


def turbo_schedule(shift: float = 3.0) -> list[float]:
    return SHIFT_TIMESTEPS[shift]


def euler_step(xt, vt, t: float, t_next: float):
    """The ODE update between two schedule points."""
    return xt - vt * (t - t_next)


def final_x0(xt, vt, t: float):
    """The last step returns the clean estimate: x0 = xt − vt · t."""
    return xt - vt * t


def load_silence_latent(path: Path) -> torch.Tensor:
    """silence_latent.pt is stored [1, 64, T]; the pipeline transposes it to [1, T, 64]."""
    return torch.load(path, weights_only=True).transpose(1, 2)


def silence_slice(silence_latent: torch.Tensor, length: int) -> torch.Tensor:
    """[length, 64] of silence, tiled when the stored latent is shorter."""
    available = silence_latent.shape[1]
    if length <= available:
        return silence_latent[0, :length, :]
    repeats = (length + available - 1) // available
    return silence_latent[0].repeat(repeats, 1)[:length]


@dataclass
class Conditions:
    """The inputs AceStepConditionGenerationModel.generate_audio / prepare_condition take, for one text-to-music request."""
    text_hidden_states: torch.Tensor          # [1, Lt, 1024] Qwen3-Embedding last hidden state of the caption prompt
    text_attention_mask: torch.Tensor         # [1, Lt] bool
    lyric_hidden_states: torch.Tensor         # [1, Ll, 1024] Qwen3-Embedding token embeddings of the lyric prompt
    lyric_attention_mask: torch.Tensor        # [1, Ll] bool
    refer_audio_acoustic_hidden_states_packed: torch.Tensor   # [1, 750, 64] silence (no timbre audio)
    refer_audio_order_mask: torch.Tensor      # [1] long
    src_latents: torch.Tensor                 # [1, T, 64] silence
    chunk_masks: torch.Tensor                 # [1, T, 64] ones: generate everything
    is_covers: torch.Tensor                   # [1] bool, False
    silence_latent: torch.Tensor              # [1, Ts, 64]
    attention_mask: torch.Tensor              # [1, T] ones
    text_ids: torch.Tensor = field(default=None, repr=False)    # the token IDs, kept for the encoder exports
    lyric_ids: torch.Tensor = field(default=None, repr=False)

    def as_kwargs(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("text_ids", "lyric_ids")}


@torch.no_grad()
def build_conditions(tokenizer, text_encoder, silence_latent: torch.Tensor, caption: str, lyrics: str, language: str,
                     metas_block: str, frames: int, dtype=torch.float32) -> Conditions:
    """One request's conditions, as the pipeline builds them for text2music without reference audio or an LM."""
    text = tokenizer(caption_prompt(caption, metas_block), padding="longest", truncation=True, max_length=256,
                     return_tensors="pt")
    lyric = tokenizer(lyric_prompt(lyrics, language), padding="longest", truncation=True, max_length=2048,
                      return_tensors="pt")
    text_hidden = text_encoder(input_ids=text.input_ids).last_hidden_state
    lyric_hidden = text_encoder.embed_tokens(lyric.input_ids)
    silence = silence_latent.to(dtype)
    source = silence_slice(silence, frames)[None]
    return Conditions(
        text_hidden_states=text_hidden.to(dtype), text_attention_mask=text.attention_mask.bool(),
        lyric_hidden_states=lyric_hidden.to(dtype), lyric_attention_mask=lyric.attention_mask.bool(),
        refer_audio_acoustic_hidden_states_packed=silence[:, :REFERENCE_FRAMES, :],
        refer_audio_order_mask=torch.tensor([0], dtype=torch.long),
        src_latents=source, chunk_masks=torch.ones(1, frames, source.shape[-1], dtype=dtype),
        is_covers=torch.tensor([False]), silence_latent=silence, attention_mask=torch.ones(1, frames, dtype=dtype),
        text_ids=text.input_ids, lyric_ids=lyric.input_ids)


def pad_encoder(hidden: torch.Tensor, mask: torch.Tensor, length: int):
    """Pad the packed condition sequence to a fixed length for a static-shape DiT export; padding is masked out."""
    n = hidden.shape[1]
    if n > length:
        raise ValueError(f"condition sequence {n} exceeds the exported length {length}")
    padded = torch.zeros(hidden.shape[0], length, hidden.shape[2], dtype=hidden.dtype)
    padded[:, :n] = hidden
    padded_mask = torch.zeros(mask.shape[0], length, dtype=mask.dtype)
    padded_mask[:, :n] = mask
    return padded, padded_mask
