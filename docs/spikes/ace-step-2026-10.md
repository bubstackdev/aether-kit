# Spike: ACE-Step music generation on Core AI (2026-10)

**Verdict: conditional go for 0.0.3. Target ACE-Step 1.5, not v1, and ship text-to-music on Mac first, without the optional 5 Hz LM.**

- **Both expensive stages export cleanly** on the first try: the 1.58B DiT and the 48 kHz audio VAE decoder.
- **They are fast on the M5 Max.**
  - The DiT takes 50–68 ms per step for 60 s of audio and 361 ms per step for 4 minutes. The turbo model uses 8 steps with no CFG, so the diffusion part of a 4-minute song is about 2.9 s.
  - The VAE decodes 10 s of stereo 48 kHz audio in 67 ms.
- **The condition:** fp16 numerics need qualification at trajectory level with real prompts. The DiT step differs from fp32 by 5–6% relative L2, though eager bf16 (the authors' own dtype) is much further off on the same synthetic inputs.

This is a feasibility spike, not product work. The code under `Spikes/ace-step/` is throwaway and is not wired into `Package.swift`.

## Licensing (confirmed from model cards)

| Repository | Licence | Notes |
|---|---|---|
| `ACE-Step/ACE-Step-v1-3.5B@82cd0d7` | **Apache-2.0** | Card front matter `license: apache-2.0`; body "License: Apache 2.0". Ungated. |
| `ACE-Step/Ace-Step1.5@19671f4` | **MIT** | Card front matter `license: mit`; body "License: [MIT]". Ungated. |
| `Qwen/Qwen3-Embedding-0.6B` | Apache-2.0 | 1.5's text encoder, bundled inside the 1.5 repo. |
| `ACE-Step/acestep-5Hz-lm-0.6B`, `acestep-v15-base` | MIT | |

The 1.5 card also states the training data is licensed, royalty-free or synthetic, and that outputs may be used commercially. Nothing gated was accepted.

## Architecture walkthrough

### ACE-Step v1 (3.5B, Apache-2.0)

- Text encoder: UMT5-base, 1.1 GB.
- Lyric encoder.
- Diffusion transformer: Sana-style linear attention, 24 layers, 2560-d, 20 heads. It works on a DCAE latent: `AutoencoderDC` f8c8, a 2-D autoencoder over 128-bin mel spectrograms, with EfficientViT blocks in the deepest stage.
- Vocoder: an ADaMoS HiFi-GAN at 44.1 kHz, hop 512, mel→waveform.
- Sampling: 27–60 steps with CFG.
- Four networks plus a mel/vocoder path. Linear attention and the multi-scale ReLU attention of the EfficientViT blocks would be new export territory, and the step count is 3–7× higher than 1.5's.

### ACE-Step 1.5 (MIT), the recommended target

| Stage | Network | Size | Notes |
|---|---|---|---|
| Planner (optional) | 5 Hz LM, a Qwen3 decoder | 0.6B / 1.7B / 4B | Writes song metadata and lyrics by chain of thought, plus 5 Hz audio codes. This is a plain Qwen3 decoder that the dense language runtime already handles. |
| Text encoder | Qwen3-Embedding-0.6B | 0.6B | Caption → 1024-d hidden states. |
| Condition encoder | Text projector; 8-layer bidirectional lyric encoder (Qwen3 blocks, sliding window 128); 4-layer timbre encoder over reference-audio latents | 0.6B | Lyric embeddings and timbre are packed with text into one cross-attention sequence. |
| Audio tokenizer/detokenizer | Attention pooler + ResidualFSQ (levels 8,8,8,5,5,5) | 2 × 105M | Only for cover/LM-hint paths. For text-to-music the result is discarded (`where(is_covers, …)`), so it can be constant-folded away. |
| **DiT** | 24 layers, 2048-d, 16 query and 8 KV heads with QK-norm and RoPE. Alternating sliding-window (128) / full *bidirectional* self-attention; cross-attention; adaLN scale/shift from t and (t − r) | **1.58B** | Input is [noisy latent 64 ‖ source latent 64 ‖ chunk mask 64] at 25 Hz, patchified ×2 to 12.5 tokens/s. Turbo: 8 fixed steps (shift 1, 2 or 3), no CFG, ODE or SDE update. |
| **Audio VAE decoder** | `AutoencoderOobleck` (diffusers): Snake activations, weight-normed Conv1d, ConvTranspose1d upsampling ×(10, 6, 4, 4, 2) = ×1920 | 161 MB fp16 | 64-ch latents at 25 Hz → stereo 48 kHz. |

## Method

- Device: M5 Max, 64 GB, macOS 27.0.1 (26A434). Exporter: coreai-torch 0.4.2 and torch 2.11 from the Forge venv (read-only), plus a spike venv with `vector-quantize-pytorch` so the upstream modeling file imports.
- Runtime: `coreai.runtime` with OS Core AI and a GPU-preferred specialization. Each stage is compared with eager fp32 CPU references on fixed random inputs.
- **VAE decoder:**
  - Weight norm is folded with `remove_weight_norm`.
  - It is exported at static windows of 250 latent frames (10 s) and 1500 frames (60 s), in fp16, plus a 10 s fp32 build.
- **DiT:**
  - The upstream `modeling_acestep_v15_turbo.py` loads from the snapshot, with only `decoder.*` weights.
  - One velocity step, `xt, t, context, encoder → v`, is exported at static T = 1500 (60 s) and T = 6000 (4 min) latent frames, with L = 512 condition tokens.
  - With `use_cache=False` the encoder K/V are recomputed each step; the authors cache them across steps.
- Every GPU run went through `with-gpu-lock.sh`. Conversion was CPU-only: peak RSS 16.1 GB for the DiT export.

## Measurements (M5 Max GPU)

| Stage | Shape | Warm latency | vs eager fp32 |
|---|---|---|---|
| VAE decode fp16 | 10 s window → [1,2,480000] | **67 ms** (first call 2.5 s) | rel L2 1.6e-3, cosine 0.999999 |
| VAE decode fp32 | 10 s window | 195 ms | rel L2 3.7e-6 |
| VAE decode fp16 | 60 s window → [1,2,2,880,000] | 719 ms | rel L2 1.5e-3 |
| DiT step fp16 | 60 s (750 tokens) + 512 condition tokens | **50–68 ms** | rel L2 5.1%, cosine 0.9987 |
| DiT step fp16 | 4 min (3000 tokens) + 512 condition tokens | **361 ms** (first call 0.63 s) | rel L2 6.0%, cosine 0.9982 |

- Exports: DiT 2.9 GB per shape (fp16); VAE 161 MB.
- Load with a cached specialization: 9–11 s for the DiT, 0.5–0.8 s for the VAE.
- **Precision context for the DiT error.** On the same synthetic inputs, eager CPU bf16 (the authors' inference dtype) reaches **rel L2 29.9%**, and eager fp16 29.3% (`results/dtype-check.log`). Random inputs push this network into a regime that is very sensitive to precision, so 5–6% for Core AI fp16 is not evidence of a defect. Single-step error on random inputs is not a quality gate either: qualification must run the 8-step trajectory from real conditioning and compare decoded audio.
- **Memory.** The DiT process showed a max RSS of 15–18 GB, but its peak physical footprint was 0.9 GB. RSS counts mapped weights and specialization buffers. Real resident cost must be measured kit-side under the admission machinery. Expect at least the fp16 weights: about 3.2 GB for the DiT.

## End-to-end estimate (M5 Max, text-to-music without the LM)

| Component | 60 s clip | 4 min song |
|---|---|---|
| Text encoder + condition encoders (about 1.2B, one pass) | est. < 0.2 s | est. < 0.3 s |
| DiT, 8 steps | 0.4–0.55 s | 2.9 s |
| VAE decode (10 s windows with overlap, or 60 s windows) | 0.4–0.7 s | 1.6–2.9 s |
| **Total** | **≈ 1–1.5 s** | **≈ 5–6 s** |

- Adding the 5 Hz LM costs roughly 1,200 codes plus metadata tokens per 4 minutes: about 7–8 s for a 1.7B LM at the kit's measured int8 rate (~190 tok/s).
- Resident weights in fp16 come to about 5.8 GB without the LM: DiT 3.2 GB, text encoder 1.2 GB, condition encoder 1.2 GB, VAE 0.16 GB. The LM adds 1.2 GB (0.6B) to 3.4 GB (1.7B).
- That is a Mac target. iPhone would need int8 or quantized DiT and encoders, and a T3-style audio quality gate.

## Most uncertain stage

**Going in, the most uncertain stage was the VAE decoder:** long 1-D convolutions at 48 kHz, Snake (`sin²`) activations, weight norm, ×1920 transposed-conv upsampling, and multi-megasample tensors. It exported and ran first time at both 10 s and 60 s windows, in fp16 and fp32, with tight numerics.

**The uncertainty that remains:**

1. fp16 versus the model's bf16 training. Qualify on real trajectories.
2. Tiled VAE decoding for full songs: choose the overlap and crossfade. Each residual unit has a receptive field of a few latent frames.
3. The upstream pipeline code that builds lyric embeddings and packs the condition sequence. It lives in ACE-Step's GitHub repo, not in the model file, and was not ported here.
4. Text and lyric tokenization in Swift: Qwen3 BPE, which swift-transformers already covers.

## Effort estimate (0.0.3)

About 3–4 engineer-weeks for Mac text-to-music with qualification:

- Export recipes: DiT at bucketed lengths (for example 30/60/120/240 s), VAE window, Qwen3-Embedding, and the condition encoder with lyric and timbre paths and condition packing; also encoder-K/V reuse across steps. About 1 week.
- Swift pipeline: tokenization, condition assembly, turbo schedule and ODE/SDE update, tiled VAE decode with crossfade, and audio delivery (shared with Kokoro and H3's audio contract). About 1 week.
- Independent references and trajectory and audio qualification (latents, waveforms, a spectral metric), plus a subjective review. About 1 week.
- Optional: the 5 Hz LM planner through the language runtime, plus cover/repaint (FSQ tokenizer). About 1 week.

## Recommendation

**Conditional go for 0.0.3** with ACE-Step 1.5, text-to-music, Mac-first, no LM.

The **gate** is an 8-step trajectory qualification from real prompts against an fp32 reference, with decoded-audio metrics, before any API work. If fp16 fails that gate, fall back to fp32 for the DiT's sensitive parts (adaLN/modulation, norms), or to fp32 throughout. The fp32 VAE is 3× slower, but still about 20× faster than real time.

Leave v1 out: older, slower to sample, and with more exotic stages.

## Disk footprint

- Peak: about 10.6 GB.
  - Spike exports: DiT 2 × 2.9 GB, VAE 0.6 GB, references about 0.3 GB.
  - ACE-Step 1.5 DiT + VAE weights in the HF hub cache: 4.8 GB.
- After cleanup: **0 B** of weights and exports. Both were deleted once these results were committed. ACE-Step is a conditional go but not scheduled, and re-downloading the 5.1 GB costs only minutes.
- The model cards and configs were deleted too. So were this spike's Core AI specialization-cache entries under `~/Library/Caches/coreai-cache/26A434/python`: 5.9 GB for the two DiT shapes.

## Evidence

- Code: `Spikes/ace-step/`
  - `export_vae.py`, `export_dit.py`: export plus fp32 references;
  - `run_ace.py`: GPU timing and comparisons;
  - `dtype_check.py`.
- Results:
  - `Spikes/ace-step/results/gpu-run.json`: every stage;
  - `results/dtype-check.log`.
- GPU lock holds, 50 s total:
  - `spikes: triposr+ace-step stages`, 14:44:57–14:45:25, 28 s, shared with TripoSR;
  - `spikes: ace-step dit 4min`, 15:02:35–15:02:57, 22 s.
