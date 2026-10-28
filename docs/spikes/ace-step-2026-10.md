# Spike: ACE-Step music generation on Core AI (2026-10)

**Status: 0.0.3 go/no-go result (runs on 2026-10-28).** This is the ACE-Step decision for the [0.0.3 roadmap](../roadmap-0.0.3.md#spikes). The [0.0.3 decision](#003-decision-2026-10-28) answers what the 0.0.2 feasibility result left open; the [0.0.2 feasibility result](#002-feasibility-result-runs-on-2026-10-26) follows it, kept as it was apart from a correction note.

**Verdict: go** for ACE-Step 1.5 text-to-music on Mac, without the 5 Hz LM: an fp16 DiT and fp16 text and condition encoders, with the **VAE in fp32**, decoded in 10 s tiles. On four fixed 60 s prompts (two instrumental, two with lyrics), the Core AI fp16 trajectory is closer to the eager fp32 reference than the authors' own bf16 is, on every gate measure and every prompt, both with the reference's conditions and end to end through Core AI's own encoders. Two conditions come with it:
- **Kit must mask the padded condition sequence itself.** Upstream's DiT ignores its cross-attention mask, which is harmless for its unpadded single requests but not for a static-shape export padded to a fixed length. Unmasked, the trajectory is unrelated to upstream's (final latents relative L2 about 1.0), and the 0.0.2 step check couldn't have caught it.
- **The fp16 VAE isn't good enough.** Its waveform is 58–61 dB from the fp32 VAE's, but it lays a noise floor around −73 to −77 dBFS over bins the music leaves near-silent, and that fails the gate's log-spectral bars against bf16 on every prompt. The fp32 VAE in 10 s tiles costs about 1.3 s per minute of audio, and the tiler alone peaks at a 2.1 GB footprint.

Every figure is from a single run, and the timings come from a lightly loaded shared host (one-minute load 1.1–5.5, `host-load.log`). Export sizes come from `du` and are binary (GiB, MiB); RSS and footprints come from `/usr/bin/time -l` and are decimal (GB).

## 0.0.3 decision (2026-10-28)

### Setup

- Same device and toolchain as the 0.0.2 run below: M5 Max, 64 GB, macOS 27.0.1 (26A434); coreai-torch 0.4.2 and torch 2.11 from Forge's locked environment, plus the spike venv in `Spikes/ace-step/requirements-spike.txt`; OS Core AI with GPU-preferred specialization.
- `ACE-Step/Ace-Step1.5@19671f4`, only `acestep-v15-turbo/`, `vae/` and `Qwen3-Embedding-0.6B/` (no 5 Hz LM). The text-to-music conditioning is ported from the pipeline's source at `ace-step/ACE-Step-1.5@ca1e85fe9430` (MIT) into `acestep_port.py`, and `check_port.py` matches the port's prompt template, instruction, metas, caption and lyric prompts and turbo schedules against upstream's own functions.
- **Prompts** (`prompts.json`): four fixed 60 s requests with fixed seeds. Two are instrumental (a lo-fi beat; orchestral trailer music) and two have short original lyrics (indie pop; a folk ballad). Their packed condition sequences are 90–131 rows.
- **References** (`reference_trajectory.py`), eager on the CPU, from the same conditions and noise:
  - fp32, the reference, and bf16, the authors' inference dtype and the yardstick Core AI has to meet;
  - 8 turbo steps at shift 3, the DiT run with `use_cache=False` as the export runs it;
  - on the first prompt, the ported loop matches the pinned model's own `generate_audio` to relative L2 1.0e-5.
- **Exports**, all fp16 unless named:
  - the DiT velocity step at T = 1,500 frames (60 s) and L = 512 condition rows, through `dit_step.Step` (2.9 GiB);
  - the Qwen3-Embedding text encoder at 256 tokens (1.1 GiB);
  - the condition encoder's text projector and lyric encoder at 256 × 512 (777 MiB);
  - the VAE decoder at 250 frames (10 s) and 1,500 frames, each in fp16 (161 MiB) and fp32 (322 MiB).
  - The silence-timbre row is a constant (no reference audio for text-to-music), and the lyric embeddings are a table lookup, so both stay on the host, as does the packing.
- **Core AI runs**, under `with-gpu-lock.sh`:
  - `run_trajectory_coreai.py`: the trajectory from the reference's conditions, which judges the DiT alone;
  - `run_encoders_coreai.py`: the text and condition encoders, host packing, then the trajectory end to end, plus tiled decoding;
  - `decode_latents_coreai.py`: every trajectory's latents decoded by one VAE.
  - The final runs used one hold, 10:55:17–10:55:59. The VAE measurements used two more, 10:56:38–10:58:21, and the go-configuration measurements a fourth, 11:09:03–11:09:18. Host load was 1.1–2.4 with no heavy job alongside (`host-load.log`).

### The gate

The gate, from the 0.0.2 report: Core AI must be no further from the fp32 trajectory than bf16 is, on every prompt, in three measures (`gate_trajectory.py`):
- the final latents' relative L2;
- a multi-resolution STFT log-magnitude distance on the decoded audio;
- the log-mel cosine on the decoded audio.

Every trajectory's latents are decoded by the same Core AI fp32 VAE, which is 1.2–2.2e-6 from the eager fp32 VAE (`decoded.json`). So the audio measures judge the DiT, not the VAE.

**With the reference's conditions** (`gate-coreai.json`):

| Prompt | Final latents rel. L2, Core AI / bf16 | STFT distance, Core AI / bf16 | Log-mel cosine, Core AI / bf16 | SNR dB, Core AI / bf16 |
| --- | --- | --- | --- | --- |
| lofi-instrumental | **0.307** / 0.437 | **0.258** / 0.485 | **0.99547** / 0.98159 | −0.68 / −0.80 |
| orchestral-instrumental | **0.034** / 0.114 | **0.052** / 0.179 | **0.99983** / 0.99835 | 27.9 / 16.9 |
| indie-pop-vocal | **0.063** / 0.204 | **0.069** / 0.257 | **0.99983** / 0.99719 | 14.9 / 9.0 |
| folk-ballad-vocal | **0.108** / 0.243 | **0.141** / 0.306 | **0.99918** / 0.99606 | 14.1 / 8.0 |

**End to end, through Core AI's own text and condition encoders** (`gate-e2e.json`):
- final latents 0.018–0.310;
- STFT distance 0.038–0.267;
- log-mel cosine 0.99524–0.99994.

Again every prompt beats bf16 on every measure.

- The encoders are close: the text encoder's hidden states have cosine ≥ 0.9999981 (relative L2 ≤ 0.002) against eager fp32, and the packed condition sequence has cosine ≥ 0.9999982 (relative L2 ≤ 0.0019), with the same rows and mask (`encoders-run.json`).
- **The lo-fi prompt is a sensitive one.** Its trajectory amplifies small differences: bf16 ends 0.437 away and Core AI 0.307, at SNRs of −0.80 dB and −0.68 dB. So the two outputs are different takes of the same prompt, not near-copies of the fp32 audio. The other three prompts stay close.
- A 20 s excerpt of Core AI against fp32 for this prompt and for the indie-pop prompt is staged for a listening check.

### The padding mask: upstream ignores it, Kit must apply it

- **The behaviour upstream.** The DiT's forward discards its `encoder_attention_mask` argument and rebuilds the cross-attention mask from the self-attention mask, which is all ones for text-to-music. Upstream runs each request with its condition sequence unpadded, so this does no harm there, and the gate's reference stays that unpadded upstream step.
- **Why it matters for Kit.** A static-shape export needs a fixed L and pads the sequence to it, and the padding rows then take softmax weight in every cross-attention.
  - In eager, on the lo-fi prompt's first step, the padded step with its mask equals the padded step without one. Both are at cosine 0.660 (relative L2 0.795) against the unpadded step (`dit-mask-eager.json`).
  - An export of that graph followed it faithfully. The first Core AI run ended at final-latent relative L2 1.01 and 1.03 on the instrumentals, against bf16's 0.437 and 0.114 (`gate-unmasked-dit.json`).
- **The fix.** `dit_step.Step` hands every DiT layer an additive cross-attention mask built from `encoder_mask`, through a forward pre-hook: 0 for condition rows and the dtype's most negative value for padding.
  - Eager, the masked padded step is 1.7e-5 from the unpadded one.
  - The masked Core AI step 0 has cosine 0.982 against the unpadded fp32 step (`dit-mask-coreai.json`).
  - Kit's port must keep this mask, or export exact condition lengths.
- **Upstream's batched generation does pad.** At the pinned pipeline, caption and lyric tokens are padded to the batch's longest (`conditioning_text.py`, lines 154–160), and the diffusion handler documents `encoder_attention_mask` as unused (`diffusion.py`, line 57). In a batch of different-length prompts, the shorter ones attend padding upstream too. This was read in the code, not run. Single requests, which the gate's reference is, aren't padded.
- **The 0.0.2 step check couldn't have caught this:** it used random inputs with an all-ones mask. On those inputs the masked export still gives the 0.0.2 figure (relative L2 0.051, cosine 0.9987).

### A reference bug, and which references were redone

`reference_trajectory.py` cast `model.decoder` to bf16 in place for the bf16 trajectory and back with `.float()`. That left bf16-rounded weights, so every later "fp32" trajectory in the same process ran bf16 weights in fp32 arithmetic.
- **Affected:** the second prompt of each reference process, orchestral and folk. Their old fp32 final latents were 0.105 and 0.234 (relative L2) from the corrected ones.
- **Unaffected:** the first prompts, lo-fi and indie pop, and every bf16 trajectory, because re-casting rounded weights to bf16 changes nothing.
- **Redone:** fp32 for orchestral and folk, with the script fixed (the bf16 run now uses a copy). The merged references keep the original bf16 halves; their conditions, noise and token IDs are byte-identical between runs (`references-provenance.json`).

### VAE: fp32, in 10 s tiles

| VAE | Warm decode, 60 s | Peak footprint |
| --- | --- | --- |
| fp16, one 60 s window | 0.42 s | 4.5 GB |
| fp32, one 60 s window | 1.06 s | 9.6 GB |
| fp16, one 10 s window | 0.065 s | 0.94 GB |
| fp32, one 10 s window | 0.18 s | 1.8 GB |
| **fp32, 10 s tiles, 32-frame overlap (7 windows)** | **1.27–1.36 s** | **2.1 GB, the tiler alone** |

- **fp16's error** (`vae/noise-floor.json`): on the fp32 latents, the fp16 VAE's waveform is 58.0–60.6 dB SNR from the fp32 VAE's, with the residual at −72.7 to −76.5 dBFS RMS. But it exceeds the fp32 signal in 25–41% of time-frequency bins: the high, near-silent bins that a log-spectral measure weighs like loud ones.
- **Against the gate**, the Core AI trajectory decoded by the fp16 VAE (`asRunAudio` in the gate records) fails against bf16 on every prompt. For example, the STFT distance is 0.556 against bf16's 0.179 on orchestral and 0.893 against 0.306 on folk.
- Whether that floor is audible at normal levels wasn't checked. The fp32 VAE removes the question for about 0.9 s more per minute of audio.
- **Tiling** (open item 4): 10 s windows with a 32-frame (1.28 s) overlap and a linear crossfade are 53.9–64.0 dB from one fp32 60 s window, with STFT distance ≤ 0.0013 (`vae/tiled-fp32.json`).
  - A 16-frame overlap gives 49.7–53.5 dB; 64 frames give 59.9–64.0 dB for one more window.
  - Tiles bound the VAE's memory, and let a song of any length decode in pieces. The tiler alone, without the one-window comparison decode, took 1.29–1.36 s per 60 s at a 2.08 GB footprint, about a fifth of the one-window decode's (`vae/tiled-fp32-alone.json`, `time-excerpts.log`). The tile timings in the table range over both runs.

### Time and memory (60 s of audio, warm)

**The go configuration in one process** (`go_pipeline_coreai.py`: fp16 text and condition encoders, the masked fp16 DiT, the fp32 VAE in 10 s tiles, nothing else loaded; `go-pipeline.json`):
- **Per prompt, wall clock from token IDs to audio:** 1.80–2.07 s. That's the text encoder 10–52 ms, the condition encoder 12–22 ms, the DiT's 8 steps 0.42–0.51 s and the tiled VAE 1.31–1.52 s. Host load was 3.4–5.5 during this hold.
- **It computes what the gate judged:** every prompt's final latents equal the earlier end-to-end run's exactly, and its tiled audio is 53.9–63.9 dB from that run's one-window fp32 decode.
- **Process peak footprint 2.68 GB, maximum RSS 8.39 GB** (`time-excerpts.log`). The footprint is below the four assets' 5.1 GiB on disk (`data-footprint.log`) because the assets' weights are file-backed pages, which RSS counts and the footprint doesn't. Kit's admission measurement is port work.

**Per stage, from the gate's runs** (load 1.1–2.4):

| Stage | Time | Source |
| --- | --- | --- |
| Text encoder (256 tokens) | 20–41 ms | `encoders-run.json` |
| Condition encoder | 18–19 ms | `encoders-run.json` |
| DiT, 8 steps | 0.39–0.44 s: 49 ms per step, except folk's first step at 95 ms, the process's first call | `coreai-run.json` |
| VAE, fp32 tiles | 1.27–1.30 s | `vae/tiled-fp32.json` |

- **Loads**, with cached specialization: the DiT 0.48–2.0 s, the text encoder 0.29–0.90 s, the condition encoder 0.16–0.50 s, a VAE 0.07–0.31 s.
- **Earlier process footprints, a different configuration:** 6.3 GB for the process holding every asset of the gate's end-to-end run (two fp16 VAEs, one of them the 60 s window), and 4.8 GB for the DiT-plus-fp16-VAE trajectory process.
- Kit-side resident measurement, encoder K/V reuse across steps (open item 3) and longer DiT shapes remain port work.

### Costs on the CPU side (for whoever reruns this)

- **Exports:**
  - the DiT export peaked at 16.1 GB RSS;
  - the condition encoder at 8.4 GB, the text encoder at 5.0 GB;
  - the 60 s fp16 VAE export at 28.3 GB footprint, because it also writes an eager fp32 reference decode of the full window.
- **References:** the reference script peaked at about 40 GB.
  - An eager fp32 trajectory takes 12–24 s, but a bf16 one takes 21–25 min on the CPU (1,268–1,514 s), which is what made the bf16 yardstick slow.
  - These ran as announced heavy jobs, one at a time.

### What the 0.0.2 open items came to

1. **The gate:** passed, with fp16 for the DiT and the encoders, and fp32 for the VAE.
2. **The front of the pipeline:** the text and condition encoders export and run at cosine ≥ 0.9999981 against eager fp32. The conditioning port matches upstream, and the pipeline code is MIT.
3. **Encoder K/V reuse:** not measured. At about 50 ms per step it doesn't decide the go.
4. **Full-song decoding:** 10 s fp32 tiles with a 32-frame overlap and a linear crossfade.
5. **Resident memory:** the go configuration's process peaked at a 2.68 GB footprint and 8.39 GB RSS. Kit's own measurement is port work.

### Evidence (0.0.3)

- **Code** (`Spikes/ace-step/`):
  - conditioning: `acestep_port.py` and `check_port.py`;
  - references: `reference_trajectory.py` and `merge_references.py`;
  - exports: `export_dit.py` with `dit_step.py`, `export_encoders.py`, `export_vae.py`;
  - Core AI runs: `run_trajectory_coreai.py`, `gpu_run_trajectory.sh`, `run_encoders_coreai.py`, `decode_latents_coreai.py`, `tiled_vae_coreai.py`, `time_vae.py`, `go_pipeline_coreai.py`;
  - judging: `gate_trajectory.py`, `vae_noise_floor.py`;
  - the mask diagnosis: `diag_dit_mask.py`.
- **Results** (`Spikes/ace-step/results/0.0.3/`):
  - `gate-coreai.json`, `gate-e2e.json` and `gate-unmasked-dit.json`;
  - `coreai-run.json`, `encoders-run.json`, `decoded.json` and `go-pipeline.json`;
  - `dit-mask-eager.json` and `dit-mask-coreai.json`;
  - `references-instrumentals.json`, `references-vocal.json`, `references-fp32-redo.json` and `references-provenance.json`;
  - `vae/` (timings, tiling with and without the comparison decode, the noise floor);
  - `time-excerpts.log` (`/usr/bin/time -l` for every job);
  - `gpu-lock.log`, `host-load.log` and `data-footprint.log`.
- **Not committed:** weights, exports, latents and audio stay in the spike data directory, 17 GiB with the earlier unmasked DiT and runs (`data-footprint.log`). That includes the listening pairs.

### Reproducing (0.0.3)

With `HF_HUB_CACHE` pointing at a cache holding the pinned snapshot, and the spike venv:
1. `check_port.py <pipeline source> <snapshot>/acestep-v15-turbo`.
2. `reference_trajectory.py <references>`. It is heavy (about 40 GB, and 21–25 min per bf16 trajectory); `--only` and `--skip-bf16` split it.
3. `export_dit.py`, `export_encoders.py --only text`, then `--only condition`, and `export_vae.py --frames 250|1500 [--dtype float32]`, one at a time. The DiT export is about 16 GB.
4. `merge_references.py <references-all> --first <dir> <id> … --redo <bf16 dir> <fp32 dir> <id> …`. It assembles the four references the gate reads (here, lofi and indie first; orchestral and folk redone) and writes `provenance.json`.
5. Under the GPU lock, reading `<references-all>`: `run_trajectory_coreai.py`, `run_encoders_coreai.py` and `decode_latents_coreai.py`; then `time_vae.py`, `tiled_vae_coreai.py` (also `--no-full`) and `go_pipeline_coreai.py`.
6. `gate_trajectory.py <references-all> <coreai> gate.json coreai|e2e`, which records each input file's SHA-256, then `vae_noise_floor.py`.

## 0.0.2 feasibility result (runs on 2026-10-26)

The rest of this report is the 0.0.2 result as it was imported for 0.0.3. **Correction (0.0.3):** its DiT step check ran fixed random inputs with an all-ones condition mask, so it couldn't show that the export ignores the mask. On real, padded conditions the unmasked export is unrelated to upstream (see [the padding mask](#the-padding-mask-upstream-ignores-it-kit-must-apply-it)). Its 5.1% step error on those random inputs still reproduces with the masked export (relative L2 0.051, cosine 0.9987).

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
  - `run_ace.py`: GPU timing and comparisons, and `gpu_run.sh`, which runs it per stage under `/usr/bin/time -l`;
  - `dtype_check.py`.
- Results: `Spikes/ace-step/results/`
  - `gpu-run.json`: every stage;
  - `dit-run2.log`: the 4-minute DiT run, including the peak physical footprint;
  - `dtype-check.log`.
- GPU lock holds, 50 s total:
  - `spikes: triposr+ace-step stages`, 14:44:57–14:45:25, 28 s, shared with TripoSR: the three VAE stages in one process, then `dit-1500x512-fp16`;
  - `spikes: ace-step dit 4min`, 15:02:35–15:02:57, 22 s: `dit-6000x512-fp16`.

## Reproducing

- **Weights.** The scripts read the newest snapshot of `ACE-Step/Ace-Step1.5` from `~/.cache/huggingface/hub/` (the spike used revision `19671f4`; nothing is downloaded by the scripts). Exports and references go to the directory given on the command line; `gpu_run.sh` expects `~/.cache/huggingface/aether-kit/spikes/ace-step/` (`SPIKE_ASSETS` overrides the root).
- **Python.** Forge's locked environment has the torch 2.11, coreai-torch 0.4.2 and diffusers (`AutoencoderOobleck`) that the scripts use, so `export_vae.py` and `run_ace.py` run in it. The DiT's upstream modeling file also imports `vector-quantize-pytorch`, which is not in Forge's lock: `export_dit.py` and `dtype_check.py` need it added in a separate venv, as the spike did. `gpu_run.sh` takes that interpreter from `SPIKE_PYTHON`.
- **Memory.** The DiT export is CPU-only but peaked at 16.1 GB, which counts as a heavy CPU job on a shared machine.

## Open for 0.0.3

What these findings leave unanswered before a go/no-go:

1. **The gate itself.** No 8-step trajectory from real conditioning was run. fp16 was checked only as single steps on random inputs, so whether the fp16 DiT passes, or needs fp32 in its modulation and norms, is still open.
2. **The front of the pipeline.** Neither the Qwen3-Embedding text encoder nor the condition encoder (lyric and timbre encoders, condition packing) was exported, and the upstream code that builds lyric embeddings was not ported. Their timing (under 0.3 s) is an estimate. The license of that GitHub pipeline code wasn't checked; only the model repositories were.
3. **Encoder K/V reuse.** The DiT was exported with `use_cache=False`, so each step recomputes the cross-attention K/V. The authors' cached form, and its effect on the 361 ms step, is unmeasured.
4. **Full-song decoding.** The overlap and crossfade for tiled VAE decoding weren't chosen or tested; only single 10 s and 60 s windows ran.
5. **Resident memory.** The DiT process showed 15–18 GB maximum RSS but a 0.9 GB peak footprint. The real resident cost under Kit's admission is unknown, beyond the expected 5.8 GB of fp16 weights.
