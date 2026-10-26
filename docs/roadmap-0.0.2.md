# 0.0.2 roadmap

Status: scope agreed, 2026-10-26. Prototype evidence from `aether-engine` lowers porting cost but is not AetherKit support; every item below needs AetherKit's own implementation, independent references and qualification. License notes come from secondary sources and must be confirmed against model cards before adoption.

## Scope

| Item | New capability | License (to confirm) | Device fit | Prototype evidence |
| --- | --- | --- | --- | --- |
| FLUX.2 [klein] 4B | Image editing and multi-reference image conditioning | Apache-2.0 | Mac and iPhone | Engine: built, verified on Mac and iPhone, published |
| Qwen3-VL-2B | Image input to the language runtime (vision encoder → decoder) | Apache-2.0 | Mac and iPhone | Engine: verified, published |
| Fixed-shape language decode export | Removes per-token input-shape growth and the 128-submission model reload | n/a | All | Kit finding (2026-10-26 audit) |
| Kokoro (82M) | Speech synthesis as a first-class modality | Apache-2.0 | Mac and iPhone | None; reportedly runs on iPhone via Core ML |
| **Complete H3** | Text-to-video-with-audio generation, replacing the qualification-only label | Apache-2.0 (pipeline code); confirm checkpoint/adapter terms | Mac | Kit: eight-step provided-conditioning gate passed 2026-10-18 |

Spikes, without a ship commitment: a hybrid recurrent-state language model (Granite 4.0-H / Zamba2: Mamba-2 state alongside KV), ACE-Step music generation, and TripoSR image-to-3D. These decide whether each enters 0.0.3.

## Complete H3

H3 currently qualifies a linked eight-step denoising trajectory from **provided** conditioning. Completing it means prompt in, video and audio out, through a public generation API. The 0.0.1 qualification inputs were deleted by macOS temporary-file cleanup (see the [release checklist](release-0.0.1-checklist.md)), so the work starts by rebuilding them durably.

### H0. Regenerate assets durably

- Re-run the recorded export chain from `lab/aether-prototype/forge`: canonical-QKV front, base bundle merge, Turbo8, spatial video decoder, weight audit, phase oracles, binding oracles.
- Write everything to `/Volumes/Scratch/aether/aether-prototype/.local/h3-qkv-<date>/` in separate subdirectories, never `/tmp`. Copy every manifest that evidence will cite into `docs/evidence`.
- Remove the inline merge steps' writes into the prototype's `docs/evidence`, which would otherwise overwrite historical records.
- Produce the native preparation cache that `h3_diagnose.py` requires. Today only the prototype's Rust `h3_generate` creates it, and its build environment is gone. Either rebuild that environment or add a CPU path that produces the cache.
- **Accept when:** all 1,100 body bindings match the `exported_sha256` values in the prototype's `h3-qkv-weight-audit.json`, and the CPU fixtures (oracle hidden, production temb/tags, prepare oracle) match their recorded hashes. Model and bundle hashes will change, because Core AI exports aren't byte-reproducible.

### H1. Re-pin and requalify the trajectory

- Update the hard-coded bundle and phase hashes in the `Scripts/prepare-h3-*-oracle.py` and `Scripts/qualify-h3-*.py` wrappers and in the probe plans.
- Redo the chain against the new assets: block, chain and backbone CPU references, the capture, the denoising CPU references, and the phase and denoising qualifications. Use the unchanged numerical limits (0.02 relative L2) and memory limits.
- Re-land `998f72f` (H3 verification speedup) and qualify it in the same pass.
- Decide modulation-table retention across runs while the lifecycle contract is being re-established anyway.
- **Accept when:** the eight-step trajectory passes with fresh capture hashes, all scoped CPU comparisons pass, and peak and settled memory stay within the limits.

### H2. Whole-run lifecycle

- Complete `qualify-h3-denoising --lifecycle`: submitted-body cancellation, retained recovery, concurrent-unload cancellation and reopened recovery across full runs. The 2026-10-19 attempt was stopped at user request.

### H3. Prompt conditioning

- Port the pinned conditioner presentation and the unnormalized 50-layer text tap, so prompts replace provided conditioning.
- **Accept when:** complete-tensor CPU comparisons for conditioner outputs pass, and the end-to-end trajectory from a tokenized prompt matches an independent reference within limits.

### H4. Decoders and stitching

- Port the video VAE with spatial tiling, the audio decoder, and temporal/spatial stitching.
- Qualify keyframe encoding for first/last-frame conditioning, which the 0.0.1 layout captures only with stub features.
- **Accept when:** decoded frames and audio match independent references numerically, and tiled versus untiled decoding agree within limits.

### H5. Media acceptance and public API

- Add an `H3GenerationSession` (prompt → video frames + audio) with the same admission, cancellation and terminal-unload contract as Krea. Keep the qualification sessions available to the probe.
- Define output containers and bounded delivery for frames and audio.
- Run a media audit (frame count, resolution, audio sample rate and duration, A/V sync) and a recorded subjective review. Publish memory and timing requirements at the qualified geometry (960×544, 124 frames).
- **Accept when:** the generation API passes native qualification, the media audit and review are recorded, and the README and DocC drop the "qualification only" label.

### Risks

- **Asset regeneration depends on a prototype build environment** that no longer exists (Rust `h3_generate`, `CARGO_HOME`).
- **Re-qualification is about a day of runs:** long CPU-reference jobs plus roughly an hour of GPU time.
- **New native outputs may not reproduce** the 0.0.1 capture hashes. Qualification relies on fresh captures plus CPU comparisons, not hash continuity.
- **Peak memory** for the full pipeline at the qualified geometry is unmeasured; the denoising stage alone peaked at 10.6 GiB.

## Sequencing

1. Start H0 and the language decode-export change early, since both unblock later work.
2. FLUX.2 klein and Qwen3-VL-2B can proceed in parallel; both have engine prototypes.
3. Kokoro is small and can slot in once the audio delivery contract exists, which H5 needs too.
4. H1 → H2 → H3/H4 → H5 are sequential.
