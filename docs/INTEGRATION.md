# Integrating AetherKit 0.0.x into Aether-Kit, and taking over `bubstackdev/aether-kit`

Status: **in progress**, 2026-11-05. The owner approved bringing in the validated work and taking over the remote.

`github.com/bubstackdev/aether-kit` holds an earlier, independent AetherKit: 0.0.1–0.0.7, 659 commits, an API freeze from
0.0.4, and a macOS 14 / iOS 17 floor. This kit is the from-scratch re-architecture (DESIGN.md, MANIFEST.md). Two rules
govern the integration:

1. **Our architecture wins.** Old code is *re-expressed* on Aether-Kit's contracts: the manifest and drivers, one
   runtime, actor ownership, no `@unchecked Sendable` or global state, and the 27-only floor. Nothing is copied in
   wholesale.
2. **Don't over-engineer validation.** A ported feature brings its old tests and fixtures as specs, plus **one** gate
   against the same reference it was validated against before. The old evidence stays where it is, in the `0.0.7`
   tag, as history. There's no qualification ceremony beyond that.

## What comes over, and how

| From 0.0.x | Decision | Where in Aether-Kit | Gate |
|---|---|---|---|
| `SharedGPULease`, `GPUAdmission`, `FairInferenceQueue` (Atelier wire-compatible) | **Port** | `AetherHost` | Old CPU tests plus the Atelier interop check |
| Lifecycle policy: idle unload, memory-pressure release, iOS easing rule | **Port** into `ModelHost` | `AetherHost` (§10) | Old CPU tests |
| Store gaps: reader leases, cross-process install lock, abandoned-stage recovery, free-space check, HF token scoped to huggingface.co, auth headers dropped on cross-host redirect, 416 restart | **Port** into `ModelStore` | `AetherCore` | Old tests as specs |
| `BackgroundTransport` (iOS background downloads, qualified on a device) | **Port** as a second `HTTPTransport` | `AetherCore` | Old tests; one device run later |
| `ModelReadiness` states and `ModelRequirements` | **Port the design** into `capability()`/`prepare()`, without `@unchecked Sendable` | `AetherHost` (M6) | Old tests |
| Sampler (penalty → constraint → temperature/top-k/top-p, no full sort), xoshiro256** RNG, token-set constraint + JSON-schema matcher | **Port** | `AetherText` Generator (M2/M3) | The Transformers-recorded sampler fixtures; the structured-output tests |
| `LanguageTextStream` (UTF-8 byte table, grapheme-safe deltas, stop-string hold-back, reasoning splitter) | **Port the behaviour** | `AetherText` Generator | Old facade CPU tests as a corpus |
| `LanguageTokenBytes`, the template trailing-newline fix, the 24-case tokenizer oracle | **Port** | `AetherText` TokenizerAdapter | The oracle cases |
| FoundationModels adapter (option/transcript/error mapping, guided generation) | **Port onto the Generator** | `AetherFoundationModels` (M3) | Old native facade checks, plus tools (new) |
| Qwen3-VL preprocessing (EXIF, alpha, grid snapping, bicubic, block-major patchify) | **Port** | `AetherVision` ImagePipeline (M4) | Old CPU tests |
| Generative foundation: output image and audio types, a manifest `schedule` block, per-graph `residency`, measured admission and memory bars, seeded Gaussian noise (SplitMix64 + Box–Muller, stream-compatible with 0.0.x) | **Built** (G0): typed `schedule`/`latent`/`residency` blocks; `FlowMatchSchedule`, `SeededGaussianNoise`, `GeneratedImage`/`GeneratedAudio` and `GraphResidency` in `AetherRuntime/Generative/`; `ResourceCoordinator` in AetherHost. Qualified-entry memory bars wait for H3's numbers (G4) | `AetherRuntime`/`AetherHost` + MANIFEST.md §5 | The old noise and schedule fixtures pass bit-exact. A gate shows phase measurement sees Core AI memory. |
| SD-Turbo / epsilon diffusion seam | **Ported** (G1): `latent-diffusion@1` in the new `AetherImage` product, `EpsilonEulerSchedule`, the CLIP tokenizer, and the Forge exporter `Forge/sdturbo/` (float32, fixed 512²). Local-only until the bundle is uploaded to `aether-models/aether-kit-sd-turbo` and pinned in the catalog. The iPhone run follows (SD-Turbo is 0.1.0's iOS-validated generative model) | `AetherImage` driver family | Its CPU-Diffusers gate: graph check (text, every U-Net step, VAE cosine 1.0 / RMSE < 1e-6) and the Swift gate from the seed (image RMSE 1.0e-5 / 3.3e-5) |
| FLUX.2 [klein] 4B | **Ported** (G2): `flux2-klein@1` (`Flux2KleinPipeline`) in `AetherImage` and the Forge exporter `Forge/flux2/` (0.0.x's int8 recipe). macOS only in 0.1.0. Local-only until the bundle is uploaded to `aether-models/aether-kit-flux2-klein-4b` and pinned. Reference images (the VAE encoder is exported and gated) are deferred | `AetherImage` | The exporter's GPU gate (31/31) and the Swift gate: prompt ids exact, text real-row cosine ≥ 0.999, image from the reference's rows SSIM ≥ 0.94 (0.991), whole pipeline recorded (0.908) |
| **Krea 2 Turbo**, vanilla and **realism** | **In scope.** First a **reproducible Forge exporter** from `krea/Krea-2-Turbo@98e0fe1` (diffusers; Krea 2 Community License). It emits two catalog entries: `krea-2-turbo`, and `krea-2-turbo-realism`, which has the realism adapter **merged into the transformer before export**, like the 0.0.x realism bundle. Then the port (G3). | `AetherImage` | Parity against CPU Diffusers on each new export. For realism, the CPU reference applies the same merged weights. The old schedule and noise fixtures also apply. The 0.0.x hash-exact gate qualified a different (prototype) bundle, so it doesn't carry over. |
| **MiniMax H3** video + audio | **In scope** (G4). Step 1 **built**: `GraphChain` (device-chained calls), `WeightBanks` + `WeightTable` (per-block weights as inputs, from bundle files), `ShapeProfile` (measured admission keyed by pinned shapes), `SerialGate` (queue, not fail), in `AetherRuntime/Chain/` (DESIGN.md §6.6; spike: `lab/h3-spike`). Step 2: the H3 exporter (weights-as-inputs graphs plus the weight table) and the driver, with its CPU oracle. | `AetherVideo` + `AetherRuntime` | Its per-graph oracle gate (0.02 rel L2) and its memory gate, run once on the qualified canvas |
| Kokoro-82M speech | **Port** (G5), with its Forge exporter | `AetherSpeech` | Its upstream-reference gate |
| The `AetherLanguage` facade API, `AetherLanguageMacros`, `LanguageBackend`, AetherCoreAI executor, `ResourceCoordinator`, tensors/`AetherError` | **Drop** (keep as reference in the tag) | — | — |
| CI shape: hosted `xcode-27`, an aggregate **CI passed** check, pinned actions, read-only token, frozen-API diff once an API is frozen | **Adopt** | `.github/workflows/ci.yml` | — |

## Order of work, and what gates 0.1.0

1. M2 (engine, Generator, ChatSession) merges; the foundation port merges.
2. Language ports onto the Generator: sampler, RNG, constraint, text stream, tokenizer fixes, then the FoundationModels adapter.
3. **Takeover PR** lands on the remote's `main` (see below).
4. G0 → G1 SD-Turbo → G2 FLUX.2 → G3 Krea (exporter first) → G4 H3 (runtime prerequisites first) → G5 Kokoro, each as a
   PR into the new `main`.
5. **0.1.0 is tagged only when Krea and H3 are ported and pass their gates**, so no release lacks them. Until then,
   0.0.x users stay on the `0.0.7` tag or `legacy/0.0.x`.

**The realism adapter** is `realism_engine_krea2_v3.1.safetensors` (sha256 `a6712629…b5b54a5`, 1.56 GB), supplied by
the owner. It's an ostris ai-toolkit **LyCORIS LoKr**: per module `lokr_w1` [4×4] and `lokr_w2` [1536×1536] with a scalar
`alpha`, in BF16, using ComfyUI key names. It covers all 28 transformer blocks (attention `wq/wk/wv/wo/gate`, MLP
`up/down/gate`) and the text-fusion `layerwise_blocks` and `refiner_blocks`. The exporter maps those keys onto the
diffusers transformer and adds `scale · kron(w1, w2)`, verifying the key mapping is total (no unmatched adapter key)
and recording the adapter hash in the bundle provenance. It isn't on Hugging Face, so it's a local Forge input
(`aether/forge-inputs/krea2-realism/`), and the **exported** realism bundle is what gets hosted and pinned. Its source is
**Civitai model 2688234 "Realism Engine Ideogram 4 + Krea 2"** (by `razzz`, version "Krea2 v3.0", matched by SHA-256).
Civitai's permissions allow sharing and selling merges, without credit, but `allowDifferentLicense: false` means
**a merge keeps those same permissions**. Civitai marks the model NSFW: it's trained for explicit adult content.

**Krea 2 Community License v.1 (2026-06-22)** covers both Krea variants and is the binding one:
- **Derivatives:** merges are permitted Derivatives.
- **Distribution:** ship the Agreement; the model name begins with "Krea"; a NOTICE carries Krea's attribution line
  and states the model was modified; nothing is presented as official.
- **Commercial use:** only while company-wide trailing-12-month revenue is under $1M; otherwise an enterprise
  license.
- **§4.2 content filtering:** **any deployment must implement content filters** against prohibited, harmful or
  unlawful output. The Acceptable Use Policy prohibits, among other things, CSAM or any sexual depiction of minors,
  and non-consensual intimate imagery, including deepfakes of real people.
- **Consequence for G3:** the Krea driver takes an explicit `ContentFilter` from the app. It has no silent default;
  the app chooses a classifier or moderation service and owns that choice. This applies to both variants, and it's a
  requirement for the realism one.

This is **the same adapter the 0.0.x prototype merged** (the owner confirmed Civitai "Krea2 v3.0", which ships this file).
The full record is in `forge-inputs/krea2-realism/SOURCE.json`.

`Forge/`, the Python export tooling with locked uv environments, comes over from 0.0.x as a top-level tooling
directory outside SwiftPM. Its exporters write **this kit's** `aether.json`. The 0.0.x prototype bundles also call
their manifest `aether.json` (`schema: 1, driver: "diffusion"`); this kit's loader rejects them as `unknownDriver`,
and no ported pipeline reuses them.

## Release decisions (2026-11-06)

- **Hosting:** one Hugging Face repo per exported bundle in the `aether-models` org, pinned by revision
  (`aether-models/aether-kit-<model>`, the 0.0.x naming). Krea repos begin with "Krea" (its license). A catalog
  entry is added only after its bundle is uploaded and pinned; until then gates install from the local export.
- **iOS in 0.1.0:** SD-Turbo is qualified on the iPhone. FLUX.2, Krea and H3 are macOS-only; on iOS, admission refuses
  them with a clear error before loading.
- **Krea `ContentFilter`:** a protocol the app implements, with no default; the driver can't be built without one.
  The kit can't enforce what a filter does: the license (§4.2) binds the deployer, and the API docs say so.
- **Realism:** its own **gated** repo, apart from vanilla Krea, carrying the Krea license, NOTICE, use policy and
  the adapter's Civitai permissions; its catalog entry is marked mature.

## Lessons carried into the design (measured in 0.0.x)

- **Core AI retains Metal memory on every input-shape change when `expectFrequentReshapes` is on,** until the model
  is released. Engines need a measured mitigation: a reload budget, or recycling every N submissions.
- **`phys_footprint` doesn't count Core AI's Metal allocations** (400 MiB of footprint against 8.57 GiB of Metal for
  a 4B model), and `os_proc_available_memory` misses mapped address space. Residency must account GPU memory too.
- **Non-contiguous or oversized input views cost 3–8× per step,** because Core AI copies them. Bind KV whole and
  contiguous.
- **Use the near-tie gate rule against upstream float32 references:** every row cosine ≥ 0.995, and an argmax
  difference is allowed only when the reference's top-1 minus top-2 probability is < 0.1 and our token is in its top
  five. Engine-internal invariants stay exact.
- **Dynamic-shape vision encoders computed wrong GPU outputs about 2.5 % of the time,** so use grid-static encoders.

## Takeover method

`main` on the remote is protected: no force-push, no deletion, and PRs need the **CI passed** check. Its owner can
merge a PR past failing checks, but nobody can force-push. So the takeover is:

1. **Preserve.** The `0.0.1`–`0.0.7` tags stay; consumers on `exact:`/`.upToNextMinor` keep resolving. Push a
   `legacy/0.0.x` branch at the old `main` (`a013c45`) for maintenance. Leave the existing branches untouched.
2. **One PR, `takeover/0.1.0` → `main`:** a merge commit whose parents are the old `main` and this kit's `main`
   (`--allow-unrelated-histories`), with **this kit's tree taken wholesale**. Both histories stay reachable, and the
   ruleset isn't touched.
3. **Version.** `0.1.0` marks the new architecture, a semver-allowed break from 0.0.x. Before tagging: README
   migration notes, a CHANGELOG `Breaking` section, and this document. `from: "0.0.7"` users would pick up `0.1.0`
   and break, so the README says so and points them to `exact: "0.0.7"` or `legacy/0.0.x`.
4. **Licensing:** Apache-2.0 (the same as 0.0.x). 0.0.x is this project's own earlier code, so ports from it need no
   attribution; NOTICE lists third-party work only (Diffusers, Transformers-recorded oracles, publishers' tokenizer
   files).
