# ``AetherKit``

Build generative AI pipelines with explicit tensor ownership and model-specific execution contracts.

## Overview

AetherKit 0.0.5 is an experimental developer preview. Its app-facing API (the language facade, model preparation and readiness, and this module's lifecycle policy and state types) is frozen for the rest of 0.0.x; everything else has no compatibility promise. This shared module provides owned float32, fp16 and int32 tensors, typed errors, decoded images with PNG encoding, PCM audio with WAV output, seeded Gaussian noise, language model registration, GPU coordination (a cross-process GPU lease compatible with Atelier and an opt-in in-process admission object for sessions), and the managed lifecycle policy for idle unload and memory pressure. It has no Core AI dependency.

Add `AetherAssets` for verified local asset installation, model preparation and readiness, `AetherCoreAI` for local Core AI model loading and the reference executor, and `AetherKreaCoreAI` with `AetherKrea` for Krea 2 Turbo text-to-image, reference images and refine. `AetherFlux2CoreAI` (FLUX.2 [klein]), `AetherSpeechCoreAI` (Kokoro speech) and `AetherLanguageCoreAI` with `AetherLanguageTokenizers` (including Qwen3-VL image input) are experimental; the H3 products are Preview since 0.0.3. `AetherDiffusion`, `AetherDiffusionCoreAI` and `AetherSDTurbo` (SD-Turbo text-to-image) are an experimental spike, new in 0.0.5. Qualification covers one Apple M5 Max running macOS 27.0.1 (and, on iOS, Kokoro speech on two iPhones and the Qwen3-0.6B on one).

## Topics

### Tensor ownership

- ``FloatTensor``
- ``Float16Tensor``
- ``Int32Tensor``
- ``TensorValue``

### Image artifacts

- ``GeneratedImage``

### Language model registration

- ``LanguageModelManifest``

### GPU coordination

Kit tools take the cross-process lease by default; app sessions opt into a host-shared admission object.

- ``SharedGPULease``
- ``SharedGPULeaseError``
- ``GPUAdmission``
- ``GPUAdmissionOwner``
- ``GPUAdmissionGrant``
- ``GPUAdmissionError``
- ``FairInferenceQueue``
- ``InferenceAdmissionError``

### Errors

- ``AetherError``
