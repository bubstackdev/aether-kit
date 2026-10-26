# ``AetherKit``

Build generative AI pipelines with explicit tensor ownership and model-specific execution contracts.

## Overview

AetherKit 0.0.1 is an experimental developer preview with no API compatibility promise. This shared module provides owned float32, fp16 and int32 tensors, typed errors, decoded images and language model registration, and has no Core AI dependency.

Add `AetherAssets` for verified local asset installation, `AetherCoreAI` for local Core AI model loading and the reference executor, and `AetherKreaCoreAI` with `AetherKrea` for Krea 2 Turbo text-to-image, reference images and refine. `AetherLanguageCoreAI` and the H3 products are experimental. Qualification covers one Apple M5 Max running macOS 27.0.1.

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

### Errors

- ``AetherError``
