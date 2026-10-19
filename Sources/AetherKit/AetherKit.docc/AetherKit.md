# ``AetherKit``

Build generative AI pipelines with explicit tensor ownership and model-specific execution contracts.

## Overview

AetherKit is in early development. The shared module provides owned float32, fp16 and int32 tensors and typed errors. Add `AetherCoreAI` for the experimental local-model reference executor, or `AetherKrea` for Krea asset and conditioning contracts. Tokenization and native text encoding are separately selectable products.

The reference executor supports stateless float32, fp16 and int32 functions and copies data across the native boundary. AetherKreaCoreAI implements the reference Krea image pipeline, with scoped desktop native parity, cancellation/recovery and lifecycle qualification. Native functions are reused for the model session lifetime. Its prepared-prompt workflow encodes once, releases the encoder and reuses owned conditioning and generation models across images. Prepared values are caller-owned and checked against the bundle identity. Optimized device-resident execution and other generative modalities remain future work.

## Topics

### Tensor ownership

- ``FloatTensor``
- ``Float16Tensor``
- ``Int32Tensor``
- ``TensorValue``

### Image artifacts

- ``GeneratedImage``

### Errors

- ``AetherError``
