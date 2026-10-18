# ``AetherKit``

Build generative AI pipelines with explicit tensor ownership and model-specific execution contracts.

## Overview

AetherKit is in early development. The shared module provides owned float32, fp16 and int32 tensors and typed errors. Add `AetherCoreAI` for the experimental local-model reference executor, or `AetherKrea` for Krea asset and conditioning contracts. Tokenization and native text encoding are separately selectable products.

The reference executor supports stateless float32, fp16 and int32 functions and copies data across the native boundary. It is intended to establish correctness and lifecycle behavior before introducing device-resident generation paths. No complete text, image, or video generator is shipped yet.

## Topics

### Tensor ownership

- ``FloatTensor``
- ``Float16Tensor``
- ``Int32Tensor``
- ``TensorValue``

### Errors

- ``AetherError``
