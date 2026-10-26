# ``AetherLanguageCoreAI``

Run a registered local dense language model on Core AI.

## Overview

> Important: Experimental. The API and export contract may change in any 0.0.x release.

``LanguageGenerationSession`` loads a registered bundle, prefills and greedily decodes into a Kit-owned, growable fp16 KV cache, and streams tokens while the next step runs. Each session admits one request at a time; ``LanguageGenerationSession/unload()`` waits for active work and releases native resources.

The session detects the asset's ``LanguageStepContract`` from its function signature. The 0.0.1 ``LanguageStepContract/runningPositions`` contract submits a new input shape every decode step, so the session recycles the native model every 128 submissions. The ``LanguageStepContract/bucketedPositions`` contract keeps input shapes in a fixed, bounded set (power-of-two prefill chunks over caches grown in 256-row steps) and writes the returned cache rows itself, so it needs no recycling; a bucketed step costs somewhat more than a running one. Bucketed assets declare their `LanguageExportBounds`.

## Topics

### Sessions

- ``LanguageGenerationSession``
- ``LanguageStepContract``
- ``LanguageGenerationError``
