# ``AetherLanguageCoreAI``

Run a registered local dense language model on Core AI.

## Overview

> Important: Experimental. The API and export contract may change in any 0.0.x release.

``LanguageGenerationSession`` loads a registered bundle, prefills and greedily decodes into a Kit-owned, growable fp16 KV cache, and streams tokens while the next step runs. Each session admits one request at a time; ``LanguageGenerationSession/unload()`` waits for active work and releases native resources. The current export contract recycles the native model every 128 submissions.

## Topics

### Sessions

- ``LanguageGenerationSession``
- ``LanguageGenerationError``
