# ``AetherH3CoreAI``

Native Core AI stages for MiniMax H3 and the sessions used to qualify them.

## Overview

> Important: Preview since 0.0.3. The profile stages are qualified on one profile and one Mac (Turbo8, M5 Max), with committed graph, trajectory, memory, session lifecycle and media-audit records. Measured admission refuses other profiles, and Turbo4 is outside it. The block, chain, phase and denoising sessions remain qualification tools, not product API.

``H3ProfileEngine`` builds the native prompt-conditioner, keyframe, denoiser (double-buffered weights) and decoder stages that `H3VideoSession` consumes. The block, chain, phase and denoising sessions are qualification probes with one-call admission, cancellation and awaited unload.

## Topics

### Profile stages

- ``H3ProfileEngine``

### Qualification sessions

- ``H3DenoisingSession``
- ``H3ChainSession``
- ``H3PhaseSession``
- ``H3BlockSession``
