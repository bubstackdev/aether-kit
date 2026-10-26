# ``AetherH3CoreAI``

Native Core AI stages for MiniMax H3 and the sessions used to qualify them.

## Overview

> Important: Experimental. Profile stages run natively, but committed graph, trajectory, decoder, lifecycle and media qualification records are pending.

``H3ProfileEngine`` builds the native prompt-conditioner, keyframe, denoiser (double-buffered weights) and decoder stages that `H3VideoSession` consumes. The block, chain, phase and denoising sessions are qualification probes with one-call admission, cancellation and awaited unload.

## Topics

### Profile stages

- ``H3ProfileEngine``

### Qualification sessions

- ``H3DenoisingSession``
- ``H3ChainSession``
- ``H3PhaseSession``
- ``H3BlockSession``
