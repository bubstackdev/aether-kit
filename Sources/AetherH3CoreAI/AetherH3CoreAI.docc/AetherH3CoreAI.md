# ``AetherH3CoreAI``

BF16 native sessions for MiniMax H3 GPU qualification.

## Overview

> Important: Experimental and qualification only. There is no conditioner, decoder or media output, and whole-run lifecycle qualification is pending.

Sessions bind verified BF16 shared Metal buffers for block, chain, outer-phase and linked eight-step denoising probes, with one-call admission, cancellation and awaited unload.

## Topics

### Sessions

- ``H3DenoisingSession``
- ``H3ChainSession``
- ``H3PhaseSession``
- ``H3BlockSession``
