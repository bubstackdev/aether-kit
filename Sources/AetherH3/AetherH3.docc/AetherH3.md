# ``AetherH3``

MiniMax H3 text-to-video-with-audio contracts and the video session API.

## Overview

> Important: Preview since 0.0.3. ``H3VideoSession`` is qualified on one profile and one Mac (Turbo8, M5 Max): graph, trajectory, memory, session lifecycle and media audits, and a subjective audio and video review. Measured admission refuses other profiles, Turbo4 is outside it, and the API may change in any 0.0.x release.

``H3VideoSession`` runs an ``H3VideoRequest`` (prompt, geometry, seed, optional first/last ``H3Keyframe`` and audio) through injected stages and returns an ``H3VideoResult``. Native stages come from `AetherH3CoreAI`'s `H3ProfileEngine` over a validated schema-3 ``H3Profile``. ``H3ResourceAdmission`` reserves host-qualified stage estimates.

## Topics

### Video generation

- ``H3VideoSession``
- ``H3VideoRequest``
- ``H3VideoResult``
- ``H3Keyframe``

### Profiles and contracts

- ``H3Profile``
- ``H3Geometry``
- ``H3Schedule``
- ``H3ResourceAdmission``
