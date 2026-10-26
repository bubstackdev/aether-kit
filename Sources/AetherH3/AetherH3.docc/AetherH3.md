# ``AetherH3``

MiniMax H3 text-to-video-with-audio contracts and the video session API.

## Overview

> Important: Experimental. End-to-end media, lifecycle and memory qualification of ``H3VideoSession`` is pending, and the API may change in any 0.0.x release.

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
