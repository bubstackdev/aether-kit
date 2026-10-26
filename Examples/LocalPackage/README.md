# Separate consumer package

From this directory, run `swift run -c release` on macOS 27 with Swift 6.4.

This package depends on every AetherKit library product as a local dependency. Running it prepares the tower profile's schedule and eight noise values, and a FLUX.2 [klein] 512² schedule and rotary table. It also type-checks the README's Krea, installed-asset, language, FLUX.2 and reference-executor examples without calling them, so it needs no weights, Python or Core AI execution.
