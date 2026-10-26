# Separate consumer package

From this directory, run `swift run -c release` on macOS 27 with Swift 6.4.

This package depends on every AetherKit library product as a local dependency. Running it prepares the tower profile's schedule and eight noise values and prints the H3 Turbo8 step count. It also type-checks the README's Krea, language and reference-executor examples without calling them, so it needs no weights, Python or Core AI execution.
