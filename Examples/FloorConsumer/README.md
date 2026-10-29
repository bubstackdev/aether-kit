# Floor consumer

A separate consumer package at AetherKit's deployment floor, macOS 14 and iOS 17. It links `AetherLanguage` and prints
the facade's availability: `unsupportedOS` below macOS 27 / iOS 27, and `bundleMissing` on 27 when no bundle path is
given (pass an installed bundle's directory as the first argument to check one).

From this directory, run `swift run -c release` with Swift 6.4. CI builds it on the `xcode-27` runner, checks that
`CoreAI` is weak-linked, and launches the binary on a GitHub-hosted `macos-15` runner, which has no CoreAI, where it
must print `unsupportedOS` ([deployment floor](../../docs/deployment-floor.md)).
