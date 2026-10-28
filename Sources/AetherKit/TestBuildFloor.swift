/// Every test target that links AetherKit refers to this, so a test build without the OS 27 triple fails with
/// "'RunTestsWithTheOS27Triple' is only available in macOS 27 or newer". The package floor is macOS 14 / iOS 17, but the
/// tests exercise Core AI ungated: build them with `swift test --triple arm64-apple-macosx27.0` (or
/// `MACOSX_DEPLOYMENT_TARGET=27.0` under xcodebuild). See docs/deployment-floor.md.
@available(macOS 27, iOS 27, *)
package enum RunTestsWithTheOS27Triple {}
