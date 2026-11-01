// swift-tools-version: 6.4
// Qwen3.5 hybrid spike harness (0.0.4): not part of the AetherKit package.
import PackageDescription

let package = Package(
    name: "Qwen35StepHarness",
    platforms: [.macOS("27.0")],
    targets: [
        .executableTarget(name: "q35-step", path: "Sources",
                          swiftSettings: [.swiftLanguageMode(.v5), .enableExperimentalFeature("Lifetimes")],
                          linkerSettings: [.linkedFramework("CoreAI"), .linkedFramework("Metal")]),
    ]
)
