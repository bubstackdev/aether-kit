// swift-tools-version: 6.4
// Hybrid-state 0.0.3 spike harness: not part of the AetherKit package.
import PackageDescription

let package = Package(
    name: "HybridStepHarness",
    platforms: [.macOS("27.0")],
    targets: [
        .executableTarget(name: "hybrid-step", path: "Sources",
                          swiftSettings: [.swiftLanguageMode(.v5), .enableExperimentalFeature("Lifetimes")],
                          linkerSettings: [.linkedFramework("CoreAI"), .linkedFramework("Metal")]),
    ]
)
