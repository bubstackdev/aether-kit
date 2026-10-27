// swift-tools-version: 6.4
// Throwaway spike harness: not part of the AetherKit package.
import PackageDescription

let package = Package(
    name: "HybridDecodeHarness",
    platforms: [.macOS("27.0")],
    targets: [
        .executableTarget(name: "hybrid-decode", path: "Sources",
                          swiftSettings: [.swiftLanguageMode(.v5), .enableExperimentalFeature("Lifetimes")],
                          linkerSettings: [.linkedFramework("CoreAI"), .linkedFramework("Metal")]),
    ]
)
