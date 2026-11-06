// swift-tools-version: 6.2
import PackageDescription

// Module layout: docs/DESIGN.md §3. Only AetherCore and AetherRuntime carry real code in M1; the task modules,
// AetherHost and the umbrella exist so the product graph (and its dependency rules) is fixed from the start.

/// Settings every target shares. Strict concurrency comes from Swift 6 language mode (the tools-6.2 default).
let common: [SwiftSetting] = [
    // Non-Sendable types (Graph, TensorArena, Bindings) are driven by their owning actor; nonisolated async
    // methods must run on the caller's executor rather than hop to the global one.
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
    // `import X` is internal unless spelled `public import`: Core AI types cannot leak out of AetherRuntime.
    .enableUpcomingFeature("InternalImportsByDefault"),
    .enableUpcomingFeature("MemberImportVisibility"),
    .enableUpcomingFeature("ExistentialAny"),
]

let package = Package(
    name: "Aether",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "AetherText", targets: ["AetherText"]),
        .library(name: "AetherFoundationModels", targets: ["AetherFoundationModels"]),
        .library(name: "AetherVision", targets: ["AetherVision"]),
        .library(name: "AetherEmbeddings", targets: ["AetherEmbeddings"]),
        .library(name: "AetherDecide", targets: ["AetherDecide"]),
        .library(name: "Aether", targets: ["Aether"]),
    ],
    targets: [
        // Identity, manifests and bytes on disk. No Core AI import (§3).
        .target(name: "AetherCore", swiftSettings: common),
        // The only module that touches CoreAIRuntime types.
        .target(
            name: "AetherRuntime",
            dependencies: ["AetherCore"],
            swiftSettings: common + [.enableExperimentalFeature("Lifetimes")],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .target(name: "AetherHost", dependencies: ["AetherRuntime", "AetherCore"], swiftSettings: common),
        // swift-transformers joins AetherText in M2, with the tokenizer adapter; M1 has no third-party dependency.
        .target(name: "AetherText", dependencies: ["AetherHost", "AetherRuntime", "AetherCore"], swiftSettings: common),
        .target(name: "AetherFoundationModels", dependencies: ["AetherText"], swiftSettings: common),
        .target(name: "AetherVision", dependencies: ["AetherHost", "AetherRuntime", "AetherCore"], swiftSettings: common),
        .target(name: "AetherEmbeddings", dependencies: ["AetherHost", "AetherRuntime", "AetherCore"], swiftSettings: common),
        .target(name: "AetherDecide", dependencies: ["AetherText", "AetherHost", "AetherRuntime", "AetherCore"], swiftSettings: common),
        .target(
            name: "Aether",
            dependencies: ["AetherText", "AetherVision", "AetherEmbeddings", "AetherDecide"],
            swiftSettings: common
        ),

        .testTarget(
            name: "AetherCoreTests",
            dependencies: ["AetherCore"],
            resources: [.copy("Fixtures")],
            swiftSettings: common
        ),
        .testTarget(
            name: "AetherHostTests",
            dependencies: ["AetherHost"],
            swiftSettings: common
        ),
        // Test-only child process for the cross-process GPU lease cases; not a product and never shipped.
        .executableTarget(
            name: "AetherGPULeaseHelper",
            dependencies: ["AetherHost"],
            path: "Tests/AetherGPULeaseHelper",
            swiftSettings: common
        ),
        .testTarget(
            name: "AetherRuntimeTests",
            dependencies: ["AetherRuntime", "AetherCore"],
            // Reads AetherCoreTests/Fixtures by path: the zoo fixtures have one home.
            swiftSettings: common + [.enableExperimentalFeature("Lifetimes")]
        ),
    ]
)
