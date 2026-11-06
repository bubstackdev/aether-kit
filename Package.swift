// swift-tools-version: 6.2
import PackageDescription

// Module layout: docs/DESIGN.md §3. AetherCore, AetherRuntime and (from M2) AetherText carry real code; the other task
// modules, AetherHost and the umbrella exist so the product graph (and its dependency rules) is fixed from the start.

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
        .library(name: "AetherImage", targets: ["AetherImage"]),
        .library(name: "AetherEmbeddings", targets: ["AetherEmbeddings"]),
        .library(name: "AetherDecide", targets: ["AetherDecide"]),
        .library(name: "Aether", targets: ["Aether"]),
    ],
    dependencies: [
        // The only third-party dependency (DESIGN.md §3): tokenizers and Jinja chat templates, for AetherText alone.
        // Pinned exactly: 1.4.0 is the version the lab's 33/33 HF template parity was measured with.
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.4.0"),
    ],
    targets: [
        // Identity, manifests and bytes on disk. No Core AI import (§3).
        .target(name: "AetherCore", swiftSettings: common),
        // The only module that touches CoreAIRuntime types.
        .target(
            name: "AetherRuntime",
            dependencies: ["AetherCore"],
            swiftSettings: common + [.enableExperimentalFeature("Lifetimes")],
            linkerSettings: [.linkedFramework("CoreAI"), .linkedFramework("Metal")]
        ),
        .target(name: "AetherHost", dependencies: ["AetherRuntime", "AetherCore"], swiftSettings: common),
        .target(
            name: "AetherText",
            dependencies: [
                "AetherHost", "AetherRuntime", "AetherCore",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: common
        ),
        // The FoundationModels `LanguageModel` bridge (M3), on the Generator. Links FoundationModels.
        .target(name: "AetherFoundationModels", dependencies: ["AetherText"], swiftSettings: common),
        .target(name: "AetherVision", dependencies: ["AetherHost", "AetherRuntime", "AetherCore"], swiftSettings: common),
        // Image generation (G1+): `latent-diffusion@1` (SD-Turbo), with its own CLIP BPE (no swift-transformers).
        .target(name: "AetherImage", dependencies: ["AetherHost", "AetherRuntime", "AetherCore"], swiftSettings: common),
        .target(name: "AetherEmbeddings", dependencies: ["AetherHost", "AetherRuntime", "AetherCore"], swiftSettings: common),
        .target(name: "AetherDecide", dependencies: ["AetherText", "AetherHost", "AetherRuntime", "AetherCore"], swiftSettings: common),
        .target(
            name: "Aether",
            dependencies: ["AetherText", "AetherVision", "AetherImage", "AetherEmbeddings", "AetherDecide"],
            swiftSettings: common
        ),

        .testTarget(
            name: "AetherCoreTests",
            dependencies: ["AetherCore"],
            resources: [.copy("Fixtures")],
            swiftSettings: common
        ),
        // Test doubles shared by the test targets (the fake decoder). Not part of any product.
        .target(
            name: "AetherTestSupport",
            dependencies: ["AetherRuntime", "AetherCore", "AetherText"],
            path: "Tests/AetherTestSupport",
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
            dependencies: ["AetherRuntime", "AetherCore", "AetherTestSupport"],
            // Reads AetherCoreTests/Fixtures by path: the zoo fixtures have one home. References are gate inputs,
            // read by path too.
            exclude: ["References"],
            // Fixtures: the sampler's recorded Transformers rows and RNG known answers (0.0.x).
            resources: [.copy("Fixtures")],
            swiftSettings: common + [.enableExperimentalFeature("Lifetimes")]
        ),
        .testTarget(
            name: "AetherFoundationModelsTests",
            dependencies: ["AetherFoundationModels", "AetherText", "AetherRuntime", "AetherCore", "AetherTestSupport"],
            swiftSettings: common
        ),
        .testTarget(
            name: "AetherImageTests",
            dependencies: ["AetherImage", "AetherRuntime", "AetherCore", "AetherHost"],
            swiftSettings: common
        ),
        .testTarget(
            name: "AetherTextTests",
            dependencies: ["AetherText", "AetherRuntime", "AetherCore", "AetherTestSupport"],
            resources: [.copy("Fixtures")],
            swiftSettings: common
        ),
    ]
)
