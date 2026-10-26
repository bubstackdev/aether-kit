// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "AetherKit",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "AetherKit", targets: ["AetherKit"]),
        .library(name: "AetherAssets", targets: ["AetherAssets"]),
        .library(name: "AetherCoreAI", targets: ["AetherCoreAI"]),
        .library(name: "AetherLanguageCoreAI", targets: ["AetherLanguageCoreAI"]),
        .library(name: "AetherH3CoreAI", targets: ["AetherH3CoreAI"]),
        .library(name: "AetherH3", targets: ["AetherH3"]),
        .library(name: "AetherKrea", targets: ["AetherKrea"]),
        .library(name: "AetherKreaTokenizers", targets: ["AetherKreaTokenizers"]),
        .library(name: "AetherH3Tokenizers", targets: ["AetherH3Tokenizers"]),
        .library(name: "AetherKreaCoreAI", targets: ["AetherKreaCoreAI"]),
        .library(name: "AetherFlux2", targets: ["AetherFlux2"]),
        .library(name: "AetherFlux2Tokenizers", targets: ["AetherFlux2Tokenizers"]),
        .library(name: "AetherFlux2CoreAI", targets: ["AetherFlux2CoreAI"]),
        .executable(name: "aether-probe", targets: ["AetherProbe"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.1.0"),
    ],
    targets: [
        .target(name: "AetherKit"),
        .target(name: "AetherAssets"),
        .target(
            name: "AetherCoreAI", dependencies: ["AetherKit", "AetherAssets"],
            swiftSettings: [.enableExperimentalFeature("Lifetimes")],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .target(name: "AetherLanguageCoreAI", dependencies: ["AetherKit", "AetherCoreAI"],
                swiftSettings: [.enableExperimentalFeature("Lifetimes")]),
        .target(name: "AetherH3CoreAI", dependencies: ["AetherH3", "AetherKit", "AetherCoreAI"],
                swiftSettings: [.enableExperimentalFeature("Lifetimes")]),
        .target(name: "AetherH3", dependencies: ["AetherKit", "AetherAssets"], resources: [.copy("Resources")]),
        .target(name: "AetherKrea", dependencies: ["AetherKit", "AetherAssets"]),
        .target(name: "AetherKreaTokenizers", dependencies: [
            "AetherKrea", .product(name: "Tokenizers", package: "swift-transformers"),
        ]),
        .target(name: "AetherH3Tokenizers", dependencies: [
            "AetherH3", "AetherKit", .product(name: "Tokenizers", package: "swift-transformers"),
        ]),
        .target(name: "AetherKreaCoreAI", dependencies: ["AetherKrea", "AetherCoreAI", "AetherKit", "AetherAssets"]),
        .target(name: "AetherFlux2", dependencies: ["AetherKit", "AetherAssets"]),
        .target(name: "AetherFlux2Tokenizers", dependencies: [
            "AetherFlux2", .product(name: "Tokenizers", package: "swift-transformers"),
        ]),
        .target(name: "AetherFlux2CoreAI", dependencies: ["AetherFlux2", "AetherCoreAI", "AetherKit", "AetherAssets"]),
        .executableTarget(name: "AetherProbe", dependencies: ["AetherH3", "AetherH3CoreAI", "AetherH3Tokenizers", "AetherKit", "AetherCoreAI", "AetherKrea", "AetherKreaTokenizers", "AetherKreaCoreAI"]),
        .testTarget(name: "AetherH3CoreAITests", dependencies: ["AetherH3CoreAI", "AetherH3", "AetherCoreAI", "AetherKit"], resources: [.copy("Fixtures")]),
        .testTarget(name: "AetherH3Tests", dependencies: ["AetherH3", "AetherKit", "AetherAssets"], resources: [.copy("Fixtures")]),
        // Test-only child process for the GPU lease contention and signal tests. Not a product; never shipped.
        .executableTarget(name: "AetherGPULeaseHelper", dependencies: ["AetherKit"], path: "Tests/AetherGPULeaseHelper"),
        .testTarget(name: "AetherKitTests", dependencies: ["AetherKit"]),
        .testTarget(name: "AetherLanguageCoreAITests", dependencies: ["AetherLanguageCoreAI", "AetherKit"]),
        .testTarget(name: "AetherAssetsTests", dependencies: ["AetherAssets"]),
        .testTarget(name: "AetherCoreAITests", dependencies: ["AetherCoreAI", "AetherKit", "AetherAssets"]),
        .testTarget(name: "AetherKreaTests", dependencies: ["AetherKrea", "AetherKit", "AetherCoreAI", "AetherKreaCoreAI", "AetherAssets"], resources: [.copy("Fixtures")]),
        .testTarget(name: "AetherH3TokenizersTests", dependencies: ["AetherH3Tokenizers", "AetherH3", "AetherKit"], resources: [.copy("Fixtures")]),
        .testTarget(name: "AetherFlux2Tests", dependencies: ["AetherFlux2", "AetherFlux2CoreAI", "AetherFlux2Tokenizers", "AetherCoreAI", "AetherKit", "AetherAssets"]),
        .testTarget(name: "AetherKreaTokenizersTests", dependencies: ["AetherKreaTokenizers", "AetherKrea"]),
        .testTarget(name: "AetherProbeTests", dependencies: ["AetherProbe", "AetherKit", "AetherKrea", "AetherKreaCoreAI"]),
        .testTarget(name: "AetherNativeTests", dependencies: [
            "AetherKit", "AetherCoreAI", "AetherKrea", "AetherKreaCoreAI", "AetherKreaTokenizers",
            "AetherFlux2", "AetherFlux2CoreAI", "AetherFlux2Tokenizers", "AetherLanguageCoreAI",
        ]),
    ]
)
