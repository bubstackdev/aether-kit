// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "AetherKit",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "AetherKit", targets: ["AetherKit"]),
        .library(name: "AetherCoreAI", targets: ["AetherCoreAI"]),
        .library(name: "AetherKrea", targets: ["AetherKrea"]),
        .library(name: "AetherKreaTokenizers", targets: ["AetherKreaTokenizers"]),
        .library(name: "AetherKreaCoreAI", targets: ["AetherKreaCoreAI"]),
        .executable(name: "aether-probe", targets: ["AetherProbe"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.1.0"),
    ],
    targets: [
        .target(name: "AetherKit"),
        .target(
            name: "AetherCoreAI", dependencies: ["AetherKit"],
            swiftSettings: [.enableExperimentalFeature("Lifetimes")],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .target(name: "AetherKrea", dependencies: ["AetherKit"]),
        .target(name: "AetherKreaTokenizers", dependencies: [
            "AetherKrea", .product(name: "Tokenizers", package: "swift-transformers"),
        ]),
        .target(name: "AetherKreaCoreAI", dependencies: ["AetherKrea", "AetherCoreAI", "AetherKit"]),
        .executableTarget(name: "AetherProbe", dependencies: ["AetherKit", "AetherCoreAI", "AetherKrea", "AetherKreaTokenizers"]),
        .testTarget(name: "AetherKitTests", dependencies: ["AetherKit"]),
        .testTarget(name: "AetherCoreAITests", dependencies: ["AetherCoreAI", "AetherKit"]),
        .testTarget(name: "AetherKreaTests", dependencies: ["AetherKrea", "AetherKit", "AetherCoreAI", "AetherKreaCoreAI"], resources: [.copy("Fixtures")]),
        .testTarget(name: "AetherKreaTokenizersTests", dependencies: ["AetherKreaTokenizers", "AetherKrea"]),
        .testTarget(name: "AetherNativeTests", dependencies: [
            "AetherKit", "AetherCoreAI", "AetherKrea", "AetherKreaCoreAI", "AetherKreaTokenizers",
        ]),
    ]
)
