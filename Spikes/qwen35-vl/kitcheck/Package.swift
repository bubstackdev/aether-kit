// swift-tools-version: 6.4
// Qwen3.5 Kit-level check (0.0.4 spike): not part of the AetherKit package.
import PackageDescription

let package = Package(
    name: "Qwen35KitCheck",
    platforms: [.macOS("27.0")],
    dependencies: [.package(path: "../../..")],
    targets: [
        .executableTarget(name: "q35-kit-check", dependencies: [
            .product(name: "AetherLanguageCoreAI", package: "qwen35-vl"), .product(name: "AetherVision", package: "qwen35-vl"),
        ], path: "Sources/q35-kit-check", swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(name: "q35-open-timing", dependencies: [.product(name: "AetherLanguageCoreAI", package: "qwen35-vl")],
                          path: "Sources/q35-open-timing", swiftSettings: [.swiftLanguageMode(.v5)]),
    ]
)
