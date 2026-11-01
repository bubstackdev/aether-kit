// swift-tools-version: 6.4
// Asset hash cache measurement (0.0.4): not part of the AetherKit package. The dependency is this checkout.
import PackageDescription

let package = Package(
    name: "AssetHashCacheTiming",
    platforms: [.macOS("27.0")],
    dependencies: [.package(path: "../..")],
    targets: [
        .executableTarget(name: "open-timing", dependencies: [
            .product(name: "AetherLanguageCoreAI", package: "asset-hash-cache"),
        ], path: "Sources/open-timing", swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(name: "asset-verify-timing", dependencies: [
            .product(name: "AetherKrea", package: "asset-hash-cache"), .product(name: "AetherH3", package: "asset-hash-cache"),
        ], path: "Sources/asset-verify-timing", swiftSettings: [.swiftLanguageMode(.v5)]),
    ]
)
