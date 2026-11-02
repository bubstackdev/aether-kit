// swift-tools-version: 6.4
// Asset hash cache measurement (0.0.4): not part of the AetherKit package. The dependency is this checkout; SwiftPM names
// a path dependency after its directory, so in a checkout not named `aether-kit` (a worktree) set `kit` to that name.
import PackageDescription

let kit = "aether-kit"

let package = Package(
    name: "AssetHashCacheTiming",
    platforms: [.macOS("27.0")],
    dependencies: [.package(path: "../..")],
    targets: [
        .executableTarget(name: "open-timing", dependencies: [
            .product(name: "AetherLanguageCoreAI", package: kit),
        ], path: "Sources/open-timing", swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(name: "asset-verify-timing", dependencies: [
            .product(name: "AetherKrea", package: kit), .product(name: "AetherH3", package: kit),
        ], path: "Sources/asset-verify-timing", swiftSettings: [.swiftLanguageMode(.v5)]),
    ]
)
