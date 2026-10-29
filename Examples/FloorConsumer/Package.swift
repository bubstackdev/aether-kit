// swift-tools-version: 6.4
import PackageDescription

// An app-shaped consumer at AetherKit's floor: it links the language facade on macOS 14 / iOS 17, where the facade
// must launch and report "unsupported OS". CI launches it on a macOS 15 runner, which has no CoreAI
// (docs/deployment-floor.md).
let package = Package(
    name: "AetherFloorConsumer",
    platforms: [.macOS(.v14), .iOS(.v17)],
    dependencies: [.package(name: "AetherKit", path: "../..")],
    targets: [
        .executableTarget(name: "AetherFloorConsumer", dependencies: [
            .product(name: "AetherLanguage", package: "AetherKit"),
        ])
    ]
)
