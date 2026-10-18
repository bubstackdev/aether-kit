// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "AetherLocalExample",
    platforms: [.macOS("27.0")],
    dependencies: [.package(name: "AetherKit", path: "../..")],
    targets: [
        .executableTarget(name: "AetherLocalExample", dependencies: [
            .product(name: "AetherKrea", package: "AetherKit"),
        ])
    ]
)
