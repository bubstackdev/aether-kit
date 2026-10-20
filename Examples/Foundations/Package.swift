// swift-tools-version: 6.4
import PackageDescription
let package = Package(
    name: "AetherFoundationConsumer",
    platforms: [.macOS("27.0")],
    dependencies: [.package(name: "AetherKit", path: "../..")],
    targets: [.executableTarget(name: "AetherFoundationConsumer", dependencies: [
        .product(name: "AetherKit", package: "AetherKit"),
        .product(name: "AetherAssets", package: "AetherKit"),
    ])]
)
