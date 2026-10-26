// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "AetherLocalExample",
    platforms: [.macOS("27.0")],
    dependencies: [.package(name: "AetherKit", path: "../..")],
    targets: [
        .executableTarget(name: "AetherLocalExample", dependencies: [
            .product(name: "AetherKit", package: "AetherKit"),
            .product(name: "AetherCoreAI", package: "AetherKit"),
            .product(name: "AetherKrea", package: "AetherKit"),
            .product(name: "AetherKreaTokenizers", package: "AetherKit"),
            .product(name: "AetherKreaCoreAI", package: "AetherKit"),
            .product(name: "AetherLanguageCoreAI", package: "AetherKit"),
            .product(name: "AetherH3", package: "AetherKit"),
            .product(name: "AetherH3CoreAI", package: "AetherKit"),
        ])
    ]
)
