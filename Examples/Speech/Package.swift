// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "AetherSpeak",
    platforms: [.macOS("27.0")],
    dependencies: [.package(name: "AetherKit", path: "../..")],
    targets: [
        .executableTarget(name: "AetherSpeak", dependencies: [
            .product(name: "AetherKit", package: "AetherKit"),
            .product(name: "AetherSpeech", package: "AetherKit"),
            .product(name: "AetherSpeechCoreAI", package: "AetherKit"),
        ])
    ]
)
