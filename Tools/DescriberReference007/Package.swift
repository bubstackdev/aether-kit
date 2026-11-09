// swift-tools-version: 6.2
// Records the reference for the Qwen3-VL describer gate (docs/VLM.md §6, G4) with AetherKit 0.0.7's
// `VisionLanguageSession`, the earlier describer this driver is gated against. A standalone package, never part of
// the kit's products: it links the 0.0.7 release, by default from its tag, or from a local checkout of the 0.0.7 line
// named by `AETHER_KIT_007` (an absolute path).
import Foundation
import PackageDescription

let local = ProcessInfo.processInfo.environment["AETHER_KIT_007"]
let kit: Package.Dependency = local.map { .package(name: "AetherKit", path: $0) }
    ?? .package(url: "https://github.com/bubstackdev/aether-kit.git", exact: "0.0.7")

let package = Package(
    name: "DescriberReference007",
    platforms: [.macOS("27.0")],
    dependencies: [kit],
    targets: [
        .executableTarget(
            name: "DescriberReference007",
            dependencies: [
                .product(name: "AetherLanguageCoreAI", package: "AetherKit"),
                .product(name: "AetherLanguageTokenizers", package: "AetherKit"),
                .product(name: "AetherVision", package: "AetherKit"),
                .product(name: "AetherKit", package: "AetherKit"),
            ]),
    ])
