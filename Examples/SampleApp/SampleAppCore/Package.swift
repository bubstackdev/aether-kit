// swift-tools-version: 6.4
import PackageDescription

// The sample app's logic, outside its views, so it is tested with `swift test` (docs/sample-app.md). The app target in
// ../project.yml links it; the committed catalog entries are its resources.
let package = Package(
    name: "SampleAppCore",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "SampleAppCore", targets: ["SampleAppCore"]),
    ],
    dependencies: [
        .package(name: "AetherKit", path: "../../.."),
    ],
    targets: [
        .target(
            name: "SampleAppCore",
            dependencies: [
                .product(name: "AetherKit", package: "AetherKit"),
                .product(name: "AetherAssets", package: "AetherKit"),
                .product(name: "AetherLanguage", package: "AetherKit"),
                .product(name: "AetherLanguageMacros", package: "AetherKit"),
            ],
            resources: [.copy("Catalog")]
        ),
        .testTarget(name: "SampleAppCoreTests", dependencies: ["SampleAppCore"]),
    ]
)
