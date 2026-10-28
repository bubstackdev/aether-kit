// swift-tools-version: 6.4
import CompilerPluginSupport
import PackageDescription

// The floor is macOS 14 / iOS 17, so apps can link the language facade there; it reports "unsupported OS" below 27.
// Every target that uses Core AI is gated at macOS 27 / iOS 27 declaration by declaration (Scripts/annotate-floor.py),
// and CoreAI links weakly because every reference to it is availability-gated. Tests build with
// `--triple arm64-apple-macosx27.0` (docs/deployment-floor.md).
let package = Package(
    name: "AetherKit",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "AetherKit", targets: ["AetherKit"]),
        .library(name: "AetherAssets", targets: ["AetherAssets"]),
        .library(name: "AetherCoreAI", targets: ["AetherCoreAI"]),
        .library(name: "AetherLanguageCoreAI", targets: ["AetherLanguageCoreAI"]),
        .library(name: "AetherLanguageRuntime", targets: ["AetherLanguageRuntime"]),
        .library(name: "AetherLanguage", targets: ["AetherLanguage"]),
        .library(name: "AetherLanguageFoundationModels", targets: ["AetherLanguageFoundationModels"]),
        .library(name: "AetherLanguageMacros", targets: ["AetherLanguageMacros"]),
        .library(name: "AetherLanguageTokenizers", targets: ["AetherLanguageTokenizers"]),
        .library(name: "AetherVision", targets: ["AetherVision"]),
        .library(name: "AetherH3CoreAI", targets: ["AetherH3CoreAI"]),
        .library(name: "AetherH3", targets: ["AetherH3"]),
        .library(name: "AetherKrea", targets: ["AetherKrea"]),
        .library(name: "AetherKreaTokenizers", targets: ["AetherKreaTokenizers"]),
        .library(name: "AetherH3Tokenizers", targets: ["AetherH3Tokenizers"]),
        .library(name: "AetherKreaCoreAI", targets: ["AetherKreaCoreAI"]),
        .library(name: "AetherFlux2", targets: ["AetherFlux2"]),
        .library(name: "AetherFlux2Tokenizers", targets: ["AetherFlux2Tokenizers"]),
        .library(name: "AetherFlux2CoreAI", targets: ["AetherFlux2CoreAI"]),
        .library(name: "AetherSpeech", targets: ["AetherSpeech"]),
        .library(name: "AetherSpeechCoreAI", targets: ["AetherSpeechCoreAI"]),
        .executable(name: "aether-probe", targets: ["AetherProbe"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.1.0"),
        // The @AetherGenerable macro's compiler plugin only (it runs on the build host); pinned to the Swift 6.4 toolchain's.
        .package(url: "https://github.com/swiftlang/swift-syntax", exact: "604.0.0"),
    ],
    targets: [
        .target(name: "AetherKit"),
        .target(name: "AetherVision"),
        .target(name: "AetherAssets"),
        .target(
            name: "AetherCoreAI", dependencies: ["AetherKit", "AetherAssets"],
            swiftSettings: [.enableExperimentalFeature("Lifetimes")],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        // Backend-neutral language runtime: no Core AI, Metal, Synchronization or AetherKit.
        .target(name: "AetherLanguageRuntime"),
        .target(name: "AetherLanguageCoreAI", dependencies: ["AetherKit", "AetherCoreAI", "AetherVision", "AetherLanguageRuntime"],
                swiftSettings: [.enableExperimentalFeature("Lifetimes")]),
        // The app-facing text facade. It links and launches on macOS 14 and iOS 17, and reaches Core AI only behind
        // availability checks (docs/language-facade.md).
        .target(name: "AetherLanguage", dependencies: [
            "AetherKit", "AetherAssets", "AetherLanguageRuntime", "AetherLanguageTokenizers", "AetherLanguageCoreAI",
        ]),
        // The facade as a FoundationModels LanguageModel (OS 27): apps keep LanguageModelSession. FoundationModels links
        // weakly, like CoreAI.
        .target(name: "AetherLanguageFoundationModels", dependencies: ["AetherLanguage", "AetherLanguageTokenizers"]),
        // @AetherGenerable and @AetherGuide: a separate product, so facade users who write schemas by hand (or use
        // FoundationModels' @Generable through the adapter) don't build swift-syntax.
        .target(name: "AetherLanguageMacros", dependencies: ["AetherLanguage", "AetherLanguageMacrosPlugin"]),
        .macro(name: "AetherLanguageMacrosPlugin", dependencies: [
            .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
            .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
        ]),
        .target(name: "AetherLanguageTokenizers", dependencies: [
            "AetherKit", .product(name: "Tokenizers", package: "swift-transformers"),
        ]),
        .target(name: "AetherH3CoreAI", dependencies: ["AetherH3", "AetherKit", "AetherCoreAI"],
                swiftSettings: [.enableExperimentalFeature("Lifetimes")]),
        .target(name: "AetherH3", dependencies: ["AetherKit", "AetherAssets"], resources: [.copy("Resources")]),
        .target(name: "AetherKrea", dependencies: ["AetherKit", "AetherAssets"]),
        .target(name: "AetherKreaTokenizers", dependencies: [
            "AetherKrea", .product(name: "Tokenizers", package: "swift-transformers"),
        ]),
        .target(name: "AetherH3Tokenizers", dependencies: [
            "AetherH3", "AetherKit", .product(name: "Tokenizers", package: "swift-transformers"),
        ]),
        .target(name: "AetherKreaCoreAI", dependencies: ["AetherKrea", "AetherCoreAI", "AetherKit", "AetherAssets"]),
        .target(name: "AetherFlux2", dependencies: ["AetherKit", "AetherAssets"]),
        .target(name: "AetherFlux2Tokenizers", dependencies: ["AetherFlux2", "AetherKit", "AetherLanguageTokenizers"]),
        .target(name: "AetherFlux2CoreAI", dependencies: ["AetherFlux2", "AetherCoreAI", "AetherKit", "AetherAssets"]),
        .target(name: "AetherSpeech", dependencies: ["AetherKit", "AetherAssets"]),
        .target(name: "AetherSpeechCoreAI", dependencies: ["AetherSpeech", "AetherCoreAI", "AetherKit", "AetherAssets"]),
        .executableTarget(name: "AetherProbe", dependencies: ["AetherH3", "AetherH3CoreAI", "AetherH3Tokenizers", "AetherKit", "AetherCoreAI", "AetherKrea", "AetherKreaTokenizers", "AetherKreaCoreAI", "AetherFlux2", "AetherFlux2CoreAI", "AetherFlux2Tokenizers", "AetherAssets", "AetherLanguage", "AetherSpeech"]),
        .testTarget(name: "AetherH3CoreAITests", dependencies: ["AetherH3CoreAI", "AetherH3", "AetherCoreAI", "AetherKit"], resources: [.copy("Fixtures")]),
        .testTarget(name: "AetherH3Tests", dependencies: ["AetherH3", "AetherKit", "AetherAssets"], resources: [.copy("Fixtures")]),
        // Test-only child process for the GPU lease contention and signal tests. Not a product; never shipped.
        .executableTarget(name: "AetherGPULeaseHelper", dependencies: ["AetherKit"], path: "Tests/AetherGPULeaseHelper"),
        .testTarget(name: "AetherKitTests", dependencies: ["AetherKit"]),
        .testTarget(name: "AetherLanguageTokenizersTests", dependencies: ["AetherLanguageTokenizers", "AetherKit"],
                    resources: [.copy("Fixtures")]),
        .testTarget(name: "AetherLanguageTests", dependencies: [
            "AetherLanguage", "AetherLanguageFoundationModels", "AetherLanguageRuntime", "AetherLanguageTokenizers", "AetherKit",
            "AetherAssets",
        ]),
        .testTarget(name: "AetherLanguageMacrosTests", dependencies: [
            "AetherLanguageMacros", "AetherLanguageMacrosPlugin",
            .product(name: "SwiftSyntaxMacroExpansion", package: "swift-syntax"),
            .product(name: "SwiftSyntaxMacrosGenericTestSupport", package: "swift-syntax"),
        ]),
        .testTarget(name: "AetherLanguageRuntimeTests", dependencies: ["AetherLanguageRuntime"], resources: [.copy("Fixtures")]),
        .testTarget(name: "AetherLanguageCoreAITests", dependencies: ["AetherLanguageCoreAI", "AetherKit", "AetherVision"], resources: [.copy("Fixtures")]),
        .testTarget(name: "AetherVisionTests", dependencies: ["AetherVision"], resources: [.copy("Fixtures")]),
        .testTarget(name: "AetherAssetsTests", dependencies: ["AetherAssets"]),
        .testTarget(name: "AetherCoreAITests", dependencies: ["AetherCoreAI", "AetherKit", "AetherAssets"]),
        .testTarget(name: "AetherKreaTests", dependencies: ["AetherKrea", "AetherKit", "AetherCoreAI", "AetherKreaCoreAI", "AetherAssets"], resources: [.copy("Fixtures")]),
        .testTarget(name: "AetherH3TokenizersTests", dependencies: ["AetherH3Tokenizers", "AetherH3", "AetherKit"], resources: [.copy("Fixtures")]),
        .testTarget(name: "AetherFlux2Tests", dependencies: [
            "AetherFlux2", "AetherFlux2CoreAI", "AetherFlux2Tokenizers", "AetherLanguageTokenizers",
            "AetherCoreAI", "AetherKit", "AetherAssets",
        ]),
        .testTarget(name: "AetherSpeechTests", dependencies: [
            "AetherSpeech", "AetherSpeechCoreAI", "AetherCoreAI", "AetherKit", "AetherAssets",
        ], resources: [.copy("Fixtures")]),
        .testTarget(name: "AetherKreaTokenizersTests", dependencies: ["AetherKreaTokenizers", "AetherKrea"]),
        .testTarget(name: "AetherProbeTests", dependencies: ["AetherProbe", "AetherKit", "AetherKrea", "AetherKreaCoreAI", "AetherFlux2", "AetherFlux2CoreAI"]),
        .testTarget(name: "AetherNativeTests", dependencies: [
            "AetherKit", "AetherCoreAI", "AetherKrea", "AetherKreaCoreAI", "AetherKreaTokenizers",
            "AetherFlux2", "AetherFlux2CoreAI", "AetherFlux2Tokenizers", "AetherLanguage", "AetherLanguageFoundationModels", "AetherLanguageMacros", "AetherLanguageCoreAI", "AetherLanguageTokenizers",
            "AetherSpeech", "AetherSpeechCoreAI", "AetherVision",
        ]),
    ]
)
