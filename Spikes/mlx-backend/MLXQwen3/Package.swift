// swift-tools-version: 6.2
// MLX go/no-go spike (not part of AetherKit's package): Qwen3 on mlx-swift at a macOS 14 floor, qualified against the
// same references as Kit's Core AI backend. Build with `swift build -c release --build-system swiftbuild`; mlx-swift
// documents that the native build system cannot compile its Metal shaders (see docs/spikes/mlx-backend-2026-10.md).
import PackageDescription

let package = Package(
    name: "MLXQwen3",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [.library(name: "MLXQwen3", targets: ["MLXQwen3"])],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.33.0"),
    ],
    targets: [
        .target(name: "MLXQwen3", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
        ]),
        .executableTarget(name: "mlx-qwen3-check", dependencies: ["MLXQwen3"]),
    ]
)
