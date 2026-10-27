// Runs the toy decoder layouts from Forge/language/probe_function_weights.py on the GPU and prints, after each
// function's first use, Metal's allocated size and the process's mapped-file pages (wired and clean) from footprint.
// Development tooling only (docs/qwen3-vl-single-decoder.md). Build and run under the GPU lock:
//
//     xcrun swiftc -O -swift-version 5 -o .build/probe-function-weights Scripts/probe-function-weights.swift \
//         -framework CoreAI -framework Metal
//     ~/.cache/aether-kit/with-gpu-lock.sh probe-function-weights .build/probe-function-weights <asset> <mode>
//
// Modes: main, image, two, twin, embeds, aligned (the asset's layout), or retain-two / retain-image / retain-aligned:
// 42 requests shaped like a Qwen3-VL image request (a 64-token text chunk, a 48-token image span, an 8-token text run,
// four decode steps) on the `two`, `image` or `aligned` asset, printing Metal's growth per request once warm.
@preconcurrency import CoreAI
import Darwin
import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let hidden = 1024, levels = 2, imageToken: Int32 = 7

func mappedFileMiB() -> (wired: Int, clean: Int) {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent("probe-function-weights-\(UUID().uuidString).json")
    defer { try? FileManager.default.removeItem(at: file) }
    let tool = Process()
    tool.executableURL = URL(filePath: "/usr/bin/footprint")
    tool.arguments = ["-p", String(getpid()), "-f", "bytes", "-w", "-j", file.path]
    tool.standardOutput = FileHandle.nullDevice; tool.standardError = FileHandle.nullDevice
    guard (try? tool.run()) != nil else { return (0, 0) }
    tool.waitUntilExit()
    guard let data = try? Data(contentsOf: file), let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let process = (json["processes"] as? [[String: Any]])?.first,
          let mapped = (process["categories"] as? [String: [String: Any]])?["mapped file"] else { return (0, 0) }
    return (((mapped["wired"] as? Int) ?? 0) >> 20, ((mapped["clean"] as? Int) ?? 0) >> 20)
}

func phase(_ name: String) {
    let pages = mappedFileMiB()
    print("\(name): metal \(device.currentAllocatedSize >> 20) MiB, mapped file wired \(pages.wired) MiB, clean \(pages.clean) MiB")
}

/// `count` token IDs; the `image` tokens after the first are image pads.
func tokens(_ count: Int, image: Int = 0) -> NDArray {
    var ids = (0..<count).map { Int32(100 + $0) }
    for i in 0..<min(image, count - 1) { ids[1 + i] = imageToken }
    return NDArray(scalars: ids, shape: [1, count])
}

func rows(_ shape: [Int]) -> NDArray { NDArray(scalars: [Float16](repeating: 0.01, count: shape.reduce(1, *)), shape: shape) }

func call(_ function: InferenceFunction, _ inputs: [String: NDArray], _ stream: ComputeStream) async throws {
    let outputs = try function.encode(inputs: inputs.mapValues { InferenceFunction.AsyncValue($0) },
                                      states: InferenceFunction.AsyncMutableViews(), to: stream)
    await stream.currentWorkCompleted()
    for (_, value) in outputs { _ = try await value.ndArray }
}

func imageInputs(_ count: Int, image: Int, rowCount: Int) -> [String: NDArray] {
    ["input_ids": tokens(count, image: image), "image_embeds": rows([rowCount, hidden]), "deepstack": rows([levels, rowCount, hidden])]
}

let arguments = CommandLine.arguments
guard arguments.count == 3 else { fatalError("usage: probe-function-weights <asset> <mode>") }
let mode = arguments[2]
var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
options.expectFrequentReshapes = true
phase("before open")
let model = try await AIModel(contentsOf: URL(filePath: arguments[1]), options: options)
phase("open")
nonisolated(unsafe) let stream = ComputeStream(commandQueue: queue)

func runText(_ name: String) async throws {
    let function = try model.loadFunction(named: name)!
    try await call(function, ["input_ids": tokens(64)], stream); phase("\(name), q 64")
    try await call(function, ["input_ids": tokens(1)], stream); phase("\(name), q 1")
}

func runImage(textSteps minimum: Int) async throws {
    let function = try model.loadFunction(named: "withImage")!
    try await call(function, imageInputs(64, image: 16, rowCount: 16), stream); phase("withImage, q 64 with 16 image rows")
    try await call(function, imageInputs(64, image: 0, rowCount: 1), stream); phase("withImage, q 64 text (one zero row)")
    try await call(function, imageInputs(minimum, image: 0, rowCount: 1), stream); phase("withImage, q \(minimum) text (one zero row)")
}

switch mode {
case "main": try await runText("main")
case "image": try await runImage(textSteps: 1)
case "two": try await runText("main"); try await runImage(textSteps: 2)
case "twin": try await runText("main"); try await runText("main2")
case "embeds":
    let embed = try model.loadFunction(named: "embed")!
    try await call(embed, ["input_ids": tokens(64)], stream); phase("embed, q 64")
    let function = try model.loadFunction(named: "main")!
    for count in [64, 1] {
        try await call(function, ["inputs_embeds": rows([1, count, hidden]), "image_mask": rows([1, count, 1]),
                                  "deepstack": rows([levels, count, hidden])], stream)
        phase("main from embeddings, q \(count)")
    }
case "aligned":
    let function = try model.loadFunction(named: "main")!
    try await call(function, imageInputs(64, image: 16, rowCount: 64), stream); phase("main, q 64 with 16 image tokens")
    try await call(function, imageInputs(64, image: 0, rowCount: 64), stream); phase("main, q 64 text (zero rows)")
    try await call(function, imageInputs(1, image: 0, rowCount: 1), stream); phase("main, q 1 text (zero row)")
case "retain-two", "retain-image", "retain-aligned":
    let first = try model.loadFunction(named: mode == "retain-image" ? "withImage" : "main")!
    let image = mode == "retain-two" ? try model.loadFunction(named: "withImage")! : first
    func textStep(_ count: Int) async throws {
        switch mode {
        case "retain-two": try await call(first, ["input_ids": tokens(count)], stream)
        case "retain-image": try await call(first, imageInputs(count, image: 0, rowCount: 1), stream)
        default: try await call(first, imageInputs(count, image: 0, rowCount: count), stream)
        }
    }
    func request() async throws {
        try await textStep(64)
        try await call(image, imageInputs(48, image: 47, rowCount: mode == "retain-aligned" ? 48 : 47), stream)
        try await textStep(8)
        for _ in 0..<4 { try await textStep(1) }
    }
    for i in 1...2 { try await request(); phase("warm request \(i)") }
    let warm = device.currentAllocatedSize
    for _ in 0..<40 { try await request() }
    phase("after 40 more requests")
    print("growth per request: \(Double(device.currentAllocatedSize - warm) / 40 / 1_048_576) MiB")
default: fatalError("unknown mode \(mode)")
}
