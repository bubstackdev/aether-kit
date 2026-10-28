// Runs a split-attention probe asset (fused / split / cond, from split_attention_probe.py) on the GPU: wired mapped-file
// pages after the first q = 64 and q = 1 calls, warm median wall time for q = 1 decode and q = 64 prefill at capacity
// 2,048, and a logits digest. Development tooling only (docs/spikes/split-attention-2026-10.md). Under the GPU lock:
//
//     xcrun swiftc -O -swift-version 5 -o .build/split-attention-probe Spikes/split-attention/split-attention-probe.swift \
//         -framework CoreAI -framework Metal
//     ~/.cache/aether-kit/with-gpu-lock.sh split-attention-probe .build/split-attention-probe <asset>
@preconcurrency import CoreAI
import Darwin
import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let layers = 2, kvHeads = 8, headDim = 128, capacity = 2048

func mappedFileMiB() -> (wired: Int, clean: Int) {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent("split-attention-probe-\(UUID().uuidString).json")
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

let cacheCount = layers * kvHeads * capacity * headDim
let keyCache = NDArray(scalars: (0..<cacheCount).map { Float16(sin(Double($0 % 9973) * 0.37) * 0.5) },
                       shape: [layers, 1, kvHeads, capacity, headDim])
let valueCache = NDArray(scalars: (0..<cacheCount).map { Float16(cos(Double($0 % 7919) * 0.53) * 0.5) },
                         shape: [layers, 1, kvHeads, capacity, headDim])

func inputs(start: Int, count: Int) -> [String: NDArray] {
    ["input_ids": NDArray(scalars: (0..<count).map { Int32((start * 31 + $0 * 97) % 8192) }, shape: [1, count]),
     "cache_positions": NDArray(scalars: (start..<(start + count)).map(Int32.init), shape: [1, count]),
     "keyCache": keyCache, "valueCache": valueCache]
}

func call(_ function: InferenceFunction, _ feed: [String: NDArray], _ stream: ComputeStream) async throws -> [Float] {
    let outputs = try function.encode(inputs: feed.mapValues { InferenceFunction.AsyncValue($0) },
                                      states: InferenceFunction.AsyncMutableViews(), to: stream)
    await stream.currentWorkCompleted()
    var logits: [Float] = []
    for (name, value) in outputs {
        guard let array = try await value.ndArray, name == "logits" else { continue }
        logits = array.view(as: Float16.self).withUnsafePointer { pointer, shape, strides in
            (0..<shape[2]).map { Float(pointer[(shape[1] - 1) * strides[1] + $0 * strides[2]]) }
        }
    }
    return logits
}

func digest(_ name: String, _ logits: [Float]) {
    let top = logits.indices.max { logits[$0] < logits[$1] }!
    print("\(name) logits: argmax \(top), max \(logits[top]), sum \(logits.reduce(0, +)), first \(logits.prefix(4).map { ($0 * 1000).rounded() / 1000 })")
}

func median(_ function: InferenceFunction, start: Int, count: Int, rounds: Int, _ stream: ComputeStream) async throws -> Double {
    let feed = inputs(start: start, count: count)
    for _ in 0..<5 { _ = try await call(function, feed, stream) }
    var times: [Double] = []
    for _ in 0..<rounds {
        let t = ContinuousClock.now
        _ = try await call(function, feed, stream)
        let d = ContinuousClock.now - t
        times.append(Double(d.components.seconds) * 1e3 + Double(d.components.attoseconds) / 1e15)
    }
    return times.sorted()[times.count / 2]
}

let arguments = CommandLine.arguments
guard arguments.count == 2 else { fatalError("usage: split-attention-probe <asset>") }
var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
options.expectFrequentReshapes = true
phase("before open")
let model = try await AIModel(contentsOf: URL(filePath: arguments[1]), options: options)
phase("open")
nonisolated(unsafe) let stream = ComputeStream(commandQueue: queue)
let function = try model.loadFunction(named: "main")!
digest("q 64 at 1024", try await call(function, inputs(start: 1024, count: 64), stream)); phase("after q 64")
digest("q 1 at 2000", try await call(function, inputs(start: 2000, count: 1), stream)); phase("after q 1")
digest("q 6 at 40", try await call(function, inputs(start: 40, count: 6), stream))
for round in 1...2 {
    let decode = try await median(function, start: 2000, count: 1, rounds: 60, stream)
    let prefill = try await median(function, start: 1024, count: 64, rounds: 30, stream)
    print(String(format: "round %d: q 1 decode median %.3f ms, q 64 prefill median %.3f ms", round, decode, prefill))
}
phase("end")
