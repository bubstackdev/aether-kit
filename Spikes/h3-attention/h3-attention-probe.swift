// Runs an H3 attention probe asset (from h3_attention_probe.py) on the GPU at each row count: the first call's time
// (specialization included), the warm median and minimum, Metal allocation growth from the first call, and a sample
// of the output for h3_attention_probe.py compare. Development tooling only (docs/spikes/h3-attention-2026-10.md).
// Under the GPU lock:
//
//     xcrun swiftc -O -swift-version 5 -o .build/h3-attention-probe Spikes/h3-attention/h3-attention-probe.swift \
//         -framework CoreAI -framework Metal
//     ~/.cache/aether-kit/with-gpu-lock.sh h3-attention-probe .build/h3-attention-probe <asset> bf16|fp16 <name> \
//         <results directory> <rows,rows,...> [rounds]
@preconcurrency import CoreAI
import Foundation
import Metal

let heads = 56, dim = 128
let sampleHeads = [0, 13, 27, 41, 55], sampleRows = 64
let seeds = ["q": 1, "k": 2, "v": 3], scales: [String: Double] = ["q": 8, "k": 1, "v": 1]

let arguments = CommandLine.arguments
guard arguments.count == 6 || arguments.count == 7, ["bf16", "fp16"].contains(arguments[2]) else {
    fatalError("usage: h3-attention-probe <asset> bf16|fp16 <name> <results directory> <rows,...> [rounds]")
}
let bf16 = arguments[2] == "bf16"
let name = arguments[3]
let results = URL(filePath: arguments[4])
let rowCounts = arguments[5].split(separator: ",").map { Int($0)! }
let rounds = arguments.count == 7 ? Int(arguments[6])! : 5
try FileManager.default.createDirectory(at: results, withIntermediateDirectories: true)

let device = MTLCreateSystemDefaultDevice()!

/// The input both sides compute: u = ((i * 2654435761 + seed * 40503) mod 2^32) / 2^32, mapped to [-scale, scale).
func fill(_ buffer: any MTLBuffer, _ tensor: String, count: Int) {
    let seed = UInt64(seeds[tensor]! * 40503), scale = scales[tensor]!
    let target = buffer.contents().assumingMemoryBound(to: UInt16.self)
    DispatchQueue.concurrentPerform(iterations: 64) { part in
        let lower = count * part / 64, upper = count * (part + 1) / 64
        for i in lower..<upper {
            let bits = (UInt64(i) &* 2654435761 &+ seed) & 0xFFFF_FFFF
            let value = Float((Double(bits) / 4294967296 * 2 - 1) * scale)
            if bf16 {
                let raw = value.bitPattern
                target[i] = UInt16(truncatingIfNeeded: (raw &+ 0x7FFF &+ ((raw >> 16) & 1)) >> 16)
            } else {
                target[i] = Float16(value).bitPattern
            }
        }
    }
}

func scalar(_ raw: UInt16) -> Float { bf16 ? Float(bitPattern: UInt32(raw) << 16) : Float(Float16(bitPattern: raw)) }

func milliseconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) * 1e3 + Double(duration.components.attoseconds) / 1e15
}

var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
options.expectFrequentReshapes = true
let model = try await AIModel(contentsOf: URL(filePath: arguments[1]), options: options)
let function = try model.loadFunction(named: "main")!
nonisolated(unsafe) let stream = ComputeStream()
let type: NDArray.ScalarType = bf16 ? .bfloat16 : .float16

for rows in rowCounts {
    let count = heads * rows * dim
    let shape = [1, heads, rows, dim]
    var buffers: [String: any MTLBuffer] = [:]
    for tensor in ["q", "k", "v", "out"] {
        buffers[tensor] = device.makeBuffer(length: count * 2, options: .storageModeShared)!
    }
    for tensor in ["q", "k", "v"] { fill(buffers[tensor]!, tensor, count: count) }

    func call() async throws -> Double {
        let inputs = Dictionary(uniqueKeysWithValues: ["q", "k", "v"].map {
            ($0, InferenceFunction.AsyncValue(unsafeBuffer: buffers[$0]!, scalarType: type, shape: shape))
        })
        var out = InferenceFunction.AsyncMutableValue(unsafeBuffer: buffers["out"]!, scalarType: type, shape: shape)
        var views = InferenceFunction.AsyncMutableViews()
        views.insert(&out, for: "out")
        let start = ContinuousClock.now
        _ = try function.encode(inputs: inputs, states: InferenceFunction.AsyncMutableViews(), outputViews: views, to: stream)
        await stream.currentWorkCompleted()
        return milliseconds(ContinuousClock.now - start)
    }

    let before = device.currentAllocatedSize
    let first = try await call()
    let afterFirst = device.currentAllocatedSize
    _ = try await call()
    var times: [Double] = []
    for _ in 0..<rounds { times.append(try await call()) }
    times.sort()

    let out = buffers["out"]!.contents().assumingMemoryBound(to: UInt16.self)
    var sample: [Float] = []
    for head in sampleHeads {
        for j in 0..<sampleRows {
            let row = j * (rows - 1) / (sampleRows - 1)
            for d in 0..<dim { sample.append(scalar(out[(head * rows + row) * dim + d])) }
        }
    }
    try sample.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: results.appendingPathComponent("\(name)-\(rows).f32"))
    let line: [String: Any] = [
        "variant": name, "rows": rows, "firstMs": first, "medianMs": times[times.count / 2], "minMs": times[0],
        "rounds": rounds, "metalGrowthFromFirstCallMiB": Double(afterFirst - before) / 1_048_576,
        "metalAllocatedMiB": Double(device.currentAllocatedSize) / 1_048_576,
        "finite": sample.allSatisfy(\.isFinite),
    ]
    let json = try JSONSerialization.data(withJSONObject: line, options: [.sortedKeys])
    print(String(decoding: json, as: UTF8.self))
    fflush(stdout)
    buffers.removeAll()
}
