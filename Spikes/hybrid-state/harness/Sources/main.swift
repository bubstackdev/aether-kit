// Sequential (kit-style) decode timing for stateful Core AI language assets with any number of states.
//
// usage: hybrid-decode <asset> <function> <prompt-ids-file> <decode-steps> <capacity> <out.json>
//                      [--prefill-function NAME] [--cpu] [--logits-dump FILE] [--dump-steps N] [--position-scalar]
//
// Prompt tokens are fed one per call through <function> unless --prefill-function names a function that takes the
// whole prompt in one call. Every state the function declares is allocated from its descriptor (dynamic axes resolved
// to <capacity>) as a zeroed shared MTLBuffer that the harness owns, exactly like the kit's KV caches.
@preconcurrency import CoreAI
import Foundation
import Metal

struct Failure: Error, CustomStringConvertible { let description: String }

var args = Array(CommandLine.arguments.dropFirst())
func flag(_ name: String) -> Bool { if let i = args.firstIndex(of: name) { args.remove(at: i); return true }; return false }
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    let v = args[i + 1]; args.removeSubrange(i...(i + 1)); return v
}
let cpu = flag("--cpu")
let positionScalar = flag("--position-scalar")   // fixed-shape assets: position_ids is [1, 1] = current position
let prefillName = option("--prefill-function")
let dumpPath = option("--logits-dump")
let dumpSteps = Int(option("--dump-steps") ?? "0") ?? 0
guard args.count == 6, let decodeSteps = Int(args[3]), let capacity = Int(args[4]) else {
    print("usage: hybrid-decode <asset> <function> <prompt-ids-file> <decode-steps> <capacity> <out.json> [options]")
    exit(2)
}
let assetURL = URL(fileURLWithPath: args[0])
let functionName = args[1]
let prompt = try String(contentsOfFile: args[2], encoding: .utf8)
    .split(whereSeparator: { $0 == "," || $0.isWhitespace }).compactMap { Int32($0) }
let outURL = URL(fileURLWithPath: args[5])

func seconds(_ d: Duration) -> Double { Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }

struct StateBuffer { let name: String; let buffer: any MTLBuffer; let descriptor: NDArrayDescriptor }

let clock = ContinuousClock()
let loadStart = clock.now
var options: SpecializationOptions = cpu ? .cpuOnly : SpecializationOptions(preferredComputeUnitKind: .gpu)
options.expectFrequentReshapes = true
let model = try await AIModel(contentsOf: assetURL, options: options)
guard let function = try model.loadFunction(named: functionName),
      let descriptor = model.functionDescriptor(for: functionName) else { throw Failure(description: "no function") }
let prefillFunction = try prefillName.map { name -> InferenceFunction in
    guard let f = try model.loadFunction(named: name) else { throw Failure(description: "no prefill function") }
    return f
}
let loadSeconds = seconds(loadStart.duration(to: clock.now))
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    throw Failure(description: "no Metal device")
}
let stream = ComputeStream(commandQueue: queue)
var states: [StateBuffer] = []
var stateBytes = 0
for name in descriptor.stateNames {
    guard case .ndArray(let template)? = descriptor.stateDescriptor(of: name) else { throw Failure(description: "state \(name)") }
    let shape = template.shape.map { $0 < 0 ? capacity : $0 }
    let resolved = template.resolvingDynamicDimensions(shape)
    guard let buffer = device.makeBuffer(length: resolved.minimumByteCount, options: .storageModeShared) else {
        throw Failure(description: "alloc \(name)")
    }
    buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
    stateBytes += buffer.length
    states.append(StateBuffer(name: name, buffer: buffer, descriptor: resolved))
}
guard case .ndArray(let logitsTemplate)? = descriptor.outputDescriptor(of: "logits") else { throw Failure(description: "logits") }
let vocab = logitsTemplate.shape.last!

var dump: [Float] = []

/// AsyncMutableValue is ~Copyable, so a variable number of states is bound by recursion: each frame owns one value
/// and passes the views (which borrow it) down; the innermost frame encodes.
func encode(_ f: InferenceFunction, inputs: [String: InferenceFunction.AsyncValue], index: Int,
            views: consuming InferenceFunction.AsyncMutableViews) throws -> [String: InferenceFunction.AsyncValue] {
    if index == states.count { return try f.encode(inputs: inputs, states: views, to: stream) }
    let s = states[index]
    var value = InferenceFunction.AsyncMutableValue(unsafeBuffer: s.buffer, scalarType: s.descriptor.scalarType,
        shape: s.descriptor.shape, strides: s.descriptor.preferredStrides, interleaveLayout: s.descriptor.interleaveLayout)
    var next = views
    next.insert(&value, for: s.name)
    return try encode(f, inputs: inputs, index: index + 1, views: next)
}

/// One joined call; returns the greedy token of the last logits row.
func call(_ f: InferenceFunction, tokens: [Int32], position: Int, keep: Bool) async throws -> Int {
    let inputs = [
        "input_ids": InferenceFunction.AsyncValue(NDArray(scalars: tokens, shape: [1, tokens.count])),
        "position_ids": positionScalar
            ? InferenceFunction.AsyncValue(NDArray(scalars: [Int32(position)], shape: [1, 1]))
            : InferenceFunction.AsyncValue(NDArray(scalars: (0..<(position + tokens.count)).map(Int32.init),
                                                   shape: [1, position + tokens.count])),
    ]
    let outputs = try encode(f, inputs: inputs, index: 0, views: InferenceFunction.AsyncMutableViews())
    await stream.currentWorkCompleted()
    guard let logits = try await outputs["logits"]?.ndArray else { throw Failure(description: "no logits") }
    let row: [Float]
    switch logits.scalarType {
    case .float16:
        row = logits.view(as: Float16.self).withUnsafePointer { p, shape, strides in
            let base = (shape[1] - 1) * strides[1]; return (0..<shape[2]).map { Float(p[base + $0 * strides[2]]) }
        }
    case .float32:
        row = logits.view(as: Float.self).withUnsafePointer { p, shape, strides in
            let base = (shape[1] - 1) * strides[1]; return (0..<shape[2]).map { p[base + $0 * strides[2]] }
        }
    default: throw Failure(description: "logits dtype")
    }
    if keep { dump.append(contentsOf: row) }
    var best = 0
    for i in 1..<row.count where row[i] > row[best] { best = i }
    return best
}

// Prefill.
let prefillStart = clock.now
var next: Int
if let prefillFunction {
    next = try await call(prefillFunction, tokens: prompt, position: 0, keep: dumpSteps > 0)
} else {
    next = 0
    for (i, t) in prompt.enumerated() {
        next = try await call(function, tokens: [t], position: i, keep: dumpSteps > 0 && i == prompt.count - 1)
    }
}
let prefillSeconds = seconds(prefillStart.duration(to: clock.now))

// Decode: greedy, EOS ignored, every step joined before the next (the kit's sequential contract).
var generated: [Int] = [next]
var stepSeconds: [Double] = []
for s in 0..<decodeSteps {
    let t0 = clock.now
    next = try await call(function, tokens: [Int32(next)], position: prompt.count + s, keep: s < dumpSteps - 1)
    stepSeconds.append(seconds(t0.duration(to: clock.now)))
    generated.append(next)
}

let warm = Array(stepSeconds.dropFirst(min(8, stepSeconds.count / 4))).sorted()
let median = warm.isEmpty ? 0 : warm[warm.count / 2]
let mean = warm.isEmpty ? 0 : warm.reduce(0, +) / Double(warm.count)
let p90 = warm.isEmpty ? 0 : warm[Int(Double(warm.count - 1) * 0.9)]
if let dumpPath {
    try dump.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: URL(fileURLWithPath: dumpPath))
}
let report: [String: Any] = [
    "asset": assetURL.path, "function": functionName, "prefillFunction": prefillName ?? NSNull(),
    "compute": cpu ? "cpuOnly" : "preferGPU", "promptTokens": prompt.count, "decodeSteps": decodeSteps,
    "capacity": capacity, "stateNames": descriptor.stateNames, "stateBytes": stateBytes, "vocab": vocab,
    "loadSeconds": loadSeconds, "prefillSeconds": prefillSeconds,
    "warmMedianMs": median * 1e3, "warmMeanMs": mean * 1e3, "warmP90Ms": p90 * 1e3,
    "tokPerSecMedian": median > 0 ? 1 / median : 0, "tokPerSecMean": mean > 0 ? 1 / mean : 0,
    "generated": generated, "stepMs": stepSeconds.map { ($0 * 1e5).rounded() / 100 },
    "device": AIModel.deviceArchitectureName,
]
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: outURL)
print(String(format: "%@ %@: load %.2fs prefill %.3fs decode median %.3f ms (%.1f tok/s) mean %.1f tok/s p90 %.3f ms",
             assetURL.lastPathComponent, functionName, loadSeconds, prefillSeconds, median * 1e3, 1 / median,
             1 / mean, p90 * 1e3))
