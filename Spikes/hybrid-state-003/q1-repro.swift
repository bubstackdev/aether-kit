// Runs a q1_repro.py asset the way Kit runs a language step (GPU preferred, expectFrequentReshapes, states as owned
// MTLBuffers): submissions of q = 4, 2, then 1 from position 0, printing before and after each. On the GPU the q = 1
// call aborts with an MPSNDArrayMatrixMultiplication assertion once the stack has 4+ Mamba layers. Under the GPU lock:
//
//     xcrun swiftc -O -swift-version 5 -o .build/q1-repro Spikes/hybrid-state-003/q1-repro.swift -framework CoreAI -framework Metal
//     ~/.cache/aether-kit/with-gpu-lock.sh q1-repro .build/q1-repro <asset> [--cpu]
@preconcurrency import CoreAI
import Foundation
import Metal

setvbuf(stdout, nil, _IOLBF, 0)
let arguments = CommandLine.arguments
guard arguments.count >= 2 else { fatalError("usage: q1-repro <asset> [--cpu]") }
var options: SpecializationOptions = arguments.contains("--cpu") ? .cpuOnly : SpecializationOptions(preferredComputeUnitKind: .gpu)
options.expectFrequentReshapes = true
let device = MTLCreateSystemDefaultDevice()!
nonisolated(unsafe) let stream = ComputeStream(commandQueue: device.makeCommandQueue()!)
let model = try await AIModel(contentsOf: URL(filePath: arguments[1]), options: options)
let function = try model.loadFunction(named: "main")!
let descriptor = model.functionDescriptor(for: "main")!
print("opened")

struct Owned { let name: String; let buffer: any MTLBuffer; let descriptor: NDArrayDescriptor }
let states: [Owned] = try descriptor.stateNames.sorted().map { name in
    guard case .ndArray(let t)? = descriptor.stateDescriptor(of: name) else { fatalError("state \(name)") }
    let d = t.resolvingDynamicDimensions(t.shape)
    let b = device.makeBuffer(length: d.minimumByteCount, options: .storageModeShared)!
    b.contents().initializeMemory(as: UInt8.self, repeating: 0, count: b.length)
    return Owned(name: name, buffer: b, descriptor: d)
}

func encode(_ inputs: [String: InferenceFunction.AsyncValue], _ index: Int,
            _ views: consuming InferenceFunction.AsyncMutableViews) throws -> [String: InferenceFunction.AsyncValue] {
    if index == states.count { return try function.encode(inputs: inputs, states: views, to: stream) }
    let s = states[index]
    var value = InferenceFunction.AsyncMutableValue(unsafeBuffer: s.buffer, scalarType: s.descriptor.scalarType, shape: s.descriptor.shape,
                                                    strides: s.descriptor.preferredStrides, interleaveLayout: s.descriptor.interleaveLayout)
    var next = views
    next.insert(&value, for: s.name)
    return try encode(inputs, index + 1, next)
}

let capacity = 64
var position = 0
for q in [4, 2, 1] {
    print("q \(q) at position \(position)…")
    let inputs: [String: InferenceFunction.AsyncValue] = [
        "x": .init(NDArray(scalars: (0..<(q * 768)).map { Float16(sin(Double($0 + position)) * 0.5) }, shape: [1, q, 768])),
        "positions": .init(NDArray(scalars: (position..<(position + q)).map(Int32.init), shape: [1, q])),
        "keyCache": .init(NDArray(scalars: [Float16](repeating: 0, count: 4 * capacity * 64), shape: [1, 4, capacity, 64])),
        "valueCache": .init(NDArray(scalars: [Float16](repeating: 0, count: 4 * capacity * 64), shape: [1, 4, capacity, 64])),
    ]
    let outputs = try encode(inputs, 0, InferenceFunction.AsyncMutableViews())
    await stream.currentWorkCompleted()
    _ = try await outputs["y"]?.ndArray
    print("q \(q) done")
    position += q
}
