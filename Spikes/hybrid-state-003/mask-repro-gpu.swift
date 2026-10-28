// Runs the assets `mask_repro.py --save DIR` wrote, the way Kit runs a language step: GPU preferred,
// `expectFrequentReshapes`, the state (when the variant has one) as a Kit-owned MTLBuffer. Prints, per variant,
// OK (cosine >= 0.999 against the eager float32 output), WRONG, or FAIL. Development tooling only. Under the GPU lock:
//
//     xcrun swiftc -O -swift-version 5 -o .build/mask-repro-gpu Spikes/hybrid-state-003/mask-repro-gpu.swift \
//         -framework CoreAI -framework Metal
//     ~/.cache/aether-kit/with-gpu-lock.sh mask-repro-gpu .build/mask-repro-gpu DIR [--cpu]
@preconcurrency import CoreAI
import Foundation
import Metal

let arguments = CommandLine.arguments
guard arguments.count >= 2 else { fatalError("usage: mask-repro-gpu DIR [--cpu]") }
let root = URL(filePath: arguments[1])
let cpu = arguments.contains("--cpu")
let device = MTLCreateSystemDefaultDevice()!
nonisolated(unsafe) let stream = ComputeStream(commandQueue: device.makeCommandQueue()!)

func load<T>(_ url: URL, as: T.Type) throws -> [T] {
    let data = try Data(contentsOf: url)
    return data.withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
}

for variant in try FileManager.default.contentsOfDirectory(atPath: root.path).sorted() {
    let dir = root.appending(path: variant)
    guard let meta = try? String(contentsOf: dir.appending(path: "meta.txt"), encoding: .utf8) else { continue }
    let m = meta.split(separator: " ").map { Int($0.trimmingCharacters(in: .whitespacesAndNewlines))! }
    let (q, hidden, kvHeads, capacity, dim, hasState) = (m[0], m[1], m[2], m[3], m[4], m[5] == 1)
    do {
        var options: SpecializationOptions = cpu ? .cpuOnly : SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model = try await AIModel(contentsOf: dir.appending(path: "m.aimodel"), options: options)
        let function = try model.loadFunction(named: "main")!
        var inputs: [String: InferenceFunction.AsyncValue] = [
            "x": .init(NDArray(scalars: try load(dir.appending(path: "x.bin"), as: Float16.self), shape: [1, q, hidden])),
            "positions": .init(NDArray(scalars: try load(dir.appending(path: "positions.bin"), as: Int32.self), shape: [1, q])),
            "keyCache": .init(NDArray(scalars: try load(dir.appending(path: "keyCache.bin"), as: Float16.self), shape: [1, kvHeads, capacity, dim])),
            "valueCache": .init(NDArray(scalars: try load(dir.appending(path: "valueCache.bin"), as: Float16.self), shape: [1, kvHeads, capacity, dim])),
        ]
        let buffer = device.makeBuffer(length: 2 * 4 * 4, options: .storageModeShared)!
        buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        let outputs: [String: InferenceFunction.AsyncValue]
        if hasState {
            var state = InferenceFunction.AsyncMutableValue(unsafeBuffer: buffer, scalarType: .float32, shape: [2, 4], strides: [4, 1],
                                                            interleaveLayout: nil)
            var views = InferenceFunction.AsyncMutableViews()
            views.insert(&state, for: "state")
            outputs = try function.encode(inputs: inputs, states: views, to: stream)
        } else {
            inputs["state"] = .init(NDArray(scalars: [Float](repeating: 0, count: 8), shape: [2, 4]))
            outputs = try function.encode(inputs: inputs, states: InferenceFunction.AsyncMutableViews(), to: stream)
        }
        await stream.currentWorkCompleted()
        let y = try await outputs["y"]!.ndArray!
        let got: [Float] = y.view(as: Float16.self).withUnsafePointer { p, shape, strides in
            var out: [Float] = []
            for t in 0..<shape[1] { for c in 0..<shape[2] { out.append(Float(p[t * strides[1] + c * strides[2]])) } }
            return out
        }
        let ref = try load(dir.appending(path: "reference.bin"), as: Float.self)
        var ab = 0.0, aa = 0.0, bb = 0.0
        for i in 0..<ref.count { ab += Double(got[i]) * Double(ref[i]); aa += Double(got[i] * got[i]); bb += Double(ref[i] * ref[i]) }
        let cosine = ab / (aa.squareRoot() * bb.squareRoot())
        print(String(format: "%-28@ %@ (cosine %.4f)", variant as NSString, cosine >= 0.999 ? "OK" : "WRONG", cosine))
    } catch {
        print(String(format: "%-28@ FAIL (%@)", variant as NSString, "\(error)" as NSString))
    }
}
