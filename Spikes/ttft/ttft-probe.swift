// Splits long-prompt prefill time on Core AI between submission count, per-submission cost and graph shape, for a
// bucketed and a running export of the same Qwen3 checkpoint. Development tooling only. Under the GPU lock:
//
//     xcrun swiftc -O -swift-version 5 -o .build/ttft-probe Spikes/ttft/ttft-probe.swift -framework CoreAI -framework Metal
//     ~/.cache/aether-kit/with-gpu-lock.sh ttft-probe .build/ttft-probe <mode> <bucketed .aimodel> <running .aimodel>
//
// Modes:
//   prefill  whole prefills at 1,024 and 2,048 tokens, C = 64 and 512, both contracts, interleaved rounds; per prefill
//            the encode and join time, the process CPU time inside encode, and (bucketed) host growth and row copies.
//   micro    steady-state single submissions (same shape repeated) at q = 1, 64, 512 and context 256 / 2,048.
//   views    bucketed q = 64 and 512 at capacity 2,048 with and without preallocated output views.
//   spin     bucketed q = 64 at capacity 2,048 in a loop for `sample` (prints the pid, runs about 20 s).
//   spinrun  the same for the running contract (q = 64 at 2,048 rows).
@preconcurrency import CoreAI
import Darwin
import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
// LAYERS overrides the layer count (random-weight spike exports at Qwen3-0.6B dimensions).
let layers = Int(ProcessInfo.processInfo.environment["LAYERS"] ?? "") ?? 28, kvHeads = 8, headDim = 128
let rowBytes = headDim * 2

func ms(_ d: Duration) -> Double { Double(d.components.seconds) * 1e3 + Double(d.components.attoseconds) / 1e15 }
func cpuMs() -> Double {
    var u = rusage(); getrusage(RUSAGE_SELF, &u)
    return Double(u.ru_utime.tv_sec + u.ru_stime.tv_sec) * 1e3 + Double(u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e3
}
func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count / 2] }

final class Cache {
    var keys: MTLBuffer, values: MTLBuffer, capacity: Int
    init(capacity: Int) {
        self.capacity = capacity
        keys = device.makeBuffer(length: layers * kvHeads * capacity * rowBytes, options: .storageModeShared)!
        values = device.makeBuffer(length: layers * kvHeads * capacity * rowBytes, options: .storageModeShared)!
        memset(keys.contents(), 0, keys.length); memset(values.contents(), 0, values.length)
    }
    var shape: [Int] { [layers, 1, kvHeads, capacity, headDim] }
    /// Kit's bucketed growth: a fresh buffer, prefix rows copied, the rest zero.
    func grow(to next: Int, valid: Int) {
        for which in 0..<2 {
            let old = which == 0 ? keys : values
            let new = device.makeBuffer(length: layers * kvHeads * next * rowBytes, options: .storageModeShared)!
            let src = old.contents(), dst = new.contents()
            DispatchQueue.concurrentPerform(iterations: layers * kvHeads) { r in
                memcpy(dst + r * next * rowBytes, src + r * capacity * rowBytes, valid * rowBytes)
                memset(dst + r * next * rowBytes + valid * rowBytes, 0, (next - valid) * rowBytes)
            }
            if which == 0 { keys = new } else { values = new }
        }
        capacity = next
    }
    func value(_ buffer: MTLBuffer) -> InferenceFunction.AsyncValue {
        InferenceFunction.AsyncValue(unsafeBuffer: buffer, scalarType: .float16, shape: shape)
    }
    func mutable(_ buffer: MTLBuffer) -> InferenceFunction.AsyncMutableValue {
        InferenceFunction.AsyncMutableValue(unsafeBuffer: buffer, scalarType: .float16, shape: shape)
    }
}

struct StepTime { var encode = 0.0, join = 0.0, encodeCPU = 0.0, host = 0.0, total = 0.0, argmax = -1 }

func tokens(_ start: Int, _ count: Int) -> [Int32] { (start..<(start + count)).map { Int32(1000 + ($0 * 37) % 5000) } }

func lastArgmax(_ outputs: [String: InferenceFunction.AsyncValue]) async throws -> Int {
    let logits = try await outputs["logits"]!.ndArray!
    return logits.view(as: Float16.self).withUnsafePointer { p, shape, strides in
        var best = 0; var top = -Float.infinity
        let base = (shape[1] - 1) * strides[1]
        for i in 0..<shape[2] { let v = Float(p[base + i * strides[2]]); if v > top { top = v; best = i } }
        return best
    }
}

/// One submission. Bucketed: cache rows are inputs, update rows copied back after the join. Running: states.
func submit(_ function: InferenceFunction, bucketed: Bool, cache: Cache, start: Int, count: Int, stream: ComputeStream,
            outputViews: (MTLBuffer, MTLBuffer)? = nil, readLogits: Bool = true, keyPositions: Bool = false) async throws -> StepTime {
    var t = StepTime()
    let ids = NDArray(scalars: tokens(start, count), shape: [1, count])
    var inputs: [String: InferenceFunction.AsyncValue] = ["input_ids": .init(ids)]
    let c0 = cpuMs(); let t0 = ContinuousClock.now
    let outputs: [String: InferenceFunction.AsyncValue]
    if bucketed {
        inputs["cache_positions"] = .init(NDArray(scalars: (start..<(start + count)).map(Int32.init), shape: [1, count]))
        if keyPositions {
            inputs["key_positions"] = .init(NDArray(scalars: (0..<cache.capacity).map(Int32.init), shape: [1, cache.capacity]))
        }
        inputs["keyCache"] = cache.value(cache.keys)
        inputs["valueCache"] = cache.value(cache.values)
        if let (ku, vu) = outputViews {
            var k = InferenceFunction.AsyncMutableValue(unsafeBuffer: ku, scalarType: .float16, shape: [layers, 1, kvHeads, count, headDim])
            var v = InferenceFunction.AsyncMutableValue(unsafeBuffer: vu, scalarType: .float16, shape: [layers, 1, kvHeads, count, headDim])
            var views = InferenceFunction.AsyncMutableViews()
            views.insert(&k, for: "keyUpdate"); views.insert(&v, for: "valueUpdate")
            outputs = try function.encode(inputs: inputs, outputViews: views, to: stream)
        } else {
            outputs = try function.encode(inputs: inputs, to: stream)
        }
    } else {
        inputs["position_ids"] = .init(NDArray(scalars: (0..<(start + count)).map(Int32.init), shape: [1, start + count]))
        var k = cache.mutable(cache.keys), v = cache.mutable(cache.values)
        var states = InferenceFunction.AsyncMutableViews()
        states.insert(&k, for: "keyCache"); states.insert(&v, for: "valueCache")
        outputs = try function.encode(inputs: inputs, states: states, to: stream)
    }
    let t1 = ContinuousClock.now; let c1 = cpuMs()
    await stream.currentWorkCompleted()
    let t2 = ContinuousClock.now
    t.encode = ms(t1 - t0); t.encodeCPU = c1 - c0; t.join = ms(t2 - t1)
    if bucketed && outputViews == nil {
        for (name, buffer) in [("keyUpdate", cache.keys), ("valueUpdate", cache.values)] {
            let update = try await outputs[name]!.ndArray!
            let dst = buffer.contents(); let cap = cache.capacity
            update.view(as: Float16.self).withUnsafePointer { p, shape, strides in
                precondition(strides[4] == 1 && strides[3] == headDim)
                let rowStride = strides[2]
                let src = UnsafeRawPointer(p)
                DispatchQueue.concurrentPerform(iterations: layers * kvHeads) { r in
                    let l = r / kvHeads, h = r % kvHeads
                    memcpy(dst + (r * cap + start) * rowBytes, src + (l * strides[0] + h * rowStride) * 2, count * rowBytes)
                }
            }
        }
    }
    if readLogits { t.argmax = try await lastArgmax(outputs) }
    t.host = ms(ContinuousClock.now - t2)
    t.total = ms(ContinuousClock.now - t0)
    return t
}

struct PrefillTime { var total = 0.0, encode = 0.0, join = 0.0, encodeCPU = 0.0, host = 0.0, growth = 0.0, submissions = 0, argmax = -1 }

func prefill(_ function: InferenceFunction, bucketed: Bool, length: Int, chunk: Int, stream: ComputeStream,
             runningCache: Cache, keyPositions: Bool = false) async throws -> PrefillTime {
    var p = PrefillTime()
    let t0 = ContinuousClock.now
    // Bucketed requests start again at 256 rows (Kit); running sessions keep their grown cache.
    let cache = bucketed ? Cache(capacity: 256) : runningCache
    var start = 0
    while start < length {
        let n = min(chunk, length - start)
        if bucketed && start + n > cache.capacity {
            let g = ContinuousClock.now
            cache.grow(to: (start + n + 255) / 256 * 256, valid: start)
            p.growth += ms(ContinuousClock.now - g)
        }
        let s = try await submit(function, bucketed: bucketed, cache: cache, start: start, count: n, stream: stream,
                                 readLogits: start + n == length, keyPositions: keyPositions)
        p.encode += s.encode; p.join += s.join; p.encodeCPU += s.encodeCPU; p.host += s.host; p.submissions += 1
        p.argmax = s.argmax
        start += n
    }
    p.total = ms(ContinuousClock.now - t0)
    return p
}

let args = CommandLine.arguments
if args.count >= 3 && args[1] == "compare" {
    // compare <kind>=<asset> ...: kinds bucketed, keypos, running. Steady-state submissions at q = 1, 64, 512 with
    // 2,048 rows of context, then whole prefills at 2,048 tokens (C = 64 and 512), interleaved rounds.
    var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
    options.expectFrequentReshapes = true
    var entries: [(kind: String, function: InferenceFunction, model: AIModel)] = []
    for spec in args.dropFirst(2) {
        let parts = spec.split(separator: "=", maxSplits: 1).map(String.init)
        let model = try await AIModel(contentsOf: URL(filePath: parts[1]), options: options)
        entries.append((parts[0], try model.loadFunction(named: "main")!, model))
    }
    nonisolated(unsafe) let stream = ComputeStream(commandQueue: queue)
    let runningCache = Cache(capacity: 4096)
    func one(_ e: (kind: String, function: InferenceFunction, model: AIModel), q: Int) async throws -> StepTime {
        let bucketed = e.kind != "running"
        let cache = bucketed ? Cache(capacity: 2048) : runningCache
        return try await submit(e.function, bucketed: bucketed, cache: cache, start: 2048 - q, count: q, stream: stream,
                                keyPositions: e.kind == "keypos")
    }
    var micro: [String: [StepTime]] = [:]
    for e in entries { for q in [1, 64, 512] { for _ in 0..<3 { _ = try await one(e, q: q) } } }
    for round in 0..<9 {
        for e in round % 2 == 0 ? entries : entries.reversed() {
            for q in [1, 64, 512] { micro["\(e.kind) q=\(q)", default: []].append(try await one(e, q: q)) }
        }
    }
    for key in micro.keys.sorted() {
        let s = micro[key]!
        print("micro \(key) ctx=2048: total \(String(format: "%.2f", median(s.map(\.total)))) encode \(String(format: "%.2f", median(s.map(\.encode)))) encodeCPU \(String(format: "%.2f", median(s.map(\.encodeCPU)))) join \(String(format: "%.2f", median(s.map(\.join)))) argmax \(s.last!.argmax)")
    }
    var pre: [String: [PrefillTime]] = [:]
    for e in entries { for c in [64, 512] { _ = try await prefill(e.function, bucketed: e.kind != "running", length: 2048, chunk: c, stream: stream, runningCache: runningCache, keyPositions: e.kind == "keypos") } }
    for round in 0..<5 {
        for e in round % 2 == 0 ? entries : entries.reversed() {
            for c in [64, 512] {
                pre["\(e.kind) C=\(c)", default: []].append(try await prefill(e.function, bucketed: e.kind != "running", length: 2048, chunk: c,
                    stream: stream, runningCache: runningCache, keyPositions: e.kind == "keypos"))
            }
        }
    }
    for key in pre.keys.sorted() {
        let s = pre[key]!
        let m = s.sorted { $0.total < $1.total }[s.count / 2]
        print("prefill \(key) N=2048: total \(String(format: "%.1f", m.total)) encode \(String(format: "%.1f", m.encode)) encodeCPU \(String(format: "%.1f", m.encodeCPU)) join \(String(format: "%.1f", m.join)) growth \(String(format: "%.1f", m.growth)) host \(String(format: "%.1f", m.host)) argmax \(m.argmax)")
    }
    exit(0)
}
if args.count == 3 && args[1] == "retain" {
    // retain <bucketed asset>: Metal growth per warm request (prefill + 8 decode steps) at several prefill widths, with
    // Kit's 256-row growth ladder ("ladder") or with the whole request's capacity allocated up front ("fixed"), then
    // the cost and effect of reloading only the function, then the whole model.
    var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
    options.expectFrequentReshapes = true
    func mib() -> Int { device.currentAllocatedSize >> 20 }
    var t = ContinuousClock.now
    var model: AIModel? = try await AIModel(contentsOf: URL(filePath: args[2]), options: options)
    print("model open \(String(format: "%.0f", ms(ContinuousClock.now - t))) ms, metal \(mib()) MiB")
    t = ContinuousClock.now
    var function: InferenceFunction? = try model!.loadFunction(named: "main")!
    print("function load \(String(format: "%.0f", ms(ContinuousClock.now - t))) ms, metal \(mib()) MiB")
    nonisolated(unsafe) var stream = ComputeStream(commandQueue: queue)
    func request(_ length: Int, _ chunk: Int, fixed: Bool = false) async throws -> (ms: Double, prefill: Double, argmax: Int) {
        let t0 = ContinuousClock.now
        let cache = Cache(capacity: fixed ? (length + 8 + 255) / 256 * 256 : 256)
        var start = 0, argmax = -1, prefill = 0.0
        while start < length + 8 {
            let n = start < length ? min(chunk, length - start) : 1
            if start + n > cache.capacity { cache.grow(to: (start + n + 255) / 256 * 256, valid: start) }
            let s = try await submit(function!, bucketed: true, cache: cache, start: start, count: n, stream: stream,
                                     readLogits: start + n >= length)
            if start + n == length { argmax = s.argmax; prefill = ms(ContinuousClock.now - t0) }
            start += n
        }
        return (ms(ContinuousClock.now - t0), prefill, argmax)
    }
    // RETAIN_CONFIGS=<length:width:ladder|fixed,...>, RETAIN_REQUESTS=<per config, default 8>.
    let configs: [(Int, Int, Bool)] = ProcessInfo.processInfo.environment["RETAIN_CONFIGS"].map {
        $0.split(separator: ",").map { item in
            let f = item.split(separator: ":"); return (Int(f[0])!, Int(f[1])!, f[2] == "fixed")
        }
    } ?? [(2048, 64, false), (2048, 512, false), (2048, 64, true), (2048, 256, true), (2048, 512, true),
          (1024, 64, false), (1024, 512, false), (1024, 512, true), (449, 64, false), (449, 512, false)]
    let requests = Int(ProcessInfo.processInfo.environment["RETAIN_REQUESTS"] ?? "") ?? 8
    for (n, c, f) in configs { _ = try await request(n, c, fixed: f); _ = try await request(n, c, fixed: f) }
    print("warm: metal \(mib()) MiB")
    for (n, c, f) in configs {
        let before = mib()
        var times: [Double] = [], prefills: [Double] = [], argmax = -1
        for _ in 0..<requests { let r = try await request(n, c, fixed: f); times.append(r.ms); prefills.append(r.prefill); argmax = r.argmax }
        print("N=\(n) C=\(c) \(f ? "fixed" : "ladder"): +\(String(format: "%.1f", Double(mib() - before) / Double(requests))) MiB per request over \(requests) (metal \(mib()) MiB), median prefill \(String(format: "%.0f", median(prefills))) ms, request \(String(format: "%.0f", median(times))) ms, argmax \(argmax)")
    }
    let beforeReload = mib()
    t = ContinuousClock.now
    function = nil
    stream = ComputeStream(commandQueue: queue)
    print("released function and stream: metal \(mib()) MiB (was \(beforeReload))")
    function = try model!.loadFunction(named: "main")!
    let loadMs = ms(ContinuousClock.now - t)
    let first = try await request(2048, 512)
    print("function reload \(String(format: "%.0f", loadMs)) ms, then first N=2048 C=512 request \(String(format: "%.0f", first.ms)) ms, metal \(mib()) MiB")
    function = nil; model = nil
    stream = ComputeStream(commandQueue: queue)
    print("released model: metal \(mib()) MiB")
    t = ContinuousClock.now
    model = try await AIModel(contentsOf: URL(filePath: args[2]), options: options)
    function = try model!.loadFunction(named: "main")!
    let reopen = ms(ContinuousClock.now - t)
    let afterReopen = try await request(2048, 512)
    let again = try await request(2048, 512)
    print("model reopen \(String(format: "%.0f", reopen)) ms, then first N=2048 C=512 request \(String(format: "%.0f", afterReopen.ms)) ms, second \(String(format: "%.0f", again.ms)) ms, metal \(mib()) MiB")
    exit(0)
}
guard args.count == 4 else { fatalError("usage: ttft-probe <prefill|micro|views|spin> <bucketed .aimodel> <running .aimodel>") }
let mode = args[1]
var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
options.expectFrequentReshapes = true
let bucketedModel = try await AIModel(contentsOf: URL(filePath: args[2]), options: options)
let bucketedMain = try bucketedModel.loadFunction(named: "main")!
let runningModel = mode == "prefill" || mode == "micro" || mode == "spinrun" ? try await AIModel(contentsOf: URL(filePath: args[3]), options: options) : nil
let runningMain = try runningModel?.loadFunction(named: "main")
nonisolated(unsafe) let stream = ComputeStream(commandQueue: queue)
print("opened; metal \(device.currentAllocatedSize >> 20) MiB")

func fmt(_ x: Double) -> String { String(format: "%.1f", x) }

switch mode {
case "prefill":
    let runningCache = Cache(capacity: 4096)
    let cases: [(String, Bool, Int, Int)] = [("bucketed", true, 64, 0), ("bucketed", true, 512, 0), ("running", false, 64, 0), ("running", false, 512, 0)]
    let lengths = [1024, 2048]
    for (_, bucketed, chunk, _) in cases {      // warm every shape once
        for length in lengths {
            _ = try await prefill(bucketed ? bucketedMain : runningMain!, bucketed: bucketed, length: length, chunk: chunk,
                                  stream: stream, runningCache: runningCache)
        }
    }
    var samples: [String: [PrefillTime]] = [:]
    for round in 0..<5 {
        for (label, bucketed, chunk, _) in round % 2 == 0 ? cases : cases.reversed() {
            for length in lengths {
                let p = try await prefill(bucketed ? bucketedMain : runningMain!, bucketed: bucketed, length: length, chunk: chunk,
                                          stream: stream, runningCache: runningCache)
                samples["\(label) C=\(chunk) N=\(length)", default: []].append(p)
            }
        }
    }
    for key in samples.keys.sorted() {
        let s = samples[key]!
        let m = s.sorted { $0.total < $1.total }[s.count / 2]
        print("\(key): total \(fmt(m.total)) ms (median), subs \(m.submissions), encode \(fmt(m.encode)), encodeCPU \(fmt(m.encodeCPU)), join \(fmt(m.join)), host \(fmt(m.host)), growth \(fmt(m.growth)), argmax \(m.argmax); all totals \(s.map { fmt($0.total) })")
    }
case "micro":
    for (label, bucketed) in [("bucketed", true), ("running", false)] {
        let function = bucketed ? bucketedMain : runningMain!
        for context in [256, 2048] {
            let cache = Cache(capacity: bucketed ? context : 4096)
            for q in [1, 64, 512] {
                let start = bucketed ? max(0, context - q) : max(0, context - q)
                for _ in 0..<3 { _ = try await submit(function, bucketed: bucketed, cache: cache, start: start, count: q, stream: stream) }
                var e: [Double] = [], j: [Double] = [], c: [Double] = [], h: [Double] = [], tot: [Double] = []
                for _ in 0..<9 {
                    let s = try await submit(function, bucketed: bucketed, cache: cache, start: start, count: q, stream: stream)
                    e.append(s.encode); j.append(s.join); c.append(s.encodeCPU); h.append(s.host); tot.append(s.total)
                }
                print("\(label) q=\(q) context=\(context): total \(fmt(median(tot))) encode \(fmt(median(e))) encodeCPU \(fmt(median(c))) join \(fmt(median(j))) host \(fmt(median(h)))")
            }
        }
    }
case "views":
    let cache = Cache(capacity: 2048)
    for q in [64, 512] {
        let bytes = layers * kvHeads * q * rowBytes
        let ku = device.makeBuffer(length: bytes, options: .storageModeShared)!, vu = device.makeBuffer(length: bytes, options: .storageModeShared)!
        for round in 0..<2 {
            for withViews in [false, true] {
                var tot: [Double] = [], e: [Double] = [], c: [Double] = []
                for i in 0..<12 {
                    let s = try await submit(bucketedMain, bucketed: true, cache: cache, start: 2048 - q, count: q, stream: stream,
                                             outputViews: withViews ? (ku, vu) : nil)
                    if i >= 3 { tot.append(s.total); e.append(s.encode); c.append(s.encodeCPU) }
                }
                print("round \(round) q=\(q) outputViews=\(withViews): total \(fmt(median(tot))) encode \(fmt(median(e))) encodeCPU \(fmt(median(c)))")
            }
        }
    }
case "spin":
    print("pid \(getpid())"); fflush(stdout)
    let cache = Cache(capacity: 2048)
    let end = ContinuousClock.now + .seconds(20)
    var n = 0
    while ContinuousClock.now < end {
        _ = try await submit(bucketedMain, bucketed: true, cache: cache, start: 1984, count: 64, stream: stream, readLogits: false)
        n += 1
    }
    print("spin: \(n) submissions in 20 s")
case "spinrun":
    print("pid \(getpid())"); fflush(stdout)
    let cache = Cache(capacity: 4096)
    let end = ContinuousClock.now + .seconds(20)
    var n = 0
    while ContinuousClock.now < end {
        _ = try await submit(runningMain!, bucketed: false, cache: cache, start: 1984, count: 64, stream: stream, readLogits: false)
        n += 1
    }
    print("spinrun: \(n) submissions in 20 s")
default:
    fatalError("unknown mode")
}
