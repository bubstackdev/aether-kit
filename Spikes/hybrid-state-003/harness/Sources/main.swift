// Hybrid-state 0.0.3 spike harness: runs a `hybrid`-contract Granite asset (Forge/language/export_granite_hybrid.py)
// the way a Kit session would, and checks it against the fp32 CPU reference (reference_granite_hybrid.py).
//
// usage: hybrid-step <asset> <reference dir> <out.json> [--decode] [--cpu] [--chunk 64] [--rounds 5]
//                    [--phases numerics,long,ttft,decode,memory]
//
// - KV caches are Kit-owned shared MTLBuffers passed as read-only inputs, grown in 256-row steps with the rows kept
//   (Kit's bucketed growth); each step's keyUpdate/valueUpdate rows are copied in after the join.
// - convState and ssmState are Kit-owned shared MTLBuffers bound as states, zeroed per request.
// - Prompts go in the hybrid plan: `--chunk`-token submissions, then the remainder in descending powers of two,
//   never resubmitting a row. One-token steps use `decode` with --decode, otherwise `main`.
@preconcurrency import CoreAI
import Foundation
import Metal

struct Failure: Error, CustomStringConvertible { let description: String }
setvbuf(stdout, nil, _IOLBF, 0)   // progress lines reach a piped log as they happen

var args = Array(CommandLine.arguments.dropFirst())
func flag(_ name: String) -> Bool { if let i = args.firstIndex(of: name) { args.remove(at: i); return true }; return false }
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    let v = args[i + 1]; args.removeSubrange(i...(i + 1)); return v
}
let decodeOnly = flag("--decode-only")   // an asset with only `decode`: every step is one token through it
let useDecode = flag("--decode") || decodeOnly
let baseline = flag("--baseline")   // export_baseline.py: [1, 1] steps, position scalar, fixed-capacity KV states
let cpu = flag("--cpu")
let chunk = Int(option("--chunk") ?? "64")!
let rounds = Int(option("--rounds") ?? "5")!
let decodeContexts = (option("--decode-contexts") ?? "256,2048").split(separator: ",").map { Int($0)! }
// --capacity N: allocate the KV caches at N rows once (one capacity, so a run touches few shapes); default: Kit's
// 256-row growth, a new shape per capacity step.
let fixedCapacity = option("--capacity").map { Int($0)! }
// --widths 4,2,1: the `widths` phase submits these widths in order from position 0, printing before each one.
let probeWidths = (option("--widths") ?? "4,2,1,1").split(separator: ",").map { Int($0)! }
let phases = Set((option("--phases") ?? "memory,numerics,long,ttft,decode").split(separator: ",").map(String.init))
guard args.count == 3 else {
    print("usage: hybrid-step <asset> <reference dir> <out.json> [--decode] [--cpu] [--chunk N] [--rounds N] [--phases …]")
    exit(2)
}
let assetURL = URL(fileURLWithPath: args[0]), referenceURL = URL(fileURLWithPath: args[1]), outURL = URL(fileURLWithPath: args[2])
let reference = try JSONSerialization.jsonObject(with: Data(contentsOf: referenceURL.appendingPathComponent("reference.json"))) as! [String: Any]

func seconds(_ d: Duration) -> Double { Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }
let clock = ContinuousClock()

func mappedFileMiB() -> (wired: Int, clean: Int) {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent("hybrid-step-\(UUID().uuidString).json")
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

// MARK: model

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
var memory: [[String: Any]] = []
func phase(_ name: String) {
    let pages = mappedFileMiB()
    memory.append(["phase": name, "metalMiB": device.currentAllocatedSize >> 20, "wiredMiB": pages.wired, "cleanMiB": pages.clean])
    print("[memory] \(name): metal \(device.currentAllocatedSize >> 20) MiB, mapped file wired \(pages.wired) MiB, clean \(pages.clean) MiB")
}
phase("before open")
let loadStart = clock.now
var options: SpecializationOptions = cpu ? .cpuOnly : SpecializationOptions(preferredComputeUnitKind: .gpu)
options.expectFrequentReshapes = true
let model = try await AIModel(contentsOf: assetURL, options: options)
let mainFunction: InferenceFunction = try model.loadFunction(named: decodeOnly ? "decode" : "main")!
let descriptor = model.functionDescriptor(for: decodeOnly ? "decode" : "main")!
let decodeFunction: InferenceFunction? = useDecode ? try model.loadFunction(named: "decode") : nil
if useDecode && decodeFunction == nil { throw Failure(description: "no decode function") }
let loadSeconds = seconds(loadStart.duration(to: clock.now))
phase("open")
nonisolated(unsafe) let stream = ComputeStream(commandQueue: queue)

guard case .ndArray(let keyTemplate)? = baseline ? descriptor.stateDescriptor(of: "keyCache") : descriptor.inputDescriptor(of: "keyCache"),
      case .ndArray(let logitsTemplate)? = descriptor.outputDescriptor(of: "logits") else { throw Failure(description: "descriptors") }
let vocab = logitsTemplate.shape.last!
let kvLayers = keyTemplate.shape[0], kvHeads = keyTemplate.shape[2], headDim = keyTemplate.shape[4]

struct StateBuffer { let name: String; let buffer: any MTLBuffer; let descriptor: NDArrayDescriptor }
var states: [StateBuffer] = []
for name in descriptor.stateNames.sorted() {
    guard case .ndArray(let template)? = descriptor.stateDescriptor(of: name) else { throw Failure(description: "state \(name)") }
    let resolved = template.resolvingDynamicDimensions(template.shape)
    guard let buffer = device.makeBuffer(length: resolved.minimumByteCount, options: .storageModeShared) else { throw Failure(description: "alloc") }
    states.append(StateBuffer(name: name, buffer: buffer, descriptor: resolved))
}

// MARK: session

/// One request's caches: KV grown in 256-row steps, recurrent states zeroed.
final class Session {
    var capacity = 0
    var keys: (any MTLBuffer)?
    var values: (any MTLBuffer)?
    var position = 0
    var growSeconds = 0.0
    init() { for s in states { s.buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: s.buffer.length) } }

    func grow(to needed: Int) throws {
        guard needed > capacity, !baseline else { return }
        let start = clock.now
        if let fixedCapacity, needed > fixedCapacity { throw Failure(description: "needs \(needed) rows, capacity is \(fixedCapacity)") }
        let newCapacity = fixedCapacity ?? (needed + 255) / 256 * 256
        let rows = kvLayers * kvHeads, rowBytes = headDim * 2
        func relayout(_ old: (any MTLBuffer)?) throws -> any MTLBuffer {
            guard let new = device.makeBuffer(length: rows * newCapacity * rowBytes, options: .storageModeShared) else { throw Failure(description: "alloc kv") }
            new.contents().initializeMemory(as: UInt8.self, repeating: 0, count: new.length)
            if let old {
                for r in 0..<rows {
                    memcpy(new.contents() + r * newCapacity * rowBytes, old.contents() + r * capacity * rowBytes, position * rowBytes)
                }
            }
            return new
        }
        keys = try relayout(keys); values = try relayout(values); capacity = newCapacity
        growSeconds += seconds(start.duration(to: clock.now))
    }

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

    /// Baseline: one token, KV as states.
    func baselineStep(_ token: Int32) async throws -> [Float] {
        let inputs: [String: InferenceFunction.AsyncValue] = [
            "input_ids": .init(NDArray(scalars: [token], shape: [1, 1])),
            "position_ids": .init(NDArray(scalars: [Int32(position)], shape: [1, 1])),
        ]
        let outputs = try encode(mainFunction, inputs: inputs, index: 0, views: InferenceFunction.AsyncMutableViews())
        await stream.currentWorkCompleted()
        guard let logits = try await outputs["logits"]?.ndArray else { throw Failure(description: "no logits") }
        position += 1
        return logits.view(as: Float16.self).withUnsafePointer { p, s, st in
            (0..<s[2]).map { Float(p[(s[1] - 1) * st[1] + $0 * st[2]]) }
        }
    }

    /// One joined submission at the current position; returns the last row's logits.
    func step(_ tokens: [Int32]) async throws -> [Float] {
        if baseline {
            var last: [Float] = []
            for t in tokens { last = try await baselineStep(t) }
            return last
        }
        try grow(to: position + tokens.count)
        let shape = [kvLayers, 1, kvHeads, capacity, headDim]
        let strides = [kvHeads * capacity * headDim, kvHeads * capacity * headDim, capacity * headDim, headDim, 1]
        let inputs: [String: InferenceFunction.AsyncValue] = [
            "input_ids": .init(NDArray(scalars: tokens, shape: [1, tokens.count])),
            "cache_positions": .init(NDArray(scalars: (position..<(position + tokens.count)).map(Int32.init), shape: [1, tokens.count])),
            "keyCache": .init(unsafeBuffer: keys!, scalarType: .float16, shape: shape, strides: strides, interleaveLayout: nil),
            "valueCache": .init(unsafeBuffer: values!, scalarType: .float16, shape: shape, strides: strides, interleaveLayout: nil),
        ]
        let f = tokens.count == 1 && (position > 0 || decodeOnly) ? (decodeFunction ?? mainFunction) : mainFunction
        let outputs = try encode(f, inputs: inputs, index: 0, views: InferenceFunction.AsyncMutableViews())
        await stream.currentWorkCompleted()
        for (name, buffer) in [("keyUpdate", keys!), ("valueUpdate", values!)] {
            guard let update = try await outputs[name]?.ndArray else { throw Failure(description: "no \(name)") }
            let cache = buffer.contents().bindMemory(to: Float16.self, capacity: buffer.length / 2)
            update.view(as: Float16.self).withUnsafePointer { p, s, st in
                for r in 0..<(s[0] * s[2]) {
                    let (l, h) = (r / s[2], r % s[2])
                    for t in 0..<s[3] {
                        let src = l * st[0] + h * st[2] + t * st[3], dst = (r * capacity + position + t) * headDim
                        for d in 0..<s[4] { cache[dst + d] = p[src + d * st[4]] }
                    }
                }
            }
        }
        guard let logits = try await outputs["logits"]?.ndArray else { throw Failure(description: "no logits") }
        position += tokens.count
        return logits.view(as: Float16.self).withUnsafePointer { p, s, st in
            (0..<s[2]).map { Float(p[(s[1] - 1) * st[1] + $0 * st[2]]) }
        }
    }

    func prefill(_ tokens: [Int32], every: ((Int, [Float]) -> Void)? = nil) async throws -> [Float] {
        var last: [Float] = []
        for width in plan(tokens.count) {
            last = try await step(Array(tokens[position..<(position + width)]))
            every?(position - 1, last)
        }
        return last
    }
}

func plan(_ count: Int) -> [Int] {
    var widths = Array(repeating: chunk, count: count / chunk), rest = count % chunk, bit = chunk / 2
    while rest > 0 { if rest >= bit { widths.append(bit); rest -= bit }; bit /= 2 }
    return widths
}

// MARK: metrics

func argmax(_ v: [Float]) -> Int { var b = 0; for i in 1..<v.count where v[i] > v[b] { b = i }; return b }
func cosine(_ a: [Float], _ b: [Float]) -> Double {
    var ab = 0.0, aa = 0.0, bb = 0.0
    for i in 0..<a.count { ab += Double(a[i]) * Double(b[i]); aa += Double(a[i]) * Double(a[i]); bb += Double(b[i]) * Double(b[i]) }
    return ab / (aa.squareRoot() * bb.squareRoot() + 1e-30)
}
func probabilities(_ v: [Float]) -> [Double] {
    let m = Double(v.max()!); let e = v.map { exp(Double($0) - m) }; let s = e.reduce(0, +); return e.map { $0 / s }
}
/// Kit's near tie: top-1 minus top-2 probability of `judge` below 0.1 and `other`'s token in its top 5.
func nearTie(judge: [Float], token: Int) -> (Bool, Double) {
    let p = probabilities(judge)
    let order = p.indices.sorted { p[$0] > p[$1] }
    let margin = p[order[0]] - p[order[1]]
    return (margin < 0.1 && order.prefix(5).contains(token), margin)
}
func readRows(_ file: String, rows: Int) throws -> [[Float]] {
    let data = try Data(contentsOf: referenceURL.appendingPathComponent(file))
    let all = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    return (0..<rows).map { Array(all[($0 * vocab)..<(($0 + 1) * vocab)]) }
}
func ints(_ any: Any?) -> [Int32] { (any as! [Any]).map { Int32(($0 as! NSNumber).intValue) } }
func median(_ v: [Double]) -> Double { let s = v.sorted(); return s.isEmpty ? 0 : s[s.count / 2] }

var report: [String: Any] = ["asset": assetURL.lastPathComponent, "decodeFunction": useDecode, "compute": cpu ? "cpuOnly" : "preferGPU",
                             "chunk": chunk, "loadSeconds": loadSeconds, "device": AIModel.deviceArchitectureName,
                             "stateNames": states.map(\.name)]

// MARK: submission widths, one by one

if phases.contains("widths") {
    let s = Session()
    var results: [[String: Any]] = []
    for width in probeWidths {
        let function = width == 1 && s.position > 0 && useDecode ? "decode" : "main"
        print("[widths] q \(width) at position \(s.position) via \(function)…")
        let t0 = clock.now
        let logits = try await s.step((0..<width).map { Int32(100 + s.position + $0) })
        let ms = seconds(t0.duration(to: clock.now)) * 1e3
        let finite = logits.allSatisfy { $0.isFinite }
        print(String(format: "[widths] q %d done in %.1f ms, logits finite %@", width, ms, finite ? "yes" : "no"))
        results.append(["width": width, "function": function, "ms": ms, "finite": finite])
    }
    report["widths"] = results
}

// MARK: cold specialization per shape

if phases.contains("compile") {
    // First use of each shape (Core AI specializes it), then the same shape again, warm.
    let tokens = ints((reference["long"] as! [String: Any])["tokens"])
    var compile: [String: Any] = [:]
    func timed(_ body: () async throws -> Void) async rethrows -> Double {
        let t0 = clock.now; try await body(); return seconds(t0.duration(to: clock.now)) * 1e3
    }
    let cold = Session()
    let chunkCold = try await timed { _ = try await cold.step(Array(tokens.prefix(chunk))) }
    let stepCold = try await timed { _ = try await cold.step([tokens[chunk]]) }
    let warm = Session()
    let chunkWarm = try await timed { _ = try await warm.step(Array(tokens.prefix(chunk))) }
    let stepWarm = try await timed { _ = try await warm.step([tokens[chunk]]) }
    compile = ["capacity": cold.capacity, "chunkColdMs": chunkCold, "chunkWarmMs": chunkWarm, "stepColdMs": stepCold,
               "stepWarmMs": stepWarm, "stepFunction": useDecode ? "decode" : "main"]
    report["compile"] = compile
    print(String(format: "[compile] capacity %d: q %d cold %.0f ms (warm %.1f), q 1 (%@) cold %.0f ms (warm %.1f)", cold.capacity,
                 chunk, chunkCold, chunkWarm, useDecode ? "decode" : "main", stepCold, stepWarm))
}

if phases.contains("memory") {
    let s = Session()
    _ = try await s.step([100, 101, 102, 103]); phase("after main q 4")
    _ = try await s.step([104]); phase("after first one-token step")
}

// MARK: numerics (chat cases)

if phases.contains("numerics") {
    var cases: [[String: Any]] = []
    for c in reference["cases"] as! [[String: Any]] {
        let name = c["name"] as! String
        let prompt = ints(c["promptTokens"]), forced = ints(c["forcedTokens"]), greedyRef = ints(c["greedyTokens"])
        let refRows = try readRows((c["forcedLogits"] as! [String: Any])["file"] as! String, rows: forced.count)
        // Teacher-forced rows.
        let s = Session()
        var rows = [try await s.prefill(prompt)]
        for t in forced.dropLast() { rows.append(try await s.step([t])) }
        var minCos = 1.0, mismatches = 0, accepted = 0
        for (r, row) in rows.enumerated() {
            minCos = min(minCos, cosine(row, refRows[r]))
            if argmax(row) != argmax(refRows[r]) {
                mismatches += 1
                if nearTie(judge: refRows[r], token: argmax(row)).0 { accepted += 1 }
            }
        }
        // Greedy continuation.
        let g = Session()
        var logits = try await g.prefill(prompt)
        var generated: [Int32] = []
        var divergence: [String: Any] = [:]
        for i in 0..<greedyRef.count {
            let token = Int32(argmax(logits))
            if divergence.isEmpty && token != greedyRef[i] {
                let (tie, margin) = nearTie(judge: logits, token: Int(greedyRef[i]))
                divergence = ["index": i, "nativeMargin": margin, "acceptedNearTie": tie]
            }
            generated.append(token)
            if i + 1 < greedyRef.count { logits = try await g.step([token]) }
        }
        let forcedPassed = minCos >= 0.995 && mismatches == accepted
        let greedyPassed = divergence.isEmpty || (divergence["acceptedNearTie"] as! Bool)
        cases.append(["name": name, "promptTokens": prompt.count, "forcedRows": rows.count, "minCosine": minCos,
                      "argmaxMismatches": mismatches, "acceptedNearTies": accepted, "forcedPassed": forcedPassed,
                      "greedyTokens": generated.count, "greedyIdentical": divergence.isEmpty,
                      "greedyDivergence": divergence.isEmpty ? NSNull() : divergence, "greedyPassed": greedyPassed,
                      "generated": generated.map(Int.init)])
        print("[numerics] \(name): min cos \(String(format: "%.6f", minCos)), mismatches \(mismatches) (near ties \(accepted)), greedy \(divergence.isEmpty ? "identical" : "\(divergence)")")
    }
    report["numerics"] = cases
}

// MARK: long-context drift

if phases.contains("long") {
    let long = reference["long"] as! [String: Any]
    let tokens = ints(long["tokens"]), rows = (long["rows"] as! [NSNumber]).map(\.intValue)
    let refRows = try readRows(long["file"] as! String, rows: rows.count)
    let want = Dictionary(uniqueKeysWithValues: rows.enumerated().map { ($1, $0) })
    var got: [Int: [Float]] = [:]
    let s = Session()
    let start = clock.now
    _ = try await s.prefill(tokens) { row, logits in if want[row] != nil { got[row] = logits } }
    let elapsed = seconds(start.duration(to: clock.now))
    var drift: [[String: Any]] = []
    for (i, r) in rows.enumerated() {
        guard let row = got[r] else { continue }
        let (tie, _) = nearTie(judge: refRows[i], token: argmax(row))
        drift.append(["row": r, "cosine": cosine(row, refRows[i]), "argmaxEqual": argmax(row) == argmax(refRows[i]), "nearTie": tie])
    }
    report["long"] = ["tokens": tokens.count, "prefillSeconds": elapsed, "rows": drift]
    print("[long] \(tokens.count) tokens in \(String(format: "%.2f", elapsed)) s; cosines \(drift.map { String(format: "%.5f", $0["cosine"] as! Double) }.joined(separator: " "))")
}

// MARK: time to first token

if phases.contains("ttft") {
    let tokens = ints((reference["long"] as! [String: Any])["tokens"])
    var ttft: [String: Any] = [:]
    for length in [64, 256, 1024, 2048] {
        var times: [Double] = [], grow: [Double] = []
        for _ in 0..<(rounds + 1) {
            let s = Session()
            let t0 = clock.now
            _ = try await s.prefill(Array(tokens.prefix(length)))
            times.append(seconds(t0.duration(to: clock.now))); grow.append(s.growSeconds)
        }
        let warm = Array(times.dropFirst())
        ttft[String(length)] = ["medianMs": median(warm) * 1e3, "minMs": warm.min()! * 1e3, "coldMs": times[0] * 1e3,
                                "growMedianMs": median(Array(grow.dropFirst())) * 1e3, "submissions": plan(length).count]
        print(String(format: "[ttft] %5d tokens: warm median %.1f ms (min %.1f, cold %.1f), %d submissions", length,
                     median(warm) * 1e3, warm.min()! * 1e3, times[0] * 1e3, plan(length).count))
    }
    report["timeToFirstToken"] = ttft
}

// MARK: decode rate

if phases.contains("decode") {
    let tokens = ints((reference["long"] as! [String: Any])["tokens"])
    var decode: [String: Any] = [:]
    for context in decodeContexts {
        var rates: [Double] = []
        for _ in 0..<max(2, rounds / 2) {
            let s = Session()
            try s.grow(to: context + 257)                   // no growth inside the timed steps
            var logits = try await s.prefill(Array(tokens.prefix(context)))
            var steps: [Double] = []
            for _ in 0..<256 {
                let t0 = clock.now
                logits = try await s.step([Int32(argmax(logits))])
                steps.append(seconds(t0.duration(to: clock.now)))
            }
            rates.append(1 / median(Array(steps.dropFirst(16))))
        }
        decode[String(context)] = ["tokPerSecMedian": median(rates), "rounds": rates]
        print(String(format: "[decode] after %d tokens: %.1f tok/s (rounds %@)", context, median(rates),
                     rates.map { String(format: "%.1f", $0) }.joined(separator: " ")))
    }
    report["decode"] = decode
}

phase("end")
report["memory"] = memory
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: outURL)
