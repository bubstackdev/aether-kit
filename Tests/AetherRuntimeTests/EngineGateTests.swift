import Foundation
import Testing
@_spi(Diagnostics) @testable import AetherRuntime
import AetherTestSupport

/// DecoderEngine gates on real bundles (DESIGN.md §14, M2). Part of the `Gates` suite: serialized, `AETHER_GATES=1`.
/// Every pipelined result is compared with the same device's `run()` path (the invariant is "pipelined == same-device
/// run()"), and with the lab's references, which were recorded on this Mac.
extension GateTests {
    static var lab: URL { models.deletingLastPathComponent() }

    struct TokenFile: Decodable { var tokens: [Int32] }
    struct RunFile: Decodable { var promptTokens: [Int32]; var generated: [Int32] }

    static func labJSON<T: Decodable>(_ type: T.Type, _ path: String) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(contentsOf: lab.appending(path: "engine-spike/results/\(path)")))
    }

    static func reference(_ name: String) throws -> RunFile {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "References/\(name)")
        return try JSONDecoder().decode(RunFile.self, from: Data(contentsOf: url))
    }

    static func engine(_ id: String, _ options: DecoderEngine.Options) async throws -> DecoderEngine {
        try await DecoderEngine(bundle: try labBundle(id), manifest: try Fixtures.manifest(id), device: mac, options: options)
    }

    static func runOptions(capacity: Int) -> DecoderEngine.Options {
        DecoderEngine.Options(capacity: capacity, initialCapacity: capacity, execution: .run)
    }

    static func pipelinedOptions(capacity: Int) -> DecoderEngine.Options {
        DecoderEngine.Options(capacity: capacity, initialCapacity: capacity)
    }

    /// Greedy: prefill, the first token from the prompt's last row, then `count - 1` more in the engine's mode.
    /// No stop tokens (fixed-length runs, as the lab references).
    static func greedy(_ engine: DecoderEngine, prompt: [Int32], count: Int) async throws -> (tokens: [Int32], seconds: Double) {
        try await engine.reset()
        let logits = try await engine.prefill(prompt)!
        var out = [Logits.argmax(logits.last)]
        let clock = ContinuousClock()
        let start = clock.now
        try await engine.beginGreedy(first: out[0], limit: count - 1)
        while let t = try await engine.nextGreedy() { out.append(t) }
        try await engine.endGreedy()
        return (out, (clock.now - start) / .seconds(1))
    }

    static func firstMismatch(_ a: [Int32], _ b: [Int32]) -> Int? {
        zip(a, b).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? (a.count == b.count ? nil : min(a.count, b.count))
    }

    // MARK: - Greedy parity, 3 chat models

    @Test(arguments: [("qwen3-0.6b", "parity/qwen3-0.6b-spike.json"), ("qwen3.5-0.8b", "parity/qwen3.5-0.8b-spike.json"),
                      ("lfm2.5-1.2b", "")])
    func engineGreedyParity(_ id: String, _ labFile: String) async throws {
        let ref = labFile.isEmpty ? try Self.reference("\(id)-spike.json") : try Self.labJSON(RunFile.self, labFile)
        let run = try await Self.engine(id, Self.runOptions(capacity: 2048))
        let (runTokens, _) = try await Self.greedy(run, prompt: ref.promptTokens, count: ref.generated.count)
        #expect(Self.firstMismatch(runTokens, ref.generated) == nil, "run() vs lab spike")

        let pipelined = try await Self.engine(id, Self.pipelinedOptions(capacity: 4096))
        let mode = await pipelined.decodeMode
        let canary = await pipelined.canary
        let (tokens, seconds) = try await Self.greedy(pipelined, prompt: ref.promptTokens, count: ref.generated.count)
        print("GATE parity \(id): \(tokens.count) tokens, mode \(mode), canary \(String(describing: canary)), "
              + "\(String(format: "%.0f", Double(tokens.count - 1) / seconds)) tok/s, "
              + "vs run() \(Self.firstMismatch(tokens, runTokens).map { "mismatch at \($0)" } ?? "equal"), "
              + "vs lab \(Self.firstMismatch(tokens, ref.generated).map { "mismatch at \($0)" } ?? "equal")")
        #expect(mode == .gpu(depth: 3))
        #expect(canary?.passed == true)
        #expect(Self.firstMismatch(tokens, runTokens) == nil, "pipelined vs same-device run()")
        #expect(Self.firstMismatch(tokens, ref.generated) == nil, "pipelined vs lab spike")
    }

    // MARK: - Depth 3 == run() over ≥ 1000 tokens at capacity 4096

    @Test(arguments: [("qwen3-0.6b", "qwen3"), ("qwen3.5-0.8b", "qwen3.5")])
    func depth3MatchesRunOver1200Tokens(_ id: String, _ short: String) async throws {
        let prompt = try Self.labJSON(TokenFile.self, "perf/depthN-exact/count-prompt-\(short).json").tokens
        let lab = try Self.labJSON(RunFile.self, "perf/depthN-exact/\(short)-count-run-4096.json")
        let run = try await Self.engine(id, Self.runOptions(capacity: 4096))
        let (runTokens, runSeconds) = try await Self.greedy(run, prompt: prompt, count: 1200)
        let pipelined = try await Self.engine(id, Self.pipelinedOptions(capacity: 4096))
        #expect(await pipelined.decodeMode == .gpu(depth: 3))
        let (tokens, seconds) = try await Self.greedy(pipelined, prompt: prompt, count: 1200)
        print("GATE depth3-vs-run \(id) cap 4096: run() \(String(format: "%.0f", 1199 / runSeconds)) tok/s, "
              + "depth 3 \(String(format: "%.0f", 1199 / seconds)) tok/s, "
              + "\(Self.firstMismatch(tokens, runTokens).map { "MISMATCH at \($0)" } ?? "1200/1200 equal"), "
              + "lab run() \(Self.firstMismatch(runTokens, lab.generated).map { "mismatch at \($0)" } ?? "equal")")
        #expect(Self.firstMismatch(tokens, runTokens) == nil)
        #expect(Self.firstMismatch(runTokens, lab.generated) == nil)
    }

    // MARK: - Chunked prefill parity at ~3k tokens

    @Test func chunkedPrefillMatchesTokenByTokenAt3k() async throws {
        let prompt = try Self.labJSON(TokenFile.self, "perf/depthN-exact/long-prompt-qwen3.json").tokens
        let lab = try Self.labJSON(RunFile.self, "perf/depthN-exact/qwen3-long-d3-tokens.json")
        let engine = try await Self.engine("qwen3-0.6b", Self.pipelinedOptions(capacity: 4096))
        let clock = ContinuousClock()
        var t0 = clock.now
        let chunked = try #require(try await engine.prefill(prompt))
        let chunkedSeconds = (clock.now - t0) / .seconds(1)
        try await engine.reset()
        t0 = clock.now
        var stepwise: Logits?
        for t in prompt { stepwise = try await engine.step(t) }
        let stepSeconds = (clock.now - t0) / .seconds(1)
        let a = chunked.last, b = try #require(stepwise).last
        let delta = zip(a, b).map { abs($0 - $1) }.max() ?? 0
        let peak = a.map(abs).max() ?? 1
        // 8 fp16 ulps at the row's peak magnitude (the lab's tolerance for equivalent paths).
        let tolerance = 8 * Float(Float16(peak).ulp)
        print("GATE chunked prefill qwen3 \(prompt.count) tok: 512-chunks \(String(format: "%.3f", chunkedSeconds)) s, "
              + "token-by-token \(String(format: "%.2f", stepSeconds)) s, last row max|Δ| \(delta) (tol \(tolerance))")
        #expect(Logits.argmax(a) == Logits.argmax(b))
        #expect(delta <= tolerance)
        // The continuation from the chunked prefill equals the lab's exact 1024-token output.
        let (tokens, seconds) = try await Self.greedy(engine, prompt: prompt, count: lab.generated.count)
        print("GATE long 3k→4k qwen3 depth 3: \(String(format: "%.0f", Double(tokens.count - 1) / seconds)) tok/s, "
              + "\(Self.firstMismatch(tokens, lab.generated).map { "MISMATCH at \($0)" } ?? "equal to lab")")
        #expect(Self.firstMismatch(tokens, lab.generated) == nil)
    }

    // MARK: - LFM3 pf64 through the prefill phase

    static var lfm3: URL { models.appending(path: "lfm3-1.7b") }
    static let lfm3Subtree = "gpu-pipelined/lfm3_1_7b_decode_int8hu_block32_sym_pf64"

    /// LFM3 is not in the catalog yet: a hand-written manifest over the lab bundle (MANIFEST §1.1 lists it).
    static func lfm3Manifest() -> Manifest {
        Manifest(
            id: "lfm3-1.7b", task: "text-generation", driver: .decoderLM, variant: .init(platform: .macOS),
            assets: ["main": .init(subtree: lfm3Subtree)],
            graphs: ["decoder": Manifest.Graph(
                file: .single(AssetPath(asset: "main", path: "lfm3_1_7b_decode_int8hu_block32_sym_pf64.aimodel")),
                functions: ["step": "main", "prefill": "prefill"],
                inputs: ["tokens": "input_ids", "positions": "position_ids"], outputs: ["logits": "logits"],
                states: Manifest.States(
                    sequence: [Manifest.SequenceState(tensors: ["keyCache", "valueCache"], seqAxis: 3)],
                    recurrent: [Manifest.RecurrentState(name: "convState")]),
                limits: Manifest.Limits(maxContext: 4096, staticSequence: .perPhase(["step": .one(1), "prefill": .one(64)])),
                compute: .gpu, expectFrequentReshapes: true)],
            tokenizer: .init(source: .dir(AssetPath(asset: "main", path: "tokenizer")), stop: ["<|im_end|>"]))
    }

    @Test(.enabled(if: FileManager.default.fileExists(
        atPath: GateTests.lfm3.appending(path: GateTests.lfm3Subtree).path)))
    func lfm3PrefillPhase() async throws {
        let ref = ModelRef(
            catalogID: "lfm3-1.7b", repo: "mlboydaisuke/LFM3-1.7B-CoreAI", revision: "8621dc4", variant: "macos",
            driver: .decoderLM, assets: [.init(name: "main", repo: "mlboydaisuke/LFM3-1.7B-CoreAI", revision: "8621dc4",
                                               subtree: Self.lfm3Subtree, files: [])])
        let bundle = InstalledBundle(ref: ref, url: Self.lfm3, isStale: false)
        let engine = try await DecoderEngine(
            bundle: bundle, manifest: Self.lfm3Manifest(), device: Self.mac, options: Self.pipelinedOptions(capacity: 4096))
        #expect(engine.description.prefillChunk == 64)
        let prompt = try JSONDecoder().decode(TokenFile.self, from: Data(contentsOf: URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().appending(path: "References/lfm3-1.7b-prefill-prompt.json"))).tokens
        let clock = ContinuousClock()
        var t0 = clock.now
        let chunked = try #require(try await engine.prefill(prompt))
        let chunkedSeconds = (clock.now - t0) / .seconds(1)
        let continuation = try await Self.greedy(engine, prompt: prompt, count: 32).tokens
        // Reference: the same engine fed one token per call on the step phase.
        try await engine.reset()
        t0 = clock.now
        var stepwise: Logits?
        for t in prompt { stepwise = try await engine.step(t) }
        let stepSeconds = (clock.now - t0) / .seconds(1)
        let delta = zip(chunked.last, try #require(stepwise).last).map { abs($0 - $1) }.max() ?? 0
        let tolerance = 8 * Float(Float16(chunked.last.map(abs).max() ?? 1).ulp)  // the lab's rule
        let first = Logits.argmax(stepwise!.last)
        try await engine.beginGreedy(first: first, limit: 31)
        var stepContinuation = [first]
        while let t = try await engine.nextGreedy() { stepContinuation.append(t) }
        try await engine.endGreedy()
        print("GATE lfm3 pf64 \(prompt.count) tok: prefill phase \(String(format: "%.0f", Double(prompt.count) / chunkedSeconds)) tok/s, "
              + "S=1 \(String(format: "%.0f", Double(prompt.count) / stepSeconds)) tok/s, last row max|Δ| \(delta) (tol \(tolerance)), "
              + "continuation \(continuation == stepContinuation ? "equal" : "DIFFERS")")
        #expect(delta <= tolerance)
        #expect(Logits.argmax(chunked.last) == first)
        #expect(continuation == stepContinuation)
    }

    // MARK: - Canary and speed

    @Test(arguments: ["qwen3-0.6b", "qwen3.5-0.8b", "lfm2.5-1.2b"])
    func canaryPassesAtCapacity4096(_ id: String) async throws {
        let engine = try await Self.engine(id, DecoderEngine.Options(capacity: 4096))
        let report = try #require(await engine.canary)
        print("GATE canary \(id): \(report)")
        #expect(report.passed && report.capacity >= 4096 && report.steps >= 60)
    }

    /// Reported, not asserted (DESIGN §14 target: ≥ 480 tok/s on an M5 Max).
    @Test func decodeSpeedQwen3Depth3() async throws {
        let prompt = try Self.labJSON(TokenFile.self, "perf/depthN-exact/count-prompt-qwen3.json").tokens
        let engine = try await Self.engine("qwen3-0.6b", Self.pipelinedOptions(capacity: 2048))
        let run = try await Self.engine("qwen3-0.6b", Self.runOptions(capacity: 2048))
        var rates: [Double] = [], runRates: [Double] = []
        for _ in 0..<3 {
            let (tokens, seconds) = try await Self.greedy(engine, prompt: prompt, count: 1200)
            rates.append(Double(tokens.count - 1) / seconds)
        }
        for _ in 0..<3 {
            let (tokens, seconds) = try await Self.greedy(run, prompt: prompt, count: 400)
            runRates.append(Double(tokens.count - 1) / seconds)
        }
        let t = await engine.greedyTimings
        let ms = { (x: Double) in String(format: "%.3f", 1000 * x / Double(max(t.steps, 1))) }
        print("GATE speed qwen3-0.6b host ms/step over \(t.steps) steps: bindings \(ms(t.bindings)), encode \(ms(t.encode)), "
              + "argmax commit \(ms(t.commit)), waiting for tokens \(ms(t.wait))")
        print("GATE speed qwen3-0.6b depth 3, 1200 tokens, cap 2048: \(rates.map { String(format: "%.0f", $0) }) tok/s; "
              + "run(): \(runRates.map { String(format: "%.0f", $0) }) tok/s (target ≥ 480, not asserted)")
    }
}

// MARK: - Core AI shape-change retention

extension GateTests {
    /// The process's `phys_footprint` (TASK_VM_INFO), the figure iOS enforces its limit against.
    static func physFootprint() -> Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Int64(info.phys_footprint) : -1
    }

    /// The earlier AetherKit 0.0.7 measured that Core AI keeps Metal memory on every input-shape change of a function
    /// specialized with `expectFrequentReshapes`, until the model is released (_reference/aether-kit-remote
    /// docs/language-generation.md:188-200, 325-336). Every decode step here binds a new `position_ids` extent.
    /// Records the device's Metal allocations across 2400 steps; reported, and asserted only against runaway growth.
    @Test(arguments: ["qwen3-0.6b", "qwen3.5-0.8b"])
    func metalRetentionOverLongDecode(_ id: String) async throws {
        let prompt = try Self.labJSON(TokenFile.self, "perf/depthN-exact/count-prompt-\(id == "qwen3-0.6b" ? "qwen3" : "qwen3.5").json").tokens
        let beforeLoad = DecoderEngine.metalAllocatedBytes()
        let engine = try await Self.engine(id, Self.pipelinedOptions(capacity: 4096))
        let afterLoad = DecoderEngine.metalAllocatedBytes()
        _ = try await Self.greedy(engine, prompt: prompt, count: 64)  // warm
        let start = DecoderEngine.metalAllocatedBytes()
        var marks: [Int64] = []
        for _ in 0..<3 {
            _ = try await Self.greedy(engine, prompt: prompt, count: 800)
            marks.append(DecoderEngine.metalAllocatedBytes())
        }
        let mib = { (b: Int64) in String(format: "%.0f", Double(b) / 1_048_576) }
        let footprint = Self.physFootprint()
        print("GATE metal retention \(id): load \(mib(beforeLoad)) → \(mib(afterLoad)) MiB (with canary and KV), after warm-up \(mib(start)) MiB; after each 800-step request "
              + "\(marks.map(mib)) MiB; growth \(mib(marks.last! - start)) MiB over 2400 steps "
              + "(model \(await engine.loadedModelBytes)); phys_footprint now \(mib(footprint)) MiB")
        #expect(marks.last! - start < 2 << 30, "Metal allocations grew by more than 2 GiB")
    }
}

extension GateTests {
    /// The canary's negative control: every pipelined step reads a stale token (the argmax writes elsewhere), as an
    /// unordered read would. The canary must fail and the engine must fall back to depth 1.
    @Test func canaryCatchesAStaleHandOff() async throws {
        var options = DecoderEngine.Options(capacity: 4096)
        options.staleTokenHandoff = true
        let engine = try await Self.engine("qwen3-0.6b", options)
        let report = try #require(await engine.canary)
        print("GATE canary negative control (stale hand-off) qwen3-0.6b: \(report), mode \(await engine.decodeMode)")
        #expect(!report.passed && report.firstMismatch != nil)
        #expect(await engine.decodeMode == .gpu(depth: 1))
    }

    /// The lab's racing hand-off (a view into the page-aligned token history, `--handoff offset`), which still races
    /// in the lab spike on macOS 27.2 26B123. Reported, not asserted: through Aether's engine it has not raced so far.
    @Test func canaryOnTheLabsRacingHandOff() async throws {
        var options = DecoderEngine.Options(capacity: 4096)
        options.racingTokenHandoff = true
        let engine = try await Self.engine("qwen3-0.6b", options)
        let report = try #require(await engine.canary)
        print("GATE canary probe (lab racing history hand-off) qwen3-0.6b: \(report), mode \(await engine.decodeMode)")
    }
}
