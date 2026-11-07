import AetherCore
import AetherImage
import AetherRuntime
import AetherSpeech
import Darwin
import Foundation

/// The on-device gates (docs/DEVICE.md): SD-Turbo, Kokoro and the M4 models (`VisionGates.swift`) on a physical
/// iPhone, against the same CPU references and bars as the Mac gates (`LatentDiffusionGateTests`, `KokoroGateTests`,
/// the M4 `*GateTests`). `run-on-device.sh` copies the bundles and references into Documents and launches with
/// `--run-id` and `--gates`; the report goes to Documents/gate.json and each SD-Turbo and Kokoro output to
/// Documents/out/ (raw little-endian float32) for the Mac-versus-device comparison.
struct Report: Codable {
    struct Check: Codable { var name: String; var passed: Bool; var detail: String }
    var runID: String
    var device: String
    var architecture: String
    var availableAtLaunchMB: Int
    var gates: [String] = []
    var checks: [Check] = []
    var measurements: [String: Double] = [:]
    /// `ComputeUnitKind.availableKinds` on this device.
    var computeUnitKinds: [String]?
    /// Model id → SHA-256 of the bundle's SHA256SUMS as copied to the device: which export was measured.
    var bundles: [String: String] = [:]
    /// The M4 placement measurement: each model under each compute unit. Not checks: the Neural Engine run never
    /// fails the gate.
    var placement: [Placement] = []
    /// Set when every requested gate has reported.
    var finished = false
    var passed: Bool { checks.allSatisfy(\.passed) }

    struct Placement: Codable {
        var model: String
        var compute: String
        var status = "not run"
        /// Within the Mac gate's bars on every parity case.
        var withinBars: Bool?
        var failed: [String] = []
        var parity: [String] = []
        /// Load, the first request's specialization (or cache read) and the parity cases.
        var parityAndLoadSeconds: Double?
        var iterations: Int?
        var loopSeconds: Double?
        var medianMS: Double?
        var p90MS: Double?
        var minMS: Double?
        /// Per inference, over the sustained run: the CPU energy billed to the app (`task_energy`), the GPU time billed
        /// to it (`task_gpu_utilisation`) and its CPU time.
        var cpuEnergyMJPerRun: Double?
        var gpuMSPerRun: Double?
        var cpuMSPerRun: Double?
        var ledgerPeakMB: Double?
        /// The app's physical footprint after the run, less before the load (the model is still loaded).
        var footprintDeltaMB: Double?
        /// The process's peak so far (monotonic over the run).
        var physFootprintPeakMB: Double?
        var thermalStart: String?
        var thermalEnd: String?
    }
}

@MainActor
enum Gate {
    static let documents = URL.documentsDirectory
    static var report = Report(runID: "manual", device: "", architecture: "", availableAtLaunchMB: 0)
    static var log: (String) async -> Void = { _ in }

    static func run(log: @escaping (String) async -> Void) async -> Report {
        self.log = log
        let args = CommandLine.arguments
        let runID = args.firstIndex(of: "--run-id").flatMap { args.indices.contains($0 + 1) ? args[$0 + 1] : nil } ?? "manual"
        var name = utsname()
        uname(&name)
        let machine = withUnsafeBytes(of: name.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        report = Report(runID: runID, device: machine, architecture: Device.current.architecture,
                        availableAtLaunchMB: Int(os_proc_available_memory() >> 20))
        let gates = args.firstIndex(of: "--gates").flatMap { args.indices.contains($0 + 1) ? args[$0 + 1] : nil }
            .map { $0.split(separator: ",").map(String.init) } ?? ["sd-turbo", "kokoro", "m4"]
        report.gates = gates
        try? FileManager.default.createDirectory(at: documents.appending(path: "out"), withIntermediateDirectories: true)
        await log("run \(runID) on \(machine) (\(Device.current.architecture)), \(report.availableAtLaunchMB) MB available, "
                  + "gates \(gates)")
        if gates.contains("sd-turbo") { await step("sd-turbo", sdTurbo) }
        if gates.contains("kokoro") { await step("kokoro", kokoro) }
        if gates.contains("m4") { await step("m4", m4) }
        report.finished = true
        save()
        return report
    }

    static func step(_ name: String, _ body: () async throws -> Void) async {
        do { try await body() } catch { check("\(name).completed", false, "\(error)") }
        save()
    }

    static func check(_ name: String, _ passed: Bool, _ detail: String) {
        report.checks.append(.init(name: name, passed: passed, detail: detail))
        Task { await log("\(passed ? "PASS" : "FAIL") \(name): \(detail)") }
    }

    static func measure(_ name: String, _ value: Double) { report.measurements[name] = value }

    static func save() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: documents.appending(path: "gate.json"), options: .atomic)
    }

    static func availableMB() -> Double { Double(os_proc_available_memory()) / 1_048_576 }

    static func localBundle(_ url: URL) throws -> InstalledBundle {
        let manifest = try Manifest.load(fromBundle: url)
        let ref = ModelRef(
            catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: manifest.variant.platform.rawValue,
            driver: manifest.driver, assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
        return InstalledBundle(ref: ref, url: url, isStale: false)
    }

    static func writeF32(_ values: [Float], _ name: String) {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        try? data.write(to: documents.appending(path: "out/\(name).f32"))
    }

    static func readF32(_ url: URL) throws -> [Float] {
        try Data(contentsOf: url).withUnsafeBytes { raw in
            (0..<(raw.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
    }

    // MARK: SD-Turbo (bars: LatentDiffusionGateTests)

    struct SDReference: Decodable {
        struct Case: Decodable { var id: String; var prompt: String; var seed: UInt64; var steps: Int; var ids: [Int32] }
        var cases: [Case]
    }

    static func sdTurbo() async throws {
        let bundleURL = documents.appending(path: "models/sd-turbo")
        let referenceURL = documents.appending(path: "references/sd-turbo")
        let reference = try JSONDecoder().decode(SDReference.self, from: Data(contentsOf: referenceURL.appending(path: "reference.json")))
        let before = availableMB()
        let started = ContinuousClock.now
        let pipeline = try await Registry.image.load(
            try localBundle(bundleURL), as: LatentDiffusionDriver.self,
            options: LoadOptions(device: .current, cachePolicy: .default, cache: .default)).pipeline()
        measure("sd-turbo.loadSeconds", seconds(ContinuousClock.now - started))
        measure("sd-turbo.budgetMB", Double((pipeline.options.memory ?? .imageGeneration).bytes(on: .current) >> 20))
        for (index, c) in reference.cases.enumerated() {
            let t0 = ContinuousClock.now
            let image = try await pipeline.generate(.init(prompt: c.prompt, seed: c.seed, steps: c.steps))
            let first = seconds(ContinuousClock.now - t0)
            let expected = try readNPY(referenceURL.appending(path: "\(c.id)/image.npy"))
            let rmse = rootMeanSquare(image.planarRGB, expected)
            check("sd-turbo.\(c.id).rmse", rmse <= 0.03, "RMSE \(rmse) (bar 0.03), PSNR \(10 * log10(4 / (rmse * rmse))) dB")
            writeF32(image.planarRGB, "sd-turbo-\(c.id)")
            try? image.pngData().write(to: documents.appending(path: "out/sd-turbo-\(c.id).png"))
            let t1 = ContinuousClock.now
            let repeated = try await pipeline.generate(.init(prompt: c.prompt, seed: c.seed, steps: c.steps))
            measure("sd-turbo.\(c.id).\(index == 0 ? "coldSeconds" : "firstSeconds")", first)
            measure("sd-turbo.\(c.id).warmSeconds", seconds(ContinuousClock.now - t1))
            check("sd-turbo.\(c.id).repeat", repeated == image, "a repeat generation is bit-identical")
            measure("sd-turbo.\(c.id).availableAfterMB", availableMB())
        }
        for (key, footprint) in pipeline.options.ledger.snapshot {
            measure("sd-turbo.\(key).peakMB", Double(footprint.peakBytes >> 20))
            measure("sd-turbo.\(key).heldMB", Double(footprint.heldBytes >> 20))
        }
        measure("sd-turbo.availableBeforeMB", before)
        measure("sd-turbo.physFootprintPeakMB", Double(physFootprintPeak() >> 20))
        await pipeline.unload()
        await log("sd-turbo done")
    }

    // MARK: Kokoro (bars: KokoroGateTests; the frame cap is the manifest's iOS cap)

    struct KokoroIndex: Decodable { struct Entry: Decodable { var name: String }; var cases: [Entry] }
    struct KokoroCase: Decodable {
        struct Floor: Decodable { var rel_l2: Double }
        var name: String
        var voice: String
        var speed: Float
        var seed: UInt64
        var phonemes: String
        var input_ids: [Int32]
        var frames: [Int]
        var float64_upstream: Floor
    }

    static func kokoro() async throws {
        let bundleURL = documents.appending(path: "models/kokoro-82m")
        let referenceURL = documents.appending(path: "references/kokoro")
        let index = try JSONDecoder().decode(KokoroIndex.self, from: Data(contentsOf: referenceURL.appending(path: "references.json")))
        let started = ContinuousClock.now
        let session = try await Registry.speech.load(
            try localBundle(bundleURL), as: KokoroDriver.self, options: LoadOptions(device: .current, cachePolicy: .default, cache: .default)).session()
        measure("kokoro.loadSeconds", seconds(ContinuousClock.now - started))
        check("kokoro.frameCap", session.frameCap == 512, "iOS frame cap \(session.frameCap) (manifest: 512)")
        for entry in index.cases {
            let directory = referenceURL.appending(path: entry.name)
            let c = try JSONDecoder().decode(KokoroCase.self, from: Data(contentsOf: directory.appending(path: "case.json")))
            let phonemes = try session.phonemes(c.phonemes)
            check("kokoro.\(c.name).ids", phonemes.tokenIDs == c.input_ids, "token ids equal upstream's")
            let request = SpeechRequest(voice: try KokoroVoice(c.voice), speed: c.speed, seed: c.seed)
            let total = c.frames.reduce(0, +)
            let t0 = ContinuousClock.now
            do {
                let speech = try await session.synthesize(phonemes, request)
                measure("kokoro.\(c.name).seconds", seconds(ContinuousClock.now - t0))
                check("kokoro.\(c.name).frames", speech.frames == c.frames, "\(total) frames → bucket \(speech.bucket)")
                if total > session.frameCap {
                    check("kokoro.\(c.name).cap", false, "\(total) frames exceed the cap but synthesis was not refused")
                    continue
                }
                let expected = try readF32(directory.appending(path: "audio.f32"))
                guard speech.audio.samples.count == expected.count else {
                    check("kokoro.\(c.name).length", false, "\(speech.audio.samples.count) samples, expected \(expected.count)")
                    continue
                }
                let bound = min(0.05, 4 * c.float64_upstream.rel_l2)
                let value = KokoroMetrics.compare(speech.audio.samples, expected)
                check("kokoro.\(c.name).relL2", value.rel_l2 <= bound, "rel L2 \(value.rel_l2) (bound \(bound))")
                check("kokoro.\(c.name).logMel", value.log_mel_mean_abs <= 0.03, "log-mel mean \(value.log_mel_mean_abs) (bound 0.03)")
                writeF32(speech.audio.samples, "kokoro-\(c.name)")
                let repeated = try await session.synthesize(phonemes, request)
                check("kokoro.\(c.name).repeat", repeated.audio == speech.audio, "a repeat synthesis is bit-identical")
            } catch let error as AetherError {
                let refusedByCap = total > session.frameCap
                if refusedByCap, case .generation(.invalidParameters(let message)) = error, message.contains("frame cap") {
                    check("kokoro.\(c.name).cap", true, "\(total) frames refused above the \(session.frameCap)-frame cap: \(message)")
                } else {
                    check("kokoro.\(c.name).completed", false, "\(error)")
                }
            }
        }
        for (key, footprint) in session.options.ledger.snapshot {
            measure("kokoro.\(key).peakMB", Double(footprint.peakBytes >> 20))
            measure("kokoro.\(key).heldMB", Double(footprint.heldBytes >> 20))
        }
        measure("kokoro.physFootprintPeakMB", Double(physFootprintPeak() >> 20))
        await session.unload()
        await log("kokoro done")
    }

    // MARK: helpers

    static func seconds(_ d: Duration) -> Double {
        Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }

    static func rootMeanSquare(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return .infinity }
        var sum = 0.0
        for (x, y) in zip(a, b) { sum += Double(x - y) * Double(x - y) }
        return (sum / Double(a.count)).squareRoot()
    }

    /// A little-endian float32 `.npy` (format 1.0), flattened.
    static func readNPY(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count > 10, data.prefix(6) == Data([0x93] + Array("NUMPY".utf8)) else { throw CocoaError(.fileReadCorruptFile) }
        let headerLength = Int(data[8]) | Int(data[9]) << 8
        let body = data.dropFirst(10 + headerLength)
        return body.withUnsafeBytes { raw in
            (0..<(raw.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
    }

    /// The process's peak physical footprint (`TASK_VM_INFO`'s `ledger_phys_footprint_peak`): what jetsam is enforced
    /// against, and unlike the Metal counter it sees mapped weights the GPU touches.
    static func physFootprintPeak() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? UInt64(info.ledger_phys_footprint_peak) : 0
    }
}
