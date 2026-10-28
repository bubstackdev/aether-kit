import AetherFlux2
import AetherFlux2CoreAI
import AetherFlux2Tokenizers
import AetherKit
import CoreAI
import Darwin
import Foundation
import Metal

@available(macOS 27, iOS 27, *)
struct Flux2BenchmarkPreparedCase: Codable, Sendable {
    let id: String
    let fixtureCase: String
    let width: Int
    let height: Int
    let steps: Int
    let seed: UInt64
    let references: Int
    let tokenIDsMatchFixture: Bool
    let noiseSHA256: String
}

@available(macOS 27, iOS 27, *)
struct Flux2BenchmarkSetup: Codable, Sendable {
    let caseID: String
    let referencesPreparedSeconds: Double
    let excludedFromIterationTimings: Bool
}

@available(macOS 27, iOS 27, *)
struct Flux2BenchmarkReport: Encodable {
    let schema = 1
    let purpose = "Opt-in FLUX.2 [klein] generation benchmark"
    let requestedMode: String
    let startedAt = ISO8601DateFormatter().string(from: Date())
    var stage = "preparation"
    var passed = false
    var benchmarkExecuted = false
    var error: String?
    var plan: Flux2BenchmarkPlan?
    var planPath: String?
    var planSHA256: String?
    var executablePath: String?
    var executableSHA256: String?
    var sourceRoot: String?
    var sourceSHA256: [String: String] = [:]
    var installedToolchain: String?
    var device: [String: String] = [:]
    var bundlePath: String?
    var bundleFingerprint: String?
    var fixtures: [String: String] = [:]
    var modelStreamsVerified = false
    var provenanceFilesStableAfterRun = false
    var memoryBeforeIterations: BenchmarkMemory?
    var memoryAfterUnload: BenchmarkMemory?
    var preparedCases: [Flux2BenchmarkPreparedCase] = []
    var setups: [Flux2BenchmarkSetup] = []
    var iterations: [BenchmarkIteration] = []
    /// First measured iteration of each case against the fixture's float32 Diffusers image.
    var ssimAgainstFixture: [String: Double] = [:]
    var statistics: [String: [String: BenchmarkStatistics]] = [:]
    let computePolicy = "preferGPU"
    let memorySampleIntervalMilliseconds = 50
    let cachePolicy = "No OS cache eviction; one generation session per case, reused across its warmups and measurements, then unloaded"
    var timingScope = "Application-stage wall clock, not GPU kernel timing. Encoded: conditioning includes balanced cleanup and text-encoder load, call and discard; references include the VAE encoder's load, calls and release. denoiserSetup includes segment acquisition and validation, which is quick because Core AI opens models lazily: the first native call after a (re)open specializes the model, so in balanced mode step0 carries the denoisers' reload cost on every iteration. Steps include native calls, host copies and Euler updates; decode includes decoder loading when needed, copies and image validation. Stage timings exclude observers (hashing, order checks); total includes them. Preflight, stream verification, one-time setup, SSIM, report IO and unload are outside timed generation."
    let modelVerificationScope = "Model streams hashed before iterations; postflight rechecks source, executable, plan, fixtures and asset declarations. Files must remain immutable during execution."
    let memoryScope = "Per-iteration maximum observed physical footprint, resident bytes and default Metal device currentAllocatedSize, sampled every 50 ms plus stage boundaries. Counters overlap and must not be summed; brief peaks may be missed."
    let correctness = "Every iteration of a case must reproduce its first iteration's pixels exactly. Provided-profile cases must reach the plan's SSIM against the fixture image on the first measured iteration; encoded cases record it (the whole pipeline is not gated in parity either)."
    let hostConditions = "Quiet-window availability must be arranged by the operator; this command does not establish system idleness or control other workloads"
}

@available(macOS 27, iOS 27, *)
final class Flux2BenchmarkReportBox {
    var value: Flux2BenchmarkReport
    init(_ value: Flux2BenchmarkReport) { self.value = value }
}

@available(macOS 27, iOS 27, *)
enum Flux2BenchmarkCommand {
    static let usage = "use benchmark-flux2 BUNDLE T2I_FIXTURE REFERENCE_FIXTURE|- PLAN.json SOURCE_ROOT OUTPUT_DIRECTORY --prepare-only|--run"

    static func run(_ arguments: [String]) async throws -> Bool {
        guard arguments.first == "benchmark-flux2" else { return false }
        guard arguments.count == 8, ["--prepare-only", "--run"].contains(arguments[7]) else {
            throw AetherError.invalidRequest(usage)
        }
        let execute = arguments[7] == "--run"
        #if DEBUG
        guard !execute else { throw AetherError.invalidRequest("benchmark measurement requires a Release build") }
        #endif
        let output = URL(fileURLWithPath: arguments[6], isDirectory: true).standardizedFileURL
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard mkdir(output.path, 0o755) == 0 else {
            throw AetherError.invalidRequest("benchmark output must be a new directory (mkdir errno \(errno))")
        }
        // One report, written after every step; a class so the per-case loop and `write` share it without overlapping
        // inout access.
        let box = Flux2BenchmarkReportBox(Flux2BenchmarkReport(requestedMode: execute ? "run" : "prepare-only"))
        func write() throws {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(box.value).write(to: output.appendingPathComponent("report.json"), options: .atomic)
        }
        try write()
        do {
            let planURL = URL(fileURLWithPath: arguments[4]).standardizedFileURL.resolvingSymlinksInPath()
            let planData = try Data(contentsOf: planURL)
            let plan = try JSONDecoder().decode(Flux2BenchmarkPlan.self, from: planData)
            try plan.validate()
            box.value.plan = plan
            if plan.profile == .provided {
                box.value.timingScope = "Provided: the fixture's text rows replace prompt encoding, and references are prepared once per case outside iteration timing (recorded under setups). Conditioning measures only the provided-tensor handoff. " + box.value.timingScope
            }
            box.value.planPath = planURL.path
            box.value.planSHA256 = benchmarkHash(planData)
            let source = URL(fileURLWithPath: arguments[5], isDirectory: true).standardizedFileURL.resolvingSymlinksInPath()
            box.value.sourceRoot = source.path
            box.value.sourceSHA256 = try KreaBenchmarkCommand.sourceIdentities(source)
            let executable = (Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])).resolvingSymlinksInPath()
            box.value.executablePath = executable.path
            box.value.executableSHA256 = try benchmarkFileHash(executable)
            box.value.installedToolchain = try KreaBenchmarkCommand.toolchain()
            box.value.device = ["os": ProcessInfo.processInfo.operatingSystemVersionString,
                "architecture": AIModel.deviceArchitectureName,
                "physicalMemoryBytes": String(ProcessInfo.processInfo.physicalMemory),
                "logicalProcessors": String(ProcessInfo.processInfo.processorCount),
                "thermalStateAtPreparation": String(ProcessInfo.processInfo.thermalState.rawValue),
                "lowPowerModeAtPreparation": String(ProcessInfo.processInfo.isLowPowerModeEnabled)]
            let assets = try Flux2Assets.open(at: URL(fileURLWithPath: arguments[1], isDirectory: true))
            guard assets.fingerprint == plan.bundleFingerprint else {
                throw AetherError.invalidModel("benchmark bundle fingerprint differs from the pinned plan")
            }
            box.value.bundlePath = assets.root.path
            box.value.bundleFingerprint = assets.fingerprint
            let t2i = URL(fileURLWithPath: arguments[2], isDirectory: true)
            let reference = arguments[3] == "-" ? nil : URL(fileURLWithPath: arguments[3], isDirectory: true)
            box.value.fixtures["t2i"] = plan.textToImageFixture
            if let id = plan.referenceFixture { box.value.fixtures["reference"] = id }
            let tokenizer = try await Flux2Tokenizer.open(assets: assets)
            var prepared: [(Flux2BenchmarkPlan.Case, Flux2BenchmarkWorkload, Flux2Prompt)] = []
            for item in plan.cases {
                let root: URL, pinned: String
                if item.fixture == "reference" {
                    guard let reference, let id = plan.referenceFixture else {
                        throw AetherError.invalidRequest("case \(item.id) needs the reference fixture directory")
                    }
                    root = reference; pinned = id
                } else { root = t2i; pinned = plan.textToImageFixture }
                let workload = try Flux2BenchmarkWorkload.load(fixture: root, expectedID: pinned, caseID: item.fixtureCase, benchmarkID: item.id)
                _ = try Flux2Configuration(width: workload.width, height: workload.height, steps: workload.steps, seed: workload.seed)
                if !workload.references.isEmpty {
                    guard assets.supportsReferences, let limits = assets.references else {
                        throw AetherError.invalidRequest("case \(item.id) needs a bundle with the reference contract")
                    }
                    try Flux2ReferencePolicy.check(workload.references.map { ($0.rows, $0.columns) },
                                                   generatedTokens: workload.imageTokens, limits: limits)
                }
                let prompt = try tokenizer.prepare(workload.prompt)
                guard prompt.tokenIDs == workload.tokenIDs else {
                    throw AetherError.invalidRequest("case \(item.id): the bundle tokenizer's ids differ from the fixture's")
                }
                box.value.preparedCases.append(Flux2BenchmarkPreparedCase(id: item.id, fixtureCase: item.fixtureCase,
                    width: workload.width, height: workload.height, steps: workload.steps, seed: workload.seed,
                    references: workload.references.count, tokenIDsMatchFixture: true, noiseSHA256: floatHash(workload.noise.values)))
                prepared.append((item, workload, prompt))
            }
            guard execute else {
                box.value.stage = "ready"
                box.value.passed = true
                try write()
                try AetherProbe.printJSON(box.value)
                return true
            }
            box.value.stage = "verifying-model-streams"
            try write()
            try assets.verify()
            box.value.modelStreamsVerified = true
            guard let device = MTLCreateSystemDefaultDevice() else { throw AetherError.unsupported("benchmark requires a Metal device") }
            box.value.memoryBeforeIterations = try BenchmarkMemory.read(device: device)
            box.value.device["metalDevice"] = device.name
            box.value.device["recommendedMaxWorkingSetBytes"] = String(device.recommendedMaxWorkingSetSize)
            let journalURL = output.appendingPathComponent("iterations.jsonl")
            guard FileManager.default.createFile(atPath: journalURL.path, contents: nil) else {
                throw AetherError.invalidRequest("cannot create benchmark iteration journal")
            }
            let journalFile = try FileHandle(forWritingTo: journalURL)
            defer { try? journalFile.close() }
            for (item, workload, prompt) in prepared {
                try await runCase(item: item, workload: workload, prompt: prompt, plan: plan, assets: assets, device: device,
                                  report: box, write: write, journal: journalFile)
            }
            box.value.memoryAfterUnload = try BenchmarkMemory.read(device: device)
            box.value.device["thermalStateAfterRun"] = String(ProcessInfo.processInfo.thermalState.rawValue)
            box.value.device["lowPowerModeAfterRun"] = String(ProcessInfo.processInfo.isLowPowerModeEnabled)
            guard try benchmarkFileHash(planURL) == box.value.planSHA256,
                  try benchmarkFileHash(executable) == box.value.executableSHA256,
                  try KreaBenchmarkCommand.sourceIdentities(source) == box.value.sourceSHA256,
                  try Flux2Assets.open(at: assets.root).fingerprint == box.value.bundleFingerprint else {
                throw AetherError.invalidRequest("benchmark source, executable, plan or asset declarations changed during execution")
            }
            for (item, _, _) in prepared {
                _ = try Flux2BenchmarkWorkload.load(fixture: item.fixture == "reference" ? reference! : t2i,
                    expectedID: item.fixture == "reference" ? plan.referenceFixture! : plan.textToImageFixture,
                    caseID: item.fixtureCase, benchmarkID: item.id)
            }
            box.value.provenanceFilesStableAfterRun = true
            for (item, workload, _) in prepared {
                box.value.statistics[item.id] = BenchmarkStatistics.flux2(box.value.iterations, caseID: item.id, steps: workload.steps)
            }
            let failedGates = prepared.filter { plan.profile == .provided && (box.value.ssimAgainstFixture[$0.0.id] ?? -1) < plan.minimumProvidedSSIM }
            guard failedGates.isEmpty else {
                throw AetherError.invalidTensor("provided-profile SSIM below \(plan.minimumProvidedSSIM): \(failedGates.map(\.0.id))")
            }
            box.value.stage = "complete"
            box.value.passed = true
            try write()
            try AetherProbe.printJSON(box.value)
        } catch {
            box.value.stage = "failed"
            box.value.passed = false
            box.value.error = String(describing: error)
            box.value.statistics = [:]
            try write()
            throw error
        }
        return true
    }

    private static func runCase(item: Flux2BenchmarkPlan.Case, workload: Flux2BenchmarkWorkload, prompt: Flux2Prompt,
                                plan: Flux2BenchmarkPlan, assets: Flux2Assets, device: any MTLDevice,
                                report box: Flux2BenchmarkReportBox, write: () throws -> Void, journal: FileHandle) async throws {
        let configuration = try Flux2Configuration(width: workload.width, height: workload.height, steps: workload.steps, seed: workload.seed)
        let session = Flux2GenerationSession(assets: assets, compute: .preferGPU, residency: plan.residency == .resident ? .resident : .balanced)
        do {
            var preparedReferences: Flux2PreparedReferences?
            if plan.profile == .provided, !workload.references.isEmpty {
                box.value.stage = "\(item.id).referenceSetup"
                box.value.benchmarkExecuted = true
                try write()
                let started = ContinuousClock.now
                preparedReferences = try await session.prepare(references: workload.references, generatedTokens: workload.imageTokens)
                box.value.setups.append(Flux2BenchmarkSetup(caseID: item.id, referencesPreparedSeconds: AetherProbe.seconds(started.duration(to: .now)),
                                                         excludedFromIterationTimings: true))
            }
            let conditioning = plan.profile == .provided ? try Flux2Conditioning(provided: workload.text) : nil
            let referenceTokens = workload.references.reduce(0) { $0 + $1.tokens }
            var firstPixels: String?
            for index in 0..<(plan.warmupIterations + plan.measuredIterations) {
                try Task.checkCancellation()
                let role = index < plan.warmupIterations ? "warmup" : "measured"
                let ordinal = index < plan.warmupIterations ? index : index - plan.warmupIterations
                box.value.benchmarkExecuted = true
                box.value.stage = "\(item.id).\(role).\(ordinal)"
                try write()
                let memory = BenchmarkMemorySampler { try BenchmarkMemory.read(device: device) }
                memory.sample()
                let sampler = Task.detached(priority: .utility) {
                    while !Task.isCancelled {
                        do { try await Task.sleep(for: .milliseconds(50)) } catch { break }
                        memory.sample()
                    }
                }
                let started = ContinuousClock.now
                let rows = Flux2BenchmarkJournal(caseID: item.id, role: role, iteration: ordinal, profile: plan.profile,
                                                 steps: workload.steps, imageTokens: workload.imageTokens,
                                                 referenceTokens: referenceTokens, memory: memory, started: started)
                do {
                    let image: GeneratedImage
                    if let conditioning {
                        image = try await session.generate(configuration: configuration, conditioning: conditioning,
                                                           references: preparedReferences, noise: workload.noise, progress: rows.accept)
                    } else {
                        image = try await session.generate(configuration: configuration, prompt: prompt, references: workload.references,
                                                           noise: workload.noise, progress: rows.accept)
                    }
                    let ended = ContinuousClock.now
                    memory.sample()
                    sampler.cancel()
                    await sampler.value
                    let pixels = try rows.finish(image, ended: ended, total: AetherProbe.seconds(started.duration(to: ended)), expectedPixels: firstPixels)
                    firstPixels = firstPixels ?? pixels
                    if role == "measured", ordinal == 0 {
                        box.value.ssimAgainstFixture[item.id] = flux2SSIM(flux2UnitHWC(image.pixelsCHW), workload.image,
                                                                       width: workload.width, height: workload.height)
                    }
                } catch {
                    sampler.cancel()
                    await sampler.value
                    try append(rows.fail(error, total: AetherProbe.seconds(started.duration(to: .now))), to: box, journal: journal)
                    try write()
                    throw error
                }
                try append(rows.row, to: box, journal: journal)
                try write()
            }
            await session.unload()
        } catch {
            await session.unload()
            throw error
        }
    }

    private static func append(_ row: BenchmarkIteration, to box: Flux2BenchmarkReportBox, journal: FileHandle) throws {
        box.value.iterations.append(row)
        var data = try JSONEncoder().encode(row)
        data.append(0x0a)
        try journal.write(contentsOf: data)
    }
}
