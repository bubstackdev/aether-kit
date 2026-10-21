import AetherKit
import AetherKrea
import AetherKreaCoreAI
import AetherKreaTokenizers
import CoreAI
import Darwin
import Foundation
import Metal

struct BenchmarkPreparedCase: Codable, Sendable {
    let id: String
    let tokenIDs: [Int32]
    let conditioningRows: Int
    let noiseSHA256: String
    let referencePath: String
    let referenceSHA256: String
    let selectionIdentity: String
}

struct KreaBenchmarkReport: Encodable {
    let schema = 1
    let purpose = "Opt-in Krea generation benchmark"
    let requestedMode: String
    let startedAt = ISO8601DateFormatter().string(from: Date())
    var stage = "preparation"
    var profile = BenchmarkProfile.encoded
    var passed = false
    var benchmarkExecuted = false
    var error: String?
    var plan: KreaBenchmarkPlan?
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
    var declaredModelSHA256: [String: String] = [:]
    var modelStreamsVerified = false
    var provenanceFilesStableAfterRun = false
    var memoryBeforeIterations: BenchmarkMemory?
    var memoryAfterUnload: BenchmarkMemory?
    var preparedCases: [BenchmarkPreparedCase] = []
    var conditioningSetups: [BenchmarkConditioningSetup] = []
    var iterations: [BenchmarkIteration] = []
    var statistics: [String: [String: BenchmarkStatistics]] = [:]
    let computePolicy = "preferGPU"
    let residency = "balanced"
    let memorySampleIntervalMilliseconds = 50
    let cachePolicy = "No OS cache eviction; same generation session across each case's warmups and measurements, new session for the next case"
    var timingScope = "Application-stage wall clock, not GPU kernel timing. Conditioning includes balanced cleanup and encoder model/function loading; denoiserSetup includes model acquisition/validation. step0 includes rotary/schedule setup and lazy denoiser function loading. Steps include native calls, host copies and Euler updates. Decode includes decoder model/function loading, copying and image validation. Stage timings exclude progress observers; total includes observers. Preflight, final pixel hashing, report IO and unload are outside timed generation."
    let modelVerificationScope = "Model streams hashed before iterations; postflight rechecks source, executable, plan, references and asset declarations. Stream files must remain immutable during execution."
    let memoryScope = "Per-iteration maximum observed physical footprint, resident bytes and default Metal device currentAllocatedSize, sampled every 50 ms plus stage boundaries. Validation observers are included. These separate counters must not be summed. Kernel process-lifetime peaks are not reset per iteration; brief peaks may be missed by sampling."
    let hostConditions = "Quiet-window availability must be arranged by the operator; this command does not establish system idleness or control other workloads"
}

enum KreaBenchmarkCommand {
    static func run(_ arguments: [String]) async throws -> Bool {
        guard arguments.first == "benchmark-krea" else { return false }
        guard arguments.count == 6, ["--prepare-only", "--run"].contains(arguments[5]) else {
            throw AetherError.invalidRequest("use benchmark-krea BUNDLE PLAN.json SOURCE_ROOT OUTPUT_DIRECTORY --prepare-only|--run")
        }
        let execute = arguments[5] == "--run"
        #if DEBUG
        guard !execute else { throw AetherError.invalidRequest("benchmark measurement requires a Release build") }
        #endif
        let output = URL(fileURLWithPath: arguments[4], isDirectory: true).standardizedFileURL
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard mkdir(output.path, 0o755) == 0 else {
            throw AetherError.invalidRequest("benchmark output must be a new directory (mkdir errno \(errno))")
        }
        var report = KreaBenchmarkReport(requestedMode: execute ? "run" : "prepare-only")
        func write() throws {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: output.appendingPathComponent("report.json"), options: .atomic)
        }
        try write()
        do {
            let planURL = URL(fileURLWithPath: arguments[2]).standardizedFileURL.resolvingSymlinksInPath()
            let planData = try Data(contentsOf: planURL)
            let plan = try JSONDecoder().decode(KreaBenchmarkPlan.self, from: planData)
            try plan.validate()
            report.plan = plan
            report.profile = plan.effectiveProfile
            if plan.effectiveProfile == .provided {
                report.timingScope = "Provided conditioning is encoded, hash-validated and its encoder unloaded once per case, recorded separately outside iteration timing. Iterations reuse one denoiser/decoder session; conditioning measures only the provided-tensor handoff. Warmup step0 includes denoiser function loading; measured steps reuse retained functions. Steps include native calls, host copies and Euler updates. Decode includes loading when needed, copies and image validation. Stage timings exclude progress observers; total includes them. Preflight, one-time conditioning setup, final pixel hashing, report IO and unload are excluded. This profile does not measure fresh-prompt end-to-end latency."
            }
            report.planPath = planURL.path
            report.planSHA256 = benchmarkHash(planData)
            let source = URL(fileURLWithPath: arguments[3], isDirectory: true).standardizedFileURL.resolvingSymlinksInPath()
            report.sourceRoot = source.path
            report.sourceSHA256 = try sourceIdentities(source)
            let executable = (Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])).resolvingSymlinksInPath()
            report.executablePath = executable.path
            report.executableSHA256 = try benchmarkFileHash(executable)
            report.installedToolchain = try toolchain()
            report.device = ["os": ProcessInfo.processInfo.operatingSystemVersionString,
                "architecture": AIModel.deviceArchitectureName,
                "computeUnits": ComputeUnitKind.availableKinds.map { String(describing: $0) }.sorted().joined(separator: ", "),
                "physicalMemoryBytes": String(ProcessInfo.processInfo.physicalMemory),
                "logicalProcessors": String(ProcessInfo.processInfo.processorCount),
                "thermalStateAtPreparation": String(ProcessInfo.processInfo.thermalState.rawValue),
                "lowPowerModeAtPreparation": String(ProcessInfo.processInfo.isLowPowerModeEnabled)]
            let assets = try KreaAssets.open(at: URL(fileURLWithPath: arguments[1], isDirectory: true))
            guard assets.fingerprint == plan.bundleFingerprint else {
                throw AetherError.invalidModel("benchmark bundle fingerprint differs from pinned plan")
            }
            report.bundlePath = assets.root.path
            report.bundleFingerprint = assets.fingerprint
            for model in [assets.textEncoder, assets.decoder] + assets.denoisers {
                report.declaredModelSHA256[model.url.lastPathComponent] = model.declaredSHA256
            }
            let tokenizer = try await KreaTokenizer.open(assets: assets)
            var prepared: [(BenchmarkCase, KreaPrompt, FloatTensor, BenchmarkExpected)] = []
            for item in plan.cases {
                let prompt = try tokenizer.prepare(item.prompt)
                let configuration = try item.configuration
                let noise = try FloatTensor(shape: [configuration.imageTokenCount, 64],
                    values: KreaNoise.generate(seed: item.seed, count: configuration.imageTokenCount * 64))
                let noiseHash = floatHash(noise.values)
                let referenceURL = URL(fileURLWithPath: item.reference, relativeTo: planURL.deletingLastPathComponent())
                    .standardizedFileURL.resolvingSymlinksInPath()
                let referenceData = try Data(contentsOf: referenceURL)
                guard benchmarkHash(referenceData) == item.referenceSHA256 else {
                    throw AetherError.invalidRequest("benchmark reference file changed: \(item.id)")
                }
                let expected = try BenchmarkExpected.load(referenceData, item: item, noiseSHA256: noiseHash)
                let selection = try assets.denoiserSelection(imageTokens: configuration.imageTokenCount, textRows: prompt.conditioningRows)
                guard !selection.specialized, !prompt.wasTruncated else {
                    throw AetherError.invalidRequest("benchmark requires untruncated prompts and original dynamic assets")
                }
                report.preparedCases.append(BenchmarkPreparedCase(id: item.id, tokenIDs: prompt.tokenIDs,
                    conditioningRows: prompt.conditioningRows, noiseSHA256: noiseHash,
                    referencePath: referenceURL.path, referenceSHA256: item.referenceSHA256, selectionIdentity: selection.identity))
                prepared.append((item, prompt, noise, expected))
            }
            guard execute else {
                report.stage = "ready"
                report.passed = true
                try write()
                try AetherProbe.printJSON(report)
                return true
            }
            report.stage = "verifying-model-streams"
            try write()
            try assets.verify()
            report.modelStreamsVerified = true
            guard let device = MTLCreateSystemDefaultDevice() else {
                throw AetherError.unsupported("benchmark requires a Metal device")
            }
            report.memoryBeforeIterations = try BenchmarkMemory.read(device: device)
            report.device["metalDevice"] = device.name
            report.device["metalRegistryID"] = String(device.registryID)
            report.device["recommendedMaxWorkingSetBytes"] = String(device.recommendedMaxWorkingSetSize)
            let journalURL = output.appendingPathComponent("iterations.jsonl")
            guard FileManager.default.createFile(atPath: journalURL.path, contents: nil) else {
                throw AetherError.invalidRequest("cannot create benchmark iteration journal")
            }
            let journal = try FileHandle(forWritingTo: journalURL)
            defer { try? journal.close() }
            for (item, prompt, noise, expected) in prepared {
                let configuration = try item.configuration
                let provided: KreaConditioning?
                if plan.effectiveProfile == .provided {
                    report.stage = "\(item.id).conditioningSetup"
                    report.benchmarkExecuted = true
                    try write()
                    let setup = try await BenchmarkConditioningPreparation.prepare(caseID: item.id, expected: expected.conditioning) {
                        let encoder = try await KreaTextEncoder.open(assets: assets, compute: .preferGPU)
                        return BenchmarkConditioner(encode: { try await encoder.encode(prompt) },
                                                    unload: { await encoder.unload() })
                    }
                    provided = setup.conditioning
                    report.conditioningSetups.append(setup.record)
                    try write()
                } else { provided = nil }
                let pipeline = KreaGenerationSession(assets: assets, compute: .preferGPU, residency: .balanced)
                let operation = BenchmarkOperation(generate: { progress in
                    if let provided {
                        return try await pipeline.generate(configuration: configuration, conditioning: provided,
                                                           noise: noise, progress: progress)
                    }
                    return try await pipeline.generate(configuration: configuration, prompt: prompt, noise: noise, progress: progress)
                }, unload: { await pipeline.unload() })
                try await BenchmarkLoop.run(item: item, expected: expected, warmups: plan.warmupIterations,
                    measurements: plan.measuredIterations, profile: plan.effectiveProfile, operation: operation,
                    readMemory: { try BenchmarkMemory.read(device: device) }, onStart: { role, iteration in
                        report.benchmarkExecuted = true
                        report.stage = "\(item.id).\(role).\(iteration)"
                        try write()
                    }, onIteration: { row in
                        report.iterations.append(row)
                        var data = try JSONEncoder().encode(row)
                        data.append(0x0a)
                        try journal.write(contentsOf: data)
                        try write()
                    })
            }
            report.memoryAfterUnload = try BenchmarkMemory.read(device: device)
            report.device["thermalStateAfterRun"] = String(ProcessInfo.processInfo.thermalState.rawValue)
            report.device["lowPowerModeAfterRun"] = String(ProcessInfo.processInfo.isLowPowerModeEnabled)
            // Do not publish partial statistics if inputs or sources changed mid-run.
            guard try benchmarkFileHash(planURL) == report.planSHA256,
                  try benchmarkFileHash(executable) == report.executableSHA256,
                  try sourceIdentities(source) == report.sourceSHA256,
                  try KreaAssets.open(at: assets.root).fingerprint == report.bundleFingerprint else {
                throw AetherError.invalidRequest("benchmark source, executable, plan or asset declarations changed during execution")
            }
            for item in report.preparedCases {
                guard try benchmarkFileHash(URL(fileURLWithPath: item.referencePath)) == item.referenceSHA256 else {
                    throw AetherError.invalidRequest("benchmark reference changed during execution")
                }
            }
            report.provenanceFilesStableAfterRun = true
            for item in plan.cases { report.statistics[item.id] = BenchmarkStatistics.measured(report.iterations, caseID: item.id) }
            report.stage = "complete"
            report.passed = true
            try write()
            try AetherProbe.printJSON(report)
        } catch {
            report.stage = "failed"
            report.passed = false
            report.error = String(describing: error)
            report.statistics = [:]
            try write()
            throw error
        }
        return true
    }

    private static func sourceIdentities(_ root: URL) throws -> [String: String] {
        guard FileManager.default.fileExists(atPath: root.appendingPathComponent("Package.swift").path),
              FileManager.default.fileExists(atPath: root.appendingPathComponent("Sources").path) else {
            throw AetherError.invalidRequest("benchmark source root must contain Package.swift and Sources")
        }
        var files = [root.appendingPathComponent("Package.swift"), root.appendingPathComponent("Package.resolved")]
        for directory in ["Sources", "Tests", "Scripts"] {
            let parent = root.appendingPathComponent(directory)
            guard let enumerator = FileManager.default.enumerator(at: parent, includingPropertiesForKeys: [.isRegularFileKey],
                                                                  options: [.skipsHiddenFiles]) else { continue }
            for case let file as URL in enumerator where ["swift", "sh", "py"].contains(file.pathExtension) {
                guard try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else { continue }
                let resolved = file.resolvingSymlinksInPath()
                guard resolved.path.hasPrefix(root.path + "/") else {
                    throw AetherError.invalidRequest("benchmark source file escapes source root")
                }
                files.append(file)
            }
        }
        var hashes: [String: String] = [:]
        for file in files.sorted(by: { $0.path < $1.path }) {
            hashes[String(file.path.dropFirst(root.path.count + 1))] = try benchmarkFileHash(file)
        }
        return hashes
    }

    private static func toolchain() throws -> String {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        process.arguments = ["swift", "--version"]
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw AetherError.unsupported("cannot identify installed Swift toolchain") }
        return String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
