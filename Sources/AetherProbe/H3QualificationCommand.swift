import AetherKit
import AetherH3
import AetherH3CoreAI
import Darwin
import Foundation
import Metal

@available(macOS 27, iOS 27, *)
private struct H3ProbeRun: Encodable {
    let name: String
    let repetition: Int
    let fullOutputSHA256: String
    let comparison: H3ReferenceComparison
}
@available(macOS 27, iOS 27, *)
private struct H3ProbeReport: Encodable {
    let schema = 1
    let scope = "Corrected Turbo8 block 0 at the two captured T2VA cardinalities; sampled CPU oracle comparison, not full pipeline qualification or a benchmark"
    let mode: String
    var passed = false
    var stage = "preparation"
    var nativeExecuted = false
    var selectedFilesVerified = false
    var bundleSHA256: String?
    var phaseManifestSHA256: String?
    var boundBufferBytes: [String: Int] = [:]
    var resources: [String: BenchmarkMemory] = [:]
    var runs: [H3ProbeRun] = []
    var error: String?
}

@available(macOS 27, iOS 27, *)
enum H3QualificationCommand {
    static func run(_ arguments: [String]) async throws -> Bool {
        guard arguments.first == "qualify-h3" else { return false }
        guard arguments.count == 5, ["--prepare-only", "--run"].contains(arguments[4]) else {
            throw AetherError.invalidRequest("use qualify-h3 BUNDLE.json PHASES.json NEW_OUTPUT_DIRECTORY --prepare-only|--run")
        }
        let output = URL(fileURLWithPath: arguments[3], isDirectory: true).standardizedFileURL
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard mkdir(output.path, 0o755) == 0 else { throw AetherError.invalidRequest("H3 probe output must be a new directory") }
        var report = H3ProbeReport(mode: arguments[4])
        var active: H3BlockSession?
        func write() throws {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: output.appendingPathComponent("report.json"), options: .atomic)
        }
        do {
            try write()
            let plan = try H3BlockProbePlan(bundle: URL(fileURLWithPath: arguments[1]), phases: URL(fileURLWithPath: arguments[2]))
            report.bundleSHA256 = plan.header.manifestSHA256
            report.phaseManifestSHA256 = plan.phaseManifestSHA256
            report.boundBufferBytes = Dictionary(uniqueKeysWithValues: plan.cases.map { ($0.name, $0.boundBytes) })
            try write()
            try plan.verify()
            report.selectedFilesVerified = true
            guard try h3SHA256(Data(contentsOf: URL(fileURLWithPath: arguments[1]))) == plan.header.manifestSHA256,
                  try h3SHA256(Data(contentsOf: URL(fileURLWithPath: arguments[2]))) == plan.phaseManifestSHA256 else {
                throw AetherError.invalidModel("H3 manifests changed during preparation")
            }
            if arguments[4] == "--prepare-only" {
                report.stage = "ready"; report.passed = true
                try write(); try AetherProbe.printJSON(report); return true
            }
            for item in plan.cases {
                try Task.checkCancellation()
                report.stage = item.name + ".load"; report.nativeExecuted = true; try write()
                report.resources[item.name + ".beforeLoad"] = try BenchmarkMemory.read(device: MTLCreateSystemDefaultDevice())
                let session = try await H3BlockSession.open(probe: item)
                active = session
                let reference = try item.reference.read()
                var firstHash: String?
                for repetition in 1...2 {
                    report.stage = item.name + ".run\(repetition)"; try write()
                    let value = try await session.run()
                    let comparison = try H3ReferenceComparison(actual: value, reference: reference,
                        rows: item.referenceRows, limit: plan.relativeL2Limit)
                    let hash = value.sha256
                    report.runs.append(H3ProbeRun(name: item.name, repetition: repetition, fullOutputSHA256: hash, comparison: comparison))
                    try write()
                    guard comparison.passed else { throw AetherError.invalidTensor("H3 sampled block output exceeds the predefined 0.02 relative-L2 gate") }
                    guard firstHash == nil || firstHash == hash else { throw AetherError.invalidTensor("H3 repeated block output changed bytes") }
                    firstHash = hash
                }
                report.resources[item.name + ".afterRun"] = try BenchmarkMemory.read(device: MTLCreateSystemDefaultDevice())
                await session.unload(); active = nil
                report.resources[item.name + ".afterUnload"] = try BenchmarkMemory.read(device: MTLCreateSystemDefaultDevice())
                guard await session.state == .closed else { throw AetherError.invalidModel("H3 block session did not close") }
            }
            // Detect changed manifests rather than publishing a stale identity.
            guard try h3SHA256(Data(contentsOf: URL(fileURLWithPath: arguments[1]))) == plan.header.manifestSHA256,
                  try h3SHA256(Data(contentsOf: URL(fileURLWithPath: arguments[2]))) == plan.phaseManifestSHA256 else {
                throw AetherError.invalidModel("H3 manifests changed during qualification")
            }
            report.stage = "complete"; report.passed = true
            try write(); try AetherProbe.printJSON(report)
        } catch {
            if let active { await active.unload() }
            report.passed = false; report.stage = "failed"; report.error = String(describing: error)
            try write(); throw error
        }
        return true
    }
}
