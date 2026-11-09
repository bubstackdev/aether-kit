import AetherKit
import AetherKrea
import AetherKreaCoreAI
import AetherKreaTokenizers
import Foundation
import Metal
import Synchronization
import Testing

/// Explicit native measurement: does a row count the session already ran render at full speed the second time? Two
/// prompts of different lengths alternate A, B, A, B, A; each render's per-step seconds and Metal growth go to a JSON
/// report. On this code each change reopens the denoisers. Recorded 2026-11-08 (1024², 8 steps, M5 Max 64 GB) with an
/// unshipped variant that kept them open with no reshape budget: every first step took 48.7–53.9 s, revisits included,
/// so Core AI re-specializes on every shape change. No image is saved and nothing is asserted about speed (the report
/// is the result). Close any host app holding the GPU first (see KreaReshapeQualificationTests).
///
/// AETHER_NATIVE_KREA_RESHAPE_REPEAT=1 AETHER_KREA_BUNDLE=<bundle> AETHER_KREA_RESHAPE_REPORT=<new .json>
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_KREA_RESHAPE_REPEAT"] == "1"
               && nativeEnvironment["AETHER_KREA_BUNDLE"] != nil && nativeEnvironment["AETHER_KREA_RESHAPE_REPORT"] != nil),
      .timeLimit(.minutes(30)))
func keptDenoisersAtAnAlreadySeenRowCount() async throws {
    let report = URL(fileURLWithPath: try #require(nativeEnvironment["AETHER_KREA_RESHAPE_REPORT"]))
    #expect(!FileManager.default.fileExists(atPath: report.path), "the report must be a new file")
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let assets = try KreaAssets.open(at: URL(fileURLWithPath: try #require(nativeEnvironment["AETHER_KREA_BUNDLE"])))
    try assets.verify()
    let tokenizer = try await KreaTokenizer.open(assets: assets)
    let device = try #require(MTLCreateSystemDefaultDevice())
    let a = try tokenizer.prepare("A red ceramic mug on a white tabletop.")
    let b = try tokenizer.prepare("A blue glass bottle on a wooden shelf beside a stack of old books, soft afternoon light, shallow depth of field.")
    #expect(a.conditioningRows != b.conditioningRows)
    let configuration = try KreaConfiguration(width: 1024, height: 1024, steps: 8, seed: 18)

    struct Render: Codable { let label: String; let rows: Int; let seconds: Double; let stepSeconds: [Double]; let metalGrowthBytes: Int }
    var renders: [Render] = []
    let session = KreaGenerationSession(assets: assets, compute: .preferGPU, residency: .resident, lifecycle: .manual)
    do {
        for (label, prompt) in [("A", a), ("B", b), ("A", a), ("B", b), ("A", a)] {
            let marks = Mutex<[Double]>([])
            let before = device.currentAllocatedSize, started = ProcessInfo.processInfo.systemUptime
            _ = try await session.generate(configuration: configuration, prompt: prompt) { event in
                switch event {
                case .denoisersReady, .step: marks.withLock { $0.append(ProcessInfo.processInfo.systemUptime) }
                default: break
                }
            }
            let times = marks.withLock { $0 }
            let intervals = zip(times, times.dropFirst()).map { $1 - $0 }
            renders.append(Render(label: label, rows: prompt.conditioningRows, seconds: ProcessInfo.processInfo.systemUptime - started,
                                  stepSeconds: intervals, metalGrowthBytes: device.currentAllocatedSize - before))
        }
        await session.unload()
    } catch {
        await session.unload()
        throw error
    }
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(renders).write(to: report, options: .withoutOverwriting)
}
