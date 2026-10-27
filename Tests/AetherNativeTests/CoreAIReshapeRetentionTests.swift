@preconcurrency import CoreAI
import Foundation
import Metal
import Testing

/// Opt-in minimal reproduction of Core AI retaining memory on every reshape, written against Core AI alone (no Kit session), for an Apple
/// Feedback: one model function specialized with `expectFrequentReshapes`, submissions encoded and joined one at a
/// time on one stream. Phase 1 repeats one submission (identical shapes and values); phase 2 alternates two token
/// counts, each submission itself identical to an earlier one. Metal's allocated size is recorded after every
/// submission and after the function, stream and model are released. With `AETHER_RETENTION_RESHAPES=0` the same loop
/// is the control. Inputs are the bucketed language asset's: token IDs, cache positions and zeroed key/value caches.
///
/// AETHER_NATIVE_CORE_AI_RETENTION=1
/// AETHER_LANGUAGE_BUCKETED_MODEL=<bucketed model.aimodel>
/// AETHER_RETENTION_OUTPUT=<report .json>
/// AETHER_RETENTION_TOKENS=<the two token counts, default 256,1>
/// AETHER_RETENTION_SUBMISSIONS=<per phase, default 40>
/// AETHER_RETENTION_RESHAPES=<1 (default) or 0>
private struct RetentionPhase: Codable {
    var tokens: [Int] = []
    var metalMiBAfterSubmission: [Int] = []
    /// Growth from the end of the phase's first cycle to its end, per further submission.
    var growthPerSubmissionMiB = 0.0
}

private struct RetentionReport: Codable {
    var model = ""
    var expectFrequentReshapes = true
    var metalMiBBeforeOpen = 0
    var metalMiBAfterOpen = 0
    var repeated = RetentionPhase()
    var alternating = RetentionPhase()
    var metalMiBAfterRelease = 0
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_CORE_AI_RETENTION"] == "1"), .timeLimit(.minutes(10)))
func coreAIRetainsMemoryPerReshapeUntilRelease() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let url = URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_BUCKETED_MODEL"]))
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_RETENTION_OUTPUT"]))
    let counts = (nativeEnvironment["AETHER_RETENTION_TOKENS"] ?? "256,1").split(separator: ",").compactMap { Int($0) }
    try #require(counts.count == 2)
    let submissions = Int(nativeEnvironment["AETHER_RETENTION_SUBMISSIONS"] ?? "") ?? 40
    let device = try #require(MTLCreateSystemDefaultDevice())
    let queue = try #require(device.makeCommandQueue())
    let mib = { device.currentAllocatedSize >> 20 }
    var report = RetentionReport()
    report.model = url.lastPathComponent
    report.expectFrequentReshapes = nativeEnvironment["AETHER_RETENTION_RESHAPES"] != "0"
    report.metalMiBBeforeOpen = mib()

    do {
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = report.expectFrequentReshapes
        var model: AIModel? = try await AIModel(contentsOf: url, options: options)
        var function: InferenceFunction? = try #require(try model?.loadFunction(named: "main"))
        var stream: ComputeStream? = ComputeStream(commandQueue: queue)
        let signature = try #require(model?.functionDescriptor(for: "main"))
        guard case .ndArray(let key)? = signature.inputDescriptor(of: "keyCache") else { throw CocoaError(.featureUnsupported) }
        var shape = key.shape
        shape[3] = 256                                        // cache rows
        let zeros = [Float16](repeating: 0, count: shape.reduce(1, *))
        report.metalMiBAfterOpen = mib()
        func run(_ cycle: [Int]) async throws -> RetentionPhase {
            var phase = RetentionPhase()
            phase.tokens = cycle
            for index in 0..<submissions {
                let count = cycle[index % cycle.count]
                let inputs: [String: InferenceFunction.AsyncValue] = [
                    "input_ids": .init(NDArray(scalars: (0..<count).map { Int32(1000 + $0 % 500) }, shape: [1, count])),
                    "cache_positions": .init(NDArray(scalars: (0..<count).map { Int32($0) }, shape: [1, count])),
                    "keyCache": .init(NDArray(scalars: zeros, shape: shape)),
                    "valueCache": .init(NDArray(scalars: zeros, shape: shape)),
                ]
                let outputs = try function!.encode(inputs: inputs, states: InferenceFunction.AsyncMutableViews(), to: stream!)
                await stream!.currentWorkCompleted()
                _ = try await outputs["logits"]?.ndArray
                phase.metalMiBAfterSubmission.append(mib())
            }
            let after = phase.metalMiBAfterSubmission, first = cycle.count - 1
            if after.count > first + 1 { phase.growthPerSubmissionMiB = Double(after.last! - after[first]) / Double(after.count - 1 - first) }
            return phase
        }
        report.repeated = try await run([counts[0]])
        report.alternating = try await run(counts)
        function = nil
        stream = nil
        model = nil
        report.metalMiBAfterRelease = mib()
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(report).write(to: output)
}
