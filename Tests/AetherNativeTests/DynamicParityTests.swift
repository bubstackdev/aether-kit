import AetherKit
import AetherKrea
import AetherKreaTokenizers
import Foundation
import Testing
@testable import AetherCoreAI
@testable import AetherKreaCoreAI

private struct DynamicCase: Codable, Sendable {
    let id: String
    let prompt: String
    let width: Int
    let height: Int
    let seed: UInt64
    let steps: Int
    let textShape: [Int]
    let files: [String: String]

    var configuration: KreaConfiguration {
        get throws { try KreaConfiguration(width: width, height: height, steps: steps, seed: seed) }
    }
}

private struct DynamicReference: Decodable, Sendable {
    let width: Int
    let height: Int
    let steps: Int
    let arm: String
    let executableSHA256: String
    let manifestSHA256: String
    let noiseSHA256: String
    let runs: [Run]
    struct Run: Decodable, Sendable {
        let encodedTextSHA256: String
        let stepLatentsSHA256: [String]
        let pixelsSHA256: String
        let pixelShape: [Int]
    }
}

private struct DynamicRun: Codable {
    let name: String
    let conditioningSHA256: String
    let stepLatentsSHA256: [String]
    let pixelsSHA256: String
    let pixelShape: [Int]
    let specialized: Bool
    let png: String
}

private struct DynamicCaseReport: Codable {
    let specification: DynamicCase
    let tokenIDs: [Int32]
    let inputSHA256: [String: String]
    var referenceRecordSHA256: String?
    var expected: DynamicRun?
    var runs: [DynamicRun] = []
}

private struct DynamicReport: Encodable {
    var passed = false
    var stage = "preparation"
    var error: String?
    var bundleFingerprint = ""
    var modelStreamsVerified = false
    var referenceBinarySHA256 = ""
    var referenceSourceSHA256: [String: String] = [:]
    var checks: [String: Bool] = [:]
    var cases: [DynamicCaseReport] = []
    var events: [NativeEvent] = []
    let benchmark = false
    let hostLoad = "Shared host; no latency or throughput claim"
    let referenceMode = "Existing unmodified standalone KreaLab round2-final, device-encoded, balanced, baseline policy"
    let inputScope = "Common token IDs and seeded noise; each runtime independently encodes text, computes rotary/schedule, runs dynamic denoisers and decodes"
    let limits = [
        "Two prompts, two dynamic image shapes, eight steps and seed 17017; broader model coverage is not asserted",
        "Native cancellation is observed across the awaited function.run boundary, without independent GPU queue instrumentation",
        "OS specialization cache state and host contention are uncontrolled; timing fields in reference records are not benchmark results",
    ]
}

private func dynamicFileHash(_ url: URL) throws -> String { nativeByteHash(try Data(contentsOf: url)) }

private func runIndependentReference(binary: URL, bundle: URL, inputs: URL, id: String, output: URL) async throws {
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let log = output.appendingPathComponent("reference.log")
    _ = FileManager.default.createFile(atPath: log.path, contents: nil)
    let handle = try FileHandle(forWritingTo: log)
    let process = Process()
    process.executableURL = binary
    process.arguments = [bundle.path, inputs.path, id, "device-encoded", "balanced", "1", output.path, "baseline"]
    process.standardOutput = handle
    process.standardError = handle
    try process.run()
    defer {
        if process.isRunning { process.terminate(); process.waitUntilExit() }
        try? handle.close()
    }
    let deadline = ContinuousClock.now.advanced(by: .seconds(1200))
    while process.isRunning {
        try Task.checkCancellation()
        guard ContinuousClock.now < deadline else { throw NativeFailure(description: "independent reference timed out: \(id)") }
        try await Task.sleep(for: .milliseconds(100))
    }
    process.waitUntilExit()
    guard process.terminationReason == .exit, process.terminationStatus == 0 else {
        throw NativeFailure(description: "independent reference exited \(process.terminationStatus); see \(log.path)")
    }
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_DYNAMIC_QUALIFICATION"] == "1"), .timeLimit(.minutes(60)))
func nativeKreaDynamicParityAndStageCancellation() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let output = try nativePath("AETHER_NATIVE_DYNAMIC_OUTPUT")
    guard !FileManager.default.fileExists(atPath: output.path) else {
        throw NativeFailure(description: "dynamic output already exists; choose a new directory")
    }
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let trace = NativeTrace()
    var report = DynamicReport()
    var activeSession: KreaGenerationSession?
    func write() throws {
        report.events = trace.events
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: output.appendingPathComponent("report.json"), options: .atomic)
    }
    func check(_ condition: Bool, _ name: String) throws {
        report.checks[name] = condition
        try write()
        guard condition else { throw NativeFailure(description: name) }
    }
    func saveAndCompare(_ image: GeneratedImage, capture: NativeCapture, name: String, expected: DynamicRun, caseIndex: Int) throws {
        let captured = capture.snapshot
        guard let conditioning = captured.conditioning, let selection = captured.selection else {
            throw NativeFailure(description: "missing native conditioning or selection")
        }
        let run = DynamicRun(name: name, conditioningSHA256: nativeHash(conditioning.tensor.values.map(Float.init)),
            stepLatentsSHA256: captured.latents, pixelsSHA256: nativeHash(image.pixelsCHW.values),
            pixelShape: image.pixelsCHW.shape, specialized: selection.specialized, png: name + ".png")
        report.cases[caseIndex].runs.append(run)
        try image.savePNG(to: output.appendingPathComponent(run.png))
        try check(!run.specialized && captured.decoded, name + ".dynamicGenerationComplete")
        try check(run.conditioningSHA256 == expected.conditioningSHA256, name + ".encoderMatchesIndependentReference")
        try check(run.stepLatentsSHA256.count == 8 && run.stepLatentsSHA256 == expected.stepLatentsSHA256,
            name + ".allEightLatentsMatchIndependentReference")
        try check(run.pixelShape == expected.pixelShape && run.pixelsSHA256 == expected.pixelsSHA256,
            name + ".decodedPixelsMatchIndependentReference")
    }
    func cancelStage(
        session: KreaGenerationSession, configuration: KreaConfiguration, prompt: KreaPrompt,
        noise: FloatTensor, conditioning: KreaConditioning?, target: URL, phase: String
    ) async throws -> NativeCapture {
        trace.phase(phase)
        let capture = NativeCapture()
        let task = Task {
            defer { trace.add("generationTaskFinished") }
            if let conditioning {
                return try await session.generate(configuration: configuration, conditioning: conditioning, noise: noise, progress: capture.accept)
            }
            return try await session.generate(configuration: configuration, prompt: prompt, noise: noise, progress: capture.accept)
        }
        defer { task.cancel() }
        let begin = try await trace.waitForNative(in: phase, model: target.path)
        do {
            _ = try await session.generate(configuration: configuration, prompt: prompt, noise: noise)
            throw NativeFailure(description: "overlapping request accepted during native stage")
        } catch AetherError.sessionBusy { trace.add("overlapRejected") }
        guard !trace.events.contains(where: { $0.event == "nativeEnd" && $0.call == begin.call }) else {
            throw NativeFailure(description: "target native call completed before cancellation request")
        }
        trace.add("cancelRequested")
        task.cancel()
        do {
            _ = try await task.value
            throw NativeFailure(description: "cancelled native stage succeeded")
        } catch is CancellationError { trace.add("cancellationReturned") }
        let events = trace.events.filter { $0.phase == phase }
        let request = try #require(events.first { $0.event == "cancelRequested" })
        let end = try #require(events.first { $0.event == "nativeEnd" && $0.call == begin.call })
        let returned = try #require(events.first { $0.event == "cancellationReturned" })
        try check(begin.sequence < request.sequence && request.sequence < end.sequence && end.sequence < returned.sequence,
            phase + ".cancellationWaitsForNativeReturn")
        try check(await session.state == .ready, phase + ".sessionReturnsReady")
        try check(capture.snapshot.decoded == false, phase + ".noCompletedImageReported")
        if conditioning == nil {
            let close = try #require(events.first { $0.event == "modelClosed" && $0.instance == begin.instance })
            try check(end.sequence < close.sequence && close.sequence < returned.sequence, phase + ".encoderClosedAfterNativeReturn")
            try check(events.filter { $0.event == "nativeBegin" }.count == 1 && capture.snapshot.conditioning == nil,
                phase + ".denoisingNeverStarted")
            let encoderOpen = try #require(events.first { $0.event == "modelOpened" && $0.model == target.path })
            let priorDenoiserCloses = events.filter { $0.event == "modelClosed" && $0.model != target.path }
            try check(priorDenoiserCloses.count == 4 && priorDenoiserCloses.allSatisfy { $0.sequence < encoderOpen.sequence },
                phase + ".balancedDenoisersReleasedBeforeEncoderAcquisition")
        } else {
            try check(capture.snapshot.latents.count == 8, phase + ".allEightStepsPrecedeDecoderCancellation")
            try check(events.last { $0.event == "nativeBegin" }?.model == target.path,
                phase + ".decoderIsLastNativeCall")
        }
        return capture
    }
    try write()
    do {
        let bundle = try nativePath("AETHER_KREA_BUNDLE")
        let assets = try KreaAssets.open(at: bundle)
        report.bundleFingerprint = assets.fingerprint
        try assets.verify()
        report.modelStreamsVerified = true
        let binary = try nativePath("AETHER_KREA_REFERENCE_PROBE")
        report.referenceBinarySHA256 = try dynamicFileHash(binary)
        let lab = try nativePath("AETHER_KREA_LAB")
        let sourcePaths = [lab.appendingPathComponent("Probe/main.swift")] +
            (FileManager.default.enumerator(at: lab.appendingPathComponent("Sources"), includingPropertiesForKeys: nil)?
                .allObjects.compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" } ?? [])
        for path in sourcePaths { report.referenceSourceSHA256[path.path] = try dynamicFileHash(path) }
        let tokenizer = try await KreaTokenizer.open(assets: assets)
        let prompts = ["a red fox in fresh snow",
            "a watercolor painting of a red fox standing beside a frozen stream in a quiet snowy forest at sunrise, tall pine trees, soft golden light, delicate blue shadows, fine brushwork"]
        let prepared = try prompts.map { try tokenizer.prepare($0) }
        let inputs = output.appendingPathComponent("inputs", isDirectory: true)
        try FileManager.default.createDirectory(at: inputs, withIntermediateDirectories: true)
        var cases: [DynamicCase] = []
        for index in 0..<2 {
            let id = index == 0 ? "dynamic512" : "dynamic384"
            let c = DynamicCase(id: id, prompt: prompts[index], width: index == 0 ? 512 : 384, height: 512,
                seed: 17017, steps: 8, textShape: [prepared[index].conditioningRows, 30720],
                files: ["ids": id + "-ids.i32", "mask": id + "-mask.i32", "noise": id + "-noise.f32", "text": id + "-placeholder-text.f32"])
            let noise = try KreaNoise.generate(seed: c.seed, count: c.configuration.imageTokenCount * 64)
            try noise.withUnsafeBytes { try Data($0).write(to: inputs.appendingPathComponent(c.files["noise"]!)) }
            try prepared[index].tokenIDs.withUnsafeBytes { try Data($0).write(to: inputs.appendingPathComponent(c.files["ids"]!)) }
            try Array(repeating: Int32(1), count: prepared[index].tokenIDs.count).withUnsafeBytes {
                try Data($0).write(to: inputs.appendingPathComponent(c.files["mask"]!))
            }
            try Data(count: c.textShape[0] * c.textShape[1] * 4).write(to: inputs.appendingPathComponent(c.files["text"]!))
            let hashes = try c.files.mapValues { try dynamicFileHash(inputs.appendingPathComponent($0)) }
            report.cases.append(DynamicCaseReport(specification: c, tokenIDs: prepared[index].tokenIDs, inputSHA256: hashes))
            cases.append(c)
        }
        let lines = try cases.map { String(decoding: try JSONEncoder().encode($0), as: UTF8.self) }.joined(separator: "\n") + "\n"
        try Data(lines.utf8).write(to: inputs.appendingPathComponent("cases.jsonl"))
        try write()

        for (index, c) in cases.enumerated() {
            report.stage = c.id + ".independentReference"
            trace.phase(report.stage)
            try write()
            let referenceOutput = output.appendingPathComponent(c.id + "-reference", isDirectory: true)
            try await runIndependentReference(binary: binary, bundle: bundle, inputs: inputs, id: c.id, output: referenceOutput)
            let referenceFile = referenceOutput.appendingPathComponent("record.json")
            let reference = try JSONDecoder().decode(DynamicReference.self, from: Data(contentsOf: referenceFile))
            report.cases[index].referenceRecordSHA256 = try dynamicFileHash(referenceFile)
            try check(reference.width == c.width && reference.height == c.height && reference.steps == 8 &&
                reference.arm == "device-encoded" && reference.runs.count == 1, c.id + ".independentReferenceComplete")
            try check(reference.executableSHA256 == report.referenceBinarySHA256 &&
                reference.noiseSHA256 == report.cases[index].inputSHA256["noise"] &&
                reference.manifestSHA256 == dynamicFileHash(bundle.appendingPathComponent("aether.json")),
                c.id + ".independentReferenceIdentitiesMatch")
            let row = try #require(reference.runs.first)
            try check(row.stepLatentsSHA256.count == 8 && row.pixelShape == [3, c.height, c.width],
                c.id + ".independentReferenceHasEightStepsAndPixels")
            let expected = DynamicRun(name: c.id + "-reference", conditioningSHA256: row.encodedTextSHA256,
                stepLatentsSHA256: row.stepLatentsSHA256, pixelsSHA256: row.pixelsSHA256,
                pixelShape: row.pixelShape, specialized: false, png: "")
            report.cases[index].expected = expected
            let configuration = try c.configuration
            let noise = try FloatTensor(shape: [configuration.imageTokenCount, 64],
                values: KreaNoise.generate(seed: c.seed, count: configuration.imageTokenCount * 64))
            let session = KreaGenerationSession(assets: assets, factory: TrackedNativeFactory(trace: trace))
            activeSession = session
            report.stage = c.id + ".nativeParity"
            trace.phase(report.stage)
            try write()
            let capture = NativeCapture()
            let image = try await session.generate(configuration: configuration, prompt: prepared[index], noise: noise, progress: capture.accept)
            try saveAndCompare(image, capture: capture, name: c.id, expected: expected, caseIndex: index)
            let conditioning = try #require(capture.snapshot.conditioning)
            if index == 0 {
                report.stage = c.id + ".cancelEncoder"
                try write()
                _ = try await cancelStage(session: session, configuration: configuration, prompt: prepared[index], noise: noise,
                    conditioning: nil, target: assets.textEncoder.url, phase: report.stage)
            } else {
                report.stage = c.id + ".cancelDecoder"
                try write()
                let cancelledCapture = try await cancelStage(session: session, configuration: configuration, prompt: prepared[index], noise: noise,
                    conditioning: conditioning, target: assets.decoder.url, phase: report.stage)
                try check(cancelledCapture.snapshot.latents == expected.stepLatentsSHA256,
                    c.id + ".cancelledDecoderRunLatentsMatchReference")
            }
            report.stage = c.id + ".recovery"
            trace.phase(report.stage)
            try write()
            let recoveryCapture = NativeCapture()
            let recovered: GeneratedImage
            if index == 0 {
                recovered = try await session.generate(configuration: configuration, prompt: prepared[index], noise: noise, progress: recoveryCapture.accept)
            } else {
                recovered = try await session.generate(configuration: configuration, conditioning: conditioning, noise: noise, progress: recoveryCapture.accept)
            }
            try saveAndCompare(recovered, capture: recoveryCapture, name: c.id + "-recovery", expected: expected, caseIndex: index)

            // Opt-in coverage for the public prepared-prompt workflow. Preserve
            // independent references and validate model reuse, without treating
            // this shared-host correctness test as a performance measurement.
            report.stage = c.id + ".preparePrompt"
            trace.phase(report.stage)
            try write()
            let reusable = try await session.prepare(prompt: prepared[index])
            try check(reusable.bundleFingerprint == assets.fingerprint &&
                nativeHash(reusable.conditioning.tensor.values.map(Float.init)) == expected.conditioningSHA256,
                c.id + ".preparedConditioningMatchesIndependentReference")
            let preparationEvents = trace.events.filter { $0.phase == report.stage }
            try check(preparationEvents.filter { $0.event == "modelClosed" && $0.model == assets.textEncoder.url.path }.count == 1,
                c.id + ".preparedEncoderReleased")
            for repetition in 1...2 {
                report.stage = c.id + ".preparedRepeat\(repetition)"
                trace.phase(report.stage)
                try write()
                let preparedCapture = NativeCapture()
                let rendered = try await session.generate(configuration: configuration, preparedPrompt: reusable,
                    noise: noise, progress: preparedCapture.accept)
                try saveAndCompare(rendered, capture: preparedCapture, name: c.id + "-prepared-\(repetition)",
                    expected: expected, caseIndex: index)
                let events = trace.events.filter { $0.phase == report.stage }
                try check(!events.contains { $0.event == "nativeBegin" && $0.model == assets.textEncoder.url.path },
                    c.id + ".preparedRepeat\(repetition)DoesNotEncode")
                if repetition == 2 {
                    try check(!events.contains { $0.event == "modelOpened" || $0.event == "modelClosed" },
                        c.id + ".preparedRepeatRetainsModels")
                }
            }
            await session.unload()
            try check(await session.state == .closed, c.id + ".sessionClosed")
            activeSession = nil
        }
        let unchanged = try report.referenceSourceSHA256.allSatisfy { path, hash in
            try dynamicFileHash(URL(fileURLWithPath: path)) == hash
        }
        try check(unchanged, "independentReferenceSourcesUnchanged")
        try check(try dynamicFileHash(binary) == report.referenceBinarySHA256, "referenceBinaryUnchanged")
        let opened = trace.events.filter { $0.event == "modelOpened" }.compactMap(\.instance).sorted()
        let closed = trace.events.filter { $0.event == "modelClosed" }.compactMap(\.instance).sorted()
        try check(opened == closed, "allNativeModelSessionsClosedExactlyOnce")
        report.stage = "complete"
        report.passed = true
        try write()
    } catch {
        if let activeSession { await activeSession.unload() }
        report.error = String(describing: error)
        report.stage = "failed"
        try write()
        throw error
    }
}
