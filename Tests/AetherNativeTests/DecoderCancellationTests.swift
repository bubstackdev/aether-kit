import AetherKit
import AetherKrea
import Foundation
import Testing
@testable import AetherCoreAI
@testable import AetherKreaCoreAI

private struct DecoderReference: Decodable {
    let width: Int
    let height: Int
    let steps: Int
    let runs: [Run]
    struct Run: Decodable {
        let latentSHA256: String
        let pixelsSHA256: String
        let pixelShape: [Int]
    }
}

private struct DecoderReport: Encodable {
    var passed = false
    var stage = "preparation"
    var error: String?
    var checks: [String: Bool] = [:]
    var events: [NativeEvent] = []
    var referenceSHA256 = ""
    var latentsSHA256 = ""
    var baselinePixelsSHA256 = ""
    var recoveredPixelsSHA256 = ""
    var scope = "Isolated native ModelSession decoder cancellation/recovery on independent eight-step reference latents"
    var integratedPipelineQualified = false
    var stepsPerGeneration = 0
    var width = 0
    var height = 0
    var recoveryPolicy = "retain"
    var eventsFile: String?
    var functionLifetime = "session"
    let benchmark = false
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_DECODER_PIPELINE_QUALIFICATION"] == "1"), .timeLimit(.minutes(20)))
func nativePipelineDecoderCancellationAndRecovery() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let output = try nativePath("AETHER_NATIVE_DECODER_PIPELINE_OUTPUT")
    guard !FileManager.default.fileExists(atPath: output.path) else { throw NativeFailure(description: "decoder pipeline output exists") }
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let eventsURL = output.appendingPathComponent("events.jsonl")
    guard FileManager.default.createFile(atPath: eventsURL.path, contents: nil) else {
        throw NativeFailure(description: "could not create native event journal")
    }
    let journal = try FileHandle(forWritingTo: eventsURL)
    defer { try? journal.close() }
    let trace = NativeTrace(journal: journal)
    var report = DecoderReport()
    report.eventsFile = eventsURL.lastPathComponent
    let recoveryPolicy = nativeEnvironment["AETHER_NATIVE_DECODER_PIPELINE_RECOVERY"] ?? "retain"
    guard ["retain", "reopen"].contains(recoveryPolicy) else {
        throw NativeFailure(description: "decoder pipeline recovery must be retain or reopen")
    }
    report.recoveryPolicy = recoveryPolicy
    let functionLifetime = nativeEnvironment["AETHER_NATIVE_FUNCTION_LIFETIME"] ?? "session"
    guard ["call", "session"].contains(functionLifetime) else {
        throw NativeFailure(description: "function lifetime must be call or session")
    }
    report.functionLifetime = functionLifetime
    let width = Int(nativeEnvironment["AETHER_NATIVE_DECODER_PIPELINE_WIDTH"] ?? "384") ?? 0
    let height = Int(nativeEnvironment["AETHER_NATIVE_DECODER_PIPELINE_HEIGHT"] ?? "512") ?? 0
    report.width = width
    report.height = height
    guard let steps = Int(nativeEnvironment["AETHER_NATIVE_DECODER_PIPELINE_STEPS"] ?? "1"), [1, 8].contains(steps) else {
        throw NativeFailure(description: "decoder pipeline qualification supports one or eight steps")
    }
    report.scope = "Native KreaGenerationSession decoder cancellation/recovery, \(width)x\(height), \(steps) steps, \(recoveryPolicy) recovery, retained fixture conditioning; compared with its own pre-cancellation baseline"
    report.stepsPerGeneration = steps
    var session: KreaGenerationSession?
    func write() throws {
        report.events = trace.events
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: output.appendingPathComponent("report.json"), options: .atomic)
    }
    func check(_ value: Bool, _ name: String) throws {
        report.checks[name] = value
        try write()
        guard value else { throw NativeFailure(description: name) }
    }
    try write()
    do {
        struct Fixture: Decodable {
            let id: String
            let textShape: [Int]
            let files: Files
            struct Files: Decodable { let text: String }
        }
        let fixtures = try nativePath("AETHER_KREA_FIXTURES")
        let lines = try String(contentsOf: fixtures.appendingPathComponent("cases.jsonl"), encoding: .utf8)
        let cases = try lines.split(separator: "\n").map { try JSONDecoder().decode(Fixture.self, from: Data($0.utf8)) }
        let fixture = try #require(cases.first { $0.id == "tower-768x1344" })
        let data = try Data(contentsOf: fixtures.appendingPathComponent(fixture.files.text))
        report.referenceSHA256 = nativeByteHash(data)
        guard data.count.isMultiple(of: 4) else { throw NativeFailure(description: "partial conditioning value") }
        let values = data.withUnsafeBytes { bytes in stride(from: 0, to: data.count, by: 4).map {
            Float(bitPattern: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self).littleEndian)
        } }
        let conditioning = try KreaConditioning(provided: FloatTensor(shape: fixture.textShape, values: values))
        let configuration = try KreaConfiguration(width: width, height: height, steps: steps, seed: 17017)
        let noise = try FloatTensor(shape: [configuration.imageTokenCount, 64],
            values: KreaNoise.generate(seed: configuration.seed, count: configuration.imageTokenCount * 64))
        report.latentsSHA256 = nativeHash(noise.values)
        let assets = try KreaAssets.open(at: nativePath("AETHER_KREA_BUNDLE"))
        var pipeline = KreaGenerationSession(assets: assets, factory: TrackedNativeFactory(trace: trace))
        session = pipeline
        trace.phase("baseline")
        report.stage = "baseline"
        try write()
        let firstCapture = NativeCapture()
        let baseline = try await pipeline.generate(configuration: configuration, conditioning: conditioning, noise: noise, progress: firstCapture.accept)
        report.baselinePixelsSHA256 = nativeHash(baseline.pixelsCHW.values)
        try check(firstCapture.snapshot.decoded && firstCapture.snapshot.selection?.specialized == false,
            "nativeDynamicBaselineCompleted")
        trace.phase("cancel")
        report.stage = "cancel"
        try write()
        let cancelledCapture = NativeCapture()
        let cancellationPipeline = pipeline
        let run = Task {
            defer { trace.add("generationTaskFinished") }
            return try await cancellationPipeline.generate(configuration: configuration, conditioning: conditioning, noise: noise, progress: cancelledCapture.accept)
        }
        defer { run.cancel() }
        let begin = try await trace.waitForNative(in: "cancel", model: assets.decoder.url.path, observationInterval: .milliseconds(1))
        do {
            _ = try await pipeline.generate(configuration: configuration, conditioning: conditioning, noise: noise)
            throw NativeFailure(description: "pipeline accepted decoder overlap")
        } catch AetherError.sessionBusy { trace.add("overlapRejected") }
        guard !trace.events.contains(where: { $0.event == "nativeEnd" && $0.call == begin.call }) else {
            throw NativeFailure(description: "native decoder finished before cancellation could be observed")
        }
        trace.add("cancelRequested")
        run.cancel()
        do {
            _ = try await run.value
            throw NativeFailure(description: "cancelled pipeline decoder succeeded")
        } catch is CancellationError { trace.add("cancellationReturned") }
        let events = trace.events.filter { $0.phase == "cancel" }
        let request = try #require(events.first { $0.event == "cancelRequested" })
        let end = try #require(events.first { $0.event == "nativeEnd" && $0.call == begin.call })
        let returned = try #require(events.first { $0.event == "cancellationReturned" })
        try check(begin.sequence < request.sequence && request.sequence < end.sequence && end.sequence < returned.sequence,
            "pipelineCancellationWaitsForNativeDecoderReturn")
        try check(await pipeline.state == .ready, "pipelineReturnsReadyAfterDecoderCancellation")
        try check(!cancelledCapture.snapshot.decoded && cancelledCapture.snapshot.latents == firstCapture.snapshot.latents,
            "decoderCancellationFollowsCompleteSamplingAndReportsNoImage")
        if recoveryPolicy == "reopen" {
            trace.phase("reopen")
            report.stage = "reopen"
            try write()
            await pipeline.unload()
            try check(await pipeline.state == .closed, "cancelledPipelineUnloadsBeforeReopening")
            pipeline = KreaGenerationSession(assets: assets, factory: TrackedNativeFactory(trace: trace))
            session = pipeline
        }
        trace.phase("recovery")
        report.stage = "recovery"
        try write()
        let recoveryCapture = NativeCapture()
        let recovered = try await pipeline.generate(configuration: configuration, conditioning: conditioning, noise: noise, progress: recoveryCapture.accept)
        report.recoveredPixelsSHA256 = nativeHash(recovered.pixelsCHW.values)
        try check(recovered.pixelsCHW == baseline.pixelsCHW && recoveryCapture.snapshot.latents == firstCapture.snapshot.latents,
            "pipelineDecoderRecoveryIsBitExact")
        let expectedOpens = recoveryPolicy == "retain" ? 5 : 10
        try check(trace.events.filter { $0.event == "modelOpened" }.count == expectedOpens,
            recoveryPolicy == "retain" ? "recoveryReusesAllDenoisersAndDecoder" : "recoveryReopensAllDenoisersAndDecoder")
        try recovered.savePNG(to: output.appendingPathComponent("recovered.png"))
        await pipeline.unload()
        try check(await pipeline.state == .closed, "pipelineUnloadsClosed")
        let opened = trace.events.filter { $0.event == "modelOpened" }.compactMap(\.instance).sorted()
        let closed = trace.events.filter { $0.event == "modelClosed" }.compactMap(\.instance).sorted()
        try check(opened == closed, "allPipelineModelSessionsClosedExactlyOnce")
        if functionLifetime == "session" {
            let loads = trace.events.filter { $0.event == "functionLoadEnd" }.count
            let calls = trace.events.filter { $0.event == "nativeBegin" }.count
            let reuses = trace.events.filter { $0.event == "functionReused" }.count
            try check(loads == expectedOpens && loads + reuses == calls, "eachFunctionLoadsOncePerModelSession")
        }
        try check(trace.journalError == nil, "nativeEventJournalWrittenWithoutErrors")
        report.integratedPipelineQualified = true
        report.passed = true
        report.stage = "complete"
        try write()
    } catch {
        if let session { await session.unload() }
        report.error = String(describing: error)
        report.stage = "failed"
        try write()
        throw error
    }
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_DECODER_QUALIFICATION"] == "1"), .timeLimit(.minutes(15)))
func nativeDecoderCancellationAndRecovery() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let output = try nativePath("AETHER_NATIVE_DECODER_OUTPUT")
    guard !FileManager.default.fileExists(atPath: output.path) else { throw NativeFailure(description: "decoder output exists") }
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let trace = NativeTrace()
    var report = DecoderReport()
    var session: ModelSession?
    func write() throws {
        report.events = trace.events
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: output.appendingPathComponent("report.json"), options: .atomic)
    }
    func check(_ value: Bool, _ name: String) throws {
        report.checks[name] = value
        try write()
        guard value else { throw NativeFailure(description: name) }
    }
    try write()
    do {
        let referenceRoot = try nativePath("AETHER_NATIVE_DECODER_REFERENCE")
        let referenceData = try Data(contentsOf: referenceRoot.appendingPathComponent("record.json"))
        let reference = try JSONDecoder().decode(DecoderReference.self, from: referenceData)
        report.referenceSHA256 = nativeByteHash(referenceData)
        let expected = try #require(reference.runs.first)
        let data = try Data(contentsOf: referenceRoot.appendingPathComponent("latents.f32"))
        report.latentsSHA256 = nativeByteHash(data)
        try check(report.latentsSHA256 == expected.latentSHA256 && reference.steps == 8, "independentLatentsVerified")
        let values = data.withUnsafeBytes { bytes in
            stride(from: 0, to: data.count, by: 4).map {
                Float(bitPattern: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self).littleEndian)
            }
        }
        let tensor = try Float16Tensor(shape: [reference.height / 16, reference.width / 16, 64], values: values.map(Float16.init))
        let inputs = ["latents": TensorValue.float16(tensor)]
        let assets = try KreaAssets.open(at: nativePath("AETHER_KREA_BUNDLE"))
        try assets.decoder.verify()
        let instance = UUID().uuidString
        let model = assets.decoder.url.path
        let backend = try await CoreAIBackend.load(at: assets.decoder.url, compute: .preferGPU, expectFrequentReshapes: false,
            executionObserver: { trace.native(began: $0, model: model, instance: instance) })
        let decoder = ModelSession(backend: backend, functions: backend.functions)
        session = decoder
        trace.add("modelOpened", model: model, instance: instance)
        func pixels(_ result: [String: TensorValue]) throws -> FloatTensor {
            guard result.count == 1, case .float32(let image) = result["image"], image.shape == expected.pixelShape else {
                throw NativeFailure(description: "unexpected decoder output")
            }
            return image
        }
        trace.phase("baseline")
        report.stage = "baseline"
        let baseline = try pixels(await decoder.runTensors(function: "decode", inputs: inputs))
        report.baselinePixelsSHA256 = nativeHash(baseline.values)
        try check(report.baselinePixelsSHA256 == expected.pixelsSHA256, "baselinePixelsMatchIndependentReference")
        trace.phase("cancel")
        report.stage = "cancel"
        try write()
        let run = Task {
            defer { trace.add("generationTaskFinished") }
            return try await decoder.runTensors(function: "decode", inputs: inputs)
        }
        defer { run.cancel() }
        let begin = try await trace.waitForNative(in: "cancel", model: model)
        do {
            _ = try await decoder.runTensors(function: "decode", inputs: inputs)
            throw NativeFailure(description: "decoder accepted overlap")
        } catch AetherError.sessionBusy { trace.add("overlapRejected") }
        guard !trace.events.contains(where: { $0.event == "nativeEnd" && $0.call == begin.call }) else {
            throw NativeFailure(description: "decoder finished before cancellation could be observed")
        }
        trace.add("cancelRequested")
        run.cancel()
        do {
            _ = try await run.value
            throw NativeFailure(description: "cancelled decoder succeeded")
        } catch is CancellationError { trace.add("cancellationReturned") }
        let events = trace.events.filter { $0.phase == "cancel" }
        let request = try #require(events.first { $0.event == "cancelRequested" })
        let end = try #require(events.first { $0.event == "nativeEnd" && $0.call == begin.call })
        let returned = try #require(events.first { $0.event == "cancellationReturned" })
        try check(begin.sequence < request.sequence && request.sequence < end.sequence && end.sequence < returned.sequence,
            "cancellationWaitsForNativeDecoderReturn")
        try check(await decoder.state == .ready, "decoderReturnsReady")
        trace.phase("recovery")
        report.stage = "recovery"
        try write()
        let recovered = try pixels(await decoder.runTensors(function: "decode", inputs: inputs))
        report.recoveredPixelsSHA256 = nativeHash(recovered.values)
        try check(report.recoveredPixelsSHA256 == expected.pixelsSHA256 && recovered == baseline,
            "sameDecoderSessionRecoversWithBitExactPixels")
        try GeneratedImage(pixelsCHW: recovered).savePNG(to: output.appendingPathComponent("recovered.png"))
        await decoder.unload()
        trace.add("modelClosed", model: model, instance: instance)
        try check(await decoder.state == .closed, "decoderUnloadsClosed")
        report.passed = true
        report.stage = "complete"
        try write()
    } catch {
        if let session { await session.unload() }
        report.error = String(describing: error)
        report.stage = "failed"
        try write()
        throw error
    }
}
