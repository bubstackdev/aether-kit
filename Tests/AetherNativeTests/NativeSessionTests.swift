import AetherKit
import AetherKrea
import AetherKreaTokenizers
import CryptoKit
import Foundation
import Synchronization
import Testing
@testable import AetherCoreAI
@testable import AetherKreaCoreAI

let nativeEnvironment = ProcessInfo.processInfo.environment

struct NativeFailure: Error, CustomStringConvertible {
    let description: String
}

struct NativeEvent: Codable, Sendable {
    let sequence: Int
    let phase: String
    let event: String
    let model: String?
    let instance: String?
    let call: Int?
    var uptime: Double? = nil
    var detail: String? = nil
}

final class NativeTrace: Sendable {
    private struct State {
        var phase = "preparation"
        var events: [NativeEvent] = []
        var active: [String: Int] = [:]
        var journalError: String?
    }
    private let state = Mutex(State())
    private let journal: FileHandle?

    init(journal: FileHandle? = nil) { self.journal = journal }
    var journalError: String? { state.withLock { $0.journalError } }

    // Persist each event before returning to the caller. A killed or stalled test
    // retains its last native boundary without relying on test-runner log flushing.
    private func append(_ event: NativeEvent, to value: inout State) {
        var event = event
        event.uptime = ProcessInfo.processInfo.systemUptime
        value.events.append(event)
        guard let journal else { return }
        do {
            var data = try JSONEncoder().encode(event)
            data.append(0x0a)
            try journal.write(contentsOf: data)
        } catch {
            value.journalError = String(describing: error)
        }
    }

    var events: [NativeEvent] { state.withLock { $0.events } }
    func phase(_ value: String) {
        state.withLock { state in
            state.phase = value
            append(NativeEvent(sequence: state.events.count, phase: value, event: "phase",
                model: nil, instance: nil, call: nil), to: &state)
        }
        FileHandle.standardError.write(Data("native qualification: \(value)\n".utf8))
    }
    func add(_ event: String, model: String? = nil, instance: String? = nil, detail: String? = nil) {
        state.withLock { value in
            append(NativeEvent(sequence: value.events.count, phase: value.phase,
                event: event, model: model, instance: instance, call: nil, detail: detail), to: &value)
        }
    }
    func native(began: Bool, model: String, instance: String) {
        state.withLock { value in
            let call: Int?
            if began {
                call = value.events.count
                value.active[instance] = call
            } else { call = value.active.removeValue(forKey: instance) }
            append(NativeEvent(sequence: value.events.count, phase: value.phase,
                event: began ? "nativeBegin" : "nativeEnd", model: model, instance: instance, call: call), to: &value)
        }
    }
    func waitForNative(in phase: String, model: String? = nil, observationInterval: Duration = .milliseconds(10)) async throws -> NativeEvent {
        let deadline = ContinuousClock.now.advanced(by: .seconds(600))
        while ContinuousClock.now < deadline {
            let snapshot = events
            if let begin = snapshot.first(where: {
                $0.phase == phase && $0.event == "nativeBegin" && (model == nil || $0.model == model)
            }) {
                // Allow the async framework call to dispatch before requesting lifecycle work.
                // Do not insert a gate into, or artificially delay, the native operation.
                try await Task.sleep(for: observationInterval)
                guard !events.contains(where: { $0.event == "nativeEnd" && $0.call == begin.call }) else {
                    throw NativeFailure(description: "native call finished before the in-flight observation")
                }
                return begin
            }
            if snapshot.contains(where: { $0.phase == phase && $0.event == "generationTaskFinished" }) {
                throw NativeFailure(description: "generation finished before observing the requested native stage")
            }
            try await Task.sleep(for: observationInterval)
        }
        throw NativeFailure(description: "timed out waiting for native call in \(phase)")
    }
}

struct TrackedNativeModel: KreaExecutable {
    let session: ModelSession
    let trace: NativeTrace
    let model: String
    let instance: String
    var functions: [ModelFunctionDescription] { session.functions }
    func runTensors(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue] {
        let shapes = inputs.keys.sorted().map { "\($0)=\(inputs[$0]!.shape)" }.joined(separator: ", ")
        trace.add("callSubmitted", model: model, instance: instance, detail: "\(function): \(shapes)")
        do {
            let outputs = try await session.runTensors(function: function, inputs: inputs)
            trace.add("callReturned", model: model, instance: instance)
            return outputs
        } catch {
            trace.add("callFailed", model: model, instance: instance, detail: String(describing: error))
            throw error
        }
    }
    func unload() async {
        trace.add("modelCloseBegin", model: model, instance: instance)
        await session.unload()
        trace.add("modelClosed", model: model, instance: instance)
    }
}

struct TrackedNativeFactory: KreaModelFactory {
    let trace: NativeTrace
    func open(_ asset: KreaModelAsset, frequentReshapes: Bool) async throws -> any KreaExecutable {
        let model = asset.url.path
        let instance = UUID().uuidString
        trace.add("modelOpenBegin", model: model, instance: instance)
        let backend = try await CoreAIBackend.load(at: asset.url, compute: .preferGPU,
            expectFrequentReshapes: frequentReshapes, executionObserver: { began in
                trace.native(began: began, model: model, instance: instance)
            }, boundaryObserver: { boundary in
                trace.add(boundary, model: model, instance: instance)
            }, retainFunctions: nativeEnvironment["AETHER_NATIVE_FUNCTION_LIFETIME"] != "call")
        let session = ModelSession(backend: backend, functions: backend.functions)
        trace.add("modelOpened", model: model, instance: instance)
        return TrackedNativeModel(session: session, trace: trace, model: model, instance: instance)
    }
}

final class NativeCapture: Sendable {
    struct State {
        var conditioning: KreaConditioning?
        var selection: KreaDenoiserSelection?
        var latents: [String] = []
        var decoded = false
    }
    private let value = Mutex(State())
    var snapshot: State { value.withLock { $0 } }
    func accept(_ event: KreaGenerationEvent) {
        value.withLock { value in
            switch event {
            case .conditioningReady(let text): value.conditioning = text
            case .denoisersReady(let selection): value.selection = selection
            case .step(_, _, let tensor): value.latents.append(nativeHash(tensor.values))
            case .decoded: value.decoded = true
            }
        }
    }
}

private struct NativeGeneration: Codable {
    let name: String
    let width: Int
    let height: Int
    let steps: Int
    let conditioningRows: Int
    let specialized: Bool
    let selectionIdentity: String
    let latentSHA256: [String]
    let pixelsSHA256: String
    let png: String
}

private struct NativeReport: Encodable {
    var passed = false
    var stage = "preparation"
    var error: String?
    var bundleFingerprint = ""
    var modelStreamsVerified = false
    var shapeRegistrationSHA256 = ""
    var checks: [String: Bool] = [:]
    var generations: [NativeGeneration] = []
    var events: [NativeEvent] = []
    var promptTokenCounts: [Int] = []
    let compute = "preferGPU"
    let stepsPerGeneration = 1
    let benchmark = false
    let hostLoad = "Shared host under significant concurrent load; correctness checks only"
    let limits = [
        "One-step lifecycle and dynamic-shape checks; eight-step independent parity is recorded separately",
        "Native event boundaries bracket the awaited function.run before output copying; no GPU queue instrumentation",
        "Dynamic outputs checked for shape, finite values and repeat stability, without an independent image baseline",
        "Model reuse means identical ModelSession instances; OS cache eviction and physical memory release are not asserted",
    ]
}

func nativeHash(_ values: [Float]) -> String {
    values.withUnsafeBytes { nativeByteHash(Data($0)) }
}

func nativeByteHash(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func nativePath(_ name: String) throws -> URL {
    guard let path = nativeEnvironment[name] else { throw NativeFailure(description: "missing \(name)") }
    return URL(fileURLWithPath: path)
}

private func nativeFloats(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    guard data.count.isMultiple(of: 4) else { throw NativeFailure(description: "partial float fixture") }
    return data.withUnsafeBytes { bytes in
        stride(from: 0, to: data.count, by: 4).map {
            Float(bitPattern: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self).littleEndian)
        }
    }
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_QUALIFICATION"] == "1"), .timeLimit(.minutes(60)))
func nativeKreaSessionQualification() async throws {
    let output = try nativePath("AETHER_NATIVE_OUTPUT")
    guard !FileManager.default.fileExists(atPath: output.path) else {
        throw NativeFailure(description: "native output directory already exists; choose a new directory")
    }
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let trace = NativeTrace()
    var report = NativeReport()
    var session: KreaGenerationSession?
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
    func save(_ image: GeneratedImage, _ capture: NativeCapture, _ name: String) throws {
        let captured = capture.snapshot
        guard let selection = captured.selection, let conditioning = captured.conditioning,
              captured.decoded, captured.latents.count == 1 else {
            throw NativeFailure(description: "incomplete generation: \(name)")
        }
        try image.savePNG(to: output.appendingPathComponent(name + ".png"))
        report.generations.append(NativeGeneration(name: name, width: image.width, height: image.height,
            steps: 1, conditioningRows: conditioning.rows, specialized: selection.specialized,
            selectionIdentity: selection.identity, latentSHA256: captured.latents,
            pixelsSHA256: nativeHash(image.pixelsCHW.values), png: name + ".png"))
        try write()
    }
    try write()
    do {
        var assets = try KreaAssets.open(at: nativePath("AETHER_KREA_BUNDLE"))
        report.bundleFingerprint = assets.fingerprint
        try assets.verify()
        report.modelStreamsVerified = true
        let registration = try nativePath("AETHER_KREA_SHAPE")
        assets = try assets.registeringSpecialization(at: registration)
        report.shapeRegistrationSHA256 = nativeByteHash(try Data(contentsOf: registration))
        let fixtures = try nativePath("AETHER_KREA_FIXTURES")
        struct Fixture: Decodable {
            let id: String
            let files: Files
            struct Files: Decodable { let noise: String; let text: String }
        }
        let lines = try String(contentsOf: fixtures.appendingPathComponent("cases.jsonl"), encoding: .utf8)
        let cases = try lines.split(separator: "\n").map { try JSONDecoder().decode(Fixture.self, from: Data($0.utf8)) }
        let fixture = try #require(cases.first { $0.id == "tower-768x1344" })
        let noise = try FloatTensor(shape: [4032, 64], values: nativeFloats(fixtures.appendingPathComponent(fixture.files.noise)))
        let conditioning = try KreaConditioning(provided: FloatTensor(shape: [20, 30720],
            values: nativeFloats(fixtures.appendingPathComponent(fixture.files.text))))
        let tower = try KreaConfiguration(width: 768, height: 1344, steps: 1, seed: 71344)
        let pipeline = KreaGenerationSession(assets: assets, factory: TrackedNativeFactory(trace: trace))
        session = pipeline
        report.stage = "generation"

        trace.phase("first")
        let firstCapture = NativeCapture()
        let first = try await pipeline.generate(configuration: tower, conditioning: conditioning, noise: noise,
            progress: firstCapture.accept)
        try save(first, firstCapture, "first")
        try check(firstCapture.snapshot.selection?.specialized == true, "towerUsesSpecialization")
        let firstOpens = trace.events.filter { $0.event == "modelOpened" }
        try check(firstOpens.count == 5, "firstRunLoadsFourDenoisersAndDecoder")

        trace.phase("repeat")
        let repeatCapture = NativeCapture()
        let repeated = try await pipeline.generate(configuration: tower, conditioning: conditioning, noise: noise,
            progress: repeatCapture.accept)
        try save(repeated, repeatCapture, "repeat")
        try check(repeated.pixelsCHW == first.pixelsCHW && repeatCapture.snapshot.latents == firstCapture.snapshot.latents,
            "repeatedGenerationIsBitExact")
        try check(trace.events.filter { $0.event == "modelOpened" }.count == 5 &&
            !trace.events.contains { $0.event == "modelClosed" }, "repeatReusesAllFiveModelSessions")

        trace.phase("cancel")
        let cancelled = Task {
            try await pipeline.generate(configuration: tower, conditioning: conditioning, noise: noise)
        }
        let cancelBegin = try await trace.waitForNative(in: "cancel")
        trace.add("cancelRequested")
        cancelled.cancel()
        do {
            _ = try await pipeline.generate(configuration: tower, conditioning: conditioning, noise: noise)
            throw NativeFailure(description: "overlap was accepted during cancelled native work")
        } catch AetherError.sessionBusy { trace.add("overlapRejected") }
        do {
            _ = try await cancelled.value
            throw NativeFailure(description: "cancelled generation succeeded")
        } catch is CancellationError { trace.add("cancellationReturned") }
        let cancelledEvents = trace.events.filter { $0.phase == "cancel" }
        let cancelRequest = try #require(cancelledEvents.first { $0.event == "cancelRequested" })
        let cancelEnd = try #require(cancelledEvents.first { $0.event == "nativeEnd" && $0.call == cancelBegin.call })
        let cancelReturn = try #require(cancelledEvents.first { $0.event == "cancellationReturned" })
        try check(cancelBegin.sequence < cancelRequest.sequence && cancelRequest.sequence < cancelEnd.sequence &&
            cancelEnd.sequence < cancelReturn.sequence, "cancelWaitsForInFlightNativeReturn")
        try check(cancelledEvents.filter { $0.event == "nativeBegin" }.count == 1, "cancelStopsBeforeNextSegment")
        try check(await pipeline.state == .ready, "cancelRestoresReadyState")

        trace.phase("recovery")
        let recoveryCapture = NativeCapture()
        let recovered = try await pipeline.generate(configuration: tower, conditioning: conditioning, noise: noise,
            progress: recoveryCapture.accept)
        try save(recovered, recoveryCapture, "recovery")
        try check(recovered.pixelsCHW == first.pixelsCHW && recoveryCapture.snapshot.latents == firstCapture.snapshot.latents,
            "recoveryAfterCancellationIsBitExact")
        try check(trace.events.filter { $0.event == "modelOpened" }.count == 5, "recoveryReusesExistingModels")

        let tokenizer = try await KreaTokenizer.open(assets: assets)
        let shortPrompt = try tokenizer.prepare("a red fox in fresh snow")
        let longPrompt = try tokenizer.prepare("a watercolor painting of a red fox standing beside a frozen stream in a quiet snowy forest at sunrise, tall pine trees, soft golden light, delicate blue shadows, fine brushwork")
        report.promptTokenCounts = [shortPrompt.tokenIDs.count, longPrompt.tokenIDs.count]
        try check(shortPrompt.conditioningRows != 20 && shortPrompt.conditioningRows != longPrompt.conditioningRows,
            "fallbackPromptsUseDifferentLengths")
        let square = try KreaConfiguration(width: 512, height: 512, steps: 1, seed: 17017)
        trace.phase("dynamic512")
        let shortCapture = NativeCapture()
        let squareImage = try await pipeline.generate(configuration: square, prompt: shortPrompt, progress: shortCapture.accept)
        try save(squareImage, shortCapture, "dynamic512")
        try check(shortCapture.snapshot.selection?.specialized == false && squareImage.pixelsCHW.shape == [3, 512, 512],
            "dynamicFallback512RunsAndDecodes")
        let specializedIDs = Set(firstOpens.filter { $0.model != assets.decoder.url.path }.compactMap(\.instance))
        let fallbackEvents = trace.events.filter { $0.phase == "dynamic512" }
        let dynamicOpen = try #require(fallbackEvents.first { $0.event == "modelOpened" && assets.denoisers.map(\.url.path).contains($0.model ?? "") })
        let specializedCloses = fallbackEvents.filter { $0.event == "modelClosed" && specializedIDs.contains($0.instance ?? "") }
        try check(specializedCloses.count == 4 && specializedCloses.allSatisfy { $0.sequence < dynamicOpen.sequence },
            "shapeSwitchClosesSpecializedModelsBeforeDynamicLoad")
        let shortConditioning = try #require(shortCapture.snapshot.conditioning)
        let shortOpens = trace.events.filter { $0.event == "modelOpened" }.count
        trace.phase("dynamic512Repeat")
        let shortRepeatCapture = NativeCapture()
        let squareRepeat = try await pipeline.generate(configuration: square, conditioning: shortConditioning,
            progress: shortRepeatCapture.accept)
        try save(squareRepeat, shortRepeatCapture, "dynamic512-repeat")
        try check(squareRepeat.pixelsCHW == squareImage.pixelsCHW && shortRepeatCapture.snapshot.latents == shortCapture.snapshot.latents,
            "dynamicFallbackRepeatIsBitExact")
        try check(trace.events.filter { $0.event == "modelOpened" }.count == shortOpens, "dynamicRepeatReusesModels")

        trace.phase("dynamic384")
        let portrait = try KreaConfiguration(width: 384, height: 512, steps: 1, seed: 17017)
        let longCapture = NativeCapture()
        let portraitImage = try await pipeline.generate(configuration: portrait, prompt: longPrompt, progress: longCapture.accept)
        try save(portraitImage, longCapture, "dynamic384")
        try check(longCapture.snapshot.selection?.specialized == false && portraitImage.pixelsCHW.shape == [3, 512, 384],
            "secondDynamicShapeAndLongerPromptRunAndDecode")
        try check(trace.events.filter { $0.event == "modelOpened" && $0.model == assets.decoder.url.path }.count == 1,
            "decoderSessionSupportsAllThreeShapes")
        let encoderOpens = trace.events.filter { $0.event == "modelOpened" && $0.model == assets.textEncoder.url.path }
        let encoderCloses = trace.events.filter { $0.event == "modelClosed" && $0.model == assets.textEncoder.url.path }
        try check(encoderOpens.count == 2 && encoderCloses.count == 2, "balancedEncodersReleasedForBothPromptLengths")

        trace.phase("unloadActive")
        let longConditioning = try #require(longCapture.snapshot.conditioning)
        let closingCapture = NativeCapture()
        let active = Task {
            try await pipeline.generate(configuration: portrait, conditioning: longConditioning, progress: closingCapture.accept)
        }
        let closeBegin = try await trace.waitForNative(in: "unloadActive")
        trace.add("unloadRequested")
        let close = Task {
            await pipeline.unload()
            trace.add("unloadReturned")
        }
        let deadline = ContinuousClock.now.advanced(by: .seconds(30))
        while await pipeline.state == .running, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        try check(await pipeline.state == .closing, "unloadEntersClosingWhileNativeWorkIsActive")
        try check(!trace.events.contains { $0.phase == "unloadActive" && $0.event == "unloadReturned" },
            "unloadHasNotReturnedDuringNativeWork")
        do {
            _ = try await pipeline.generate(configuration: portrait, conditioning: longConditioning)
            throw NativeFailure(description: "generation accepted while closing")
        } catch AetherError.sessionClosed { trace.add("closingRequestRejected") }
        let closingImage = try await active.value
        await close.value
        try save(closingImage, closingCapture, "unload-active")
        try check(await pipeline.state == .closed, "unloadFinishesClosed")
        try check(closingImage.pixelsCHW == portraitImage.pixelsCHW, "activeGenerationCompletesAndOutputSurvivesUnload")
        let closingEvents = trace.events.filter { $0.phase == "unloadActive" }
        let unloadRequest = try #require(closingEvents.first { $0.event == "unloadRequested" })
        let firstEnd = try #require(closingEvents.first { $0.event == "nativeEnd" && $0.call == closeBegin.call })
        let lastEnd = try #require(closingEvents.last { $0.event == "nativeEnd" })
        let firstClose = try #require(closingEvents.first { $0.event == "modelClosed" })
        let unloadReturn = try #require(closingEvents.first { $0.event == "unloadReturned" })
        try check(closeBegin.sequence < unloadRequest.sequence && unloadRequest.sequence < firstEnd.sequence &&
            lastEnd.sequence < firstClose.sequence && firstClose.sequence < unloadReturn.sequence,
            "unloadReleasesModelsOnlyAfterAllNativeCallsReturn")
        let openedIDs = trace.events.filter { $0.event == "modelOpened" }.compactMap(\.instance).sorted()
        let closedIDs = trace.events.filter { $0.event == "modelClosed" }.compactMap(\.instance).sorted()
        try check(openedIDs == closedIDs, "allOpenedModelSessionsClosedExactlyOnce")
        await pipeline.unload()
        do {
            _ = try await pipeline.generate(configuration: portrait, conditioning: longConditioning)
            throw NativeFailure(description: "generation accepted after terminal unload")
        } catch AetherError.sessionClosed { trace.add("closedRequestRejected") }
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
