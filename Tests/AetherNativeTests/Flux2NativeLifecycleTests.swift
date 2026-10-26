import AetherAssets
import AetherFlux2
import AetherFlux2Tokenizers
import AetherKit
import Foundation
import Synchronization
import Testing
@testable import AetherCoreAI
@testable import AetherFlux2CoreAI

/// Opt-in native lifecycle qualification of `Flux2GenerationSession` on the real klein bundle, modeled on Krea's
/// `AETHER_NATIVE_QUALIFICATION` suite. A tracked factory opens real Core AI sessions whose backend observers bracket
/// the awaited native `function.run`; nothing gates or delays native work. Phases:
/// baseline and exact repeat with model/function reuse; cancellation observed during a native denoiser call and during
/// the native decoder call, each with overlap rejection and exact recovery; balanced prompt ordering; references;
/// resident reuse across prompts and sizes; unload during active work; admission charge/release, refusal and rollback.
/// Run alone under the shared GPU lock.
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_FLUX2_LIFECYCLE"] == "1"
               && nativeEnvironment["AETHER_FLUX2_BUNDLE"] != nil && nativeEnvironment["AETHER_FLUX2_FIXTURES"] != nil
               && nativeEnvironment["AETHER_FLUX2_LIFECYCLE_OUTPUT"] != nil),
      .timeLimit(.minutes(60)))
func flux2NativeLifecycleQualification() async throws {
    let env = nativeEnvironment
    let output = URL(fileURLWithPath: try #require(env["AETHER_FLUX2_LIFECYCLE_OUTPUT"]))
    #expect(!FileManager.default.fileExists(atPath: output.path), "output directory must be new")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    FileManager.default.createFile(atPath: output.appendingPathComponent("events.jsonl").path, contents: nil)
    let journal = try FileHandle(forWritingTo: output.appendingPathComponent("events.jsonl"))
    defer { try? journal.close() }
    let trace = NativeTrace(journal: journal)
    let report = Flux2Report(url: output.appendingPathComponent("report.json"))
    report.set("kind", "flux2-native-lifecycle")
    let started = Date()
    let assets = try Flux2Assets.open(at: URL(fileURLWithPath: try #require(env["AETHER_FLUX2_BUNDLE"])))
    try assets.verify()
    report.set("bundle", ["fingerprint": assets.fingerprint, "referenceFingerprint": assets.referenceFingerprint ?? ""])
    let t2i = try Flux2Fixture(URL(fileURLWithPath: try #require(env["AETHER_FLUX2_FIXTURES"])))
    let fox = try #require(t2i.cases.first { $0.id == "fox-512" })
    let references = try env["AETHER_FLUX2_REFERENCE_FIXTURES"].map { try Flux2Fixture(URL(fileURLWithPath: $0)) }
    let tokenizer = try await Flux2Tokenizer.open(assets: assets)
    let configuration = try Flux2Configuration(width: fox.width, height: fox.height, steps: fox.steps, seed: UInt64(fox.seed))
    let conditioning = try Flux2Conditioning(provided: FloatTensor(shape: [512, 7680], values: fox.text))
    let noise = try FloatTensor(shape: [fox.tokens, 128], values: fox.noise)
    let factory = TrackedFlux2Factory(trace: trace)
    func check(_ condition: Bool, _ name: String) {
        report.check(name, trace.currentPhase, value: condition ? 1 : 0, bar: 1)
    }
    func events(_ phase: String) -> [NativeEvent] { trace.events.filter { $0.phase == phase } }
    func opens(_ phase: String) -> [String] { events(phase).filter { $0.event == "modelOpened" }.compactMap { $0.model.map(role) } }
    func closes(_ phase: String) -> [String] { events(phase).filter { $0.event == "modelClosed" }.compactMap { $0.model.map(role) } }
    func save(_ image: GeneratedImage, _ name: String) throws -> String {
        try image.savePNG(to: output.appendingPathComponent(name + ".png"))
        return nativeHash(image.pixelsCHW.values)
    }
    func run(_ session: Flux2GenerationSession, _ phase: String) -> Task<GeneratedImage, any Error> {
        Task {
            defer { trace.add("generationTaskFinished") }
            return try await session.generate(configuration: configuration, conditioning: conditioning, noise: noise)
        }
    }

    // A. Baseline and exact repeat (balanced).
    let session = Flux2GenerationSession(assets: assets, factory: factory)
    trace.phase("baseline")
    let baseline = try await session.generate(configuration: configuration, conditioning: conditioning, noise: noise)
    let baselineHash = try save(baseline, "baseline")
    report.check("baselineSSIM", "baseline", value: ssim(unitHWC(baseline.pixelsCHW), fox.image, width: fox.width, height: fox.height), bar: 0.94)
    check(opens("baseline") == ["denoiser0", "denoiser1", "denoiser2", "vaeDecoder"], "baselineOpensDenoisersThenDecoder")
    trace.phase("repeat")
    let repeated = try await session.generate(configuration: configuration, conditioning: conditioning, noise: noise)
    check(try save(repeated, "repeat") == baselineHash, "repeatIsExact")
    check(opens("repeat").isEmpty && closes("repeat").isEmpty, "repeatReusesModels")
    check(!events("repeat").contains { $0.event == "functionLoadBegin" }, "repeatReusesNativeFunctions")
    check(events("repeat").filter { $0.event == "nativeBegin" }.count == fox.steps * 3 + 1, "repeatNativeCallCount")

    // B/C. Cancellation observed during a native denoiser call, then during the native decoder call.
    for (phase, model, interval) in [("denoiserCancellation", "denoiser1", Duration.milliseconds(5)),
                                     ("decoderCancellation", "vaeDecoder", Duration.milliseconds(1))] {
        trace.phase(phase)
        let task = run(session, phase)
        let begin = try await trace.waitForNative(in: phase, model: path(assets, model), observationInterval: interval)
        do {
            _ = try await session.prepare(prompt: tokenizer.prepare("overlap"))
            check(false, "overlapRejected")
        } catch AetherError.sessionBusy {
            check(true, "overlapRejected")
        }
        trace.add("cancelRequested")
        task.cancel()
        var cancelled = false
        do { _ = try await task.value } catch is CancellationError { cancelled = true }
        trace.add("cancellationReturned")
        check(cancelled, "cancellationThrows")
        let list = events(phase)
        let end = list.firstIndex { $0.event == "nativeEnd" && $0.call == begin.call }
        let returned = list.firstIndex { $0.event == "cancellationReturned" }
        check(end != nil && returned != nil && end! < returned!, "nativeReturnPrecedesCancellation")
        let request = list.firstIndex { $0.event == "cancelRequested" }!
        let laterCalls = list[list.index(after: request)...].filter { $0.event == "nativeBegin" }
        check(laterCalls.isEmpty, "noNativeCallAfterCancellation")
        check(await session.state == .ready, "stateReturnsToReady")
        trace.phase(phase + "Recovery")
        let recovered = try await session.generate(configuration: configuration, conditioning: conditioning, noise: noise)
        check(try save(recovered, phase + "-recovery") == baselineHash, "recoveryIsExact")
        check(opens(phase + "Recovery").isEmpty, "recoveryReusesModels")
    }

    // D. Balanced prompt: denoisers close before the encoder opens; the encoder closes before denoisers reload.
    trace.phase("balancedPrompt")
    let promptImage = try await session.generate(configuration: configuration, prompt: tokenizer.prepare(fox.prompt), noise: noise)
    _ = try save(promptImage, "balanced-prompt")
    let ordered = events("balancedPrompt").filter { ["modelOpened", "modelClosed"].contains($0.event) }
        .map { "\($0.event == "modelOpened" ? "open" : "close"):\(role($0.model!))" }
    check(ordered == ["close:denoiser0", "close:denoiser1", "close:denoiser2", "open:textEncoder", "close:textEncoder",
                      "open:denoiser0", "open:denoiser1", "open:denoiser2"], "balancedCloseBeforeAcquire")
    report.record("balancedPromptSSIM", "balancedPrompt", ssim(unitHWC(promptImage.pixelsCHW), fox.image, width: fox.width, height: fox.height))

    // E. References: encoder opened once per preparation and closed; denoisers recycled for the new token count.
    if let reference = references?.cases.first {
        trace.phase("references")
        let images = try reference.references.map { try Flux2ReferenceImage(pixels: FloatTensor(shape: [3, $0.height, $0.width], values: $0.pixels)) }
        let prepared = try await session.prepare(references: images, generatedTokens: reference.tokens)
        let image = try await session.generate(
            configuration: try Flux2Configuration(width: reference.width, height: reference.height, steps: reference.steps, seed: UInt64(reference.seed)),
            conditioning: try Flux2Conditioning(provided: FloatTensor(shape: [512, 7680], values: reference.text)), references: prepared,
            noise: try FloatTensor(shape: [reference.tokens, 128], values: reference.noise))
        _ = try save(image, "reference-" + reference.id)
        report.check("referenceSSIM", "references", value: ssim(unitHWC(image.pixelsCHW), reference.image, width: reference.width, height: reference.height), bar: 0.94)
        check(opens("references").filter { $0 == "vaeEncoder" }.count == 1 && closes("references").contains("vaeEncoder"), "referenceEncoderReleased")
        check(closes("references").contains("denoiser0") && opens("references").contains("denoiser0"), "denoisersRecycledForTokenCount")
    }

    // F. Unload during active work: rejects new work, completes the active generation, closes everything once.
    trace.phase("activeUnload")
    let active = run(session, "activeUnload")
    _ = try await trace.waitForNative(in: "activeUnload", model: path(assets, "denoiser0"))
    trace.add("unloadRequested")
    let unloading = Task { await session.unload(); trace.add("unloadReturned") }
    while await session.state != .closing { await Task.yield() }
    do { _ = try await session.prepare(prompt: tokenizer.prepare("rejected")); check(false, "closingRejectsWork") }
    catch AetherError.sessionClosed { check(true, "closingRejectsWork") }
    let finished = try await active.value
    await unloading.value
    check(try save(finished, "active-unload") == baselineHash, "activeGenerationCompletesExactly")
    check(await session.state == .closed, "unloadCloses")
    let unloadEvents = events("activeUnload")
    // Closes before the request are the shape recycle at generation start; only unload's releases count here.
    let requested = unloadEvents.firstIndex { $0.event == "unloadRequested" } ?? 0
    let lastNativeEnd = unloadEvents.lastIndex { $0.event == "nativeEnd" } ?? -1
    let releases = unloadEvents.indices.filter { $0 > requested && unloadEvents[$0].event == "modelCloseBegin" }
    check(releases.count == 4 && releases.allSatisfy { $0 > lastNativeEnd }, "nativeReturnsPrecedeModelRelease")
    do { _ = try await session.generate(configuration: configuration, conditioning: conditioning, noise: noise); check(false, "closedRejectsWork") }
    catch AetherError.sessionClosed { check(true, "closedRejectsWork") }
    await session.unload()
    check(finished.pixelsCHW.values.count == 3 * fox.width * fox.height, "ownedOutputValidAfterUnload")

    // G. Resident reuse across prompts and sizes.
    trace.phase("resident")
    let resident = Flux2GenerationSession(assets: assets, factory: factory, residency: .resident)
    _ = try await resident.generate(configuration: configuration, prompt: tokenizer.prepare(fox.prompt), noise: noise)
    trace.phase("residentSecondPrompt")
    _ = try await resident.generate(configuration: configuration, prompt: tokenizer.prepare("A lighthouse at dusk, watercolor"), noise: noise)
    check(opens("residentSecondPrompt").isEmpty && closes("residentSecondPrompt").isEmpty, "residentReusesEncoderAndDenoisers")
    trace.phase("residentNewSize")
    let other = try Flux2Configuration(width: 576, height: 448, seed: 4576)
    _ = try await resident.generate(configuration: other, prompt: tokenizer.prepare("A bowl of ramen on a wooden table, overhead photo"))
    let resized = opens("residentNewSize")
    check(!resized.contains("textEncoder"), "residentKeepsEncoderAcrossSizes")
    check(Set(resized) == ["denoiser0", "denoiser1", "denoiser2", "vaeDecoder"] && Set(closes("residentNewSize")) == Set(resized),
          "residentRecyclesShapeDependentModels")
    trace.phase("residentReturn")
    let back = try await resident.generate(configuration: configuration, conditioning: conditioning, noise: noise)
    check(nativeHash(back.pixelsCHW.values) == baselineHash, "residentReturnIsExact")
    await resident.unload()

    // H. Admission: charge during an in-flight call, release after, refusal before load, rollback on capacity.
    trace.phase("admission")
    let gib: UInt64 = 1 << 30
    let estimates = try ([assets.textEncoder, assets.decoder] + assets.denoisers).map {
        try Flux2ModelResourceEstimate(asset: $0, residentBytes: 2 * gib, runScratchBytes: gib / 4)
    }
    let coordinator = try ResourceCoordinator(capacityBytes: 64 * gib)
    let admitted = Flux2GenerationSession(assets: assets, factory: factory, resources: try Flux2ResourceAdmission(
        coordinator: coordinator, operationBytes: gib / 2, maximumImageTokens: 1024, models: estimates))
    let charged = Task { () -> GeneratedImage in
        defer { trace.add("generationTaskFinished") }
        return try await admitted.generate(configuration: configuration, conditioning: conditioning, noise: noise)
    }
    _ = try await trace.waitForNative(in: "admission", model: path(assets, "denoiser2"))
    let during = coordinator.snapshot.reservedBytes
    let admittedImage = try await charged.value
    check(during == gib / 2 + 3 * 2 * gib + gib / 4, "chargesOperationResidencyAndScratch")
    check(coordinator.snapshot.reservedBytes == 4 * 2 * gib, "retainsOnlyResidencyAfterCompletion")
    check(nativeHash(admittedImage.pixelsCHW.values) == baselineHash, "admittedGenerationIsExact")
    var refused = false
    do { _ = try await admitted.generate(configuration: try Flux2Configuration(width: 768, height: 768), conditioning: conditioning) }
    catch AetherError.invalidRequest { refused = true }
    check(refused && coordinator.snapshot.reservedBytes == 8 * gib, "oversizedRefusedBeforeLoad")
    refused = false
    if let reference = references?.cases.first?.references.first {
        do { _ = try await admitted.prepare(references: [try Flux2ReferenceImage(pixels: FloatTensor(shape: [3, reference.height, reference.width], values: reference.pixels))]) }
        catch AetherError.invalidRequest { refused = true }
        check(refused && coordinator.snapshot.reservedBytes == 8 * gib, "unknownEstimateRefusedBeforeRelease")
    }
    await admitted.unload()
    check(coordinator.snapshot.reservedBytes == 0, "unloadReleasesAllReservations")
    let small = try ResourceCoordinator(capacityBytes: gib / 2 + 2 * 2 * gib + gib / 4)
    let constrained = Flux2GenerationSession(assets: assets, factory: factory, resources: try Flux2ResourceAdmission(
        coordinator: small, operationBytes: gib / 2, maximumImageTokens: 1024, models: estimates))
    var insufficient = false
    do { _ = try await constrained.generate(configuration: configuration, conditioning: conditioning, noise: noise) }
    catch ResourceAdmissionError.insufficientCapacity { insufficient = true }
    check(insufficient && small.snapshot.reservedBytes == 0, "insufficientCapacityRollsBack")
    check(opens("admission").filter { $0 == "denoiser2" }.count == 1, "thirdSegmentNeverOpenedWithoutCapacity")
    await constrained.unload()

    // Every opened instance closed exactly once, after its native calls returned.
    let all = trace.events
    let opened = Set(all.filter { $0.event == "modelOpened" }.compactMap(\.instance))
    let closedCounts = Dictionary(grouping: all.filter { $0.event == "modelClosed" }.compactMap(\.instance), by: { $0 }).mapValues(\.count)
    trace.phase("summary")
    check(opened.allSatisfy { closedCounts[$0] == 1 } && closedCounts.count == opened.count, "everyInstanceClosedOnce")
    let pairs = all.filter { $0.event == "nativeBegin" }.count
    check(pairs == all.filter { $0.event == "nativeEnd" }.count, "everyNativeCallReturned")
    report.set("nativeCalls", pairs)
    report.set("modelInstances", opened.count)
    report.set("functionLoads", all.filter { $0.event == "functionLoadBegin" }.count)
    report.set("functionReuses", all.filter { $0.event == "functionReused" }.count)
    report.set("elapsedSeconds", String(format: "%.1f", Date().timeIntervalSince(started)))
    report.set("limits", ["Native boundaries bracket the awaited function.run; GPU queues are not instrumented.",
                          "Admission estimates are logical test values, not measured memory.",
                          "Cancellation is cooperative between native calls; an observed in-flight call always completes."])
    #expect(trace.journalError == nil)
    let failures = report.finish()
    #expect(failures.isEmpty, "\(failures)")
}

private func role(_ path: String) -> String {
    let name = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
    for suffix in ["denoiser0", "denoiser1", "denoiser2", "vae-encoder", "vae", "text"] where name.hasSuffix(suffix) {
        return ["vae-encoder": "vaeEncoder", "vae": "vaeDecoder", "text": "textEncoder"][suffix] ?? suffix
    }
    return name
}

private func path(_ assets: Flux2Assets, _ role: String) -> String {
    switch role {
    case "vaeDecoder": assets.decoder.url.path
    case "vaeEncoder": assets.referenceEncoder!.url.path
    case "textEncoder": assets.textEncoder.url.path
    default: assets.denoisers[Int(role.dropFirst("denoiser".count))!].url.path
    }
}

extension NativeTrace {
    var currentPhase: String { events.last(where: { $0.event == "phase" })?.phase ?? "" }
}

struct TrackedFlux2Model: StagedModel {
    let session: ModelSession
    let trace: NativeTrace
    let model: String
    let instance: String
    var functions: [ModelFunctionDescription] { session.functions }
    func runTensors(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue] {
        trace.add("callSubmitted", model: model, instance: instance, detail: function)
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

struct TrackedFlux2Factory: Flux2ModelFactory {
    let trace: NativeTrace
    func open(_ asset: DeclaredModelAsset, frequentReshapes: Bool) async throws -> any StagedModel {
        let model = asset.url.path, instance = UUID().uuidString
        trace.add("modelOpenBegin", model: model, instance: instance)
        let backend = try await CoreAIBackend.load(at: asset.url, compute: .preferGPU, expectFrequentReshapes: frequentReshapes,
            executionObserver: { began in trace.native(began: began, model: model, instance: instance) },
            boundaryObserver: { boundary in
                if boundary.hasPrefix("function") { trace.add(boundary, model: model, instance: instance) }
            })
        trace.add("modelOpened", model: model, instance: instance)
        return TrackedFlux2Model(session: ModelSession(backend: backend, functions: backend.functions), trace: trace,
                                 model: model, instance: instance)
    }
}
