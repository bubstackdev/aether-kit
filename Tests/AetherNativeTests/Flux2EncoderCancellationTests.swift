import AetherAssets
import AetherFlux2
import AetherFlux2Tokenizers
import AetherKit
import Foundation
import Testing
@testable import AetherCoreAI
@testable import AetherFlux2CoreAI

/// Opt-in native cancellation of the FLUX.2 text encoder, on the real klein bundle with balanced residency (every
/// prompt request opens the encoder and discards it afterwards). Measures, then checks:
/// - a baseline prompt generation and its exact repeat, with the encoder's open and native-call durations;
/// - cancellation observed during the encoder's native call, and during the encoder's model open, each repeated: the
///   call (or open) returns before cancellation does, no native call starts after the request, the encoder is
///   discarded, the session returns to ready, and the next prompt request reproduces the baseline exactly;
/// - a cancelled call on a resident session, which also discards the retained encoder;
/// - the same cancellation on a session with resource admission and in-process GPU admission: no reservation and no
///   GPU grant remains afterwards.
/// With AETHER_FLUX2_REFERENCE_FIXTURES, the same for the reference (VAE) encoder: `prepare(references:)` cancelled during
/// its first native call and during its open, and `generate(prompt:references:)` cancelled during its native call.
/// Records whether Core AI ended the in-flight call or open early, and the cancellation latency.
/// Run alone under the shared GPU lock.
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_FLUX2_ENCODER_CANCEL"] == "1"
               && nativeEnvironment["AETHER_FLUX2_BUNDLE"] != nil && nativeEnvironment["AETHER_FLUX2_FIXTURES"] != nil
               && nativeEnvironment["AETHER_FLUX2_ENCODER_CANCEL_OUTPUT"] != nil),
      .timeLimit(.minutes(30)))
func flux2NativeEncoderCancellation() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let env = nativeEnvironment
    let output = URL(fileURLWithPath: try #require(env["AETHER_FLUX2_ENCODER_CANCEL_OUTPUT"]))
    #expect(!FileManager.default.fileExists(atPath: output.path), "output directory must be new")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    FileManager.default.createFile(atPath: output.appendingPathComponent("events.jsonl").path, contents: nil)
    let journal = try FileHandle(forWritingTo: output.appendingPathComponent("events.jsonl"))
    defer { try? journal.close() }
    let trace = NativeTrace(journal: journal)
    let report = Flux2Report(url: output.appendingPathComponent("report.json"))
    report.set("kind", "flux2-native-encoder-cancellation")
    let started = Date()
    let assets = try Flux2Assets.open(at: URL(fileURLWithPath: try #require(env["AETHER_FLUX2_BUNDLE"])))
    report.set("bundle", ["fingerprint": assets.fingerprint])
    let t2i = try Flux2Fixture(URL(fileURLWithPath: try #require(env["AETHER_FLUX2_FIXTURES"])))
    let fox = try #require(t2i.cases.first { $0.id == "fox-512" })
    report.set("fixtures", ["t2i": t2i.id, "case": fox.id])
    let tokenizer = try await Flux2Tokenizer.open(assets: assets)
    let prompt = try tokenizer.prepare(fox.prompt)
    let configuration = try Flux2Configuration(width: fox.width, height: fox.height, steps: fox.steps, seed: UInt64(fox.seed))
    let noise = try FloatTensor(shape: [fox.tokens, 128], values: fox.noise)
    let factory = TrackedFlux2Factory(trace: trace)
    let encoderPath = assets.textEncoder.url.path
    let trials = Int(env["AETHER_FLUX2_ENCODER_CANCEL_TRIALS"] ?? "3") ?? 3

    func check(_ condition: Bool, _ name: String) { report.check(name, trace.currentPhase, value: condition ? 1 : 0, bar: 1) }
    func events(_ phase: String) -> [NativeEvent] { trace.events.filter { $0.phase == phase } }
    func seconds(_ list: [NativeEvent], from: (NativeEvent) -> Bool, to: (NativeEvent) -> Bool) -> Double? {
        guard let a = list.firstIndex(where: from), let b = list[a...].firstIndex(where: to),
              let start = list[a].uptime, let end = list[b].uptime else { return nil }
        return end - start
    }
    func encoderTimings(_ phase: String) {
        let list = events(phase)
        if let open = seconds(list, from: { $0.event == "modelOpenBegin" && $0.model == encoderPath },
                              to: { $0.event == "modelOpened" && $0.model == encoderPath }) {
            report.timing("encoderOpen", phase, open)
        }
        if let call = seconds(list, from: { $0.event == "nativeBegin" && $0.model == encoderPath },
                              to: { $0.event == "nativeEnd" && $0.model == encoderPath }) {
            report.timing("encoderNativeCall", phase, call)
        }
    }
    func generate(_ session: Flux2GenerationSession) -> Task<GeneratedImage, any Error> {
        Task {
            defer { trace.add("generationTaskFinished") }
            return try await session.generate(configuration: configuration, prompt: prompt, noise: noise)
        }
    }
    /// Waits for the encoder's open to begin; nothing gates or delays the open itself.
    func waitForEncoderOpen(in phase: String) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(600))
        while ContinuousClock.now < deadline {
            if events(phase).contains(where: { $0.event == "modelOpenBegin" && $0.model == encoderPath }) { return }
            try await Task.sleep(for: .milliseconds(1))
        }
        throw NativeFailure(description: "timed out waiting for the encoder open in \(phase)")
    }

    // A. Baseline and exact repeat: the encoder opens, runs once and closes in each request.
    let session = Flux2GenerationSession(assets: assets, factory: factory)
    trace.phase("baseline")
    let baseline = try await session.generate(configuration: configuration, prompt: prompt, noise: noise)
    let baselineHash = nativeHash(baseline.pixelsCHW.values)
    try baseline.savePNG(to: output.appendingPathComponent("baseline.png"))
    report.record("baselineSSIM", "baseline", ssim(unitHWC(baseline.pixelsCHW), fox.image, width: fox.width, height: fox.height))
    encoderTimings("baseline")
    for index in 1...2 {
        trace.phase("repeat\(index)")
        let repeated = try await session.generate(configuration: configuration, prompt: prompt, noise: noise)
        check(nativeHash(repeated.pixelsCHW.values) == baselineHash, "repeatIsExact")
        encoderTimings("repeat\(index)")
    }

    // B. Cancellation during the encoder's native call, then during its model open.
    func cancellation(_ session: Flux2GenerationSession, phase: String, duringOpen: Bool) async throws {
        trace.phase(phase)
        let task = generate(session)
        if duringOpen {
            try await waitForEncoderOpen(in: phase)
        } else {
            _ = try await trace.waitForNative(in: phase, model: encoderPath, observationInterval: .milliseconds(2))
        }
        trace.add("cancelRequested")
        task.cancel()
        var cancelled = false
        do { _ = try await task.value } catch is CancellationError { cancelled = true }
        trace.add("cancellationReturned")
        check(cancelled, "cancellationThrows")
        let list = events(phase)
        let request = list.firstIndex { $0.event == "cancelRequested" }!
        let returned = list.firstIndex { $0.event == "cancellationReturned" }!
        if let latency = seconds(list, from: { $0.event == "cancelRequested" }, to: { $0.event == "cancellationReturned" }) {
            report.timing("cancellationLatency", phase, latency)
        }
        if duringOpen {
            // The open either completed (and the encoder was then discarded) or Core AI ended it early.
            let opened = list.contains { $0.event == "modelOpened" && $0.model == encoderPath }
            report.record("openCompletedAfterCancel", phase, opened ? 1 : 0)
            if let open = seconds(list, from: { $0.event == "modelOpenBegin" && $0.model == encoderPath },
                                  to: { ($0.event == "modelOpened" && $0.model == encoderPath) || $0.event == "cancellationReturned" }) {
                report.timing("encoderOpenUntilReturnOrCancel", phase, open)
            }
            check(!list.contains { $0.event == "nativeBegin" }, "noNativeCallAfterCancellation")
            check(!opened || list.contains { $0.event == "modelClosed" && $0.model == encoderPath }, "encoderDiscarded")
        } else {
            let begin = list.first { $0.event == "nativeBegin" && $0.model == encoderPath }!
            let end = list.firstIndex { $0.event == "nativeEnd" && $0.call == begin.call }
            check(end != nil && end! < returned, "nativeReturnPrecedesCancellation")
            // Core AI returned the call's outputs (it ran to completion) if their copy began; Kit's own check after the
            // joined call then threw. Otherwise Core AI ended the call itself.
            let completed = list.contains { $0.event == "outputCopyBegin" && $0.model == encoderPath }
            report.record("nativeCallRanToCompletion", phase, completed ? 1 : 0)
            if let call = seconds(list, from: { $0.event == "nativeBegin" && $0.model == encoderPath },
                                  to: { $0.event == "nativeEnd" && $0.model == encoderPath }) {
                report.timing("encoderNativeCall", phase, call)
            }
            let close = list.firstIndex { $0.event == "modelClosed" && $0.model == encoderPath }
            check(close != nil && close! > end! && close! < returned, "encoderDiscardedBeforeReturn")
            check(list[list.index(after: request)...].filter { $0.event == "nativeBegin" }.isEmpty, "noNativeCallAfterCancellation")
        }
        check(await session.state == .ready, "stateReturnsToReady")
        trace.phase(phase + "Recovery")
        let recovered = try await session.generate(configuration: configuration, prompt: prompt, noise: noise)
        check(nativeHash(recovered.pixelsCHW.values) == baselineHash, "recoveryIsExact")
    }
    for trial in 1...trials {
        try await cancellation(session, phase: "encoderCallCancellation\(trial)", duringOpen: false)
    }
    for trial in 1...trials {
        try await cancellation(session, phase: "encoderOpenCancellation\(trial)", duringOpen: true)
    }
    await session.unload()

    // Resident residency keeps the encoder between requests; a cancelled call still discards it, and the next request
    // opens a new one.
    let resident = Flux2GenerationSession(assets: assets, factory: factory, residency: .resident)
    trace.phase("residentBaseline")
    let residentBaseline = try await resident.generate(configuration: configuration, prompt: prompt, noise: noise)
    check(nativeHash(residentBaseline.pixelsCHW.values) == baselineHash, "residentBaselineIsExact")
    encoderTimings("residentBaseline")
    trace.phase("residentRepeat")
    _ = try await resident.generate(configuration: configuration, prompt: prompt, noise: noise)
    encoderTimings("residentRepeat")
    try await cancellation(resident, phase: "residentEncoderCallCancellation", duringOpen: false)
    check(events("residentEncoderCallCancellationRecovery").contains { $0.event == "modelOpened" && $0.model == encoderPath },
          "residentRecoveryOpensANewEncoder")
    await resident.unload()

    // C. With resource admission and in-process GPU admission: nothing remains reserved or granted after cancellation.
    let gib: UInt64 = 1 << 30
    let estimates = try ([assets.textEncoder, assets.decoder] + assets.denoisers).map {
        try Flux2ModelResourceEstimate(asset: $0, residentBytes: 2 * gib, runScratchBytes: gib / 4)
    }
    let coordinator = try ResourceCoordinator(capacityBytes: 64 * gib)
    let admission = try GPUAdmission(waiting: .failFast)
    let admitted = Flux2GenerationSession(assets: assets, factory: factory, resources: try Flux2ResourceAdmission(
        coordinator: coordinator, operationBytes: gib / 2, maximumImageTokens: 1024, models: estimates), gpu: admission.owner("flux2"))
    trace.phase("admittedBaseline")
    let admittedBaseline = try await admitted.generate(configuration: configuration, prompt: prompt, noise: noise)
    check(nativeHash(admittedBaseline.pixelsCHW.values) == baselineHash, "admittedBaselineIsExact")
    let retained = coordinator.snapshot.reservedBytes
    check(retained == 4 * 2 * gib, "admittedRetainsDenoisersAndDecoder")
    try await cancellation(admitted, phase: "admittedEncoderCallCancellation", duringOpen: false)
    // After the recovery request the session retains the same models as after the baseline.
    check(coordinator.snapshot.reservedBytes == retained, "noReservationLeakedByCancellation")
    check(admission.snapshot.running == 0 && admission.snapshot.queued == 0, "noGPUGrantLeakedByCancellation")
    let probe = try await admission.admit(owner: "probe", waiting: .failFast)
    probe.release()
    check(true, "gpuAdmissionImmediatelyAvailable")
    await admitted.unload()
    check(coordinator.snapshot.reservedBytes == 0, "unloadReleasesAllReservations")

    // D. The reference encoder (with AETHER_FLUX2_REFERENCE_FIXTURES): `prepare(references:)` for the two-reference case,
    // cancelled during the VAE encoder's first native call and during its open, and `generate(prompt:references:)`
    // cancelled during the encoder's native call. The second image is never submitted after a cancel, the encoder is
    // discarded before the request returns, and recovery reproduces the uncancelled latents and pixels exactly.
    if let path = env["AETHER_FLUX2_REFERENCE_FIXTURES"], let referenceAsset = assets.referenceEncoder {
        let fixture = try Flux2Fixture(URL(fileURLWithPath: path))
        report.set("fixtures", ["t2i": t2i.id, "case": fox.id, "reference": fixture.id])
        let two = try #require(fixture.cases.first { $0.references.count == 2 })
        let images = try two.references.map {
            try Flux2ReferenceImage(pixels: FloatTensor(shape: [3, $0.height, $0.width], values: $0.pixels))
        }
        let vaePath = referenceAsset.url.path
        let referenceSession = Flux2GenerationSession(assets: assets, factory: factory)
        func latentsHash(_ prepared: Flux2PreparedReferences) -> String { nativeHash(prepared.images.flatMap(\.latents.values)) }
        trace.phase("referenceBaseline")
        let prepared = try await referenceSession.prepare(references: images, generatedTokens: two.tokens)
        let preparedHash = latentsHash(prepared)
        report.check("referenceLatentCosine", two.id, value: cosine(prepared.images.flatMap(\.latents.values), two.referenceLatents), bar: 0.999)
        let baselineEvents = events("referenceBaseline")
        if let open = seconds(baselineEvents, from: { $0.event == "modelOpenBegin" && $0.model == vaePath },
                              to: { $0.event == "modelOpened" && $0.model == vaePath }) {
            report.timing("referenceEncoderOpen", "referenceBaseline", open)
        }
        let begins = baselineEvents.filter { $0.event == "nativeBegin" && $0.model == vaePath }
        for begin in begins {
            if let end = baselineEvents.first(where: { $0.event == "nativeEnd" && $0.call == begin.call }), let a = begin.uptime, let b = end.uptime {
                report.timing("referenceEncoderNativeCall", "referenceBaseline", b - a)
            }
        }
        check(begins.count == 2, "baselineEncodesBothImages")
        let referencePrompt = try tokenizer.prepare(two.prompt)
        let referenceConfiguration = try Flux2Configuration(width: two.width, height: two.height, steps: two.steps, seed: UInt64(two.seed))
        let referenceNoise = try FloatTensor(shape: [two.tokens, 128], values: two.noise)
        trace.phase("referenceGenerateBaseline")
        let generatedBaseline = try await referenceSession.generate(configuration: referenceConfiguration, prompt: referencePrompt,
                                                                    references: images, noise: referenceNoise)
        let generatedHash = nativeHash(generatedBaseline.pixelsCHW.values)

        func referenceCancellation(phase: String, duringOpen: Bool, throughGenerate: Bool) async throws {
            trace.phase(phase)
            let task = Task { () -> String in
                defer { trace.add("generationTaskFinished") }
                if throughGenerate {
                    let image = try await referenceSession.generate(configuration: referenceConfiguration, prompt: referencePrompt,
                                                                    references: images, noise: referenceNoise)
                    return nativeHash(image.pixelsCHW.values)
                }
                return latentsHash(try await referenceSession.prepare(references: images, generatedTokens: two.tokens))
            }
            if duringOpen {
                let deadline = ContinuousClock.now.advanced(by: .seconds(600))
                while !events(phase).contains(where: { $0.event == "modelOpenBegin" && $0.model == vaePath }) {
                    guard ContinuousClock.now < deadline else { throw NativeFailure(description: "no reference encoder open in \(phase)") }
                    try await Task.sleep(for: .milliseconds(1))
                }
            } else {
                _ = try await trace.waitForNative(in: phase, model: vaePath, observationInterval: .milliseconds(2))
            }
            trace.add("cancelRequested")
            task.cancel()
            var cancelled = false
            do { _ = try await task.value } catch is CancellationError { cancelled = true }
            trace.add("cancellationReturned")
            check(cancelled, "cancellationThrows")
            let list = events(phase)
            let request = list.firstIndex { $0.event == "cancelRequested" }!
            let returned = list.firstIndex { $0.event == "cancellationReturned" }!
            if let latency = seconds(list, from: { $0.event == "cancelRequested" }, to: { $0.event == "cancellationReturned" }) {
                report.timing("cancellationLatency", phase, latency)
            }
            let vaeCalls = list.filter { $0.event == "nativeBegin" && $0.model == vaePath }
            if duringOpen {
                let opened = list.contains { $0.event == "modelOpened" && $0.model == vaePath }
                report.record("openCompletedAfterCancel", phase, opened ? 1 : 0)
                check(vaeCalls.isEmpty, "noEncoderCallAfterCancellation")
                check(!opened || list.contains { $0.event == "modelClosed" && $0.model == vaePath }, "encoderDiscarded")
            } else {
                let begin = vaeCalls.first!
                let end = list.firstIndex { $0.event == "nativeEnd" && $0.call == begin.call }
                check(end != nil && end! < returned, "nativeReturnPrecedesCancellation")
                report.record("nativeCallRanToCompletion", phase,
                              list.contains { $0.event == "outputCopyBegin" && $0.model == vaePath } ? 1 : 0)
                if let call = seconds(list, from: { $0.event == "nativeBegin" && $0.model == vaePath },
                                      to: { $0.event == "nativeEnd" && $0.model == vaePath }) {
                    report.timing("referenceEncoderNativeCall", phase, call)
                }
                check(vaeCalls.count == 1, "secondImageNeverSubmitted")
                let close = list.firstIndex { $0.event == "modelClosed" && $0.model == vaePath }
                check(close != nil && close! > end! && close! < returned, "encoderDiscardedBeforeReturn")
                check(list[list.index(after: request)...].filter { $0.event == "nativeBegin" }.isEmpty, "noNativeCallAfterCancellation")
            }
            check(await referenceSession.state == .ready, "stateReturnsToReady")
            trace.phase(phase + "Recovery")
            if throughGenerate {
                let image = try await referenceSession.generate(configuration: referenceConfiguration, prompt: referencePrompt,
                                                                references: images, noise: referenceNoise)
                check(nativeHash(image.pixelsCHW.values) == generatedHash, "recoveryIsExact")
            } else {
                check(latentsHash(try await referenceSession.prepare(references: images, generatedTokens: two.tokens)) == preparedHash,
                      "recoveryIsExact")
            }
        }
        for trial in 1...trials {
            try await referenceCancellation(phase: "referenceCallCancellation\(trial)", duringOpen: false, throughGenerate: false)
        }
        for trial in 1...trials {
            try await referenceCancellation(phase: "referenceOpenCancellation\(trial)", duringOpen: true, throughGenerate: false)
        }
        try await referenceCancellation(phase: "referenceGenerateCancellation", duringOpen: false, throughGenerate: true)
        await referenceSession.unload()
    }

    let all = trace.events
    let opened = Set(all.filter { $0.event == "modelOpened" }.compactMap(\.instance))
    let closedCounts = Dictionary(grouping: all.filter { $0.event == "modelClosed" }.compactMap(\.instance), by: { $0 }).mapValues(\.count)
    trace.phase("summary")
    check(opened.allSatisfy { closedCounts[$0] == 1 } && closedCounts.count == opened.count, "everyInstanceClosedOnce")
    check(all.filter { $0.event == "nativeBegin" }.count == all.filter { $0.event == "nativeEnd" }.count, "everyNativeCallReturned")
    report.set("elapsedSeconds", String(format: "%.1f", Date().timeIntervalSince(started)))
    report.set("limits", ["Native boundaries bracket the awaited function.run; GPU queues are not instrumented.",
                          "Cancellation is requested 2 ms after the encoder's native call begins, or right after its open begins.",
                          "Admission estimates are logical test values, not measured memory."])
    #expect(trace.journalError == nil)
    let failures = report.finish()
    #expect(failures.isEmpty, "\(failures)")
}
