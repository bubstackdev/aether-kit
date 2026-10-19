import AetherKit
import AetherCoreAI
import AetherKrea
import Foundation
import Synchronization
import Testing
@testable import AetherKreaCoreAI

private final class EventLog: Sendable {
    let values = Mutex<[String]>([])
    func append(_ value: String) { values.withLock { $0.append(value) } }
    var snapshot: [String] { values.withLock { $0 } }
}

private func declaration(_ type: String, _ shape: [Int]) -> [String: Any] {
    ["kind": "tensor", "scalarType": type, "shape": shape]
}

private func describe(_ name: String, inputs: [String: [String: Any]], outputs: [String: [String: Any]]) throws -> ModelFunctionDescription {
    try JSONDecoder().decode(ModelFunctionDescription.self, from: JSONSerialization.data(withJSONObject: [
        "name": name, "inputs": inputs, "outputs": outputs, "states": [:],
    ]))
}

private func segmentSignatures() throws -> [ModelFunctionDescription] {
    let image = declaration("float32", [1, 6144])
    let text = declaration("float32", [1, 6144])
    let cond = declaration("float32", [8, 6144])
    return try [
        describe("segment", inputs: [
            "latents": declaration("float16", [1, 64]), "text_embeds": declaration("float16", [1, 30720]),
            "sigma": declaration("float32", [1]), "rope_image": declaration("float32", [1, 2, 128]),
        ], outputs: ["cond": cond, "image_1": image, "text_1": text]),
        describe("segment", inputs: ["image_1": image, "text_1": text], outputs: ["image_2": image, "text_2": text]),
        describe("segment", inputs: ["image_2": image, "text_2": text, "cond": cond], outputs: ["image_3": image, "text_3": text]),
        describe("segment", inputs: ["image_3": image, "text_3": text, "cond": cond], outputs: ["velocity": declaration("float16", [1, 64])]),
    ]
}

private struct FakeModel: KreaExecutable {
    let functions: [ModelFunctionDescription]
    let run: @Sendable ([String: TensorValue]) async throws -> [String: TensorValue]
    let close: @Sendable () -> Void
    func runTensors(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue] { try await run(inputs) }
    func unload() async { close() }
}

private final class FakeFactory: KreaModelFactory, Sendable {
    let models: [String: FakeModel]
    let log: EventLog
    let failOnce: Mutex<String?>

    init(log: EventLog, gate: PipelineGate? = nil, failOnce: String? = nil, decoderGate: PipelineGate? = nil,
         encoderGate: PipelineGate? = nil, failEncodingOnce: Bool = false, checkDecoderLatents: Bool = true) throws {
        self.log = log
        self.failOnce = Mutex(failOnce)
        var models: [String: FakeModel] = [:]
        for (index, signature) in try segmentSignatures().enumerated() {
            let model = FakeModel(functions: [signature], run: { inputs in
                log.append("run\(index)")
                if index == 0, let gate { await gate.enter() }
                if index >= 2 {
                    guard case .float32(let cond) = inputs["cond"] else {
                        throw AetherError.invalidTensor("nonadjacent cond carry was lost")
                    }
                    #expect(cond.values.first == 7)
                }
                var result: [String: TensorValue] = [:]
                for (name, value) in signature.outputs {
                    let shape = value.shape!
                    if value.scalarType == "float16" {
                        result[name] = .float16(try Float16Tensor(shape: shape, values: Array(repeating: 1, count: 64)))
                    } else {
                        result[name] = .float32(try FloatTensor(shape: shape, values: Array(repeating: name == "cond" ? 7 : 0, count: FloatTensor.elementCount(for: shape))))
                    }
                }
                return result
            }, close: { log.append("close\(index)") })
            models["denoiser\(index).aimodel"] = model
            models["fixed\(index).aimodel"] = model
        }
        models["vaeDecoder.aimodel"] = FakeModel(functions: [try describe("decode", inputs: [
            "latents": declaration("float16", [1, 1, 64]),
        ], outputs: ["image": declaration("float32", [3, 16, 16])])], run: { inputs in
            if let decoderGate { await decoderGate.enter() }
            guard case .float16(let latents) = inputs["latents"] else { throw AetherError.invalidTensor("missing decoder latents") }
            if checkDecoderLatents { #expect(latents.values.allSatisfy { $0 == -1 }) }
            return ["image": .float32(try FloatTensor(shape: [3, 16, 16], values: Array(repeating: 0, count: 3 * 16 * 16)))]
        }, close: { log.append("closeDecoder") })
        let encodingFailure = Mutex(failEncodingOnce)
        models["textEncoder.aimodel"] = FakeModel(functions: [], run: { inputs in
            log.append("runEncoder")
            if let encoderGate { await encoderGate.enter() }
            if encodingFailure.withLock({ fail in let result = fail; fail = false; return result }) {
                throw AetherError.invalidTensor("injected encoding failure")
            }
            #expect(inputs["input_ids"]?.shape == [1, 35])
            guard case .int32(let ids) = inputs["input_ids"] else { throw AetherError.invalidTensor("missing token IDs") }
            return ["text_embeds": .float16(try Float16Tensor(shape: [35, 30720], values: Array(repeating: Float16(ids.values.last ?? 0), count: 35 * 30720)))]
        }, close: { log.append("closeEncoder") })
        self.models = models
    }

    func open(_ asset: KreaModelAsset, frequentReshapes: Bool) async throws -> any KreaExecutable {
        let name = asset.url.lastPathComponent
        log.append("open:" + name)
        let fail = failOnce.withLock { value in
            if value == name { value = nil; return true }
            return false
        }
        if fail { throw AetherError.invalidModel("injected load failure") }
        guard let model = models[name] else { throw AetherError.invalidModel("unexpected model") }
        return model
    }
}

private actor PipelineGate {
    var entered = false
    var completed = false
    var startWaiters: [CheckedContinuation<Void, Never>] = []
    var completion: CheckedContinuation<Void, Never>?
    func enter() async {
        if completed { return }
        await withCheckedContinuation { continuation in
            completion = continuation
            entered = true
            startWaiters.forEach { $0.resume() }
            startWaiters.removeAll()
        }
    }
    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { startWaiters.append($0) }
    }
    func release() { completed = true; completion?.resume(); completion = nil }
}

private func oneRow() throws -> KreaConditioning {
    try KreaConditioning(tensor: Float16Tensor(shape: [1, 30720], values: Array(repeating: 0, count: 30720)))
}

@Test func referencePipelineRoutesCarriesAcrossAllStepsAndReusesModels() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let assets = try KreaAssets.open(at: fixture.root)
    let registration = try shapeRegistration(fixture: fixture, assets: assets)
    let specialized = try assets.registeringSpecialization(at: registration)
    let log = EventLog()
    let session = KreaGenerationSession(assets: specialized, factory: try FakeFactory(log: log))
    let configuration = try KreaConfiguration(width: 16, height: 16)
    let noise = try FloatTensor(shape: [1, 64], values: Array(repeating: 0, count: 64))
    for _ in 0..<2 {
        let image = try await session.generate(configuration: configuration, conditioning: oneRow(), noise: noise) { event in
            switch event {
            case .step(let index, let total, _): log.append("step:\(index)/\(total)")
            case .denoisersReady(let selection): #expect(selection.specialized)
            default: break
            }
        }
        #expect(image.width == 16)
        #expect(image.rgb8().allSatisfy { $0 == 128 })
    }
    #expect(log.snapshot.filter { $0.hasPrefix("open:") }.count == 5)
    #expect(log.snapshot.filter { $0.hasPrefix("step:") }.count == 16)
    #expect(log.snapshot.filter { $0 == "run3" }.count == 16)
    await session.unload()
    #expect(await session.state == .closed)
    #expect(log.snapshot.contains("closeDecoder"))
}

@Test func partialLoadFailureIsCleanedUpAndCanRetry() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let log = EventLog()
    let factory = try FakeFactory(log: log, failOnce: "denoiser1.aimodel")
    let session = try KreaGenerationSession(assets: KreaAssets.open(at: fixture.root), factory: factory)
    let configuration = try KreaConfiguration(width: 16, height: 16)
    let noise = try FloatTensor(shape: [1, 64], values: Array(repeating: 0, count: 64))
    await #expect(throws: AetherError.invalidModel("injected load failure")) {
        try await session.generate(configuration: configuration, conditioning: oneRow(), noise: noise)
    }
    #expect(await session.state == .ready)
    #expect(log.snapshot.contains("close0"))
    _ = try await session.generate(configuration: configuration, conditioning: oneRow(), noise: noise)
    await session.unload()
}

@Test func balancedPromptGenerationReleasesEncoderBeforeDenoiserLoad() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let log = EventLog()
    let session = try KreaGenerationSession(assets: KreaAssets.open(at: fixture.root), factory: FakeFactory(log: log))
    _ = try await session.generate(configuration: KreaConfiguration(width: 16, height: 16),
        prompt: KreaPrompt(tokenIDs: Array(repeating: 0, count: 35)),
        noise: FloatTensor(shape: [1, 64], values: Array(repeating: 0, count: 64)))
    let closed = try #require(log.snapshot.firstIndex(of: "closeEncoder"))
    let opened = try #require(log.snapshot.firstIndex(of: "open:denoiser0.aimodel"))
    #expect(closed < opened)
    await session.unload()
}

@Test func residentPromptGenerationRetainsEncoderUntilUnload() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let log = EventLog()
    let session = try KreaGenerationSession(assets: KreaAssets.open(at: fixture.root), factory: FakeFactory(log: log), residency: .resident)
    for _ in 0..<2 {
        _ = try await session.generate(configuration: KreaConfiguration(width: 16, height: 16),
            prompt: KreaPrompt(tokenIDs: Array(repeating: 0, count: 35)),
            noise: FloatTensor(shape: [1, 64], values: Array(repeating: 0, count: 64)))
    }
    #expect(log.snapshot.filter { $0 == "open:textEncoder.aimodel" }.count == 1)
    #expect(!log.snapshot.contains("closeEncoder"))
    await session.unload()
    #expect(log.snapshot.filter { $0 == "closeEncoder" }.count == 1)
}

@Test func balancedRepeatedPromptReleasesDenoisersBeforeEncoderAcquisition() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let log = EventLog()
    let session = try KreaGenerationSession(assets: KreaAssets.open(at: fixture.root), factory: FakeFactory(log: log))
    let configuration = try KreaConfiguration(width: 16, height: 16)
    let noise = try FloatTensor(shape: [1, 64], values: Array(repeating: 0, count: 64))
    _ = try await session.generate(configuration: configuration, conditioning: oneRow(), noise: noise)
    let boundary = log.snapshot.count
    _ = try await session.generate(configuration: configuration,
        prompt: KreaPrompt(tokenIDs: Array(repeating: 0, count: 35)), noise: noise)
    let second = Array(log.snapshot.dropFirst(boundary))
    let encoderOpen = try #require(second.firstIndex(of: "open:textEncoder.aimodel"))
    for index in 0..<4 {
        let close = try #require(second.firstIndex(of: "close\(index)"))
        #expect(close < encoderOpen)
    }
    let encoderClose = try #require(second.firstIndex(of: "closeEncoder"))
    let denoiserOpen = try #require(second.firstIndex(of: "open:denoiser0.aimodel"))
    #expect(encoderClose < denoiserOpen)
    #expect(log.snapshot.filter { $0 == "open:vaeDecoder.aimodel" }.count == 1)
    await session.unload()
}

@Test func shapeSwitchClosesSpecializedModelsBeforeLoadingFallback() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let assets = try KreaAssets.open(at: fixture.root)
    let registered = try assets.registeringSpecialization(at: shapeRegistration(fixture: fixture, assets: assets))
    let log = EventLog()
    let session = KreaGenerationSession(assets: registered, factory: try FakeFactory(log: log))
    let configuration = try KreaConfiguration(width: 16, height: 16)
    _ = try await session.generate(configuration: configuration, conditioning: oneRow(), noise: FloatTensor(shape: [1, 64], values: Array(repeating: 0, count: 64)))
    let differentShape = try KreaConditioning(tensor: Float16Tensor(shape: [2, 30720], values: Array(repeating: 0, count: 2 * 30720)))
    // The fake dynamic models only accept one row, so the fallback must fail its
    // descriptor gate after releasing the previous specialization, before execution.
    await #expect(throws: AetherError.self) { try await session.generate(configuration: configuration, conditioning: differentShape) }
    let closed = try #require(log.snapshot.firstIndex(of: "close0"))
    let loaded = try #require(log.snapshot.firstIndex(of: "open:denoiser0.aimodel"))
    #expect(closed < loaded)
    #expect(log.snapshot.filter { $0 == "run0" }.count == 8)
    await session.unload()
}

@Test(.timeLimit(.minutes(1))) func pipelineCancellationAndUnloadWaitForSubmittedWork() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let gate = PipelineGate()
    let log = EventLog()
    let session = try KreaGenerationSession(assets: KreaAssets.open(at: fixture.root), factory: FakeFactory(log: log, gate: gate))
    let configuration = try KreaConfiguration(width: 16, height: 16)
    let run = Task { try await session.generate(configuration: configuration, conditioning: oneRow()) }
    await gate.waitUntilEntered()
    await #expect(throws: AetherError.sessionBusy) { try await session.generate(configuration: configuration, conditioning: oneRow()) }
    await #expect(throws: AetherError.sessionBusy) {
        try await session.prepare(prompt: KreaPrompt(tokenIDs: Array(repeating: 0, count: 35)))
    }
    run.cancel()
    let close = Task { await session.unload() }
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await session.state != .closing, ContinuousClock.now < deadline { await Task.yield() }
    #expect(await session.state == .closing)
    await gate.release()
    await #expect(throws: CancellationError.self) { try await run.value }
    await close.value
    #expect(!log.snapshot.contains("run1"))
    #expect(await session.state == .closed)
}

@Test func segmentContractsRejectMissingCarryWrongShapeAndEarlyVelocity() throws {
    let valid = try segmentSignatures()
    try KreaContracts.segments(valid, tokens: 1, text: 1)
    #expect(throws: AetherError.self) { try KreaContracts.segments(valid, tokens: 2, text: 1) }
    for fault in ["missing", "early", "state", "overwrite"] {
        var functions = valid
        let index = fault == "missing" ? 1 : 0
        var raw = try JSONSerialization.jsonObject(with: JSONEncoder().encode(functions[index])) as! [String: Any]
        if fault == "missing" {
            var inputs = raw["inputs"] as! [String: Any]
            inputs["image_3"] = declaration("float32", [1, 6144])
            raw["inputs"] = inputs
        } else if fault == "state" { raw["states"] = ["kv": declaration("float32", [1])] }
        else {
            var outputs = raw["outputs"] as! [String: Any]
            outputs[fault == "early" ? "velocity" : "latents"] = declaration("float16", [1, 64])
            raw["outputs"] = outputs
        }
        functions[index] = try JSONDecoder().decode(ModelFunctionDescription.self, from: JSONSerialization.data(withJSONObject: raw))
        #expect(throws: AetherError.self) { try KreaContracts.segments(functions, tokens: 1, text: 1) }
    }
    let decoder = try describe("decode", inputs: ["latents": declaration("float16", [1, 1, 64])], outputs: ["image": declaration("float32", [3, 16, 16])])
    try KreaContracts.decoder(decoder, configuration: KreaConfiguration(width: 16, height: 16))
    #expect(throws: AetherError.self) { try KreaContracts.decoder(decoder, configuration: KreaConfiguration(width: 32, height: 16)) }
}

@Test(.timeLimit(.minutes(1))) func decoderCancellationRestoresPipelineAndCanRecover() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let gate = PipelineGate()
    let log = EventLog()
    let session = try KreaGenerationSession(assets: KreaAssets.open(at: fixture.root),
        factory: FakeFactory(log: log, decoderGate: gate))
    let configuration = try KreaConfiguration(width: 16, height: 16)
    let noise = try FloatTensor(shape: [1, 64], values: Array(repeating: 0, count: 64))
    let run = Task {
        try await session.generate(configuration: configuration, conditioning: oneRow(), noise: noise) { event in
            if case .decoded = event { log.append("decoded") }
        }
    }
    await gate.waitUntilEntered()
    await #expect(throws: AetherError.sessionBusy) {
        try await session.generate(configuration: configuration, conditioning: oneRow(), noise: noise)
    }
    run.cancel()
    await gate.release()
    await #expect(throws: CancellationError.self) { try await run.value }
    #expect(await session.state == .ready)
    #expect(!log.snapshot.contains("decoded"))
    let recovered = try await session.generate(configuration: configuration, conditioning: oneRow(), noise: noise)
    #expect(recovered.rgb8().allSatisfy { $0 == 128 })
    #expect(log.snapshot.filter { $0 == "open:vaeDecoder.aimodel" }.count == 1)
    await session.unload()
}

@Test func preparedPromptReusesEncodingAndModelsAcrossSeedsAndSessions() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let assets = try KreaAssets.open(at: fixture.root)
    let log = EventLog()
    let factory = try FakeFactory(log: log, checkDecoderLatents: false)
    let session = KreaGenerationSession(assets: assets, factory: factory)
    let prepared = try await session.prepare(prompt: KreaPrompt(tokenIDs: Array(repeating: 0, count: 35)))
    #expect(prepared.bundleFingerprint == assets.fingerprint)
    #expect(prepared.conditioning == (try oneRow()))
    #expect(prepared.conditioningByteCount == 61_440)
    #expect(log.snapshot == ["open:textEncoder.aimodel", "runEncoder", "closeEncoder"])
    var finalLatents: [FloatTensor] = []
    for seed: UInt64 in [17, 18] {
        let configuration = try KreaConfiguration(width: 16, height: 16, seed: seed)
        let preparedSteps = Mutex<[FloatTensor]>([])
        let providedSteps = Mutex<[FloatTensor]>([])
        let image = try await session.generate(configuration: configuration, preparedPrompt: prepared) { event in
            if case .step(_, _, let latents) = event { preparedSteps.withLock { $0.append(latents) } }
        }
        let replay = try await session.generate(configuration: configuration, conditioning: prepared.conditioning) { event in
            if case .step(_, _, let latents) = event { providedSteps.withLock { $0.append(latents) } }
        }
        #expect(image.pixelsCHW == replay.pixelsCHW)
        #expect(preparedSteps.withLock { $0 } == providedSteps.withLock { $0 })
        let steps = preparedSteps.withLock { $0 }
        #expect(steps.count == 8)
        finalLatents.append(try #require(steps.last))
    }
    #expect(finalLatents[0] != finalLatents[1])
    #expect(log.snapshot.filter { $0.hasPrefix("open:") }.count == 6)
    #expect(log.snapshot.filter { $0.hasPrefix("close") } == ["closeEncoder"])
    await session.unload()
    await #expect(throws: AetherError.sessionClosed) {
        try await session.generate(configuration: KreaConfiguration(width: 16, height: 16), preparedPrompt: prepared)
    }
    let next = KreaGenerationSession(assets: assets, factory: factory)
    _ = try await next.generate(configuration: KreaConfiguration(width: 16, height: 16), preparedPrompt: prepared)
    #expect(log.snapshot.filter { $0 == "runEncoder" }.count == 1)
    await next.unload()
}

@Test func preparedPromptRejectsAnotherBundleBeforeOpeningModels() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let assets = try KreaAssets.open(at: fixture.root)
    let log = EventLog()
    let session = KreaGenerationSession(assets: assets, factory: try FakeFactory(log: log))
    let prepared = try await session.prepare(prompt: KreaPrompt(tokenIDs: Array(repeating: 0, count: 35)))
    await session.unload()
    try fixture.edit("tokenizer/tokenizer_config.json") { $0["changed"] = true }
    let changedAssets = try KreaAssets.open(at: fixture.root)
    #expect(changedAssets.fingerprint != assets.fingerprint)
    let otherLog = EventLog()
    let other = KreaGenerationSession(assets: changedAssets, factory: try FakeFactory(log: otherLog))
    await #expect(throws: AetherError.invalidRequest("prepared prompt belongs to a different Krea bundle")) {
        try await other.generate(configuration: KreaConfiguration(width: 16, height: 16), preparedPrompt: prepared)
    }
    #expect(otherLog.snapshot.isEmpty)
    #expect(await other.state == .ready)
    await other.unload()
}

@Test func preparationReleasesRetainedModelsEvenWithResidentPolicy() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let log = EventLog()
    let session = try KreaGenerationSession(assets: KreaAssets.open(at: fixture.root), factory: FakeFactory(log: log), residency: .resident)
    let prompt = try KreaPrompt(tokenIDs: Array(repeating: 0, count: 35))
    let configuration = try KreaConfiguration(width: 16, height: 16)
    let noise = try FloatTensor(shape: [1, 64], values: Array(repeating: 0, count: 64))
    _ = try await session.generate(configuration: configuration, prompt: prompt, noise: noise)
    let boundary = log.snapshot.count
    let prepared = try await session.prepare(prompt: prompt)
    let preparation = Array(log.snapshot.dropFirst(boundary))
    let encoding = try #require(preparation.firstIndex(of: "runEncoder"))
    for index in 0..<4 {
        #expect(try #require(preparation.firstIndex(of: "close\(index)")) < encoding)
    }
    #expect(preparation.last == "closeEncoder")
    #expect(!preparation.contains("closeDecoder"))
    _ = try await session.generate(configuration: configuration, preparedPrompt: prepared, noise: noise)
    #expect(log.snapshot.filter { $0 == "open:vaeDecoder.aimodel" }.count == 1)
    #expect(log.snapshot.filter { $0 == "open:denoiser0.aimodel" }.count == 2)
    await session.unload()
}

@Test func preparationFailureClosesEncoderAndAllowsRetry() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let log = EventLog()
    let session = try KreaGenerationSession(assets: KreaAssets.open(at: fixture.root), factory: FakeFactory(log: log, failEncodingOnce: true))
    let prompt = try KreaPrompt(tokenIDs: Array(repeating: 0, count: 35))
    await #expect(throws: AetherError.invalidTensor("injected encoding failure")) { try await session.prepare(prompt: prompt) }
    #expect(await session.state == .ready)
    #expect(log.snapshot.last == "closeEncoder")
    let prepared = try await session.prepare(prompt: prompt)
    #expect(prepared.conditioning == (try oneRow()))
    #expect(log.snapshot.filter { $0 == "open:textEncoder.aimodel" }.count == 2)
    #expect(log.snapshot.filter { $0 == "closeEncoder" }.count == 2)
    await session.unload()
}

@Test(.timeLimit(.minutes(1))) func cancelledPreparationHoldsOperationSlotUntilEncodingReturns() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let gate = PipelineGate()
    let log = EventLog()
    let session = try KreaGenerationSession(assets: KreaAssets.open(at: fixture.root), factory: FakeFactory(log: log, encoderGate: gate))
    let prompt = try KreaPrompt(tokenIDs: Array(repeating: 0, count: 35))
    let preparation = Task { try await session.prepare(prompt: prompt) }
    await gate.waitUntilEntered()
    preparation.cancel()
    await #expect(throws: AetherError.sessionBusy) { try await session.prepare(prompt: prompt) }
    await #expect(throws: AetherError.sessionBusy) {
        try await session.generate(configuration: KreaConfiguration(width: 16, height: 16), conditioning: oneRow())
    }
    #expect(!log.snapshot.contains("closeEncoder"))
    await gate.release()
    await #expect(throws: CancellationError.self) { try await preparation.value }
    #expect(await session.state == .ready)
    #expect(log.snapshot.last == "closeEncoder")
    _ = try await session.prepare(prompt: prompt)
    #expect(log.snapshot.filter { $0 == "open:textEncoder.aimodel" }.count == 2)
    await session.unload()
}

@Test(.timeLimit(.minutes(1))) func unloadWaitsForPreparationAndClosesPermanently() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let gate = PipelineGate()
    let log = EventLog()
    let session = try KreaGenerationSession(assets: KreaAssets.open(at: fixture.root), factory: FakeFactory(log: log, encoderGate: gate))
    let prompt = try KreaPrompt(tokenIDs: Array(repeating: 0, count: 35))
    let preparation = Task { try await session.prepare(prompt: prompt) }
    await gate.waitUntilEntered()
    let close = Task { await session.unload() }
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await session.state != .closing, ContinuousClock.now < deadline { await Task.yield() }
    #expect(await session.state == .closing)
    #expect(!log.snapshot.contains("closeEncoder"))
    await #expect(throws: AetherError.sessionClosed) { try await session.prepare(prompt: prompt) }
    await gate.release()
    let prepared = try await preparation.value
    await close.value
    #expect(prepared.conditioning == (try oneRow()))
    #expect(await session.state == .closed)
    #expect(log.snapshot.filter { $0 == "closeEncoder" }.count == 1)
    await #expect(throws: AetherError.sessionClosed) { try await session.prepare(prompt: prompt) }
}

@Test func preparingAnotherPromptPreservesPreviouslyOwnedConditioning() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let log = EventLog()
    let session = try KreaGenerationSession(assets: KreaAssets.open(at: fixture.root), factory: FakeFactory(log: log))
    let original = try await session.prepare(prompt: KreaPrompt(tokenIDs: Array(repeating: 0, count: 35)))
    let replacement = try await session.prepare(prompt: KreaPrompt(tokenIDs: Array(repeating: 1, count: 35)))
    await session.unload()
    #expect(original.conditioning.tensor.values.allSatisfy { $0 == 0 })
    #expect(replacement.conditioning.tensor.values.allSatisfy { $0 == 1 })
    #expect(original.conditioning != replacement.conditioning)
    #expect(log.snapshot.filter { $0 == "runEncoder" }.count == 2)
    #expect(log.snapshot.filter { $0 == "closeEncoder" }.count == 2)
    #expect(log.snapshot.filter { $0.hasPrefix("open:") }.count == 2)
}

@Test func refineStartsFromEncodedLatentsAndRunsOnlyTheRemainingSteps() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let log = EventLog()
    let session = try KreaGenerationSession(assets: KreaAssets.open(at: fixture.root), factory: FakeFactory(log: log, checkDecoderLatents: false))
    let configuration = try KreaConfiguration(width: 16, height: 16)
    let prepared = try await session.prepare(prompt: KreaPrompt(tokenIDs: Array(repeating: 0, count: 35)))
    let schedule = try KreaSchedule(steps: 8)
    let start = try KreaRefineStart(latents: FloatTensor(shape: [1, 64], values: Array(repeating: 1, count: 64)), startStep: 7)
    let noise = try FloatTensor(shape: [1, 64], values: Array(repeating: 0, count: 64))
    let first = Mutex<[Float]?>(nil)
    _ = try await session.refine(configuration: configuration, preparedPrompt: prepared, start: start, noise: noise) { event in
        if case .step(let index, let total, let latents) = event {
            log.append("step:\(index)/\(total)")
            first.withLock { if $0 == nil { $0 = latents.values } }
        }
    }
    // Only step 7 ran; the start was (1 − σ7)·1 + σ7·0, then the Euler step added (0 − σ7)·1.
    #expect(log.snapshot.filter { $0.hasPrefix("step:") } == ["step:7/8"])
    let sigma = schedule.sigmas[7]
    #expect(first.withLock { $0 }?.allSatisfy { abs($0 - ((1 - sigma) - sigma)) < 1e-6 } == true)
    #expect(abs(sigma - 0.311) < 0.01 && abs(schedule.sigmas[6] - 0.513) < 0.01)
    // Wrong shapes and step 0 are refused before any model work.
    #expect(throws: AetherError.self) { try KreaRefineStart(latents: noise, startStep: 0) }
    let wide = try KreaRefineStart(latents: FloatTensor(shape: [2, 64], values: Array(repeating: 0, count: 128)), startStep: 7)
    await #expect(throws: AetherError.self) { try await session.refine(configuration: configuration, preparedPrompt: prepared, start: wide) }
    await session.unload()
}

@Test func refineStartMixesPictureAndNoiseBySigma() throws {
    let schedule = try KreaSchedule(steps: 8)
    let mixed = try schedule.refineStart(x0: [2, 0], noise: [0, 4], step: 6)
    let sigma = schedule.sigmas[6]
    #expect(abs(mixed[0] - 2 * (1 - sigma)) < 1e-6 && abs(mixed[1] - 4 * sigma) < 1e-6)
    #expect(throws: AetherError.self) { try schedule.refineStart(x0: [1], noise: [1], step: 0) }
    #expect(throws: AetherError.self) { try schedule.refineStart(x0: [1], noise: [1, 2], step: 3) }
}
