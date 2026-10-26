import AetherAssets
import AetherCoreAI
import AetherFlux2
import AetherKit
import Foundation
import Synchronization
import Testing
@testable import AetherFlux2CoreAI

final class Flux2EventLog: Sendable {
    let values = Mutex<[String]>([])
    func append(_ value: String) { values.withLock { $0.append(value) } }
    var snapshot: [String] { values.withLock { $0 } }
    func clear() { values.withLock { $0.removeAll() } }
}

/// Suspends a scripted native call until released, so lifecycle tests observe in-flight work.
actor Flux2Gate {
    private var entered = false, released = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private var held: CheckedContinuation<Void, Never>?
    func enter() async {
        if released { return }
        await withCheckedContinuation { continuation in
            held = continuation; entered = true
            waiters.forEach { $0.resume() }; waiters.removeAll()
        }
    }
    func waitUntilEntered() async { if entered { return }; await withCheckedContinuation { waiters.append($0) } }
    func release() { released = true; held?.resume(); held = nil }
}

private func value(_ type: String, _ shape: [Int]) -> [String: Any] { ["kind": "tensor", "scalarType": type, "shape": shape] }

private func describe(_ name: String, _ inputs: [String: [String: Any]], _ outputs: [String: [String: Any]]) throws -> ModelFunctionDescription {
    try JSONDecoder().decode(ModelFunctionDescription.self, from: JSONSerialization.data(withJSONObject: [
        "name": name, "inputs": inputs, "outputs": outputs, "states": [:]]))
}

/// The exported signatures, with -1 for dynamic token dimensions.
private func segmentSignatures() throws -> [ModelFunctionDescription] {
    let image = value("float32", [-1, 3072]), text = value("float32", [-1, 3072]), cond = value("float32", [17, 3072])
    let ropes = ["rope_text": value("float32", [-1, 2, 128]), "rope_image": value("float32", [-1, 2, 128])]
    return try [
        describe("segment", ["latents": value("float16", [-1, 128]), "text_embeds": value("float16", [-1, 7680]),
                             "sigma": value("float32", [1])].merging(ropes) { $1 },
                 ["image_1": image, "text_1": text, "cond": cond]),
        describe("segment", ["image_1": image, "text_1": text, "cond": cond].merging(ropes) { $1 }, ["image_2": image, "text_2": text]),
        describe("segment", ["image_2": image, "text_2": text, "cond": cond].merging(ropes) { $1 }, ["velocity": value("float16", [-1, 128])]),
    ]
}

private struct ScriptedModel: StagedModel {
    let name: String
    let functions: [ModelFunctionDescription]
    let log: Flux2EventLog
    let body: @Sendable ([String: TensorValue]) async throws -> [String: TensorValue]
    func runTensors(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue] {
        log.append("run:" + name)
        return try await body(inputs)
    }
    func unload() async { log.append("close:" + name) }
}

/// Records shapes the pipeline submits. Denoisers return velocity = token index + 1 for generated and
/// reference tokens alike; only generated tokens may advance.
final class Flux2ScriptedFactory: Flux2ModelFactory, Sendable {
    let log: Flux2EventLog
    let gates: Mutex<[String: Flux2Gate]>
    let failures: Mutex<Set<String>>
    let observed = Mutex<[String: [String: TensorValue]]>([:])
    let runFailures = Mutex<Set<String>>([])

    init(log: Flux2EventLog, gates: [String: Flux2Gate] = [:], failures: Set<String> = []) {
        self.log = log; self.gates = Mutex(gates); self.failures = Mutex(failures)
    }

    func failNextRun(of role: String) { runFailures.withLock { _ = $0.insert(role) } }

    func gate(_ role: String) -> Flux2Gate? { gates.withLock { $0[role] } }

    func open(_ asset: DeclaredModelAsset, frequentReshapes: Bool) async throws -> any StagedModel {
        let role = asset.url.deletingPathExtension().lastPathComponent
        log.append("open:\(role):\(frequentReshapes)")
        if failures.withLock({ $0.remove(role) != nil }) { throw AetherError.invalidModel("injected load failure: \(role)") }
        let gate = self.gate(role)
        // Records inputs, then fails the call once if a run failure was injected for this role.
        let record: @Sendable ([String: TensorValue]) throws -> Void = { [self] inputs in
            self.observed.withLock { $0[role] = inputs }
            if self.runFailures.withLock({ $0.remove(role) != nil }) { throw AetherError.invalidTensor("injected run failure: \(role)") }
        }
        switch role {
        case "textEncoder":
            let signature = try describe("encode", [
                "input_ids": value("int32", [1, -1]), "position_ids": value("int32", [1, -1]),
                "attention_mask": value("int32", [1, -1])], ["text_embeds": value("float16", [-1, 7680])])
            return ScriptedModel(name: role, functions: [signature], log: log) { inputs in
                try record(inputs)
                if let gate { await gate.enter() }
                return ["text_embeds": .float16(try Float16Tensor(shape: [512, 7680], values: Array(repeating: 0.5, count: 512 * 7680)))]
            }
        case "vaeEncoder":
            let signature = try describe("encode", ["reference_pixels": value("float32", [3, -1, -1])],
                                         ["reference_latents": value("float32", [-1, -1, 128])])
            return ScriptedModel(name: role, functions: [signature], log: log) { inputs in
                try record(inputs)
                if let gate { await gate.enter() }
                guard case .float32(let pixels) = inputs["reference_pixels"] else { throw AetherError.invalidTensor("pixels") }
                let rows = pixels.shape[1] / 16, columns = pixels.shape[2] / 16
                return ["reference_latents": .float32(try FloatTensor(shape: [rows, columns, 128],
                                                                     values: Array(repeating: pixels.values[0], count: rows * columns * 128)))]
            }
        case "vaeDecoder":
            let signature = try describe("decode", ["latents": value("float16", [-1, -1, 128])], ["image": value("float32", [3, -1, -1])])
            return ScriptedModel(name: role, functions: [signature], log: log) { inputs in
                try record(inputs)
                if let gate { await gate.enter() }
                guard case .float16(let latents) = inputs["latents"] else { throw AetherError.invalidTensor("latents") }
                let h = latents.shape[0] * 16, w = latents.shape[1] * 16
                return ["image": .float32(try FloatTensor(shape: [3, h, w], values: Array(repeating: 0.25, count: 3 * h * w)))]
            }
        default:
            let index = Int(role.dropFirst("denoiser".count))!
            let signature = try segmentSignatures()[index]
            return ScriptedModel(name: role, functions: [signature], log: log) { inputs in
                try record(inputs)
                if let gate { await gate.enter() }
                guard let tokens = inputs[index == 0 ? "latents" : "image_\(index)"]?.shape.first else { throw AetherError.invalidTensor("tokens") }
                var result: [String: TensorValue] = [:]
                for (name, declared) in signature.outputs {
                    let shape = declared.shape!.enumerated().map { $0.element < 0 ? ($0.offset == 0 && name.hasPrefix("text") ? 512 : tokens) : $0.element }
                    let count = try FloatTensor.elementCount(for: shape)
                    if name == "velocity" {
                        result[name] = .float16(try Float16Tensor(shape: shape, values: (0..<count).map { Float16($0 / 128 + 1) }))
                    } else {
                        result[name] = .float32(try FloatTensor(shape: shape, values: Array(repeating: 0, count: count)))
                    }
                }
                return result
            }
        }
    }
}

struct Flux2Harness {
    let fixture: Flux2BundleFixture
    let assets: Flux2Assets
    let log = Flux2EventLog()
    let factory: Flux2ScriptedFactory
    let session: Flux2GenerationSession

    init(references: Bool = true, residency: Flux2Residency = .balanced, gates: [String: Flux2Gate] = [:],
         failures: Set<String> = [], resources: ((Flux2Assets) throws -> Flux2ResourceAdmission)? = nil,
         gpu: GPUAdmissionOwner? = nil) throws {
        fixture = try Flux2BundleFixture(references: references)
        assets = try Flux2Assets.open(at: fixture.root)
        factory = Flux2ScriptedFactory(log: log, gates: gates, failures: failures)
        session = Flux2GenerationSession(assets: assets, factory: factory, residency: residency, resources: try resources?(assets),
                                         gpu: gpu)
    }
}

private func prompt() throws -> Flux2Prompt { try Flux2Prompt(renderedTokenIDs: [1, 2, 3]) }
private func reference(rows: Int, columns: Int, fill: Float) throws -> Flux2ReferenceImage {
    try Flux2ReferenceImage(pixels: FloatTensor(shape: [3, rows * 16, columns * 16],
                                                values: Array(repeating: fill, count: 3 * rows * 16 * columns * 16)))
}

@Test func textToImageRunsStagesInBalancedOrder() async throws {
    let h = try Flux2Harness()
    defer { h.fixture.cleanup() }
    let configuration = try Flux2Configuration(width: 64, height: 32, steps: 2, seed: 3)
    let steps = Mutex<[FloatTensor]>([])
    let image = try await h.session.generate(configuration: configuration, prompt: prompt()) { event in
        if case .step(_, _, let latents) = event { steps.withLock { $0.append(latents) } }
    }
    #expect(image.width == 64 && image.height == 32)
    #expect(h.log.snapshot == [
        "open:textEncoder:true", "run:textEncoder", "close:textEncoder",
        "open:denoiser0:true", "open:denoiser1:true", "open:denoiser2:true",
        "run:denoiser0", "run:denoiser1", "run:denoiser2", "run:denoiser0", "run:denoiser1", "run:denoiser2",
        "open:vaeDecoder:false", "run:vaeDecoder",
    ])
    // Euler over the scripted velocity (token index + 1) from the seeded noise.
    let schedule = try Flux2Schedule(steps: 2, imageTokens: 8)
    var expected = try configuration.initialNoise().values
    for step in 0..<2 {
        expected = try schedule.advance(step: step, latents: expected, velocity: (0..<(8 * 128)).map { Float($0 / 128 + 1) })
    }
    #expect(steps.withLock { $0.last?.values } == expected)
    let observed = h.factory.observed.withLock { $0 }
    #expect(observed["denoiser0"]?["rope_image"]?.shape == [8, 2, 128])
    #expect(observed["denoiser0"]?["rope_text"]?.shape == [512, 2, 128])
    #expect(observed["vaeDecoder"]?["latents"]?.shape == [2, 4, 128])
    await h.session.unload()
    #expect(h.log.snapshot.suffix(4) == ["close:denoiser0", "close:denoiser1", "close:denoiser2", "close:vaeDecoder"])
}

@Test func referencesAreAppendedImmutablyAndOnlyGeneratedTokensAdvance() async throws {
    let h = try Flux2Harness()
    defer { h.fixture.cleanup() }
    let configuration = try Flux2Configuration(width: 32, height: 32, steps: 1, seed: 9)
    let refs = try [reference(rows: 1, columns: 2, fill: 0.5), reference(rows: 2, columns: 1, fill: -0.25)]
    let ready = Mutex<Flux2PreparedReferences?>(nil)
    let last = Mutex<FloatTensor?>(nil)
    _ = try await h.session.generate(configuration: configuration, prompt: prompt(), references: refs) { event in
        if case .referencesReady(let value) = event { ready.withLock { $0 = value } }
        if case .step(_, _, let latents) = event { last.withLock { $0 = latents } }
    }
    #expect(h.log.snapshot.prefix(7) == ["open:textEncoder:true", "run:textEncoder", "close:textEncoder",
                                         "open:vaeEncoder:false", "run:vaeEncoder", "run:vaeEncoder", "close:vaeEncoder"])
    let prepared = try #require(ready.withLock { $0 })
    #expect(prepared.tokens == 4 && prepared.images.map(\.rows) == [1, 2])
    let observed = h.factory.observed.withLock { $0 }
    guard case .float16(let latents) = observed["denoiser0"]?["latents"],
          case .float32(let rope) = observed["denoiser0"]?["rope_image"] else { Issue.record("missing inputs"); return }
    #expect(latents.shape == [8, 128] && rope.shape == [8, 2, 128])
    // Reference tokens follow the four generated tokens, in order, unchanged.
    #expect(Array(latents.values[(4 * 128)..<(6 * 128)]).allSatisfy { $0 == 0.5 })
    #expect(Array(latents.values[(6 * 128)...]).allSatisfy { $0 == -0.25 })
    // Reference k sits at t = 10·(k + 1): the axis-0 cosine of token 4 is cos(10), of token 6 cos(20).
    #expect(rope.values[4 * 256] == Float(cos(10.0)) && rope.values[6 * 256] == Float(cos(20.0)))
    #expect(rope.values[3 * 256] == 1)
    // μ uses generated tokens only, and the advanced latents are the four generated tokens.
    let schedule = try Flux2Schedule(steps: 1, imageTokens: 4)
    let expected = try schedule.advance(step: 0, latents: configuration.initialNoise().values,
                                        velocity: (0..<(4 * 128)).map { Float($0 / 128 + 1) })
    #expect(last.withLock { $0?.values } == expected)
    #expect(observed["vaeDecoder"]?["latents"]?.shape == [2, 2, 128])
    await h.session.unload()
}

@Test func preparedConditioningAndReferencesAreReusedWithoutEncoders() async throws {
    let h = try Flux2Harness()
    defer { h.fixture.cleanup() }
    let configuration = try Flux2Configuration(width: 32, height: 32, steps: 1)
    let prepared = try await h.session.prepare(prompt: prompt())
    let refs = try await h.session.prepare(references: [reference(rows: 1, columns: 1, fill: 0)], generatedTokens: 4)
    #expect(prepared.bundleFingerprint == h.assets.fingerprint && refs.referenceFingerprint == h.assets.referenceFingerprint)
    h.log.clear()
    for seed: UInt64 in [1, 2] {
        _ = try await h.session.generate(configuration: try Flux2Configuration(width: 32, height: 32, steps: 1, seed: seed),
                                         preparedPrompt: prepared, references: refs)
    }
    let events = h.log.snapshot
    #expect(!events.contains { $0.contains("textEncoder") || $0.contains("vaeEncoder") })
    #expect(events.filter { $0.hasPrefix("open:") }.count == 4)  // denoisers and decoder load once
    // A different total token count recycles the denoisers; a different output size recycles the decoder.
    h.log.clear()
    _ = try await h.session.generate(configuration: configuration, preparedPrompt: prepared)
    #expect(h.log.snapshot.prefix(4) == ["close:denoiser0", "close:denoiser1", "close:denoiser2", "open:denoiser0:true"])
    #expect(!h.log.snapshot.contains("open:vaeDecoder:false"))
    h.log.clear()
    _ = try await h.session.generate(configuration: try Flux2Configuration(width: 48, height: 32, steps: 1), preparedPrompt: prepared)
    #expect(h.log.snapshot.contains("close:vaeDecoder") && h.log.snapshot.contains("open:vaeDecoder:false"))

    let other = try Flux2Harness(references: false)
    defer { other.fixture.cleanup() }
    await #expect(throws: AetherError.self) {
        try await other.session.generate(configuration: configuration, preparedPrompt: prepared)
    }
    await #expect(throws: AetherError.self) {
        try await other.session.generate(configuration: configuration, conditioning: prepared.conditioning, references: refs)
    }
    #expect(other.log.snapshot.isEmpty)
    await h.session.unload()
}

@Test func residentPolicyRetainsTheTextEncoderAndDenoisers() async throws {
    let h = try Flux2Harness(residency: .resident)
    defer { h.fixture.cleanup() }
    let configuration = try Flux2Configuration(width: 32, height: 32, steps: 1)
    _ = try await h.session.generate(configuration: configuration, prompt: prompt())
    h.log.clear()
    _ = try await h.session.generate(configuration: configuration, prompt: prompt())
    #expect(!h.log.snapshot.contains { $0.hasPrefix("open:") || $0.hasPrefix("close:") })
    // Explicit preparation still releases both stages.
    h.log.clear()
    _ = try await h.session.prepare(prompt: prompt())
    #expect(h.log.snapshot == ["close:denoiser0", "close:denoiser1", "close:denoiser2", "run:textEncoder", "close:textEncoder"])
    await h.session.unload()
}

@Test func invalidReferenceRequestsRefuseBeforeAnyModelLoads() async throws {
    let h = try Flux2Harness()
    defer { h.fixture.cleanup() }
    let big = try Flux2Configuration(width: 1024, height: 1024, steps: 1)
    await #expect(throws: AetherError.self) {
        try await h.session.generate(configuration: big, prompt: prompt(), references: [reference(rows: 1, columns: 1, fill: 0)])
    }
    await #expect(throws: AetherError.self) {
        try await h.session.prepare(references: [reference(rows: 1, columns: 1, fill: 0), reference(rows: 1, columns: 1, fill: 0),
                                                 reference(rows: 1, columns: 1, fill: 0)])
    }
    await #expect(throws: AetherError.self) {
        try await h.session.generate(configuration: try Flux2Configuration(width: 32, height: 32), prompt: prompt(),
                                     noise: FloatTensor(shape: [1, 128], values: Array(repeating: 0, count: 128)))
    }
    let textOnly = try Flux2Harness(references: false)
    defer { textOnly.fixture.cleanup() }
    await #expect(throws: AetherError.self) {
        try await textOnly.session.prepare(references: [reference(rows: 1, columns: 1, fill: 0)])
    }
    #expect(h.log.snapshot.isEmpty && textOnly.log.snapshot.isEmpty)
}

@Test func overlapIsBusyAndCancellationRecoversAfterTheSubmittedCall() async throws {
    let gate = Flux2Gate()
    let h = try Flux2Harness(gates: ["denoiser1": gate])
    defer { h.fixture.cleanup() }
    let configuration = try Flux2Configuration(width: 32, height: 32, steps: 2)
    let conditioning = try Flux2Conditioning(tensor: Float16Tensor(shape: [512, 7680], values: Array(repeating: 0, count: 512 * 7680)))
    let task = Task { try await h.session.generate(configuration: configuration, conditioning: conditioning) }
    await gate.waitUntilEntered()
    await #expect(throws: AetherError.sessionBusy) { try await h.session.prepare(prompt: prompt()) }
    task.cancel()
    #expect(await h.session.state == .running)  // the submitted call has not returned yet
    await gate.release()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await h.session.state == .ready)
    #expect(!h.log.snapshot.contains("run:denoiser2"))
    // Retained models recover for the next request.
    h.log.clear()
    _ = try await h.session.generate(configuration: configuration, conditioning: conditioning)
    #expect(!h.log.snapshot.contains { $0.hasPrefix("open:denoiser") })
    await h.session.unload()
}

@Test func encoderRunFailuresDiscardTheEncoderAndRetryReopensIt() async throws {
    let h = try Flux2Harness(residency: .resident)
    defer { h.fixture.cleanup() }
    let configuration = try Flux2Configuration(width: 32, height: 32, steps: 1)
    h.factory.failNextRun(of: "textEncoder")
    await #expect(throws: AetherError.self) { try await h.session.generate(configuration: configuration, prompt: prompt()) }
    #expect(h.log.snapshot == ["open:textEncoder:true", "run:textEncoder", "close:textEncoder"])
    #expect(await h.session.state == .ready)
    // The failed encoder was discarded: the retry opens a new one, which resident mode then keeps.
    h.log.clear()
    _ = try await h.session.generate(configuration: configuration, prompt: prompt())
    #expect(h.log.snapshot.prefix(2) == ["open:textEncoder:true", "run:textEncoder"])
    #expect(!h.log.snapshot.contains("close:textEncoder"))

    h.log.clear()
    h.factory.failNextRun(of: "vaeEncoder")
    let refs = try [reference(rows: 1, columns: 1, fill: 0), reference(rows: 1, columns: 2, fill: 0)]
    await #expect(throws: AetherError.self) { try await h.session.prepare(references: refs, generatedTokens: 4) }
    #expect(h.log.snapshot == ["open:vaeEncoder:false", "run:vaeEncoder", "close:vaeEncoder"])
    h.log.clear()
    let prepared = try await h.session.prepare(references: refs, generatedTokens: 4)
    #expect(prepared.tokens == 3)
    #expect(h.log.snapshot == ["open:vaeEncoder:false", "run:vaeEncoder", "run:vaeEncoder", "close:vaeEncoder"])
    await h.session.unload()
}

@Test func cancellingAnEncoderCallDiscardsTheEncoderAndRecovers() async throws {
    let configuration = try Flux2Configuration(width: 32, height: 32, steps: 1)
    let refs = try [reference(rows: 1, columns: 1, fill: 0)]
    for role in ["textEncoder", "vaeEncoder"] {
        let gate = Flux2Gate()
        let h = try Flux2Harness(residency: .resident, gates: [role: gate])
        defer { h.fixture.cleanup() }
        @Sendable func request() async throws {
            if role == "textEncoder" { _ = try await h.session.generate(configuration: configuration, prompt: prompt()) }
            else { _ = try await h.session.prepare(references: refs, generatedTokens: 4) }
        }
        let task = Task { try await request() }
        await gate.waitUntilEntered()
        await #expect(throws: AetherError.sessionBusy, "\(role)") { try await h.session.prepare(prompt: prompt()) }
        task.cancel()
        #expect(await h.session.state == .running, "\(role)")  // the submitted call has not returned
        await gate.release()
        await #expect(throws: CancellationError.self, "\(role)") { try await task.value }
        #expect(h.log.snapshot == ["open:\(role):\(role == "textEncoder")", "run:\(role)", "close:\(role)"], "\(role)")
        #expect(await h.session.state == .ready, "\(role)")
        // Recovery opens a fresh encoder and completes.
        h.log.clear()
        try await request()
        #expect(h.log.snapshot.prefix(2) == ["open:\(role):\(role == "textEncoder")", "run:\(role)"], "\(role)")
        await h.session.unload()
    }
}

@Test func referencePreparationChecksTheAdmissionProfileBeforeLoading() async throws {
    let coordinator = try ResourceCoordinator(capacityBytes: 1_000)
    let h = try Flux2Harness { assets in
        let models = try ([assets.textEncoder, assets.decoder, assets.referenceEncoder!] + assets.denoisers).map {
            try Flux2ModelResourceEstimate(asset: $0, residentBytes: 100, runScratchBytes: 10)
        }
        return try Flux2ResourceAdmission(coordinator: coordinator, operationBytes: 50, maximumImageTokens: 6, models: models)
    }
    defer { h.fixture.cleanup() }
    // 4 reference tokens + 4 generated exceed the profile's 6 even though the bundle allows 4,096.
    await #expect(throws: AetherError.self) {
        try await h.session.prepare(references: [reference(rows: 2, columns: 2, fill: 0)], generatedTokens: 4)
    }
    await #expect(throws: AetherError.self) {
        try await h.session.prepare(references: [reference(rows: 1, columns: 1, fill: 0)], generatedTokens: 0)
    }
    #expect(h.log.snapshot.isEmpty && coordinator.snapshot.reservedBytes == 0)
    let prepared = try await h.session.prepare(references: [reference(rows: 1, columns: 2, fill: 0)], generatedTokens: 4)
    #expect(prepared.tokens == 2 && coordinator.snapshot.reservedBytes == 0)
    await h.session.unload()
}

@Test func denoiserLoadFailureClosesThePartialSetAndUnloadWaitsForActiveWork() async throws {
    let h = try Flux2Harness(failures: ["denoiser1"])
    defer { h.fixture.cleanup() }
    let configuration = try Flux2Configuration(width: 32, height: 32, steps: 1)
    await #expect(throws: AetherError.self) { try await h.session.generate(configuration: configuration, prompt: prompt()) }
    // Partial denoiser load closed the opened segment; a retry loads the full set.
    #expect(h.log.snapshot.suffix(2) == ["open:denoiser1:true", "close:denoiser0"])
    _ = try await h.session.generate(configuration: configuration, prompt: prompt())

    let gate = Flux2Gate()
    let busy = try Flux2Harness(gates: ["vaeDecoder": gate])
    defer { busy.fixture.cleanup() }
    let task = Task { try await busy.session.generate(configuration: configuration, prompt: prompt()) }
    await gate.waitUntilEntered()
    let closing = Task { await busy.session.unload() }
    while await busy.session.state != .closing { await Task.yield() }
    await #expect(throws: AetherError.sessionClosed) { try await busy.session.prepare(prompt: prompt()) }
    await gate.release()
    let image = try await task.value
    await closing.value
    let finalState = await busy.session.state
    #expect(image.width == 32 && finalState == .closed)
    #expect(busy.log.snapshot.last == "close:vaeDecoder")
    await #expect(throws: AetherError.sessionClosed) { try await busy.session.prepare(prompt: prompt()) }
    await busy.session.unload()
}

@Test func resourceAdmissionReservesAndRefusesUnknownOrOversizedWork() async throws {
    let coordinator = try ResourceCoordinator(capacityBytes: 1_000)
    let gate = Flux2Gate()
    let h = try Flux2Harness(gates: ["denoiser0": gate]) { assets in
        let models = try ([assets.textEncoder, assets.decoder] + assets.denoisers).map {
            try Flux2ModelResourceEstimate(asset: $0, residentBytes: 100, runScratchBytes: 10)
        }
        return try Flux2ResourceAdmission(coordinator: coordinator, operationBytes: 50, maximumImageTokens: 64, models: models)
    }
    defer { h.fixture.cleanup() }
    let configuration = try Flux2Configuration(width: 32, height: 32, steps: 1)
    let task = Task { try await h.session.generate(configuration: configuration, prompt: prompt()) }
    await gate.waitUntilEntered()
    // Operation + three denoisers resident + one in-flight scratch.
    #expect(coordinator.snapshot.reservedBytes == 50 + 300 + 10)
    await gate.release()
    _ = try await task.value
    #expect(coordinator.snapshot.reservedBytes == 400)  // denoisers and decoder stay resident
    await #expect(throws: AetherError.self) {
        try await h.session.generate(configuration: try Flux2Configuration(width: 256, height: 256), prompt: prompt())
    }
    // The reference encoder has no estimate: refused before loading, nothing leaks.
    await #expect(throws: AetherError.self) { try await h.session.prepare(references: [reference(rows: 1, columns: 1, fill: 0)]) }
    #expect(coordinator.snapshot.reservedBytes == 400)
    await h.session.unload()
    #expect(coordinator.snapshot.reservedBytes == 0)
}

@Test func contractsRejectIncompatibleSignatures() throws {
    var signatures = try segmentSignatures()
    try Flux2Contracts.segments(signatures, tokens: 8)
    signatures.swapAt(1, 2)
    #expect(throws: AetherError.self) { try Flux2Contracts.segments(signatures, tokens: 8) }
    let wrong = try describe("decode", ["latents": value("float32", [-1, -1, 128])], ["image": value("float32", [3, -1, -1])])
    #expect(throws: AetherError.self) { try Flux2Contracts.decoder(wrong, rows: 2, columns: 2) }
    #expect(throws: AetherError.self) {
        try Flux2Contracts.value(.float16(Float16Tensor(shape: [2, 128], values: Array(repeating: 0, count: 256))),
                                 type: "float16", shape: [3, 128])
    }
}

private func estimates(_ coordinator: ResourceCoordinator) -> (Flux2Assets) throws -> Flux2ResourceAdmission {
    { assets in
        let models = try ([assets.textEncoder, assets.decoder] + assets.denoisers).map {
            try Flux2ModelResourceEstimate(asset: $0, residentBytes: 100, runScratchBytes: 10)
        }
        return try Flux2ResourceAdmission(coordinator: coordinator, operationBytes: 50, maximumImageTokens: 64, models: models)
    }
}

@Test(.timeLimit(.minutes(1))) func sharedGPUAdmissionGatesFluxBeforeAnyModelOpensAndBeforeReserving() async throws {
    let coordinator = try ResourceCoordinator(capacityBytes: 2_000)
    let admission = try GPUAdmission()
    let gate = Flux2Gate()
    let running = try Flux2Harness(gates: ["denoiser0": gate], resources: estimates(coordinator), gpu: admission.owner("a"))
    let waiting = try Flux2Harness(resources: estimates(coordinator), gpu: admission.owner("b"))
    defer { running.fixture.cleanup(); waiting.fixture.cleanup() }
    let configuration = try Flux2Configuration(width: 32, height: 32, steps: 1)
    let first = Task { try await running.session.generate(configuration: configuration, prompt: prompt()) }
    await gate.waitUntilEntered()
    #expect(coordinator.snapshot.reservedBytes == 50 + 300 + 10)
    let second = Task { try await waiting.session.generate(configuration: configuration, prompt: prompt()) }
    while admission.snapshot.queued == 0 { try await Task.sleep(for: .milliseconds(5)) }
    // Queued behind the running session: nothing opened and nothing reserved.
    #expect(waiting.log.snapshot.isEmpty)
    #expect(coordinator.snapshot.reservedBytes == 50 + 300 + 10)
    #expect(await waiting.session.state == .running)
    await gate.release()
    _ = try await first.value
    _ = try await second.value
    #expect(waiting.log.snapshot.contains("open:denoiser0:true"))
    #expect(admission.snapshot == .init(running: 0, queued: 0, holdsCrossProcessLease: false))
    await running.session.unload(); await waiting.session.unload()
    #expect(coordinator.snapshot.reservedBytes == 0)
}

@Test(.timeLimit(.minutes(1))) func fluxUnloadWithdrawsAnOperationStillWaitingForTheGPU() async throws {
    let admission = try GPUAdmission()
    let h = try Flux2Harness(gpu: admission.owner("flux"))
    defer { h.fixture.cleanup() }
    let other = try await admission.admit(owner: "another session")
    let configuration = try Flux2Configuration(width: 32, height: 32, steps: 1)
    let task = Task { try await h.session.generate(configuration: configuration, prompt: prompt()) }
    while admission.snapshot.queued == 0 { try await Task.sleep(for: .milliseconds(5)) }
    await h.session.unload()
    await #expect(throws: AetherError.sessionClosed) { try await task.value }
    #expect(await h.session.state == .closed)
    #expect(h.log.snapshot.filter { $0.hasPrefix("open:") }.isEmpty)
    #expect(admission.snapshot == .init(running: 1, queued: 0, holdsCrossProcessLease: false))
    other.release()
}
