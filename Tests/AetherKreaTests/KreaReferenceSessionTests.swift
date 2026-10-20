import AetherKit
import AetherCoreAI
import AetherKrea
import CryptoKit
import Foundation
import Synchronization
import Testing
@testable import AetherKreaCoreAI

// Reference-image (`visionTokens`) execution through scripted models: ordering, residency, validation before any
// load, cancellation/failure cleanup and the exact tensors handed to `encodeWithImages`. No native model runs.

private final class Log: Sendable {
    let values = Mutex<[String]>([])
    func append(_ value: String) { values.withLock { $0.append(value) } }
    var snapshot: [String] { values.withLock { $0 } }
}

private func value(_ type: String, _ shape: [Int]) -> [String: Any] { ["kind": "tensor", "scalarType": type, "shape": shape] }

private func describe(_ name: String, _ inputs: [String: [String: Any]], _ outputs: [String: [String: Any]]) throws -> ModelFunctionDescription {
    try JSONDecoder().decode(ModelFunctionDescription.self, from: JSONSerialization.data(withJSONObject: [
        "name": name, "inputs": inputs, "outputs": outputs, "states": [:],
    ]))
}

/// Adds the vision asset and Krea's reference declarations to a text-only `BundleFixture`.
private func referenceBundle() throws -> BundleFixture {
    let fixture = try BundleFixture()
    let directory = fixture.root.appendingPathComponent("vision.aimodel")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let data = Data("fixture model: vision".utf8)
    try data.write(to: directory.appendingPathComponent("main.mlirb"))
    try Data(SHA256.hash(data: data)).write(to: directory.appendingPathComponent("main.hash"))
    try Data("{\"assetVersion\":\"2.0\"}".utf8).write(to: directory.appendingPathComponent("metadata.json"))
    try fixture.edit("metadata.json") { json in
        var assets = json["assets"] as! [String: String]
        assets["vision"] = "vision.aimodel"
        json["assets"] = assets
    }
    try fixture.edit("aether.json") { json in
        var diffusion = json["diffusion"] as! [String: Any]
        diffusion["referenceImages"] = ["kind": "visionTokens", "encoderAsset": "vision", "maxImages": 4, "maxSide": 384,
                                        "maxTotalTokens": 576, "label": "Picture {n}: "]
        json["diffusion"] = diffusion
        json["vision"] = ["deepstackLevels": 3, "hiddenSize": 2560, "imageMean": [0.5, 0.5, 0.5], "imageStd": [0.5, 0.5, 0.5],
                          "maxPixels": 16_777_216, "mergeSize": 2, "minPixels": 65_536, "patchSize": 16,
                          "positionTableSide": 48, "positions": "mrope", "resample": "bicubic-antialias", "rotaryTheta": 10_000,
                          "temporalPatchSize": 2, "tokens": ["image": 151_655, "visionEnd": 151_653, "visionStart": 151_652]]
    }
    return fixture
}

private func image(_ side: Int, shade: UInt8) throws -> KreaReferencePixels {
    try KreaReferencePreprocessing.prepare(KreaReferenceImage(rgb: Array(repeating: shade, count: side * side * 3), width: side, height: side))
}

/// 40 prefix/template tokens, one pad per image, 3 text tokens: 35 + 3 + image tokens = conditioning rows.
private func prompt(_ images: [KreaReferencePixels]) throws -> KreaReferencePrompt {
    let pad = Int(KreaReferenceContract.imageToken)
    let rendered = Array(1...40) + images.flatMap { _ in [7, pad, 8] } + [9, 10, 11]
    return try KreaReferencePrompt(renderedTokenIDs: rendered, textOnlyTokenCount: 43, images: images)
}

private struct Model: KreaExecutable {
    let functions: [ModelFunctionDescription]
    let run: @Sendable (String, [String: TensorValue]) async throws -> [String: TensorValue]
    let close: @Sendable () -> Void
    func runTensors(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue] { try await run(function, inputs) }
    func unload() async { close() }
}

private actor Gate {
    private var entered = false, released = false
    private var waiters: [CheckedContinuation<Void, Never>] = [], held: CheckedContinuation<Void, Never>?
    func enter() async {
        entered = true; waiters.forEach { $0.resume() }; waiters = []
        if released { return }
        await withCheckedContinuation { held = $0 }
    }
    func waitUntilEntered() async { if entered { return }; await withCheckedContinuation { waiters.append($0) } }
    func release() { released = true; held?.resume(); held = nil }
}

private final class Seen: Sendable { let values = Mutex<[String: TensorValue]>([:]) }

private final class Factory: KreaModelFactory, Sendable {
    let log: Log
    let failVisionAt: Int?
    let gate: Gate?
    let seen = Seen()
    init(log: Log, failVisionAt: Int? = nil, gate: Gate? = nil) { self.log = log; self.failVisionAt = failVisionAt; self.gate = gate }

    func open(_ asset: KreaModelAsset, frequentReshapes: Bool) async throws -> any KreaExecutable {
        let name = asset.url.lastPathComponent
        log.append("open:\(name)")
        switch name {
        case "vision.aimodel":
            #expect(!frequentReshapes) // the vision tower is specialized without reshapes
            let calls = Mutex(0)
            let vision = try describe("encode", [
                "pixels": value("float16", [-1, 1536]), "pos_index": value("int32", [-1, 4]),
                "pos_weight": value("float32", [-1, 4]), "patch_rc": value("int32", [-1, 2]),
            ], ["image_embeds": value("float16", [-1, 2560]), "deepstack": value("float16", [3, -1, 2560])])
            return Model(functions: [vision], run: { [log, failVisionAt, gate] function, inputs in
                if let gate { await gate.enter() }
                #expect(function == "encode")
                guard case .float16(let pixels) = inputs["pixels"], case .int32(let index) = inputs["pos_index"],
                      case .float32(let weight) = inputs["pos_weight"], case .int32(let rc) = inputs["patch_rc"] else {
                    throw AetherError.invalidTensor("missing vision inputs")
                }
                let patches = pixels.shape[0]
                #expect(pixels.shape == [patches, 1536] && index.shape == [patches, 4] && weight.shape == [patches, 4] && rc.shape == [patches, 2])
                let call = calls.withLock { value in defer { value += 1 }; return value }
                log.append("vision\(call)")
                if failVisionAt == call { throw AetherError.invalidTensor("injected vision failure") }
                let tokens = patches / 4
                // Values identify image and level so the concatenation order is observable.
                return [
                    "image_embeds": .float16(try Float16Tensor(shape: [tokens, 2560], values: Array(repeating: Float16(call), count: tokens * 2560))),
                    "deepstack": .float16(try Float16Tensor(shape: [3, tokens, 2560], values: (0..<3).flatMap { level in
                        Array(repeating: Float16(10 * level + call), count: tokens * 2560) })),
                ]
            }, close: { [log] in log.append("closeVision") })
        case "textEncoder.aimodel":
            let signature = try describe("encodeWithImages", [
                "input_ids": value("int32", [1, -1]), "attention_mask": value("int32", [1, -1]), "position_ids": value("int32", [1, -1]),
                "rope_positions": value("int32", [3, 1, -1]), "image_embeds": value("float16", [-1, 2560]),
                "deepstack": value("float16", [3, -1, 2560]),
            ], ["text_embeds": value("float16", [-1, 30720])])
            return Model(functions: [signature], run: { [log, seen] function, inputs in
                log.append("text:\(function)")
                seen.values.withLock { $0 = inputs }
                guard case .int32(let ids) = inputs["input_ids"] else { throw AetherError.invalidTensor("missing ids") }
                let n = ids.shape[1]
                return ["text_embeds": .float16(try Float16Tensor(shape: [n, 30720], values: Array(repeating: 0, count: n * 30720)))]
            }, close: { [log] in log.append("closeText") })
        default:
            throw AetherError.invalidModel("unexpected asset \(name)")
        }
    }
}

@Test func referenceBundleOpensWithASeparateIdentityAndVerifiesTheVisionAsset() throws {
    let text = try BundleFixture()
    defer { text.cleanup() }
    let plain = try KreaAssets.open(at: text.root)
    #expect(!plain.supportsReferences && plain.referenceFingerprint == nil && plain.maximumTextRows == 512)
    #expect(throws: AetherError.self) { _ = try plain.denoiserSelection(imageTokens: 1, textRows: 513) }

    let fixture = try referenceBundle()
    defer { fixture.cleanup() }
    let assets = try KreaAssets.open(at: fixture.root)
    #expect(assets.supportsReferences && assets.maximumTextRows == 1116)
    let reference = try #require(assets.referenceFingerprint)
    #expect(reference != assets.fingerprint)
    #expect(try assets.denoiserSelection(imageTokens: 1, textRows: 1116).textRows == 1116)
    #expect(throws: AetherError.self) { _ = try assets.denoiserSelection(imageTokens: 1, textRows: 1117) }
    try assets.verifyIncludingReferences()
    try Data("corrupt".utf8).write(to: fixture.root.appendingPathComponent("vision.aimodel/main.mlirb"))
    try assets.verify() // text-only verification never reads the vision stream
    #expect(throws: AetherError.self) { try assets.verifyIncludingReferences() }
}

@Test func referenceDeclarationsOtherThanKreasAreRefused() throws {
    let edits: [(inout [String: Any]) -> Void] = [
        { json in var d = json["diffusion"] as! [String: Any]; var r = d["referenceImages"] as! [String: Any]; r["maxImages"] = 8; d["referenceImages"] = r; json["diffusion"] = d },
        { json in var v = json["vision"] as! [String: Any]; v["resample"] = "bilinear"; json["vision"] = v },
        { json in var f = json["functions"] as! [String: Any]; f.removeValue(forKey: "vision/encode"); json["functions"] = f },
        { json in var i = json["inputs"] as! [String: Any]; i["rope_positions"] = ["bind": "positions", "dtype": "int32", "mode": "running"]; json["inputs"] = i },
        { json in json.removeValue(forKey: "vision") },
    ]
    for edit in edits {
        let fixture = try referenceBundle()
        defer { fixture.cleanup() }
        try fixture.edit("aether.json", edit)
        #expect(throws: AetherError.self) { _ = try KreaAssets.open(at: fixture.root) }
    }
}

@Test func referencePromptEncodesEachImageThenEncodeWithImagesInPromptOrder() async throws {
    let fixture = try referenceBundle()
    defer { fixture.cleanup() }
    let log = Log(), factory = Factory(log: log)
    let session = KreaGenerationSession(assets: try KreaAssets.open(at: fixture.root), factory: factory)
    let images = [try image(384, shade: 10), try image(256, shade: 200)] // 144 + 64 tokens
    let reference = try prompt(images)
    let prepared = try await session.prepare(referencePrompt: reference, references: images)
    #expect(prepared.conditioning.rows == reference.conditioningRows)
    #expect(log.snapshot == ["open:vision.aimodel", "vision0", "vision1", "closeVision",
                             "open:textEncoder.aimodel", "text:encodeWithImages", "closeText"])
    let inputs = factory.seen.values.withLock { $0 }
    #expect(inputs["rope_positions"]?.shape == [3, 1, reference.tokenIDs.count])
    guard case .float16(let embeds) = inputs["image_embeds"], case .float16(let deep) = inputs["deepstack"] else {
        Issue.record("missing image inputs"); return
    }
    #expect(embeds.shape == [208, 2560] && deep.shape == [3, 208, 2560])
    #expect(embeds.values[0] == 0 && embeds.values[144 * 2560] == 1)
    // Level-major: level 1 starts with image 0's rows (value 10), then image 1's (value 11).
    #expect(deep.values[208 * 2560] == 10 && deep.values[(208 + 144) * 2560] == 11 && deep.values[2 * 208 * 2560 + 144 * 2560] == 21)
    await session.unload()
}

@Test func mismatchedOrUnsupportedReferencesRefuseBeforeAnyModelLoads() async throws {
    let log = Log()
    let images = [try image(384, shade: 10)]
    let reference = try prompt(images)
    let text = try BundleFixture()
    defer { text.cleanup() }
    let plain = KreaGenerationSession(assets: try KreaAssets.open(at: text.root), factory: Factory(log: log))
    await #expect(throws: AetherError.self) { _ = try await plain.prepare(referencePrompt: reference, references: images) }
    let fixture = try referenceBundle()
    defer { fixture.cleanup() }
    let session = KreaGenerationSession(assets: try KreaAssets.open(at: fixture.root), factory: Factory(log: log))
    await #expect(throws: AetherError.self) {
        _ = try await session.prepare(referencePrompt: reference, references: [try image(384, shade: 11)]) // other digest
    }
    await #expect(throws: AetherError.self) { _ = try await session.prepare(referencePrompt: reference, references: images + images) }
    #expect(log.snapshot.isEmpty)
}

@Test func visionFailureDiscardsTheEncoderAndNeverReachesTheTextEncoder() async throws {
    let fixture = try referenceBundle()
    defer { fixture.cleanup() }
    let log = Log()
    let session = KreaGenerationSession(assets: try KreaAssets.open(at: fixture.root), factory: Factory(log: log, failVisionAt: 1))
    let images = [try image(384, shade: 10), try image(384, shade: 20)]
    await #expect(throws: AetherError.self) { _ = try await session.prepare(referencePrompt: try prompt(images), references: images) }
    #expect(log.snapshot == ["open:vision.aimodel", "vision0", "vision1", "closeVision"])
    // The session stays usable: the next request opens a fresh vision encoder.
    let ok = [try image(384, shade: 30)]
    _ = try await session.prepare(referencePrompt: try prompt(ok), references: ok)
    #expect(log.snapshot.suffix(6) == ["open:vision.aimodel", "vision0", "closeVision", "open:textEncoder.aimodel",
                                       "text:encodeWithImages", "closeText"])
    await session.unload()
}

@Test func cancellationBeforeEncodingLoadsNothing() async throws {
    let fixture = try referenceBundle()
    defer { fixture.cleanup() }
    let log = Log()
    let session = KreaGenerationSession(assets: try KreaAssets.open(at: fixture.root), factory: Factory(log: log))
    let images = [try image(384, shade: 10)]
    let reference = try prompt(images)
    let task = Task { try await session.prepare(referencePrompt: reference, references: images) }
    task.cancel()
    await #expect(throws: (any Error).self) { _ = try await task.value }
    #expect(!log.snapshot.contains("text:encodeWithImages"))
    #expect(log.snapshot.filter { $0.hasPrefix("open:") }.count == log.snapshot.filter { $0.hasPrefix("close") }.count)
}

@Test(.timeLimit(.minutes(1))) func cancellationDuringVisionEncodingJoinsAndDiscardsTheEncoder() async throws {
    let fixture = try referenceBundle()
    defer { fixture.cleanup() }
    let log = Log(), gate = Gate()
    let session = KreaGenerationSession(assets: try KreaAssets.open(at: fixture.root), factory: Factory(log: log, gate: gate))
    let images = [try image(384, shade: 10), try image(384, shade: 20)]
    let reference = try prompt(images)
    let task = Task { try await session.prepare(referencePrompt: reference, references: images) }
    await gate.waitUntilEntered()
    task.cancel()
    // The operation slot stays held until the in-flight native call returns.
    await #expect(throws: AetherError.sessionBusy) { _ = try await session.prepare(referencePrompt: reference, references: images) }
    await gate.release()
    await #expect(throws: (any Error).self) { _ = try await task.value }
    #expect(log.snapshot == ["open:vision.aimodel", "vision0", "closeVision"])
    await session.unload()
}

/// Vision and `encodeWithImages` as above (the text encoder's rows carry their row index, so conditioning is
/// recognisable), plus shape-generic denoiser segments that record the `text_embeds` they are handed, and a decoder.
private final class RefineFactory: KreaModelFactory, Sendable {
    let log: Log
    let reference: Factory
    final class TextLog: Sendable { let values = Mutex<[Float16Tensor]>([]) }
    let seenText = TextLog()
    init(log: Log) { self.log = log; reference = Factory(log: log) }

    func open(_ asset: KreaModelAsset, frequentReshapes: Bool) async throws -> any KreaExecutable {
        let name = asset.url.lastPathComponent
        switch name {
        case "vision.aimodel": return try await reference.open(asset, frequentReshapes: frequentReshapes)
        case "textEncoder.aimodel":
            log.append("open:\(name)")
            let signature = try describe("encodeWithImages", [
                "input_ids": value("int32", [1, -1]), "attention_mask": value("int32", [1, -1]), "position_ids": value("int32", [1, -1]),
                "rope_positions": value("int32", [3, 1, -1]), "image_embeds": value("float16", [-1, 2560]),
                "deepstack": value("float16", [3, -1, 2560]),
            ], ["text_embeds": value("float16", [-1, 30720])])
            return Model(functions: [signature], run: { [log] function, inputs in
                log.append("text:\(function)")
                guard case .int32(let ids) = inputs["input_ids"] else { throw AetherError.invalidTensor("missing ids") }
                let n = ids.shape[1]
                return ["text_embeds": .float16(try Float16Tensor(shape: [n, 30720], values: (0..<n).flatMap { row in
                    Array(repeating: Float16(row), count: 30720) }))]
            }, close: { [log] in log.append("closeText") })
        case "vaeDecoder.aimodel":
            log.append("open:\(name)")
            let decode = try describe("decode", ["latents": value("float16", [-1, -1, 64])], ["image": value("float32", [3, -1, -1])])
            return Model(functions: [decode], run: { _, inputs in
                guard case .float16(let latents) = inputs["latents"] else { throw AetherError.invalidTensor("missing latents") }
                let (h, w) = (latents.shape[0] * 16, latents.shape[1] * 16)
                return ["image": .float32(try FloatTensor(shape: [3, h, w], values: Array(repeating: 0, count: 3 * h * w)))]
            }, close: { [log] in log.append("closeDecoder") })
        default:
            guard name.hasPrefix("denoiser"), let index = Int(name.dropFirst("denoiser".count).prefix(1)) else {
                throw AetherError.invalidModel("unexpected asset \(name)")
            }
            log.append("open:\(name)")
            let image = value("float32", [-1, 6144]), text = value("float32", [-1, 6144]), cond = value("float32", [8, 6144])
            let signature = try [
                describe("segment", ["latents": value("float16", [-1, 64]), "text_embeds": value("float16", [-1, 30720]),
                                     "sigma": value("float32", [1]), "rope_image": value("float32", [-1, 2, 128])],
                         ["cond": cond, "image_1": image, "text_1": text]),
                describe("segment", ["image_1": image, "text_1": text], ["image_2": image, "text_2": text]),
                describe("segment", ["image_2": image, "text_2": text, "cond": cond], ["image_3": image, "text_3": text]),
                describe("segment", ["image_3": image, "text_3": text, "cond": cond], ["velocity": value("float16", [-1, 64])]),
            ][index]
            return Model(functions: [signature], run: { [seenText] _, inputs in
                func rows(_ key: String) -> Int? { inputs[key]?.shape.first }
                var result: [String: TensorValue] = [:]
                if index == 0, case .float16(let embeds) = inputs["text_embeds"] { seenText.values.withLock { $0.append(embeds) } }
                let tokens = rows("latents") ?? rows("image_\(index)") ?? 1, textRows = rows("text_embeds") ?? rows("text_\(index)") ?? 1
                for name in signature.outputs.keys {
                    switch name {
                    case "cond": result[name] = .float32(try FloatTensor(shape: [8, 6144], values: Array(repeating: 0, count: 8 * 6144)))
                    case "velocity": result[name] = .float16(try Float16Tensor(shape: [tokens, 64], values: Array(repeating: 0, count: tokens * 64)))
                    default:
                        let n = name.hasPrefix("image") ? tokens : textRows
                        result[name] = .float32(try FloatTensor(shape: [n, 6144], values: Array(repeating: 0, count: n * 6144)))
                    }
                }
                return result
            }, close: { [log] in log.append("closeDenoiser\(index)") })
        }
    }
}

/// A refine conditioned on reference pictures (the app's reference refine): a prompt prepared with
/// `prepare(referencePrompt:references:)` passes `refine`'s bundle guard, and its conditioning rows (vision tokens
/// included) reach the denoisers unchanged, exactly as `generate(referencePrompt:references:)` hands them over.
@Test func aReferencePreparedPromptDrivesARefineWithItsConditioningUnchanged() async throws {
    let fixture = try referenceBundle()
    defer { fixture.cleanup() }
    let log = Log(), factory = RefineFactory(log: log)
    let session = KreaGenerationSession(assets: try KreaAssets.open(at: fixture.root), factory: factory)
    let images = [try image(384, shade: 10)]
    let reference = try prompt(images)
    let prepared = try await session.prepare(referencePrompt: reference, references: images)
    #expect(prepared.conditioning.rows == reference.conditioningRows && reference.conditioningRows > 144)
    let configuration = try KreaConfiguration(width: 16, height: 16)
    let start = try KreaRefineStart(latents: FloatTensor(shape: [1, 64], values: Array(repeating: 1, count: 64)), startStep: 7)
    let noise = try FloatTensor(shape: [1, 64], values: Array(repeating: 0, count: 64))
    _ = try await session.refine(configuration: configuration, preparedPrompt: prepared, start: start, noise: noise)
    let refined = factory.seenText.values.withLock { $0 }
    #expect(refined.count == 1 && refined[0] == prepared.conditioning.tensor) // one step (7 of 8), the prepared rows exactly
    _ = try await session.generate(configuration: configuration, referencePrompt: reference, references: images, noise: noise)
    let generated = factory.seenText.values.withLock { $0 }
    #expect(generated.count == 9 && generated.dropFirst().allSatisfy { $0 == prepared.conditioning.tensor })
    // A prompt prepared by a session of another bundle never drives a refine.
    let other = try referenceBundle()
    defer { other.cleanup() }
    try other.edit("aether.json") { json in json["description"] = "another export" }
    let foreign = KreaGenerationSession(assets: try KreaAssets.open(at: other.root), factory: RefineFactory(log: Log()))
    await #expect(throws: AetherError.self) {
        _ = try await foreign.refine(configuration: configuration, preparedPrompt: prepared, start: start, noise: noise)
    }
    await session.unload(); await foreign.unload()
}
