import AetherAssets
import AetherKit
import CoreGraphics
import Foundation
import Testing
@testable import AetherFlux2

@Test func scheduleMatchesThePipelinesEmpiricalShift() throws {
    // Observed in the unmodified reference pipeline: 2 steps at 256 generated tokens (11 §B14).
    let two = try Flux2Schedule(steps: 2, imageTokens: 256)
    #expect(two.sigmas == [1, 0.8787367343902588, 0])
    // μ rule: linear in steps between the 10- and 200-step lines; above 4,300 tokens only the high line.
    let shift = Flux2Schedule.EmpiricalShift.klein
    let high = 0.00016927 * 4096 + 0.45666666, low = 8.73809524e-05 * 4096 + 1.89833333
    let slope = (high - low) / 190
    #expect(shift.mu(imageTokens: 4096, steps: 4) == slope * 4 + (high - 200 * slope))
    #expect(shift.mu(imageTokens: 5120, steps: 4) == 0.00016927 * 5120 + 0.45666666)
    let four = try Flux2Schedule(steps: 4, imageTokens: 1024)
    #expect(four.sigmas.count == 5 && four.sigmas.first == 1 && four.sigmas.last == 0)
    #expect(four.timesteps == Array(four.sigmas.dropLast()).map { $0 * 1000 / 1000 })
    #expect(zip(four.sigmas, four.sigmas.dropFirst()).allSatisfy { $0 > $1 })
    #expect(throws: AetherError.self) { try Flux2Schedule(steps: 0, imageTokens: 1) }
}

@Test func eulerRoundsTheProductBeforeTheSum() throws {
    let schedule = try Flux2Schedule(steps: 4, imageTokens: 1024)
    let x: [Float] = [0.1, -2, 3.25], v: [Float] = [1.5, 0.75, -4]
    let delta = schedule.sigmas[1] - schedule.sigmas[0]
    let expected = zip(x, v).map { pair -> Float in let product = delta * pair.1; return pair.0 + product }
    #expect(try schedule.advance(step: 0, latents: x, velocity: v) == expected)
    #expect(throws: AetherError.self) { try schedule.advance(step: 4, latents: x, velocity: v) }
    #expect(throws: AetherError.self) { try schedule.advance(step: 0, latents: x, velocity: [1]) }
}

@Test func rotaryTablesUseFourAxesAndReferenceOffsets() throws {
    let text = try Flux2Rotary.table(ids: Flux2Rotary.textIDs(length: 3))
    #expect(text.shape == [3, 2, 128])
    // Text token 2: only the fourth axis rotates; its first frequency is 1, so dims 48 and 112 hold cos(2).
    let base = 2 * 256
    #expect(text.values[base + 48] == Float(cos(2.0)) && text.values[base + 48 + 64] == Float(cos(2.0)))
    #expect(text.values[base + 128 + 48] == Float(sin(2.0)))
    #expect(text.values[base + 0] == 1 && text.values[base + 128] == 0)
    let ids = Flux2Rotary.gridIDs(rows: 2, columns: 3, t: 20)
    #expect(Array(ids[(5 * 4)..<(6 * 4)]) == [20, 1, 2, 0])
    let grid = try Flux2Rotary.table(ids: ids)
    // Token 5 = (20, 1, 2, 0): axis 0 dim 0 = cos(20), axis 1 dim 16 = cos(1), axis 2 dim 32 = cos(2).
    #expect(grid.values[5 * 256 + 0] == Float(cos(20.0)))
    #expect(grid.values[5 * 256 + 16] == Float(cos(1.0)))
    #expect(grid.values[5 * 256 + 32] == Float(cos(2.0)))
    let frequency = 1 / pow(2000.0, 2.0 / 32)
    #expect(grid.values[5 * 256 + 1] == Float(cos(20 * frequency)))
    #expect(throws: AetherError.self) { try Flux2Rotary.table(ids: [1, 2, 3]) }
}

@Test func promptPadsRightAndTruncatesThePrefix() throws {
    #expect(Flux2Prompt.render("a fox") == "<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    let prompt = try Flux2Prompt(renderedTokenIDs: [5, 6, 7])
    #expect(prompt.tokenIDs.count == 512 && Array(prompt.tokenIDs.prefix(4)) == [5, 6, 7, 151643])
    #expect(prompt.realTokens == 3 && !prompt.wasTruncated)
    let inputs = try prompt.encoderInputs
    #expect(inputs["position_ids"]?.shape == [1, 512] && inputs["attention_mask"]?.shape == [1, 512])
    let long = try Flux2Prompt(renderedTokenIDs: Array(0..<600))
    #expect(long.wasTruncated && long.tokenIDs.last == 511 && long.realTokens == 512)
    let replay = try Flux2Prompt(paddedTokenIDs: prompt.tokenIDs, attentionMask: prompt.attentionMask)
    let again = try Flux2Prompt(tokenIDs: prompt.tokenIDs, mask: prompt.attentionMask)
    #expect(replay == again && replay.tokenIDs == prompt.tokenIDs)
    #expect(throws: AetherError.self) {
        try Flux2Prompt(paddedTokenIDs: prompt.tokenIDs, attentionMask: [0] + Array(prompt.attentionMask.dropFirst()))
    }
    #expect(throws: AetherError.self) { try Flux2Prompt(renderedTokenIDs: []) }
}

extension Flux2Prompt {
    init(tokenIDs: [Int32], mask: [Int32]) throws { try self.init(paddedTokenIDs: tokenIDs, attentionMask: mask) }
}

@Test func configurationRefusesUnsupportedSizes() throws {
    let configuration = try Flux2Configuration(width: 512, height: 384, seed: 7)
    #expect(configuration.imageTokenCount == 768 && configuration.steps == 4)
    let noise = try configuration.initialNoise()
    #expect(noise.shape == [768, 128])
    #expect(noise.values == (try SeededGaussianNoise.generate(seed: 7, count: 768 * 128)))
    #expect(try Flux2Configuration(width: 1024, height: 1024).imageTokenCount == 4096)
    for (w, h) in [(500, 512), (16, 512), (1040, 1024), (0, 64)] {
        #expect(throws: AetherError.self) { try Flux2Configuration(width: w, height: h) }
    }
    #expect(throws: AetherError.self) { try Flux2Configuration(width: 512, height: 512, steps: 51) }
}

@Test func referencePolicyAndPreparationBoundsApplyBeforeLoading() throws {
    let limits = Flux2ReferenceLimits(maxImages: 2, maxSide: 512, maxTotalTokens: 4096, positionStride: 10)
    try Flux2ReferencePolicy.check([(16, 16), (12, 16)], generatedTokens: 1024, limits: limits)
    #expect(throws: AetherError.self) { try Flux2ReferencePolicy.check([(1, 1)], generatedTokens: 1, limits: nil) }
    #expect(throws: AetherError.self) {
        try Flux2ReferencePolicy.check([(1, 1), (1, 1), (1, 1)], generatedTokens: 1, limits: limits)
    }
    #expect(throws: AetherError.self) { try Flux2ReferencePolicy.check([(33, 1)], generatedTokens: 1, limits: limits) }
    #expect(throws: AetherError.self) { try Flux2ReferencePolicy.check([(32, 32)], generatedTokens: 4000, limits: limits) }
    #expect(try Flux2ReferenceImage.preparedSize(width: 1000, height: 600, maxSide: 512) == (512, 304))
    #expect(try Flux2ReferenceImage.preparedSize(width: 200, height: 90, maxSide: 512) == (192, 80))
    #expect(try Flux2ReferenceImage.preparedSize(width: 10, height: 10, maxSide: 512) == (16, 16))
    #expect(throws: AetherError.self) { try Flux2ReferenceImage(pixels: FloatTensor(shape: [3, 20, 16], values: Array(repeating: 0, count: 960))) }

    // A 40×24 red image fits without upscaling and floors to 32×16; red maps to (1, −1, −1).
    let space = CGColorSpace(name: CGColorSpace.sRGB)!
    let context = CGContext(data: nil, width: 40, height: 24, bitsPerComponent: 8, bytesPerRow: 160, space: space,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setFillColor(CGColor(srgbRed: 1, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 40, height: 24))
    let image = try Flux2ReferenceImage(image: context.makeImage()!, maxSide: 512)
    #expect(image.width == 32 && image.height == 16 && image.tokens == 2)
    let plane = 32 * 16
    #expect(image.pixels.values[0] == 1 && image.pixels.values[plane] == -1 && image.pixels.values[2 * plane] == -1)
}

@Test func bundleOpensWithIdentitiesAndRefusesOtherRecipes() throws {
    let fixture = try Flux2BundleFixture()
    defer { fixture.cleanup() }
    let assets = try Flux2Assets.open(at: fixture.root)
    #expect(assets.denoisers.count == 3 && assets.supportsReferences)
    #expect(assets.references == Flux2ReferenceLimits(maxImages: 2, maxSide: 512, maxTotalTokens: 4096, positionStride: 10))
    #expect(assets.fingerprint.count == 64 && assets.referenceFingerprint?.count == 64)
    #expect(assets.referenceFingerprint != assets.fingerprint)
    try assets.verify()

    let textOnly = try Flux2BundleFixture(references: false)
    defer { textOnly.cleanup() }
    let plain = try Flux2Assets.open(at: textOnly.root)
    #expect(!plain.supportsReferences && plain.referenceFingerprint == nil)
    // Placement comes from the manifest: per-asset overrides over the bundle-wide default.
    #expect(assets.expectsFrequentReshapes("textEncoder") && assets.expectsFrequentReshapes("denoiser2"))
    #expect(!assets.expectsFrequentReshapes("vaeDecoder") && !assets.expectsFrequentReshapes("vaeEncoder"))

    for change in ["schema", "shift", "rotary", "prompt", "target", "binding", "segments", "role", "references", "source",
                   "revision", "reshapes", "placement"] {
        let bad = try Flux2BundleFixture()
        defer { bad.cleanup() }
        try bad.edit("aether.json") { json in
            var d = json["diffusion"] as! [String: Any]
            switch change {
            case "schema": json["schema"] = 2
            case "shift": var s = d["sigmaShift"] as! [String: Any]; s["tokenThreshold"] = 4301; d["sigmaShift"] = s
            case "rotary": d["rotary"] = ["axes": [32, 32, 32, 32], "theta": 10000.0]
            case "prompt": var p = d["prompt"] as! [String: Any]; p["maxLength"] = 256; d["prompt"] = p
            case "target": var t = json["target"] as! [String: Any]; t["compiled"] = true; json["target"] = t
            case "binding": var i = json["inputs"] as! [String: Any]; i["latents"] = ["bind": "latents", "dtype": "float32"]; json["inputs"] = i
            case "segments": d["denoiserSegments"] = ["denoiser0", "denoiser1"]
            case "role": var f = json["functions"] as! [String: Any]; f["vaeDecoder/decode"] = ["role": "decode"]; json["functions"] = f
            case "references": var r = d["referenceImages"] as! [String: Any]; r["positionStride"] = 1; d["referenceImages"] = r
            case "reshapes": d["assetReshapes"] = ["denoiser9": false]
            case "placement": d.removeValue(forKey: "expectFrequentReshapes")
            case "revision": json["provenance"] = ["source": ["hfRepo": "black-forest-labs/FLUX.2-klein-4B", "revision": "x"], "files": [String: String]()]
            default: json["provenance"] = ["source": ["hfRepo": "other/model", "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294"], "files": [String: String]()]
            }
            json["diffusion"] = d
        }
        #expect(throws: AetherError.self, "\(change)") { try Flux2Assets.open(at: bad.root) }
    }
}

@Test func bundleChecksProvenanceAndDetectsMutation() throws {
    let fixture = try Flux2BundleFixture()
    defer { fixture.cleanup() }
    try fixture.edit("aether.json") { json in
        json["provenance"] = ["source": ["hfRepo": "black-forest-labs/FLUX.2-klein-4B", "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294"],
                              "files": ["vaeDecoder.aimodel/main.mlirb": "sha256:" + String(repeating: "0", count: 64)]]
    }
    #expect(throws: AetherError.self) { try Flux2Assets.open(at: fixture.root) }

    let clean = try Flux2BundleFixture()
    defer { clean.cleanup() }
    let assets = try Flux2Assets.open(at: clean.root)
    try Data("corrupt".utf8).write(to: assets.denoisers[1].url.appendingPathComponent("main.mlirb"))
    #expect(throws: AetherError.self) { try assets.verify() }
    try Data("{}".utf8).write(to: assets.tokenizerDirectory.appendingPathComponent("tokenizer.json"))
    #expect(throws: AetherError.self) { try assets.verifyTokenizer() }
    #expect(try Flux2Assets.open(at: clean.root).fingerprint != assets.fingerprint)
}

@Test func concurrentVerificationReportsTheFirstFailureInSerialOrder() throws {
    let fixture = try Flux2BundleFixture()
    defer { fixture.cleanup() }
    let assets = try Flux2Assets.open(at: fixture.root)
    // Corrupt a later and an earlier asset; the earlier one (denoiser1 precedes vaeEncoder) is reported.
    try Data("corrupt".utf8).write(to: assets.referenceEncoder!.url.appendingPathComponent("main.mlirb"))
    try Data("corrupt".utf8).write(to: assets.denoisers[1].url.appendingPathComponent("main.mlirb"))
    do { try assets.verify(); Issue.record("verification passed") } catch let error as AetherError {
        #expect(error.description.contains("denoiser1.aimodel") && !error.description.contains("vaeEncoder"))
    }
}

@Test func verificationStaysInsideTheAssetAndObservesCancellation() async throws {
    let fixture = try Flux2BundleFixture()
    defer { fixture.cleanup() }
    let assets = try Flux2Assets.open(at: fixture.root)
    // A stream replaced by a symlink to identical bytes outside the asset directory is refused.
    let stream = assets.decoder.url.appendingPathComponent("main.mlirb")
    let outside = fixture.root.appendingPathComponent("outside.mlirb")
    try FileManager.default.copyItem(at: stream, to: outside)
    try FileManager.default.removeItem(at: stream)
    try FileManager.default.createSymbolicLink(at: stream, withDestinationURL: outside)
    #expect(throws: (any Error).self) { try assets.decoder.verify() }

    let clean = try Flux2BundleFixture()
    defer { clean.cleanup() }
    let verified = try Flux2Assets.open(at: clean.root)
    let task = Task {
        withUnsafeCurrentTask { $0?.cancel() }
        try verified.verify()
    }
    await #expect(throws: CancellationError.self) { try await task.value }
}

@Test func installationRoundTripPreservesIdentities() async throws {
    let fixture = try Flux2BundleFixture()
    defer { fixture.cleanup() }
    let store = FileManager.default.temporaryDirectory.appendingPathComponent("flux2-store-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: store) }
    let assets = try Flux2Assets.open(at: fixture.root)
    let manifest = try assets.installationManifest()
    #expect(manifest.files.count == 4 + 6 * 3)
    let local = try LocalAssetStore(root: store)
    let id = try await local.install(manifest, from: fixture.root)
    let lease = try await local.acquire(id)
    let installed = try Flux2Assets.open(in: lease)
    #expect(installed.fingerprint == assets.fingerprint && installed.referenceFingerprint == assets.referenceFingerprint)
    #expect(installed.root.path.hasPrefix(lease.directory.resolvingSymlinksInPath().path))
}
