import CoreGraphics
import Foundation
import Testing
@testable import AetherRuntime
import AetherCore
import AetherTestSupport

/// `vlm-static-inject@1` on the CPU (docs/VLM.md §6, G1–G2): smart resize, the prompt plan against 0.0.7's M-RoPE
/// oracle, the manifest contract, and the engine's image binding on a fake tower and a fake decoder that reads its
/// statics.
@Suite("Vision-language (fake)")
struct VisionLanguageTests {
    // MARK: smart resize

    /// transformers' `smart_resize(h, w, factor=32, min_pixels=65536, max_pixels=524288)`, recorded (5.22.1).
    @Test func smartResizeMatchesUpstream() throws {
        var resize = Manifest.ImagePreprocess.Resize(mode: .smartResize, filter: .torchUint8Bicubic)
        resize.factor = 32
        resize.minPixels = 65536
        resize.maxPixels = 524_288
        let cases: [((h: Int, w: Int), (h: Int, w: Int))] = [
            ((80, 80), (256, 256)), ((608, 810), (608, 800)), ((1412, 1200), (768, 640)), ((2268, 4032), (512, 960)),
            ((256, 256), (256, 256)), ((704, 704), (704, 704)), ((576, 768), (576, 768)), ((512, 896), (512, 896)),
            ((48, 6000), (64, 6016)), ((100, 3000), (96, 3008)), ((80, 112), (224, 320)),
        ]
        for (source, want) in cases {
            let got = try #require(try resize.smartSize(width: source.w, height: source.h))
            #expect(got.width == want.w && got.height == want.h, "\(source)")
        }
        #expect(throws: AetherError.self) { _ = try resize.smartSize(width: 1, height: 1000) }
    }

    @Test func smartResizeValidation() throws {
        var resize = Manifest.ImagePreprocess.Resize(mode: .smartResize, filter: .torchUint8Bicubic)
        let patches = Manifest.ImagePreprocess.Patchify(patch: 16, temporal: 2, merge: 2, rowOrder: .blockMajor, withinPatch: .channelFramePixel)
        #expect(throws: AetherError.self) { try Manifest.ImagePreprocess(resize: resize, layout: .patches, patchify: patches).validate() }
        resize.factor = 32
        resize.minPixels = 65536
        resize.maxPixels = 524_288
        try Manifest.ImagePreprocess(resize: resize, layout: .patches, patchify: patches).validate()
        #expect(throws: AetherError.self) { try Manifest.ImagePreprocess(resize: resize, layout: .NCHW).validate() }
        var stretch = Manifest.ImagePreprocess.Resize(mode: .stretch, size: .init(width: 8, height: 8), filter: .vImage)
        stretch.factor = 4
        #expect(throws: AetherError.self) { try Manifest.ImagePreprocess(resize: stretch).validate() }
    }

    // MARK: the prompt plan

    /// 0.0.7's `LanguagePromptPlan.positions`, ported as the oracle: a text token takes the running rotary position p
    /// on every axis and p advances by one; an image of merged grid (h, w) takes (p, p + row, p + column) in raster
    /// order and p then advances by max(h, w).
    static func oracle(_ tokens: [Int32], images: [ImagePromptPlan.Image]) -> [[Int32]] {
        var rows: [[Int32]] = []
        var p = 0, index = 0, k = 0
        while index < tokens.count {
            if k < images.count, images[k].start == index {
                let image = images[k]
                for hi in 0..<image.gridH { for wi in 0..<image.gridW { rows.append([Int32(p), Int32(p + hi), Int32(p + wi)]) } }
                index += image.count
                p += max(image.gridH, image.gridW)
                k += 1
            } else {
                rows.append([Int32(p), Int32(p), Int32(p)])
                p += 1
                index += 1
            }
        }
        return rows
    }

    /// The decoder's rule applied to the plan's statics (what the graph computes per token).
    static func graphPositions(_ plan: ImagePromptPlan, slots: Int, entries: Int, vocab: Int32) -> [[Int32]] {
        let positions = plan.imagePositions(slots: slots), shifts = plan.ropeShifts(entries: entries)
        return plan.tokens.enumerated().map { p, id in
            if id >= vocab {
                let g = Int(id - vocab)
                return Array(positions[(g * 3)..<(g * 3 + 3)])
            }
            var r = Int32(p)
            for j in 0..<entries where Int32(p) >= shifts.start[j] { r -= shifts.amount[j] }
            return [r, r, r]
        }
    }

    static func spec(slots: Int = 64, images: Int = 4, rows: Int = 64) throws -> ImageInputSpec {
        ImageInputSpec(padId: FakeVLM.pad, openId: FakeVLM.open, closeId: FakeVLM.close, idBase: Int32(FakeVLM.vocab),
                       slots: slots, maxImages: images, patchRows: rows, merge: 2,
                       preprocessor: try ImagePreprocessor(FakeVLM.preprocess(patchRows: rows)))
    }

    static func prepared(width: Int, height: Int, seed: Int, spec: ImageInputSpec) throws -> PreparedPromptImage {
        try spec.prepare(ImageInput(FakeVLM.picture(width: width, height: height, seed: seed)))
    }

    /// A rendered prompt: text runs around one `[open, pad, close]` per image.
    static func rendered(images: Int, text: Int = 3) -> [Int32] {
        var out: [Int32] = [256] + (0..<Int32(text)).map { 40 + $0 }
        for i in 0..<images { out += [FakeVLM.open, FakeVLM.pad, FakeVLM.close] + (0..<Int32(text + i)).map { 60 + $0 } }
        return out + [257, 10]
    }

    @Test func planPositionsEqualThe007Oracle() throws {
        let spec = try Self.spec()
        let sizes = [(8, 8), (16, 8), (8, 16), (4, 4), (12, 8)]
        for count in 0...4 {
            let images = try (0..<count).map { k in try Self.prepared(width: sizes[k].0, height: sizes[k].1, seed: k, spec: spec) }
            let plan = try ImagePromptPlan(rendered: Self.rendered(images: count), images: images, spec: spec)
            #expect(plan.images.count == count)
            #expect(plan.tokens.count == Self.rendered(images: count).count - count + images.reduce(0) { $0 + $1.tokens })
            #expect(Self.graphPositions(plan, slots: spec.slots, entries: spec.maxImages, vocab: spec.idBase)
                    == Self.oracle(plan.tokens, images: plan.images), "\(count) images")
            // Slots pack in order; samplers see the placeholder.
            var base = 0
            for image in plan.images {
                #expect(image.base == base)
                #expect(Array(plan.tokens[image.start..<image.end]) == (0..<image.count).map { spec.idBase + Int32(base + $0) })
                #expect(plan.placeholderTokens[image.start..<image.end].allSatisfy { $0 == FakeVLM.pad })
                base += image.count
            }
        }
        // The grids: 8 × 16 pixels is a 4 × 8 patch grid, 2 × 4 tokens.
        let wide = try Self.prepared(width: 16, height: 8, seed: 0, spec: spec)
        #expect((wide.gridH, wide.gridW, wide.tokens) == (4, 8, 8))
    }

    @Test func planRefusesWhatTheDecoderCannotHold() throws {
        let spec = try Self.spec(slots: 10, images: 2)
        let one = try Self.prepared(width: 8, height: 8, seed: 1, spec: spec)          // 4 tokens
        let two = try Self.prepared(width: 8, height: 8, seed: 2, spec: spec)
        let three = try Self.prepared(width: 8, height: 8, seed: 3, spec: spec)
        #expect(throws: AetherError.self) { _ = try ImagePromptPlan(rendered: Self.rendered(images: 1), images: [], spec: spec) }
        #expect(throws: AetherError.self) { _ = try ImagePromptPlan(rendered: Self.rendered(images: 1), images: [one, two], spec: spec) }
        #expect(throws: AetherError.self) {
            _ = try ImagePromptPlan(rendered: Self.rendered(images: 3), images: [one, two, three], spec: spec)
        }
        #expect(throws: AetherError.self) {   // 12 tokens for 10 slots
            _ = try ImagePromptPlan(rendered: Self.rendered(images: 3), images: [one, two, three],
                                    spec: try Self.spec(slots: 10, images: 4))
        }
        #expect(throws: AetherError.self) { _ = try ImagePromptPlan(rendered: [1, FakeVLM.pad, FakeVLM.close], images: [one], spec: spec) }
        #expect(throws: AetherError.self) { _ = try ImagePromptPlan(rendered: [FakeVLM.open, FakeVLM.pad, 2], images: [one], spec: spec) }
        #expect(throws: AetherError.self) { _ = try ImagePromptPlan(rendered: [1, Int32(FakeVLM.vocab) + 1], images: [], spec: spec) }
        let plan = try ImagePromptPlan(rendered: Self.rendered(images: 2), images: [one, two], spec: spec)
        #expect(plan.images.map(\.count) == [4, 4])
    }

    @Test func reuseStopsAtTheFirstDifferentImage() throws {
        let spec = try Self.spec()
        let a = try Self.prepared(width: 8, height: 8, seed: 1, spec: spec)
        let b = try Self.prepared(width: 8, height: 8, seed: 2, spec: spec)          // same grid, other content
        let ab = try ImagePromptPlan(rendered: Self.rendered(images: 2), images: [a, b], spec: spec)
        let aa = try ImagePromptPlan(rendered: Self.rendered(images: 2), images: [a, a], spec: spec)
        let a1 = try ImagePromptPlan(rendered: Self.rendered(images: 1), images: [a], spec: spec)
        let none = try ImagePromptPlan(rendered: Self.rendered(images: 0), images: [], spec: spec)
        #expect(ab.tokens == aa.tokens)                                              // ids alone can't tell them apart
        #expect(ab.reusableLimit(bound: ab.images) == .max)
        #expect(aa.reusableLimit(bound: ab.images) == ab.images[1].start)
        #expect(a1.reusableLimit(bound: ab.images) == ab.images[1].start)
        #expect(ab.reusableLimit(bound: a1.images) == ab.images[1].start)
        #expect(none.reusableLimit(bound: ab.images) == ab.images[0].start)
        #expect(ab.reusableLimit(bound: []) == ab.images[0].start)
    }

    @Test func preparedImagesNameTheirContent() throws {
        let spec = try Self.spec()
        let a = try Self.prepared(width: 8, height: 8, seed: 1, spec: spec)
        #expect(a == (try Self.prepared(width: 8, height: 8, seed: 1, spec: spec)))
        #expect(a != (try Self.prepared(width: 8, height: 8, seed: 2, spec: spec)))
        #expect(a.values.count == 16 * FakeVLM.rowLength)
        // Larger than the tower: smart resize brings it within (64 rows of 2 × 2 pixels: 256 pixels).
        let big = try Self.prepared(width: 64, height: 32, seed: 0, spec: spec)
        #expect(big.gridH * big.gridW <= 64)
    }

    // MARK: the manifest

    /// The exporter's own manifest (`export_qwen3vl.py --sample-manifest --pins source-pins-4b-abliterated.json`)
    /// decodes, validates and meets the contract, with the typed fields the driver reads.
    @Test func theExportersManifestMeetsTheContract() throws {
        let url = try #require(Bundle.module.url(forResource: "Fixtures/qwen3vl-4b-sample-aether", withExtension: "json"))
        var manifest = try Manifest.decode(Data(contentsOf: url))
        try DriverContract.vlmStaticInject.validate(manifest)
        #expect(manifest.driver == .vlmStaticInject)
        #expect(manifest.graphs["vision"]?.kind == .visionTower)
        let resize = try #require(manifest.preprocess?.image?.resize)
        #expect((resize.mode, resize.factor, resize.minPixels, resize.maxPixels) == (.smartResize, 32, 65536, 524_288))
        let image = try #require(manifest.image)
        #expect((image.slots, image.maxImages) == (4096, 16))
        let tokens = try image.typedTokens()
        #expect((tokens.padId, tokens.openId, tokens.closeId, tokens.idBase) == (151_655, 151_652, 151_653, 151_936))
        let decoder = try #require(manifest.graphs["decoder"])
        #expect(decoder.inputEncodings?["tokens"]?.extensionIds?.base == 151_936)
        #expect(decoder.inputEncodings?["tokens"]?.extensionIds?.count == 4096)
        for role in ["imageEmbeds", "deepstackEmbeds", "imagePositions", "ropeShiftStart", "ropeShiftAmount"] {
            #expect(decoder.inputs[role]?.isStatic == true, "\(role)")
        }
        // A graph kind that isn't one is refused; a missing required static role too.
        manifest.graphs["vision"]!.kind = .decoderLM
        #expect(throws: AetherError.self) { try manifest.validate() }
        manifest.graphs["vision"]!.kind = .visionTower
        manifest.graphs["decoder"]!.inputs["imageEmbeds"] = nil
        #expect(throws: AetherError.self) { try DriverContract.vlmStaticInject.validate(manifest) }
    }

    // MARK: the engine

    static func plan(_ engine: DecoderEngine, _ images: [(Int, Int, Int)], text: Int = 3) throws -> (ImagePromptPlan, [PreparedPromptImage]) {
        let spec = try #require(engine.imageInput)
        let prepared = try images.map { try spec.prepare(ImageInput(FakeVLM.picture(width: $0.0, height: $0.1, seed: $0.2))) }
        return (try ImagePromptPlan(rendered: Self.rendered(images: images.count, text: text), images: prepared, spec: spec), prepared)
    }

    /// Binds, prefills the whole prompt and greedy-steps `steps` tokens on a fresh engine.
    static func fresh(_ images: [(Int, Int, Int)], text: Int = 3, chunk: Int = 512, steps: Int = 6) async throws -> [Int32] {
        let engine = try await FakeVLM.engine(prefillChunk: chunk)
        let (plan, prepared) = try plan(engine, images, text: text)
        try await engine.bindImages(plan, prepared: prepared)
        return try await decode(engine, plan.tokens, steps: steps)
    }

    static func decode(_ engine: DecoderEngine, _ tokens: [Int32], from start: Int = 0, steps: Int) async throws -> [Int32] {
        var token = try Logits.argmax(try #require(try await engine.prefill(Array(tokens[start...]))).last)
        var out = [token]
        for _ in 1..<steps {
            token = try Logits.argmax(try await engine.step(token).last)
            out.append(token)
        }
        return out
    }

    @Test func statesAreTheOraclePositions() async throws {
        let rope = RopeLog()
        let engine = try await FakeVLM.engine(rope: rope)
        let (plan, prepared) = try Self.plan(engine, [(8, 8, 1), (16, 8, 2), (4, 4, 3)])
        try await engine.bindImages(plan, prepared: prepared)
        _ = try await engine.prefill(plan.tokens)
        #expect(rope.positions(plan.tokens.count) == Self.oracle(plan.tokens, images: plan.images))
        #expect(await engine.boundImages == plan.images)
        #expect(await engine.imageEncodes == 3)
    }

    @Test(arguments: [1, 3, 5, 512]) func chunkedPrefillEqualsOneShot(chunk: Int) async throws {
        let images = [(8, 8, 1), (16, 8, 2)]
        #expect(try await Self.fresh(images, chunk: chunk) == (try await Self.fresh(images)))
    }

    @Test func contentSlotsAndOrderChangeTheOutput() async throws {
        let base = try await Self.fresh([(8, 8, 1), (8, 8, 2)])
        #expect(base != (try await Self.fresh([(8, 8, 1), (8, 8, 3)])))             // another picture, same grid
        #expect(base != (try await Self.fresh([(8, 8, 2), (8, 8, 1)])))             // the order
        #expect(base != (try await Self.fresh([(8, 8, 1)])))
    }

    @Test func aSecondTurnOnTheSameImagesReusesThemWithoutEncoding() async throws {
        let towerLog = CallLog()
        let engine = try await FakeVLM.engine(towerLog: towerLog)
        let images = [(8, 8, 1), (16, 8, 2)]
        let (first, prepared) = try Self.plan(engine, images, text: 3)
        try await engine.bindImages(first, prepared: prepared)
        _ = try await engine.prefill(first.tokens)
        // The next turn extends the prompt: the images stay where they were.
        let (second, again) = try Self.plan(engine, images, text: 3)
        let longer = second.tokens + [65, 66, 67]
        let keep = min(second.reusableLimit(bound: await engine.boundImages), first.tokens.count)
        #expect(keep == first.tokens.count)
        try await engine.rewind(to: keep)
        try await engine.bindImages(second, prepared: again)
        #expect(towerLog.calls.count == 2)                                           // no new encode
        let reused = try await Self.decode(engine, longer, from: keep, steps: 4)
        let fresh = try await FakeVLM.engine()
        try await fresh.bindImages(second, prepared: again)
        #expect(reused == (try await Self.decode(fresh, longer, steps: 4)))
    }

    @Test func aSameGridDifferentImageReusesNothingPastItsStart() async throws {
        let towerLog = CallLog()
        let engine = try await FakeVLM.engine(towerLog: towerLog)
        let (first, prepared) = try Self.plan(engine, [(8, 8, 1), (8, 8, 2)])
        try await engine.bindImages(first, prepared: prepared)
        _ = try await engine.prefill(first.tokens)
        let (second, other) = try Self.plan(engine, [(8, 8, 1), (8, 8, 9)])
        #expect(second.tokens == first.tokens)
        let keep = min(Generatorless.commonPrefix(first.tokens, second.tokens) - 1, second.reusableLimit(bound: await engine.boundImages))
        #expect(keep == first.images[1].start)
        try await engine.rewind(to: keep)
        #expect(await engine.boundImages == [first.images[0]])                       // the cut image is dropped
        try await engine.bindImages(second, prepared: other)
        #expect(towerLog.calls.count == 3)                                           // only the new picture is encoded
        let reused = try await Self.decode(engine, second.tokens, from: keep, steps: 5)
        #expect(reused == (try await Self.fresh([(8, 8, 1), (8, 8, 9)], steps: 5)))
    }

    @Test func textAfterARewindBeforeTheImagesEqualsAFreshTextRun() async throws {
        let engine = try await FakeVLM.engine()
        let (plan, prepared) = try Self.plan(engine, [(8, 8, 1), (16, 8, 2)])
        try await engine.bindImages(plan, prepared: prepared)
        _ = try await engine.prefill(plan.tokens)
        let start = plan.images[0].start
        try await engine.rewind(to: start)
        #expect(await engine.boundImages.isEmpty)
        let text = Array(plan.tokens[..<start]) + (0..<Int32(40)).map { 70 + $0 }
        let reused = try await Self.decode(engine, text, from: start, steps: 5)
        let fresh = try await FakeVLM.engine()
        #expect(reused == (try await Self.decode(fresh, text, steps: 5)))
        // And a reset restores the text-only statics too.
        try await engine.reset()
        #expect(try await Self.decode(engine, text, steps: 5) == reused)
    }

    @Test func aBindIsTheEnginesOperationUntilItEnds() async throws {
        let gate = TowerGate(holds: 1)
        let engine = try await FakeVLM.engine(towerGate: gate)
        let (plan, prepared) = try Self.plan(engine, [(8, 8, 1)])
        let bind = Task { try await engine.bindImages(plan, prepared: prepared) }
        for await _ in gate.entered { break }                                        // suspended in the tower
        await #expect(throws: AetherError.self) { _ = try await engine.prefill([1, 2]) }
        await #expect(throws: AetherError.self) { try await engine.beginGreedy(first: 1, limit: 2) }
        await #expect(throws: AetherError.self) { try await engine.bindImages(plan, prepared: prepared) }
        await #expect(throws: AetherError.self) { try await engine.reset() }
        await #expect(throws: AetherError.self) { _ = try await engine.snapshot() }
        #expect(await engine.position == 0)
        gate.open()
        try await bind.value
        #expect(await engine.boundImages == plan.images)
        let held = try await Self.decode(engine, plan.tokens, steps: 4)
        #expect(held == (try await Self.fresh([(8, 8, 1)], steps: 4)))
    }

    @Test func aFailedBindRefusesToFeedOnImagesUntilARewindOrReset() async throws {
        let gate = TowerGate()
        let engine = try await FakeVLM.engine(towerGate: gate)
        let (first, prepared) = try Self.plan(engine, [(8, 8, 1)])
        try await engine.bindImages(first, prepared: prepared)
        _ = try await engine.prefill(first.tokens)
        // The next turn keeps the first image and adds one; its encode fails.
        let (second, other) = try Self.plan(engine, [(8, 8, 1), (8, 8, 2)])
        let keep = min(Generatorless.commonPrefix(first.tokens, second.tokens), second.reusableLimit(bound: await engine.boundImages))
        let start = first.images[0].start
        #expect(keep > first.images[0].end)
        try await engine.rewind(to: keep)
        gate.failing = true
        await #expect(throws: AetherError.self) { try await engine.bindImages(second, prepared: other) }
        gate.failing = false
        #expect(await engine.boundImages.isEmpty)
        // The state still holds the first image's tokens and the text fed after it with its shift.
        await #expect(throws: AetherError.self) { _ = try await engine.prefill([65]) }
        await #expect(throws: AetherError.self) { _ = try await engine.step(65) }
        await #expect(throws: AetherError.self) { try await engine.beginGreedy(first: 65, limit: 2) }
        await #expect(throws: AetherError.self) { _ = try await engine.snapshot() }
        try await engine.rewind(to: start + 1)                                       // still inside the image
        await #expect(throws: AetherError.self) { _ = try await engine.prefill([65]) }
        // A rewind to the image's start: text from there equals a fresh text run.
        try await engine.rewind(to: start)
        let text = Array(first.tokens[..<start]) + (0..<Int32(20)).map { 70 + $0 }
        let reused = try await Self.decode(engine, text, from: start, steps: 4)
        #expect(reused == (try await Self.decode(try await FakeVLM.engine(), text, steps: 4)))
        // A reset clears it too.
        try await engine.reset()
        try await engine.bindImages(first, prepared: prepared)
        _ = try await engine.prefill(first.tokens)
        try await engine.rewind(to: keep)
        gate.failing = true
        await #expect(throws: AetherError.self) { try await engine.bindImages(second, prepared: other) }
        await #expect(throws: AetherError.self) { _ = try await engine.prefill([65]) }
        try await engine.reset()
        _ = try await engine.prefill([1, 2])
        // A failed bind on a state that holds no image refuses nothing.
        await #expect(throws: AetherError.self) { try await engine.bindImages(first, prepared: prepared) }
        _ = try await engine.prefill([3])
        gate.failing = false
    }

    @Test func extensionIdsOutsideTheSlotsAreRefused() async throws {
        let engine = try await FakeVLM.engine(slots: 8)
        await #expect(throws: AetherError.self) { _ = try await engine.prefill([1, Int32(FakeVLM.vocab + 8)]) }
        let text = try await FakeVLM.engine(slots: 8)
        _ = try await text.prefill([1, 2, Int32(FakeVLM.vocab + 7)])
    }

    @Test func aTextModelTakesNoImages() async throws {
        let fake = FakeLM()
        let engine = try await DecoderEngine(step: fake, graph: fake.manifestGraph(), device: DecoderEngineTests.mac,
                                             options: DecoderEngineTests.options())
        #expect(engine.imageInput == nil)
        let spec = try Self.spec()
        let image = try Self.prepared(width: 8, height: 8, seed: 1, spec: spec)
        let plan = try ImagePromptPlan(rendered: Self.rendered(images: 1), images: [image], spec: spec)
        await #expect(throws: AetherError.self) { try await engine.bindImages(plan, prepared: [image]) }
    }

    @Test func theVisionGraphMustDeclareItsKind() async throws {
        var vision = FakeVLM.visionGraph()
        vision.kind = nil
        await #expect(throws: AetherError.self) {
            _ = try await DecoderEngine(
                step: FakeVLMDecoder(slots: 24, images: 4), graph: FakeVLM.decoderGraph(slots: 24), vision: FakeVisionTower(rows: 16),
                visionGraph: vision, image: try FakeVLM.imageBlock(slots: 24, images: 4),
                preprocess: FakeVLM.preprocess(patchRows: 16), device: DecoderEngineTests.mac, options: DecoderEngineTests.options())
        }
    }

    @Test func theTowerAndStaticsMustAgree() async throws {
        // A tower whose token rows don't match the decoder's hidden width, slots that disagree with the manifest.
        await #expect(throws: AetherError.self) {
            _ = try await DecoderEngine(
                step: FakeVLMDecoder(slots: 24, images: 4), graph: FakeVLM.decoderGraph(slots: 24), vision: FakeVisionTower(rows: 16),
                visionGraph: FakeVLM.visionGraph(), image: try FakeVLM.imageBlock(slots: 32, images: 4),
                preprocess: FakeVLM.preprocess(patchRows: 16), device: DecoderEngineTests.mac, options: DecoderEngineTests.options())
        }
        await #expect(throws: AetherError.self) {
            _ = try await DecoderEngine(
                step: FakeVLMDecoder(slots: 24, images: 4), graph: FakeVLM.decoderGraph(slots: 24), vision: FakeVisionTower(rows: 16),
                visionGraph: FakeVLM.visionGraph(), image: try FakeVLM.imageBlock(slots: 24, images: 4),
                preprocess: FakeVLM.preprocess(patchRows: 32), device: DecoderEngineTests.mac, options: DecoderEngineTests.options())
        }
    }
}

enum Generatorless {
    static func commonPrefix(_ a: [Int32], _ b: [Int32]) -> Int {
        var n = 0
        while n < a.count, n < b.count, a[n] == b[n] { n += 1 }
        return n
    }
}
