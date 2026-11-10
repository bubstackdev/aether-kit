@_spi(Testing) import AetherHost
import CryptoKit
import Foundation
import Testing
@testable import AetherCore
@testable import AetherRuntime
@testable import AetherVideo

/// MiniMax H3's FL2VA keyframes, hermetic: the packed layout with first, last and first + last conditions, the
/// conditioner presentation, the vision tower's inputs, the M-RoPE gather and the condition noising against Diffusers'
/// own code (`Fixtures/h3-keyframes.json`, Forge/h3/keyframe_fixture_h3.py); and, through fake graphs on the qualified
/// canvas, the conditioning each mode assembles: the tiled encode, the tower call, the presentation the plan builds,
/// the condition rows in front of the generated rows, and the scheduler leaving them alone.
@Suite("MiniMax H3 keyframes")
struct MiniMaxH3KeyframeTests {
    struct Fixture: Decodable {
        struct Presentation: Decodable {
            let canvas: String, anchors: [String], count: Int, grids: [[Int]]
            let ids_sha256: String, tags_sha256: String, positions_sha256: String, pixels_sha256: String
        }
        struct Embedded: Decodable { let canvas: String; let ids: [Int32]; let tags: [Int32]; let positions: [[Int32]] }
        struct Vision: Decodable { let canvas: String; let grid: [Int]; let pos_index_sha256, pos_weight_sha256, patch_rc_sha256: String }
        struct Rotary: Decodable { let rows: Int; let mropeSection: [Int]; let cosTableBase64: String; let cosSha256: String }
        struct Layout: Decodable {
            struct Case: Decodable {
                let anchors: [String], conditionRows: Int, positions: [Float], anchorTimes: [Double]
                let tags, videoIndices, audioIndices, timestepIndices: [Int32]
                let timesteps: [Float]
            }
            let textTags: [Int32], latentFrames: Int, latentHeight: Int, latentWidth: Int, audioLatents: Int
            let videoTimestep: Float, audioTimestep: Float
            let cases: [Case]
        }
        struct Noising: Decodable { let timestep: Float; let condition, noise, noised: [Float] }
        let prompt: String, promptIds: [Int32], labels: [String: [Int32]]
        let presentations: [Presentation], embedded: Embedded, vision: [Vision], rotary: Rotary, layout: Layout
        let noising: Noising

        static func load() throws -> Fixture {
            let url = try #require(Bundle.module.url(forResource: "h3-keyframes", withExtension: "json", subdirectory: "Fixtures"))
            return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        }

        func tokenize(_ text: String) -> [Int32] { text == prompt ? promptIds : labels[text] ?? [] }
    }

    static func sha256<T>(_ values: [T]) -> String {
        values.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
    }

    static func canvas(_ text: String) -> (width: Int, height: Int) {
        let sides = text.split(separator: "x").map { Int($0)! }
        return (sides[0], sides[1])
    }

    static func anchors(_ names: [String]) -> [H3Anchor] { names.map { $0 == "first" ? .first : .last } }

    /// The synthetic keyframes the fixture presents: byte i = (7 i + 3) mod 256, the last (5 i + 11) mod 256.
    static func keyframe(_ width: Int, _ height: Int, last: Bool = false) throws -> VideoKeyframe {
        let (m, o) = last ? (5, 11) : (7, 3)
        return try VideoKeyframe(rgb: (0..<(width * height * 3)).map { UInt8((m * $0 + o) % 256) }, width: width, height: height)
    }

    /// The test manifest with the export's FL2VA tower at 960×544 and its `h3.vision` block.
    static func manifest(vision: Bool = true) throws -> Manifest {
        var json = try JSONSerialization.jsonObject(with: Data(contentsOf: MiniMaxH3Tests.manifestURL)) as! [String: Any]
        if vision {
            var graphs = json["graphs"] as! [String: Any]
            graphs["conditionerVision960x544"] = [
                "file": "main:models/conditioner-vision-960x544.aimodel", "function": "main",
                "inputs": ["pixels": "pixels", "posIndex": "pos_index", "posWeight": "pos_weight", "patchRC": "patch_rc"],
                "outputs": ["imageEmbeds": "image_embeds", "deepstack": "deepstack"], "residency": "perPrompt", "compute": "gpu"]
            json["graphs"] = graphs
            var h3 = json["h3"] as! [String: Any]
            h3["vision"] = ["patchSize": 16, "mergeSize": 2, "temporalPatchSize": 2, "tableSide": 48, "deepstackLevels": 3,
                            "presentation": ["imagePad": 151655, "label": "<Picture {index}>: ", "visionEnd": 151653,
                                             "visionStart": 151652]]
            json["h3"] = h3
        }
        return try Manifest.decode(JSONSerialization.data(withJSONObject: json))
    }

    static func rules() throws -> H3KeyframeRules {
        let h3 = try #require(try manifest().h3)
        return try #require(try H3KeyframeRules(h3))
    }

    static func pipeline(manifest: Manifest, fixture: Fixture) throws -> MiniMaxH3Pipeline {
        try MiniMaxH3Pipeline(
            manifest: manifest, device: Device(platform: .macOS, architecture: "h17c"), bundle: nil,
            options: .init(gpu: .unadmitted, probe: GPUMemoryProbe { 0 }), tokenize: { fixture.tokenize($0) },
            load: { role throws(AetherError) in throw .manifestInvalid(field: role, reason: "no graphs in this test") })
    }

    // MARK: Against Diffusers

    @Test func layoutWithKeyframesMatchesDiffusers() throws {
        let f = try Fixture.load().layout
        #expect(Set(f.cases.map { $0.anchors.joined(separator: "+") }) == ["", "first", "last", "first+last"])
        for c in f.cases {
            let layout = H3Layout(textTags: f.textTags, latentFrames: f.latentFrames, latentHeight: f.latentHeight,
                                  latentWidth: f.latentWidth, audioLatents: f.audioLatents, anchors: Self.anchors(c.anchors))
            let name = c.anchors.isEmpty ? "t2va" : c.anchors.joined(separator: "+")
            #expect(layout.positions == c.positions, "\(name) positions")
            #expect(layout.tags == c.tags, "\(name) tags")
            #expect(layout.videoIndices == c.videoIndices && layout.audioIndices == c.audioIndices, "\(name) indices")
            #expect(layout.conditionRows == c.conditionRows, "\(name) condition rows")
            let row = layout.timesteps(video: f.videoTimestep, audio: f.audioTimestep)
            #expect(row.values == c.timesteps && row.indices == c.timestepIndices, "\(name) row timesteps")
            #expect(row.adaln == zip(c.timestepIndices, c.tags).map { $0 * 3 + $1 })
            #expect(Set(layout.gather) == Set((0..<Int32(layout.rows))), "\(name) gather is a permutation")
        }
        // The last anchor sits 5/3 before the end of the generated span, summed as numpy sums it: over 37 frames that
        // differs in the last ulp from a sequential sum, and the fixture holds numpy's.
        let spans = (0..<f.latentFrames).map { (5.0 / 3.0) * ($0 % 5 == 0 ? 1.0 : 4.0) }
        let last = try #require(f.cases.first { $0.anchors == ["last"] })
        #expect(Double(f.textTags.count) + h3NumpySum(spans[...]) - 5.0 / 3.0 == last.anchorTimes[0])
        #expect(spans.reduce(0, +) != h3NumpySum(spans[...]))
    }

    @Test func presentationMatchesDiffusersStep() throws {
        let f = try Fixture.load(), rules = try Self.rules()
        let small = f.embedded
        let tokenize: (String) throws(AetherError) -> [Int32] = { f.tokenize($0) }
        let p = try H3Presentation(prompt: f.promptIds, canvases: [Self.canvas(small.canvas), Self.canvas(small.canvas)],
                                   rules: rules, tokenize: tokenize)
        #expect(p.ids == small.ids && p.tags == small.tags && p.positions == small.positions)
        for item in f.presentations {
            let canvases = Array(repeating: Self.canvas(item.canvas), count: item.anchors.count)
            let p = try H3Presentation(prompt: f.promptIds, canvases: canvases, rules: rules, tokenize: tokenize)
            let name = "\(item.canvas) \(item.anchors.joined(separator: "+"))"
            #expect(p.count == item.count, "\(name)")
            #expect(Self.sha256(p.ids) == item.ids_sha256, "\(name) ids")
            #expect(Self.sha256(p.tags) == item.tags_sha256, "\(name) tags")
            #expect(Self.sha256(p.positions.flatMap { $0 }) == item.positions_sha256, "\(name) positions")
            #expect(p.images.map { [1, $0.rows * 2, $0.columns * 2] } == item.grids, "\(name) grids")
        }
        // A last frame alone is presented as one keyframe, labelled "<Picture 1>": only its anchor differs.
        let first = try #require(f.presentations.first { $0.canvas == "960x544" && $0.anchors == ["first"] })
        let last = try #require(f.presentations.first { $0.canvas == "960x544" && $0.anchors == ["last"] })
        #expect(first.ids_sha256 == last.ids_sha256 && first.positions_sha256 == last.positions_sha256)
    }

    @Test func visionInputsMatchTheProcessorAndTables() throws {
        let f = try Fixture.load(), rules = try Self.rules()
        for item in f.presentations {
            let (w, h) = Self.canvas(item.canvas)
            let frames = try item.anchors.map { try Self.keyframe(w, h, last: $0 == "last") }
            let pixels = try frames.flatMap { try H3VisionInputs($0, rules: rules).pixels }
            #expect(Self.sha256(pixels) == item.pixels_sha256, "\(item.canvas) \(item.anchors) pixels")
        }
        for item in f.vision {
            let (w, h) = Self.canvas(item.canvas)
            let inputs = try H3VisionInputs(try Self.keyframe(w, h), rules: rules)
            #expect([inputs.gridHeight, inputs.gridWidth] == item.grid)
            #expect(Self.sha256(inputs.positionIndex) == item.pos_index_sha256, "\(item.canvas) taps")
            #expect(Self.sha256(inputs.positionWeight) == item.pos_weight_sha256, "\(item.canvas) weights")
            #expect(Self.sha256(inputs.patchCoordinates) == item.patch_rc_sha256, "\(item.canvas) coordinates")
        }
        // A canvas the processor would resize is refused.
        #expect(throws: AetherError.self) { _ = try rules.grid(width: 960, height: 528) }
        #expect(throws: AetherError.self) { _ = try rules.grid(width: 224, height: 224) }
    }

    @Test func rotaryIsAGatherOfTheTextTable() throws {
        let f = try Fixture.load(), rules = try Self.rules()
        let p = try H3Presentation(prompt: f.promptIds, canvases: [(256, 256), (256, 256)], rules: rules,
                                   tokenize: { f.tokenize($0) })
        let bytes = try #require(Data(base64Encoded: f.rotary.cosTableBase64))
        let table = bytes.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
        #expect(table.count == f.rotary.rows * 128 && rules.mropeSection == f.rotary.mropeSection)
        let cos = try p.rotary(table, width: 128, section: rules.mropeSection)
        #expect(Self.sha256(cos) == f.rotary.cosSha256)
        // The label rows (all axes equal) are the table's own rows.
        #expect(Array(cos.prefix(6 * 128)) == Array(table.prefix(6 * 128)))
        #expect(throws: AetherError.self) { _ = try p.rotary(Array(table.prefix(10 * 128)), width: 128, section: rules.mropeSection) }
    }

    @Test func conditionNoisingMatchesScaleNoise() throws {
        let n = try Fixture.load().noising
        let noised = zip(n.condition, n.noise).map { H3Geometry.noised($0, $1, timestep: n.timestep) }
        #expect(noised.map(\.bitPattern) == n.noised.map(\.bitPattern))
    }

    @Test func imageRowsScatterAndAddInBFloat16() {
        // Two rows of width 2 per image at rows 1 and 4 of a 6-row hidden state.
        let images = [H3Presentation.Image(start: 1, rows: 1, columns: 2), H3Presentation.Image(start: 4, rows: 1, columns: 2)]
        func bf16(_ values: [Float]) -> [UInt8] { values.map(h3BFloat16Bits).withUnsafeBytes { [UInt8]($0) } }
        func floats(_ bytes: [UInt8]) -> [Float] {
            bytes.withUnsafeBytes { $0.bindMemory(to: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) } }
        }
        var hidden = bf16([Float](repeating: 1, count: 12))
        H3ImageRows.scatter([bf16([2, 2, 2, 2]), bf16([3, 3, 3, 3])], into: &hidden, width: 2, at: images)
        #expect(floats(hidden) == [1, 1, 2, 2, 2, 2, 1, 1, 3, 3, 3, 3])
        // Level 1 of [2 levels, 2 tokens, 2]: adds 0.5 to the first image's rows and 256 to the second's; 259 is a bf16
        // tie (steps of 2 above 256) and rounds to even, 260.
        H3ImageRows.add([bf16([9, 9, 9, 9, 0.5, 0.5, 0.5, 0.5]), bf16([9, 9, 9, 9, 256, 256, 256, 256])], level: 1,
                        into: &hidden, width: 2, at: images)
        #expect(floats(hidden) == [1, 1, 2.5, 2.5, 2.5, 2.5, 1, 1, 260, 260, 260, 260])
        #expect(h3BFloat16Bits(Float(bitPattern: 0x3F80_8000)) == 0x3F80 && h3BFloat16Bits(Float(bitPattern: 0x3F81_8000)) == 0x3F82)
    }

    // MARK: Fake graphs on the qualified canvas

    /// A graph with fixed tensors whose run calls `body` with the bindings.
    final class Fake: GraphBackend {
        let signature: GraphSignature
        let body: (inout Bindings) throws(AetherError) -> Void
        init(inputs: [TensorSpec], outputs: [TensorSpec], body: @escaping (inout Bindings) throws(AetherError) -> Void) {
            signature = GraphSignature(function: "main", inputs: inputs, states: [], outputs: outputs)
            self.body = body
        }
        func run(_ bindings: inout Bindings) async throws(AetherError) { try body(&bindings) }
    }

    /// The tile graph records each tile's pixels; its moments are the tile's index and its (0, 0) red value. The tail
    /// writes each condition value from the tile it would stitch, so the plumbing (order, layout, shapes) is visible.
    final class Encoder {
        var tiles: [[Float]] = []
        lazy var tile = Fake(inputs: [TensorSpec(name: "pixels", scalarType: .float32, shape: [1, 3, 1, 256, 256])],
                             outputs: [TensorSpec(name: "moments", scalarType: .float32, shape: [1, 48, 1, 16, 16])]) { [unowned self] b throws(AetherError) in
            let pixels = try b.input("pixels")!.readFloats()
            self.tiles.append(pixels)
            try b.output("moments")!.write([Float(self.tiles.count - 1), pixels[0]] + [Float](repeating: 0, count: 48 * 256 - 2))
        }
        let tail = Fake(inputs: [TensorSpec(name: "tiles", scalarType: .float32, shape: [15, 48, 1, 16, 16])],
                        outputs: [TensorSpec(name: "condition", scalarType: .float32, shape: [1, 24, 1, 34, 60])]) { b throws(AetherError) in
            let tiles = try b.input("tiles")!.readFloats(), per = 48 * 256
            // condition[i] = the red (0, 0) value of tile i mod 15, plus the tile index read back from its moments.
            try b.output("condition")!.write((0..<(24 * 34 * 60)).map { i in tiles[(i % 15) * per + 1] + tiles[(i % 15) * per] / 100 })
        }
    }

    /// The tower records the pixels it was fed; its merger rows and DeepStack are constants per keyframe call.
    final class Tower {
        var pixels: [[UInt8]] = []
        lazy var graph = Fake(
            inputs: [TensorSpec(name: "pixels", scalarType: .bfloat16, shape: [2040, 1536]),
                     TensorSpec(name: "pos_index", scalarType: .int32, shape: [2040, 4]),
                     TensorSpec(name: "pos_weight", scalarType: .float32, shape: [2040, 4]),
                     TensorSpec(name: "patch_rc", scalarType: .int32, shape: [2040, 2])],
            outputs: [TensorSpec(name: "image_embeds", scalarType: .bfloat16, shape: [510, 5120]),
                      TensorSpec(name: "deepstack", scalarType: .bfloat16, shape: [3, 510, 5120])]) { [unowned self] b throws(AetherError) in
            self.pixels.append(try b.input("pixels")!.readBytes())
            let value = h3BFloat16Bits(Float(self.pixels.count))
            try b.output("image_embeds")!.writeBytes([UInt16](repeating: value, count: 510 * 5120).withUnsafeBytes { [UInt8]($0) })
            try b.output("deepstack")!.writeBytes([UInt16](repeating: value, count: 3 * 510 * 5120).withUnsafeBytes { [UInt8]($0) })
        }
    }

    static let visionNames = (inputs: ["pixels": "pixels", "posIndex": "pos_index", "posWeight": "pos_weight", "patchRC": "patch_rc"],
                              outputs: ["imageEmbeds": "image_embeds", "deepstack": "deepstack"])

    @Test(arguments: ["first", "last", "first+last"]) func conditioningLayoutThroughFakeGraphs(mode: String) async throws {
        let f = try Fixture.load(), manifest = try Self.manifest(), rules = try Self.rules()
        let pipeline = try Self.pipeline(manifest: manifest, fixture: f)
        let first = try Self.keyframe(960, 544), last = try Self.keyframe(960, 544, last: true)
        let keyframes: VideoKeyframes = switch mode {
        case "first": .first(first)
        case "last": .last(last)
        default: .firstAndLast(first, last)
        }
        let plan = try await pipeline.plan(VideoRequest(prompt: f.prompt, keyframes: keyframes, seed: 7))
        let anchors: [H3Anchor] = mode == "first" ? [.first] : mode == "last" ? [.last] : [.first, .last]
        #expect(plan.anchors == anchors && plan.keyframes == keyframes.frames)
        #expect(plan.recipe == manifest.h3!.recipes["turbo8/fl2va"], "a keyframed clip takes the fl2va recipe")
        #expect(plan.admissionKey.description == "960x544x124/fl2va\(anchors.count)")

        // The presentation the conditioner sees is Diffusers' (ids, tags, positions by hash).
        let expected = try #require(f.presentations.first { $0.canvas == "960x544" && $0.anchors == mode.split(separator: "+").map(String.init) })
        let presentation = try #require(plan.presentation)
        #expect(Self.sha256(plan.conditioningIDs) == expected.ids_sha256 && Self.sha256(plan.textTags) == expected.tags_sha256)
        #expect(Self.sha256(presentation.positions.flatMap { $0 }) == expected.positions_sha256)
        #expect(plan.ids == f.promptIds, "the prompt's own ids are kept apart from the presentation")

        // The tiled encode: 15 tiles at the exporter's recorded starts (`_split_tiles`), planar 0…255 pixels.
        let arena = TensorArena()
        var conditions: [[Float]] = []
        for frame in plan.keyframes {
            let encoder = Encoder()
            conditions.append(try await MiniMaxH3Pipeline.encodeKeyframe(
                frame, tile: (encoder.tile, "pixels", "moments"), tail: (encoder.tail, "tiles", "condition"), arena: arena))
            #expect(encoder.tiles.count == 15)
            let starts = [0, 144, 288].flatMap { y in [0, 176, 352, 528, 704].map { (y, $0) } }
            for (index, (y, x)) in starts.enumerated() {
                let tile = encoder.tiles[index]
                // Channel c, row r, column k of the tile is the frame's pixel (y + r, x + k), channel c.
                for (c, r, k) in [(0, 0, 0), (1, 5, 9), (2, 255, 255)] {
                    #expect(tile[(c * 256 + r) * 256 + k] == Float(frame.rgb[((y + r) * 960 + x + k) * 3 + c]))
                }
            }
            #expect(conditions.last![16] == Float(frame.rgb[176 * 3]) + 0.01, "condition 16 comes from tile 1 (row 0, x 176)")
        }

        // The tower: fed the processor's pixels (Diffusers' hash for these keyframes), one call per keyframe.
        let tower = Tower()
        var vision: [H3VisionRows] = []
        for (frame, image) in zip(plan.keyframes, presentation.images) {
            vision.append(try await MiniMaxH3Pipeline.embedKeyframe(
                frame, rules: rules, graph: tower.graph, names: Self.visionNames, tokens: image.tokens, width: 5120, arena: arena,
                bindWeights: { _ throws(AetherError) in }))
        }
        #expect(Self.sha256(tower.pixels.flatMap { $0 }) == expected.pixels_sha256)
        #expect(vision.count == anchors.count && vision.allSatisfy { $0.embeddings.count == 510 * 5120 * 2 })

        // The packed sequence: the condition rows lead the video rows, each block at its anchor's time, and the
        // vision blocks (and only they, among the text rows) are tagged video.
        let g = plan.geometry
        let layout = H3Layout(textTags: plan.textTags, geometry: g, anchors: plan.anchors)
        let t = presentation.count
        #expect(layout.conditionRows == 510 * anchors.count && layout.videoIndices.count == 510 * anchors.count + g.videoRows)
        #expect(layout.rows == t + 510 * anchors.count + g.audioRows + g.videoRows)
        #expect(layout.tags.prefix(t).filter { $0 == 0 }.count == presentation.images.count * 512)
        let spans = (0..<g.latentFrames).map { (5.0 / 3.0) * ($0 % 5 == 0 ? 1.0 : 4.0) }
        for (index, anchor) in anchors.enumerated() {
            let time = anchor == .first ? Float(t) : Float(Double(t) + h3NumpySum(spans[...]) - 5.0 / 3.0)
            #expect(layout.positions[(t + index * 510) * 3] == time, "\(anchor) anchor time")
        }
        let step = layout.timesteps(video: 0.0476190447807312, audio: 0.1666666865348816)
        #expect(step.values.last == 0.999 && step.indices[t] == Int32(step.values.count - 1), "condition rows held at 0.999")

        // The latents: the condition noise drawn first, in packed order; then the text-to-video stream's shape.
        let latents = try g.initialLatents(seed: 7, conditions: conditions)
        let frame = 24 * 34 * 60
        let stream = try SeededGaussianNoise.generate(seed: 7, count: frame * anchors.count + g.videoRows * 96 + g.audioRows * 32)
        for (index, condition) in conditions.enumerated() {
            let rows = g.packFrames((0..<frame).map { H3Geometry.noised(condition[$0], stream[index * frame + $0], timestep: 0.999) },
                                    latentFrames: 1)
            #expect(Array(latents.video[(index * 510 * 96)..<((index + 1) * 510 * 96)]) == rows, "condition \(index) rows")
        }
        #expect(Array(latents.video.dropFirst(anchors.count * 510 * 96)) == g.packFrames(Array(stream[(frame * anchors.count)..<(frame * anchors.count + g.videoRows * 96)])))

        // The scheduler advances the generated rows only.
        var advanced = latents.video
        let arm = H3Arm(plan.recipe.video)
        arm.advance(step: 0, sample: &advanced, velocity: [Float](repeating: 1, count: advanced.count), from: layout.conditionRows * 96)
        #expect(Array(advanced.prefix(layout.conditionRows * 96)) == Array(latents.video.prefix(layout.conditionRows * 96)))
        #expect(advanced[layout.conditionRows * 96] != latents.video[layout.conditionRows * 96])
    }

    @Test func keyframeRequestsAreRefusedBeforeLoading() async throws {
        let f = try Fixture.load()
        let pipeline = try Self.pipeline(manifest: try Self.manifest(), fixture: f)
        // A keyframe not at the canvas, and one at another profile's canvas.
        for frame in [try Self.keyframe(512, 288), try Self.keyframe(544, 960)] {
            await #expect(throws: AetherError.self) { _ = try await pipeline.plan(VideoRequest(prompt: f.prompt, keyframes: .first(frame))) }
        }
        // An export without the tower can't present keyframes.
        let textOnly = try Self.pipeline(manifest: try Self.manifest(vision: false), fixture: f)
        let error = await #expect(throws: AetherError.self) {
            _ = try await textOnly.plan(VideoRequest(prompt: f.prompt, keyframes: .last(try Self.keyframe(960, 544))))
        }
        guard case .unsupportedOperation? = error else { Issue.record("\(String(describing: error))"); return }
        // Text to video is unchanged: the t2va recipe and the prompt's own rows.
        let plan = try await pipeline.plan(VideoRequest(prompt: f.prompt))
        #expect(plan.presentation == nil && plan.anchors.isEmpty && plan.admissionKey.description == "960x544x124")
        #expect(plan.recipe == pipeline.h3.recipes["turbo8/t2va"] && plan.conditioningIDs == f.promptIds)
        #expect(throws: AetherError.self) { _ = try VideoKeyframe(rgb: [0, 0], width: 1, height: 1) }
    }
}
