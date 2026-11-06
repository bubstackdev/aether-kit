import Foundation
import Testing
@testable import AetherCore
@testable import AetherRuntime
@testable import AetherVideo

/// MiniMax H3's host recipe and contract, hermetic: the packed layout against Diffusers' own (a tiny geometry recorded
/// by Forge/h3/layout_fixture_h3.py), the qualified geometry, the modulation padding, the scheduler arm, the decode
/// tiling and stitching, the manifest block and the pipeline's refusals.
@Suite("MiniMax H3 host recipe")
struct MiniMaxH3Tests {
    static var manifestURL: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "AetherCoreTests/Fixtures/h3/minimax-h3.json")
    }

    static func manifest() throws -> Manifest { try Manifest.decode(Data(contentsOf: manifestURL)) }

    struct LayoutFixture: Decodable {
        let textTokens, latentFrames, latentHeight, latentWidth, audioLatents: Int
        let videoTimestep, audioTimestep: Float
        let positions: [Float]
        let tags, videoIndices, audioIndices, textIndices, timestepIndices: [Int32]
        let timesteps, cos, sin: [Float]
    }

    @Test func layoutMatchesDiffusers() throws {
        let url = try #require(Bundle.module.url(forResource: "h3-layout", withExtension: "json", subdirectory: "Fixtures"))
        let f = try JSONDecoder().decode(LayoutFixture.self, from: Data(contentsOf: url))
        let layout = H3Layout(textTokens: f.textTokens, latentFrames: f.latentFrames, latentHeight: f.latentHeight,
                              latentWidth: f.latentWidth, audioLatents: f.audioLatents)
        #expect(layout.positions == f.positions)
        #expect(layout.tags == f.tags)
        #expect(layout.textIndices == f.textIndices && layout.audioIndices == f.audioIndices && layout.videoIndices == f.videoIndices)
        let row = layout.timesteps(video: f.videoTimestep, audio: f.audioTimestep)
        #expect(row.values == f.timesteps)
        #expect(row.indices == f.timestepIndices)
        #expect(row.adaln == zip(f.timestepIndices, f.tags).map { $0 * 3 + $1 })
        let rotary = layout.rotary()
        let worst = zip(rotary.cos + rotary.sin, f.cos + f.sin).map { abs($0 - $1) }.max() ?? 0
        #expect(worst <= 2e-6, "rotary differs from MiniMaxH3RotaryPosEmbed by \(worst)")
        // The gather is a permutation into [text; video; audio].
        #expect(Set(layout.gather) == Set((0..<Int32(layout.rows))))
    }

    @Test func qualifiedGeometry() throws {
        let h3 = try #require(try Self.manifest().h3)
        let profile = try #require(h3.profile("960x544x124"))
        let g = H3Geometry(profile)
        #expect(g.videoRows == 18870 && g.audioLatents == 207 && g.audioRows == 414 && g.videoWindows == 7)
        #expect(g.latentHeight == 34 && g.latentWidth == 60 && g.rowsPerLatentFrame == 510)
        let layout = H3Layout(textTokens: 26, geometry: g)
        #expect(layout.rows == 19310)
        let tiling = H3Tiling(width: 960, height: 544)
        #expect(tiling.count == 15)
        for (starts, overlaps, extent) in [(tiling.yStarts, tiling.yOverlaps, 544), (tiling.xStarts, tiling.xOverlaps, 960)] {
            #expect(starts.first == 0 && starts.last! + 256 == extent)
            #expect(overlaps.allSatisfy { $0 >= 64 && $0 % 16 == 0 })
            #expect(zip(zip(starts, starts.dropFirst()), overlaps).allSatisfy { $0.0 + 256 - $1 == $0.1 })
        }
    }

    @Test func packingAndNoiseOrder() throws {
        let g = H3Geometry(width: 64, height: 64, frames: 22, latentFrames: 7, audioLatents: 3)
        // Source value i at [c, f, y, x] packs to row (f, y/2, x/2), slot c·4 + (y%2)·2 + x%2.
        let source = (0..<(24 * 7 * 4 * 4)).map(Float.init)
        let packed = g.packFrames(source)
        func src(_ c: Int, _ f: Int, _ y: Int, _ x: Int) -> Float { Float(((c * 7 + f) * 4 + y) * 4 + x) }
        #expect(packed[0] == src(0, 0, 0, 0) && packed[1] == src(0, 0, 0, 1) && packed[2] == src(0, 0, 1, 0))
        #expect(packed[4] == src(1, 0, 0, 0))
        #expect(packed[96] == src(0, 0, 0, 2))
        #expect(packed[4 * 96] == src(0, 1, 0, 0))
        // One stream: the source video tensor, then the audio rows.
        let latents = try g.initialLatents(seed: 7)
        let stream = try SeededGaussianNoise.generate(seed: 7, count: g.videoRows * 96 + g.audioRows * 32)
        #expect(latents.video == g.packFrames(Array(stream[..<(g.videoRows * 96)])))
        #expect(latents.audio == Array(stream[(g.videoRows * 96)...]))
    }

    @Test func modulationPaddingRepeatsRows() {
        // [6, 3·1, 2] bf16 → [6, 3·2, 2]: each slice's three rows, then copies.
        let bytes = (0..<(6 * 3 * 2 * 2)).map { UInt8($0) }
        let padded = H3Padding.table(bytes, k: 1, cardinality: 2, width: 2)
        #expect(padded.count == 6 * 6 * 4)
        for slice in 0..<6 {
            let original = Array(bytes[(slice * 12)..<(slice * 12 + 12)])
            #expect(Array(padded[(slice * 24)..<(slice * 24 + 24)]) == original + original)
        }
        #expect(H3Padding.embedding([1, 2, 3], k: 1, cardinality: 3) == [1, 2, 3, 1, 2, 3, 1, 2, 3])
        #expect(H3Padding.embedding([1, 2, 3, 4], k: 2, cardinality: 2) == [1, 2, 3, 4])
    }

    @Test func schedulerArm() throws {
        let h3 = try #require(try Self.manifest().h3)
        let recipe = try #require(h3.recipes["turbo8/t2va"])
        let arm = H3Arm(recipe.video)
        var x: [Float] = [0.5, -1]
        let v: [Float] = [0.25, 2]
        // The last step (σ_next = 0) lands on the denoised estimate x + σ·v.
        let last = recipe.steps.count - 1, sigma = 1 - arm.timesteps[last]
        arm.advance(step: last, sample: &x, velocity: v)
        #expect(x == [0.5 + sigma * 0.25, -1 + sigma * 2])
    }

    @Test func temporalStitchingEmitsEveryFrame() {
        var stitcher = H3TemporalStitcher(width: 2, height: 1)
        var frames = 0
        for w in 0..<7 { frames += stitcher.push([Float](repeating: Float(w), count: 3 * 22 * 2)).count / 6 }
        frames += stitcher.finish().count / 6
        #expect(frames == 124)
        // The cross-fade starts at the previous window's value.
        var s = H3TemporalStitcher(width: 1, height: 1)
        _ = s.push([Float](repeating: 1, count: 66))
        let second = s.push([Float](repeating: 0, count: 66))
        #expect(second[0] == 1)
    }

    @Test func manifestBlockDecodesAndRefuses() throws {
        let manifest = try Self.manifest()
        let h3 = try #require(manifest.h3)
        #expect(h3.profiles.map(\.key) == ["960x544x124", "544x960x124"])
        #expect(h3.video.noise.order == .sourceVideoThenAudio)
        #expect(try h3.recipes["turbo8/t2va"]!.steps[3].table(7) == AssetPath(asset: "main", path: "tables/turbo8/t2va/step-3/block-07.raw"))
        try DriverContract.minimaxH3.validate(manifest)
        var json = try JSONSerialization.jsonObject(with: Data(contentsOf: Self.manifestURL)) as! [String: Any]
        var block = json["h3"] as! [String: Any]
        var conditioner = block["conditioner"] as! [String: Any]
        conditioner["tap"] = 49
        block["conditioner"] = conditioner
        json["h3"] = block
        let error = #expect(throws: AetherError.self) { _ = try Manifest.decode(try JSONSerialization.data(withJSONObject: json)) }
        guard case .manifestInvalid(let field, _)? = error else { Issue.record("\(String(describing: error))"); return }
        #expect(field == "h3.conditioner.tap")
    }

    static func pipeline(device: Device = Device(platform: .macOS, architecture: "h17c")) throws -> MiniMaxH3Pipeline {
        try MiniMaxH3Pipeline(
            manifest: try manifest(), device: device, bundle: nil, options: .init(probe: GPUMemoryProbe { 0 }),
            tokenize: { $0.unicodeScalars.map { Int32($0.value) } },
            load: { role throws(AetherError) in throw .manifestInvalid(field: role, reason: "no graphs in this test") })
    }

    @Test func refusesBeforeLoading() async throws {
        let phone = Device(platform: .iOS, architecture: "h19p")
        let error = #expect(throws: AetherError.self) { _ = try Self.pipeline(device: phone) }
        guard case .unsupportedDevice? = error else { Issue.record("\(String(describing: error))"); return }
        let pipeline = try Self.pipeline()
        for request in [VideoRequest(prompt: "a fox", profile: "640x480x124"), VideoRequest(prompt: "a fox", recipe: "turbo3"),
                        VideoRequest(prompt: "x")] {
            await #expect(throws: AetherError.self) { _ = try await pipeline.generate(request) }
        }
        #expect(await pipeline.residentRoles.isEmpty)
        let plan = try await pipeline.plan(VideoRequest(prompt: "a fox in snow"))
        #expect(plan.profile.key == "960x544x124" && plan.ids.count == 13)
        #expect(plan.cardinality == plan.recipe.steps.map(\.timesteps.count).max())
    }
}
