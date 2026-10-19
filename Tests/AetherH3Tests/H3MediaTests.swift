import AetherKit
import AetherH3
import CryptoKit
import Foundation
import Testing

private func digest(_ values: [Float]) -> String {
    values.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
}

@Test func h3NoiseMatchesIndependentSplitMixBoxMullerBits() throws {
    // Golden bits from an independent Python implementation of the same stream.
    #expect(try H3Noise.generate(seed: 1043, count: 7).map(\.bitPattern) ==
            [0xbff69b06, 0xbf0b5ad4, 0xbed666d3, 0xbfe6e40c, 0xbfb30e8b, 0xbe3d4e3d, 0x3f2010f4])
    #expect(try H3Noise.generate(seed: 0, count: 3).map(\.bitPattern) == [0xbee7cfdd, 0x3e54c09f, 0x4029a387])
    #expect(try H3Noise.generate(seed: 9, count: 0).isEmpty)
    #expect(throws: AetherError.self) { try H3Noise.generate(seed: 1, count: -1) }
}

@Test func h3InitialLatentsSplitOneStreamIntoPackedVideoThenChannelMajorAudio() throws {
    let geometry = try H3Geometry(width: 64, height: 64, requestedFrames: 22, diagnostic: true)
    let latents = try geometry.initialLatents(seed: 7)
    #expect(latents.video.spec.shape == [1, geometry.targetVideoRows, 96])
    #expect(latents.audio.spec.shape == [1, geometry.audioLatentsPerChannel * 2, 32])
    let videoCount = geometry.targetVideoRows * 96
    let noise = try H3Noise.generate(seed: 7, count: videoCount + geometry.audioRows * 32)
    let packed = try geometry.packVideo(FloatTensor(shape: geometry.videoLatentShape, values: Array(noise[..<videoCount])))
    #expect(try latents.video.floatValues() == packed.values)
    #expect(try latents.audio.floatValues() == Array(noise[videoCount...]))
    #expect(try geometry.initialLatents(seed: 8).video.floatValues() != latents.video.floatValues())
}

/// Independent formulation of the source's temporal cross-fade: every output frame is
/// either a window's own frame or a linear blend of the previous window's frames 17–21
/// with the next window's frames 0–4.
private func referenceTemporal(_ windows: [[Float]], area: Int) -> [Float] {
    func sample(_ w: Int, _ f: Int, _ c: Int, _ p: Int) -> Float { windows[w][(c * 22 + f) * area + p] }
    let total = 17 * windows.count + 5
    var out = [Float](repeating: 0, count: total * 3 * area)
    for frame in 0..<total {
        let w = min(frame / 17, windows.count - 1), f = frame - 17 * w
        for c in 0..<3 {
            for p in 0..<area {
                let value: Float
                if f >= 17 { value = sample(w, f, c, p) }                       // final tail
                else if f < 5, w > 0 {
                    let weight = Float(f) / 5
                    let a = sample(w - 1, 17 + f, c, p), b = sample(w, f, c, p)
                    value = a * (1 - weight) + b * weight
                } else { value = sample(w, f, c, p) }
                out[(frame * 3 + c) * area + p] = value
            }
        }
    }
    return out
}

@Test func h3TemporalStitcherCrossFadesWindowsAndEmits17nPlus5Frames() throws {
    let (width, height) = (4, 2), area = 8
    var generator = SystemRandomNumberGenerator()
    let windows = (0..<3).map { _ in (0..<(66 * area)).map { _ in Float.random(in: -2...2, using: &generator) } }
    var stitcher = try H3VideoStitcher(width: width, height: height, frames: 56)
    #expect(stitcher.expectedWindows == 3)
    var output: [Float] = []
    for window in windows { output += try stitcher.push(window) }
    output += try stitcher.finish()
    #expect(output.count == 56 * 3 * area)
    let reference = referenceTemporal(windows, area: area)
    #expect(zip(output, reference).allSatisfy { abs($0 - $1) <= 1e-6 })
    // The first window passes through untouched; blends start at the second window's frame 1.
    var firstWindow: [Float] = []
    for f in 0..<17 { for c in 0..<3 { firstWindow += windows[0][((c * 22 + f) * area)..<((c * 22 + f + 1) * area)] } }
    #expect(Array(output[0..<(17 * 3 * area)]) == firstWindow)
}

@Test func h3TemporalStitcherRejectsWrongCountsShapesAndValues() throws {
    #expect(throws: AetherError.self) { try H3VideoStitcher(width: 4, height: 2, frames: 21) }
    #expect(throws: AetherError.self) { try H3VideoStitcher(width: 4, height: 2, frames: 40) }
    #expect(throws: AetherError.self) { try H3VideoStitcher(width: 0, height: 2, frames: 22) }
    var stitcher = try H3VideoStitcher(width: 4, height: 2, frames: 22)
    #expect(throws: AetherError.self) { try stitcher.finish() }
    #expect(throws: AetherError.self) { try stitcher.push([Float](repeating: 0, count: 65 * 8)) }
    #expect(throws: AetherError.self) { try stitcher.push([Float](repeating: .nan, count: 66 * 8)) }
    _ = try stitcher.push([Float](repeating: 0.5, count: 66 * 8))
    #expect(throws: AetherError.self) { try stitcher.push([Float](repeating: 0.5, count: 66 * 8)) }
    #expect(try stitcher.finish().count == 5 * 3 * 8)
}

@Test func h3WindowRowsAreContiguousSevenFrameSlicesAtStrideFive() throws {
    let geometry = try H3Geometry(width: 64, height: 64, requestedFrames: 39, diagnostic: true)
    #expect(geometry.frames == 39 && geometry.videoWindows == 2 && geometry.latentFrames == 12)
    let packed = (0..<(geometry.targetVideoRows * 96)).map(Float.init)
    let perFrame = geometry.rowsPerLatentFrame * 96
    #expect(try geometry.videoWindowRows(packed, index: 0) == Array(packed[0..<(7 * perFrame)]))
    #expect(try geometry.videoWindowRows(packed, index: 1) == Array(packed[(5 * perFrame)..<(12 * perFrame)]))
    #expect(throws: AetherError.self) { try geometry.videoWindowRows(packed, index: 2) }
    #expect(throws: AetherError.self) { try geometry.videoWindowRows(Array(packed.dropLast()), index: 0) }
}

@Test func h3PixelsDenormalizeImageNetClipAndRoundTiesToEven() throws {
    // One 1×1 frame: values chosen to hit clipping on both ends and one exact mid value.
    let frames: [Float] = [-10, 10, (0.5 - 0.406) / 0.225]
    let rgb = try H3Pixels.rgb8(frames: frames, width: 1, height: 1)
    #expect(rgb.count == 1 && rgb[0][0] == 0 && rgb[0][1] == 255)
    #expect((127...128).contains(rgb[0][2]))
    #expect(throws: AetherError.self) { try H3Pixels.rgb8(frames: [0, 0], width: 1, height: 1) }
    #expect(throws: AetherError.self) { try H3Pixels.rgb8(frames: [0, .infinity, 0], width: 1, height: 1) }
}

private let sourceTiling = H3SpatialTiling(tileHeight: 256, tileWidth: 256, yStarts: [0, 144, 288],
                                           xStarts: [0, 176, 352, 528, 704], yOverlaps: [112, 112], xOverlaps: [80, 80, 80, 80])

@Test func h3SpatialTilingValidatesTheSourceGridAndRejectsGaps() throws {
    try sourceTiling.validate(width: 960, height: 544)
    #expect(sourceTiling.tileCount == 15)
    #expect(throws: AetherError.self) { try sourceTiling.validate(width: 992, height: 544) }
    let gap = H3SpatialTiling(tileHeight: 256, tileWidth: 256, yStarts: [0, 160, 288], xStarts: sourceTiling.xStarts,
                              yOverlaps: [112, 112], xOverlaps: sourceTiling.xOverlaps)
    #expect(throws: AetherError.self) { try gap.validate(width: 960, height: 544) }
    let small = H3SpatialTiling(tileHeight: 128, tileWidth: 256, yStarts: [0], xStarts: [0], yOverlaps: [], xOverlaps: [])
    #expect(throws: AetherError.self) { try small.validate(width: 256, height: 128) }
}

@Test func h3SpatialStitcherOfConstantTilesIsConstantAndRequiresEveryTile() throws {
    var stitcher = try H3SpatialStitcher(tiling: sourceTiling, width: 960, height: 544)
    #expect(throws: AetherError.self) { try stitcher.finish() }
    let tile = [Float](repeating: 0.25, count: 3 * 22 * 256 * 256)
    for _ in 0..<15 { try stitcher.push(tile) }
    #expect(throws: AetherError.self) { try stitcher.push(tile) }
    let output = try stitcher.finish()
    #expect(output.count == 3 * 22 * 544 * 960)
    #expect(output.allSatisfy { abs($0 - 0.25) <= 1e-6 })
}

/// Opt-in: the prototype's original-source spatial oracle (no model inference). Set
/// `AETHER_H3_SPATIAL_ORACLE` to its `h3-video-spatial-oracle.json`; every referenced file is
/// hash-checked before use.
@Test(.enabled(if: ProcessInfo.processInfo.environment["AETHER_H3_SPATIAL_ORACLE"] != nil))
func h3SpatialTilesMatchOriginalStitchingAndOddLatentPatchOffsets() throws {
    struct Item: Decodable { let path: String; let sha256: String }
    struct Tile: Decodable { let pixels: Item; let rows: Item }
    struct Oracle: Decodable { let width: Int; let height: Int; let tiling: H3SpatialTiling; let packed_rows: Item; let tiles: [Tile]; let reference_sha256: String }
    let url = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["AETHER_H3_SPATIAL_ORACLE"]))
    let oracle = try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: url))
    func read(_ item: Item) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: item.path))
        #expect(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == item.sha256)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    let packed = try read(oracle.packed_rows)
    var stitcher = try H3SpatialStitcher(tiling: oracle.tiling, width: oracle.width, height: oracle.height)
    for (index, tile) in oracle.tiles.enumerated() {
        #expect(try oracle.tiling.extract(width: oracle.width, height: oracle.height, packed: packed, index: index) == read(tile.rows))
        try stitcher.push(read(tile.pixels))
    }
    #expect(digest(try stitcher.finish()) == oracle.reference_sha256)
    #expect(throws: AetherError.self) {
        try oracle.tiling.extract(width: oracle.width, height: oracle.height, packed: packed, index: oracle.tiling.tileCount)
    }
}

@Test func h3GeneratedTurboGridReproducesTheCapturedTurbo8GridBitForBit() throws {
    let captured = try H3Schedule(), generated = try H3Schedule.turbo(steps: 8, videoShift: 12, audioShift: 3)
    #expect(generated.count == 8 && captured.count == 8)
    for modality in [H3Schedule.Modality.video, .audio] {
        #expect(generated.sigmas(modality).map(\.bitPattern) == captured.sigmas(modality).map(\.bitPattern))
    }
    #expect(zip(generated.steps, captured.steps).allSatisfy {
        $0.videoTimestep.bitPattern == $1.videoTimestep.bitPattern && $0.audioTimestep.bitPattern == $1.audioTimestep.bitPattern
    })
}

@Test func h3FourStepTurboGridsMatchThePublishedAdapterSpecs() throws {
    func close(_ a: [Float], _ b: [Float]) -> Bool { a.count == b.count && zip(a, b).allSatisfy { abs($0 - $1) < 5e-5 } }
    // ModelTC Minimax-H3-Turbo: NFE 4 at shifts 12/3 (544p adapters).
    let s544 = try H3Schedule.turbo(steps: 4, videoShift: 12, audioShift: 3)
    #expect(close(s544.sigmas(.video), [1, 0.9730, 0.9231, 0.8000, 0]))
    #expect(close(s544.sigmas(.audio), [1, 0.9, 0.75, 0.5, 0]))
    // 768p adapters train at video shift 6.
    let s768 = try H3Schedule.turbo(steps: 4, videoShift: 6, audioShift: 3)
    #expect(close(s768.sigmas(.video), [1, 0.947368, 0.857143, 0.666667, 0]))
    // A four-step schedule advances exactly four times.
    #expect(throws: AetherError.self) { try s768.advance(.video, step: 4, sample: [1], velocity: [1]) }
    #expect(try s768.advance(.video, step: 3, sample: [1], velocity: [1]).count == 1)
    for (steps, video, audio) in [(0, 6, 3), (51, 6, 3), (4, 0.5, 3), (4, 6, Float.nan)] as [(Int, Float, Float)] {
        #expect(throws: AetherError.self) { try H3Schedule.turbo(steps: steps, videoShift: video, audioShift: audio) }
    }
}

@Test func h3TemporalStitcherMatchesThePinnedDiffusersDecodeControlFlow() throws {
    struct Item: Decodable { let path: String; let sha256: String }
    struct Oracle: Decodable { let width: Int; let height: Int; let frames: Int; let windows: [Item]; let reference: Item }
    let root = try #require(Bundle.module.url(forResource: "temporal-oracle", withExtension: nil, subdirectory: "Fixtures"))
    let oracle = try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: root.appending(path: "temporal-oracle.json")))
    func read(_ item: Item) throws -> [Float] {
        let data = try Data(contentsOf: root.appending(path: item.path))
        #expect(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == item.sha256)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    var stitcher = try H3VideoStitcher(width: oracle.width, height: oracle.height, frames: oracle.frames)
    var output: [Float] = []
    for window in oracle.windows { output += try stitcher.push(read(window)) }
    output += try stitcher.finish()
    #expect(digest(output) == oracle.reference.sha256)
}

@Test func h3SourceTilingReproducesTheOriginalSplitForProductionCanvases() throws {
    #expect(try H3SpatialTiling.source(width: 960, height: 544) == sourceTiling)
    for (width, height) in [(544, 960), (1344, 768), (768, 1344), (864, 480), (1024, 992)] {
        let tiling = try H3SpatialTiling.source(width: width, height: height)
        try tiling.validate(width: width, height: height)
        #expect(tiling.yOverlaps.allSatisfy { $0 >= 64 } && tiling.xOverlaps.allSatisfy { $0 >= 64 })
    }
    #expect(try H3SpatialTiling.source(width: 1344, height: 768).xStarts.count == 7)
    #expect(throws: AetherError.self) { try H3SpatialTiling.source(width: 256, height: 544) }
}

@Test func h3RotaryTablesMatchPinnedDiffusersForAFirstKeyframeLayout() throws {
    struct Oracle: Decodable { let width: Int; let height: Int; let requested_frames: Int; let text_rows: Int; let rows: Int
                               let positions: [Float]; let cos: [Float]; let sin: [Float] }
    let url = try #require(Bundle.module.url(forResource: "rotary-oracle", withExtension: "json", subdirectory: "Fixtures"))
    let oracle = try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: url))
    let geometry = try H3Geometry(width: oracle.width, height: oracle.height, requestedFrames: oracle.requested_frames, diagnostic: true)
    let layout = try H3Layout(geometry: geometry, textTags: Array(repeating: 1, count: oracle.text_rows), anchors: [.first])
    #expect(layout.rows == oracle.rows)
    #expect(layout.positions.values == oracle.positions)
    let (cos, sin) = try layout.rotary()
    let worst = zip(cos.values + sin.values, oracle.cos + oracle.sin).map { abs($0 - $1) }.max() ?? .infinity
    #expect(worst <= 2e-6, "worst absolute difference \(worst)")
}
