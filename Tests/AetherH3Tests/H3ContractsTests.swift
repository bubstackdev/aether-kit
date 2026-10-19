import AetherKit
import AetherH3
import CryptoKit
import Foundation
import Testing

private func fixture(_ name: String) throws -> Data {
    try Data(contentsOf: #require(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures")))
}
private func hash<T>(_ values: [T]) -> String {
    values.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
}

@Test func h3GeometrySnapsFramesAndRejectsInvalidOrOverflowingRequests() throws {
    let geometry = try H3Geometry()
    #expect(geometry.frames == 124)
    #expect(geometry.latentFrames == 37)
    #expect(geometry.videoLatentShape == [24, 37, 34, 60])
    #expect(geometry.targetVideoRows == 18_870)
    #expect(geometry.audioLatentsPerChannel == 207)
    #expect(geometry.durationSeconds == Double(124) / 24)
    #expect(try H3Geometry(requestedFrames: 124).frames == 124)
    #expect(try H3Geometry(requestedFrames: 125).frames == 141)
    #expect(try H3Geometry(requestedFrames: 345).frames == 345)
    #expect(try H3Geometry(width: 32, height: 32, requestedFrames: 1, diagnostic: true).frames == 5)
    for (width, height, frames) in [(0, 544, 120), (960, -32, 120), (961, 544, 120),
                                   (Int.max - 31, 544, 120), (2048, 2048, 120),
                                   (960, 544, 0), (960, 544, 5), (960, 544, 346), (960, 544, Int.max)] {
        #expect(throws: AetherError.self) { try H3Geometry(width: width, height: height, requestedFrames: frames) }
    }
}

@Test func h3TurboSchedulerMatchesAllEightIndependentVideoAndAudioUpdates() throws {
    struct Oracle: Decodable {
        let video: [Float]; let audio: [Float]; let steps: [Step]
        struct Step: Decodable {
            let video_velocity: [Float]; let audio_velocity: [Float]
            let video_sha256: String; let audio_sha256: String
        }
    }
    let oracle = try JSONDecoder().decode(Oracle.self, from: fixture("turbo8-numerics"))
    let schedule = try H3Schedule()
    #expect(schedule.steps.count == 8)
    #expect(schedule.steps[0].videoTimestep.bitPattern == Float(0).bitPattern)
    var video = oracle.video, audio = oracle.audio
    for (step, expected) in oracle.steps.enumerated() {
        video = try schedule.advance(.video, step: step, sample: video, velocity: expected.video_velocity)
        audio = try schedule.advance(.audio, step: step, sample: audio, velocity: expected.audio_velocity)
        #expect(hash(video) == expected.video_sha256, "video update \(step)")
        #expect(hash(audio) == expected.audio_sha256, "audio update \(step)")
    }
}

@Test func h3SchedulerRejectsBadStepsShapesAndNonfiniteArithmetic() throws {
    let schedule = try H3Schedule()
    for step in [-1, 8, Int.max] {
        #expect(throws: AetherError.self) { try schedule.advance(.video, step: step, sample: [0], velocity: [0]) }
    }
    for (sample, velocity): ([Float], [Float]) in [([], []), ([0], []), ([.nan], [0]), ([0], [.infinity]), ([.greatestFiniteMagnitude], [.greatestFiniteMagnitude])] {
        #expect(throws: AetherError.self) { try schedule.advance(.audio, step: 0, sample: sample, velocity: velocity) }
    }
}

@Test func h3LayoutsMatchEveryCapturedPositionTagAndIndex() throws {
    struct Oracle: Decodable {
        let cases: [Case]
        struct Case: Decodable { let name: String; let textTags: [Int32]; let rows: Int; let hashes: [String: String] }
    }
    let oracle = try JSONDecoder().decode(Oracle.self, from: fixture("layout-oracle"))
    for expected in oracle.cases {
        let anchors: [H3Anchor]
        switch expected.name {
        case "first": anchors = [.first]
        case "last": anchors = [.last]
        case "both": anchors = [.first, .last]
        default: anchors = []
        }
        let layout = try H3Layout(geometry: H3Geometry(), textTags: expected.textTags, anchors: anchors)
        #expect(layout.rows == expected.rows)
        #expect(hash(layout.positions.values) == expected.hashes["position_ids"], "\(expected.name) positions")
        #expect(hash(layout.tokenTags) == expected.hashes["token_tags"])
        #expect(hash(layout.videoIndices) == expected.hashes["video_indices"])
        #expect(hash(layout.audioIndices) == expected.hashes["audio_indices"])
        #expect(hash(layout.textIndices) == expected.hashes["text_indices"])
        #expect(hash(expected.textTags) == expected.hashes["text_token_tags"])
    }
}

@Test func h3TimestepPlansPreserveActualCardinalityAndModalityRows() throws {
    let geometry = try H3Geometry(), schedule = try H3Schedule()
    for anchors: [H3Anchor] in [[], [.first], [.last], [.first, .last]] {
        let layout = try H3Layout(geometry: geometry, textTags: [0, 1, 1], anchors: anchors)
        for step in schedule.steps {
            let plan = layout.timesteps(for: step)
            #expect(plan.values.count == (anchors.isEmpty ? (step.index == 0 ? 1 : 2) : (step.index == 0 ? 2 : 3)))
            #expect(plan.indices.count == layout.rows)
            for i in layout.audioIndices { #expect(plan.values[Int(plan.indices[Int(i)])] == step.audioTimestep) }
            for i in layout.textIndices { #expect(plan.values[Int(plan.indices[Int(i)])] == step.videoTimestep) }
            for i in layout.videoIndices.prefix(layout.conditionVideoRows) {
                #expect(plan.values[Int(plan.indices[Int(i)])] == max(step.videoTimestep, 0.999))
            }
            #expect(zip(plan.adalnIndices, zip(plan.indices, layout.tokenTags)).allSatisfy { $0 == $1.0 * 3 + $1.1 })
        }
    }
}

@Test func h3LayoutRejectsInvalidTagsAndAnchorOrder() throws {
    let geometry = try H3Geometry()
    for tags: [Int32] in [[], [2], [-1], Array(repeating: 1, count: 4097)] {
        #expect(throws: AetherError.self) { try H3Layout(geometry: geometry, textTags: tags) }
    }
    for anchors: [H3Anchor] in [[.last, .first], [.first, .first], [.last, .last], [.first, .last, .last]] {
        #expect(throws: AetherError.self) { try H3Layout(geometry: geometry, textTags: [1], anchors: anchors) }
    }
}

private func header() -> [String: Any] {
    ["schema": 1, "complete": true, "snapshot": H3ExportHeader.snapshot,
     "diffusers_commit": H3ExportHeader.diffusersCommit, "recipe": "turbo8",
     "transformer_weight_layout": H3ExportHeader.transformerWeightLayout,
     "adapter": ["repository": "lightx2v/Minimax-h3-Turbo", "revision": "3ec17a324ced54151364f24f8b5fb6bf7e26414f",
                 "sha256": "e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f",
                 "execution": "dynamic_lora", "rank": 128, "alpha": 8]]
}

@Test func h3HeaderRequiresCorrectedQKVAndExactTurboAdapterIdentity() throws {
    let good = try JSONSerialization.data(withJSONObject: header())
    #expect(try H3ExportHeader(data: good).recipe == .turbo8)
    #expect(try H3ExportHeader(data: good).manifestSHA256 == SHA256.hash(data: good).map { String(format: "%02x", $0) }.joined())
    for (key, value) in [("schema", 2 as Any), ("complete", false), ("snapshot", "changed"),
                         ("diffusers_commit", "changed"), ("recipe", "base50"), ("transformer_weight_layout", "old")] {
        var wrong = header(); wrong[key] = value
        #expect(throws: (any Error).self) { try H3ExportHeader(data: JSONSerialization.data(withJSONObject: wrong)) }
    }
    var missing = header(); missing.removeValue(forKey: "transformer_weight_layout")
    #expect(throws: (any Error).self) { try H3ExportHeader(data: JSONSerialization.data(withJSONObject: missing)) }
    for (key, value) in [("repository", "other" as Any), ("revision", "other"), ("sha256", "other"),
                         ("execution", "merged"), ("rank", 64), ("alpha", 128)] {
        var wrong = header(), adapter = wrong["adapter"] as! [String: Any]; adapter[key] = value; wrong["adapter"] = adapter
        #expect(throws: (any Error).self) { try H3ExportHeader(data: JSONSerialization.data(withJSONObject: wrong)) }
    }
    var noAdapter = header(); noAdapter.removeValue(forKey: "adapter")
    #expect(throws: (any Error).self) { try H3ExportHeader(data: JSONSerialization.data(withJSONObject: noAdapter)) }
}

@Test func h3VideoPatchOrderMatchesIndependentUpstreamOracle() throws {
    struct Oracle: Decodable { let sourceShape: [Int]; let patchify_sha256: String }
    let oracle = try JSONDecoder().decode(Oracle.self, from: fixture("video-packing-oracle"))
    let geometry = try H3Geometry(width: 64, height: 64, requestedFrames: 22, diagnostic: true)
    let values = (0..<(24 * 7 * 4 * 4)).map(Float.init)
    let packed = try geometry.packVideo(FloatTensor(shape: oracle.sourceShape, values: values))
    #expect(packed.shape == [28, 96])
    #expect(hash(packed.values) == oracle.patchify_sha256)
    #expect(throws: AetherError.self) {
        try geometry.packVideo(FloatTensor(shape: [values.count], values: values))
    }
}

@Test func h3HeaderAcceptsRetainedCorrectedPrototypeExport() throws {
    let value = try H3ExportHeader(data: fixture("corrected-export-header"))
    #expect(value.recipe == .turbo8)
}
