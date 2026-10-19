import AetherKit
import AetherH3
import Foundation
import Testing

/// Opt-in, CPU only: set `AETHER_H3_PROFILE` to an exported schema-3 `bundle.json`. Opens it, checks every recorded
/// step's timesteps against the layout for 5 s 960×544 clips in both orientations and both tasks, and hashes all tables.
@Test(.enabled(if: ProcessInfo.processInfo.environment["AETHER_H3_PROFILE"] != nil))
func h3ExportedProfileTablesMatchTheLayoutForTheFirstClipMenu() throws {
    let url = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["AETHER_H3_PROFILE"]))
    let profile = try H3Profile(url: url)
    #expect(Set(profile.recipes.keys) == ["turbo8", "turbo4"])
    #expect(profile.modelIdentity == "minimax-h3.fl2va-turbo/1:" + profile.manifestSHA256)
    #expect(profile.modelIdentity.hasPrefix(H3Profile.modelIdentityPrefix) && profile.manifestSHA256.count == 64)
    for (width, height) in [(960, 544), (544, 960)] {
        let geometry = try H3Geometry(width: width, height: height, requestedFrames: 120)
        for anchors in [[], [H3Anchor.first]] {
            let layout = try H3Layout(geometry: geometry, textTags: Array(repeating: 1, count: 64), anchors: anchors)
            for recipe in ["turbo8", "turbo4"] {
                let steps = try profile.steps(recipe: recipe, layout: layout)
                #expect(steps.count == (recipe == "turbo8" ? 8 : 4))
                #expect(steps.allSatisfy { $0.tables.count == 50 })
            }
        }
    }
    try profile.verifyTables()
    try profile.verifyModels()
    if !profile.complete { #expect(!profile.missing.isEmpty) }
    if profile.complete {
        let conditioner = try #require(profile.conditioner), prepare = try #require(profile.prepare), audio = try #require(profile.audioDecoder)
        #expect(profile.parameters.count == 50 && conditioner.parameters.count == 50 && prepare.textParameters.count == 47)
        #expect(audio.sampleRate == 32000 && audio.hop == 800 && audio.channels == 2)
        #expect(profile.videoDecoder.precision == "bfloat16" && profile.videoDecoder.float32 != nil)
        // The 5 s clip menu needs Na = 207 (124 frames); the dynamic decoder is CPU-only.
        #expect(try audio.model(latents: 207).url.lastPathComponent == "audio-decoder-na207.aimodel")
        #expect(throws: AetherError.self) { try audio.model(latents: 208) }
        #expect(profile.header?.recipe == .turbo8)   // the header only exists for the pinned Turbo adapter identity
        // In-place bindings read back with their recorded hashes: the first and last block, layer and refiner weight.
        for file in [profile.parameters[0][1], profile.parameters[49][21], conditioner.parameters[0][0], conditioner.parameters[49][10],
                     prepare.textParameters[0], prepare.textParameters[46]] { _ = try file.read() }
        #expect(try conditioner.embedding.rows([0, 151935]).spec.shape == [1, 2, 5120])
        if ProcessInfo.processInfo.environment["AETHER_H3_PROFILE_PARAMETERS"] != nil { try profile.verifyParameters() }
    }
}
