import AetherKit
import AetherH3
import Foundation
import Testing

/// Opt-in, CPU only: set `AETHER_H3_PROFILE` to an exported schema-2 `bundle.json`. Opens it, checks every recorded
/// step's timesteps against the layout for 5 s 960×544 clips in both orientations and both tasks, and hashes all tables.
@Test(.enabled(if: ProcessInfo.processInfo.environment["AETHER_H3_PROFILE"] != nil))
func h3ExportedProfileTablesMatchTheLayoutForTheFirstClipMenu() throws {
    let url = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["AETHER_H3_PROFILE"]))
    let profile = try H3Profile(url: url)
    #expect(Set(profile.recipes.keys) == ["turbo8", "turbo4"])
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
    if !profile.complete { #expect(!profile.missing.isEmpty) }
}
