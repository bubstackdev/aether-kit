import AetherH3
import AetherKit
import Testing
@testable import AetherProbe

/// Review L1: decode-only takes a render's final_video as the session decodes it, without FL2VA's condition rows.
@Test func decodeOnlyDropsTheKeyframeConditionRowsOfAnFL2VARender() throws {
    let geometry = try H3Geometry(width: 960, height: 544, requestedFrames: 124, diagnostic: true)
    let condition = geometry.rowsPerLatentFrame, target = geometry.targetVideoRows
    #expect(condition == 510 && target == 18_870)
    let generated = (0..<(target * 96)).map { Float($0 % 977) }
    let anchored = [Float](repeating: -1, count: condition * 96) + generated
    #expect(try H3ProfileCommand.generatedVideoRows(anchored, geometry: geometry, anchored: true) == generated)
    #expect(try H3ProfileCommand.generatedVideoRows(generated, geometry: geometry, anchored: false) == generated)
    let rows = try H3ProfileCommand.generatedVideoRows(anchored, geometry: geometry, anchored: true)
    #expect(try geometry.videoWindowRows(rows, index: geometry.videoWindows - 1).count == 7 * condition * 96)
    // A render that doesn't match its dump is refused before any tile runs.
    #expect(throws: AetherError.self) { try H3ProfileCommand.generatedVideoRows(anchored, geometry: geometry, anchored: false) }
    #expect(throws: AetherError.self) { try H3ProfileCommand.generatedVideoRows(generated, geometry: geometry, anchored: true) }
}
