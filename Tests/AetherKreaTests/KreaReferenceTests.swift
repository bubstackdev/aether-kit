import AetherKit
import Testing
@testable import AetherKrea

// Captured from independent Swift lab oracles, retained in prototype/tests/numerics.rs.
@Test func scheduleMatchesExistingOracleBits() throws {
    let schedule = try KreaSchedule()
    #expect(schedule.timesteps.map(\.bitPattern) == [
        1065353216, 1064627160, 1063751508, 1062674713,
        1061318479, 1059557808, 1057180097, 1050619501,
    ])
    let deltas = zip(schedule.sigmas.dropFirst(), schedule.sigmas).map { ($0 - $1).bitPattern }
    #expect(deltas == [
        3174122112, 3176515648, 3179508184, 3181743696,
        3184979192, 3188793276, 3192834602, 3198103149,
    ])
    #expect(try KreaSchedule(steps: 1).sigmas == [1, 0])
    #expect(throws: AetherError.self) { try KreaSchedule(steps: 0) }
    #expect(throws: AetherError.self) { try KreaSchedule(steps: 51) }
}

@Test func seededNoiseMatchesExistingOracleBits() throws {
    let cases: [(UInt64, [UInt32])] = [
        (0, [3202863069, 1045741727, 1076470663, 3204126899, 3212645673, 1072668933, 1048658635, 3219995906]),
        (42, [1054103078, 1059526174, 3211023016, 1068094895, 1071473486, 3220247501, 1057729992, 3218346802]),
        (71344, [1035851364, 1074166398, 1060776933, 3184718423, 1070719263, 3206437512, 1025533962, 1052832527]),
    ]
    for (seed, expected) in cases {
        #expect(try KreaNoise.generate(seed: seed, count: 8).map(\.bitPattern) == expected)
        #expect(try KreaNoise.generate(seed: seed, count: 3).map(\.bitPattern) == Array(expected.prefix(3)))
    }
    #expect(try KreaNoise.generate(seed: 0, count: 0).isEmpty)
    #expect(throws: AetherError.self) { try KreaNoise.generate(seed: 0, count: -1) }
}

@Test func eulerPreservesRoundingAndRejectsInvalidInputs() throws {
    let a = Float(bitPattern: 0x3f800001)
    let b = Float(bitPattern: 0x3f7ffffe)
    #expect(roundedProduct(a, b) - 1 == 0)
    #expect(Float(-1).addingProduct(a, b) != 0)
    let schedule = try KreaSchedule(steps: 1)
    #expect(try schedule.advance(step: 0, latents: [2], velocity: [3]) == [-1])
    #expect(throws: AetherError.self) { try schedule.advance(step: 1, latents: [2], velocity: [3]) }
    #expect(throws: AetherError.self) { try schedule.advance(step: 0, latents: [2], velocity: []) }
    #expect(throws: AetherError.self) { try schedule.advance(step: 0, latents: [2], velocity: [.nan]) }
    #expect(throws: AetherError.self) {
        try schedule.advance(step: 0, latents: [.greatestFiniteMagnitude], velocity: [-.greatestFiniteMagnitude])
    }
}

@Test func geometryIsBoundedBeforeAllocation() throws {
    #expect(try KreaConfiguration(width: 768, height: 1344).imageTokenCount == 4032)
    #expect(try KreaConfiguration(width: 1024, height: 1024).imageTokenCount == 4096)
    for (width, height) in [(0, 16), (17, 16), (2048, 2048), (Int.max - 15, Int.max - 15)] {
        #expect(throws: AetherError.self) { try KreaConfiguration(width: width, height: height) }
    }
}
