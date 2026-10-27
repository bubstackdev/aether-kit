import AetherKit
import CryptoKit
import Foundation
import Testing

/// SHA-256 of the first 98,304 values' float32 bits, recorded from the standalone Kokoro and H3
/// generators before they were folded into `SeededGaussianNoise`. All three streams were bit-identical
/// for these seeds at counts 0, 1, 2, 3, 7, 128, 5,400, 16,201 and 98,304.
private let recordedStreams: [UInt64: String] = [
    0: "d3c374d012de4a5bd4ff3a84c72f32836e9b3a4486d611fc3ea24fe24eafb622",
    1: "f708c4cda415178d4f593f9359d4223c150343a3bd89f9f3369a91a06e10d0f9",
    3: "8fcfdf4713b1e4a5f35681e547331b2b42b079b2f4de5e5929cd087d756a6004",
    5: "e5b1e14c75de41e28d18c03b1c78750b7c94a984b6870592004c65c98a67379e",
    100: "845e0f4548a5d846e6fe3d36249d4475dc6aae854538270ee94b7ad7f7a2e0d9",
    1043: "4a79d10bb38b2936f2fdb3efc4cf5f979d8f1931c4bfb209f032138e6ef19219",
    0x5eed: "f0ce5715969c2a5048a598dc04e6734c5dfe9a47dc10f6e3328b4ac93145bfe4",
    0xdeadbeef: "08ca528ded9a348fa76ba9dc8a296f2eae759c5898de82c6168221d0173c738a",
    .max: "8374d6935fef05f0d0ccdc4505c379acbb29b91f61ca614e8b872852c11ae5b0",
]

@Test func seededGaussianNoiseReproducesTheRecordedStreamsAndIsPrefixStable() throws {
    for (seed, expected) in recordedStreams {
        let bits = try SeededGaussianNoise.generate(seed: seed, count: 98_304).map(\.bitPattern)
        let digest = bits.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
        #expect(digest == expected, "seed \(seed)")
        for count in [0, 1, 2, 3, 7, 128, 5_400, 16_201] {
            #expect(try SeededGaussianNoise.generate(seed: seed, count: count).map(\.bitPattern) == Array(bits.prefix(count)),
                    "seed \(seed) count \(count)")
        }
    }
    #expect(throws: AetherError.self) { try SeededGaussianNoise.generate(seed: 0, count: -1) }
}
