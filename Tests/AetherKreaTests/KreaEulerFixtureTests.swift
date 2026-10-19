import AetherKrea
import CryptoKit
import Foundation
import Testing

@Test(.enabled(if: ProcessInfo.processInfo.environment["AETHER_KREA_FIXTURES"] != nil))
func eulerReplaysAllRetainedTowerSteps() throws {
    struct Fixture: Decodable {
        let id: String
        let files: Files
        struct Files: Decodable { let noise: String; let final: String; let stepLatents: [String]; let stepVelocities: [String] }
    }
    let root = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["AETHER_KREA_FIXTURES"]))
    let lines = try String(contentsOf: root.appendingPathComponent("cases.jsonl"), encoding: .utf8)
    let records = try lines.split(separator: "\n").map { try JSONDecoder().decode(Fixture.self, from: Data($0.utf8)) }
    let fixture = try #require(records.first { $0.id == "tower-768x1344" })
    #expect(fixture.files.stepLatents.count == 8)
    #expect(fixture.files.stepVelocities.count == 8)
    func read(_ relative: String) throws -> [Float] {
        let data = try Data(contentsOf: root.appendingPathComponent(relative))
        guard data.count == 4032 * 64 * 4 else { throw FixtureError.invalidByteCount }
        return data.withUnsafeBytes { bytes in
            stride(from: 0, to: data.count, by: 4).map { Float(bitPattern: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self).littleEndian) }
        }
    }
    var latents = try read(fixture.files.noise)
    let schedule = try KreaSchedule()
    for step in 0..<8 {
        // The Forge snapshots are pre-update inputs, unlike the lab's post-step hash journal.
        let input = try Data(contentsOf: root.appendingPathComponent(fixture.files.stepLatents[step]))
        #expect(latents.withUnsafeBytes { Array(SHA256.hash(data: Data($0))) } == Array(SHA256.hash(data: input)))
        latents = try schedule.advance(step: step, latents: latents, velocity: read(fixture.files.stepVelocities[step]))
        let next = step < 7 ? fixture.files.stepLatents[step + 1] : fixture.files.final
        let expected = try Data(contentsOf: root.appendingPathComponent(next))
        let actualHash = latents.withUnsafeBytes { SHA256.hash(data: Data($0)) }
        #expect(Array(actualHash) == Array(SHA256.hash(data: expected)), "Euler fixture step \(step)")
    }
}

private enum FixtureError: Error { case invalidByteCount }
