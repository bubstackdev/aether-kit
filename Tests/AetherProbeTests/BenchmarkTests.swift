import AetherKit
import AetherKreaCoreAI
import Foundation
import Testing
@testable import AetherKrea
@testable import AetherProbe

private struct BenchmarkFixture: Sendable {
    let item = BenchmarkCase(id: "fake", prompt: "test", width: 16, height: 16, steps: 8, seed: 17,
                             reference: "ref.json", referenceSHA256: String(repeating: "a", count: 64))
    let conditioning: KreaConditioning
    let latents: [FloatTensor]
    let image: GeneratedImage
    let expected: BenchmarkExpected
    init() throws {
        conditioning = try KreaConditioning(tensor: Float16Tensor(shape: [1, 30720], values: Array(repeating: 0.5, count: 30720)))
        latents = try (0..<8).map { try FloatTensor(shape: [1, 64], values: Array(repeating: Float($0) + 0.25, count: 64)) }
        image = try GeneratedImage(pixelsCHW: FloatTensor(shape: [3, 16, 16], values: Array(repeating: 0.25, count: 768)))
        expected = BenchmarkExpected(conditioning: floatHash(conditioning.tensor.values.map(Float.init)),
                                     steps: latents.map { floatHash($0.values) }, pixels: floatHash(image.pixelsCHW.values), shape: [3, 16, 16])
    }
    var selection: KreaDenoiserSelection { KreaDenoiserSelection(assets: [], imageTokens: 1, textRows: 1, specialized: false) }
}

private actor BenchmarkFakePipeline {
    let fixture: BenchmarkFixture
    let mismatch: Bool
    var calls = 0
    var closes = 0
    init(_ fixture: BenchmarkFixture, mismatch: Bool = false) { self.fixture = fixture; self.mismatch = mismatch }
    func generate(_ progress: @Sendable (KreaGenerationEvent) throws -> Void) throws -> GeneratedImage {
        calls += 1
        try progress(.conditioningReady(fixture.conditioning))
        try progress(.denoisersReady(fixture.selection))
        for index in 0..<8 {
            try progress(.step(index: index, total: 8, latents: fixture.latents[mismatch && index == 3 ? 0 : index]))
        }
        try progress(.decoded)
        return fixture.image
    }
    func unload() { closes += 1 }
    var operation: BenchmarkOperation { BenchmarkOperation(generate: { progress in try await self.generate(progress) },
                                                           unload: { await self.unload() }) }
}

private func fakeMemory() -> BenchmarkMemory {
    BenchmarkMemory(physicalFootprintBytes: 100, residentBytes: 200, metalAllocatedBytes: 300,
                    processLifetimePeakFootprintBytes: 400, processLifetimePeakResidentBytes: 500)
}

@Test func benchmarkLoopSeparatesWarmupsAndValidatesCompleteGenerations() async throws {
    let fixture = try BenchmarkFixture()
    let fake = BenchmarkFakePipeline(fixture)
    var rows: [BenchmarkIteration] = []
    try await BenchmarkLoop.run(item: fixture.item, expected: fixture.expected, warmups: 1, measurements: 3,
        operation: fake.operation, readMemory: fakeMemory, onStart: { _, _ in }, onIteration: { rows.append($0) })
    #expect(rows.map(\.role) == ["warmup", "measured", "measured", "measured"])
    #expect(rows.map(\.iteration) == [0, 0, 1, 2])
    #expect(rows.allSatisfy { $0.passed && $0.stepLatentsSHA256 == fixture.expected.steps && $0.pixelsSHA256 == fixture.expected.pixels })
    #expect(rows.allSatisfy { $0.memorySamples >= 2 && $0.sampledPeakMemory?.metalAllocatedBytes == 300 })
    for row in rows {
        let accounted = row.stages.values.reduce(0, +) + row.observerSeconds
        #expect(abs(accounted - (row.totalSeconds ?? 0)) < 0.000001)
    }
    #expect(BenchmarkStatistics.measured(rows, caseID: "fake")["totalIncludingObservers"]?.count == 3)
    #expect(await fake.calls == 4)
    #expect(await fake.closes == 1)
}

@Test func benchmarkMismatchStopsBeforeMeasurementsAndClosesPipeline() async throws {
    let fixture = try BenchmarkFixture()
    let fake = BenchmarkFakePipeline(fixture, mismatch: true)
    var rows: [BenchmarkIteration] = []
    await #expect(throws: AetherError.self) {
        try await BenchmarkLoop.run(item: fixture.item, expected: fixture.expected, warmups: 1, measurements: 3,
            operation: fake.operation, readMemory: fakeMemory, onStart: { _, _ in }, onIteration: { rows.append($0) })
    }
    #expect(rows.count == 1 && rows[0].role == "warmup" && !rows[0].passed)
    #expect(rows[0].error?.contains("latent hash mismatch at step 3") == true)
    #expect(BenchmarkStatistics.measured(rows, caseID: "fake").isEmpty)
    #expect(await fake.calls == 1)
    #expect(await fake.closes == 1)
}

@Test func benchmarkMemoryAndPersistenceFailuresCannotProducePassingMeasurements() async throws {
    let fixture = try BenchmarkFixture()
    let fake = BenchmarkFakePipeline(fixture)
    var rows: [BenchmarkIteration] = []
    await #expect(throws: AetherError.self) {
        try await BenchmarkLoop.run(item: fixture.item, expected: fixture.expected, warmups: 1, measurements: 3,
            operation: fake.operation, readMemory: { throw AetherError.unsupported("memory unavailable") },
            onStart: { _, _ in }, onIteration: { rows.append($0) })
    }
    #expect(rows.count == 1 && !rows[0].passed)
    #expect(rows[0].error?.contains("memory sampling failed") == true)
    #expect(await fake.closes == 1)
    let other = BenchmarkFakePipeline(fixture)
    var writes = 0
    await #expect(throws: AetherError.self) {
        try await BenchmarkLoop.run(item: fixture.item, expected: fixture.expected, warmups: 1, measurements: 3,
            operation: other.operation, readMemory: fakeMemory, onStart: { _, _ in }, onIteration: { _ in
                writes += 1
                throw AetherError.unsupported("disk full")
            })
    }
    #expect(writes == 1)
    #expect(await other.closes == 1)
}

@Test func benchmarkRejectsIncompleteAndOutOfOrderEvents() throws {
    let fixture = try BenchmarkFixture()
    let memory = BenchmarkMemorySampler(read: fakeMemory)
    let journal = BenchmarkIterationJournal(caseID: "fake", role: "measured", iteration: 0,
        expected: fixture.expected, memory: memory, started: .now)
    #expect(throws: AetherError.self) { try journal.accept(.decoded) }
    #expect(throws: AetherError.self) { try journal.finish(fixture.image, ended: .now, total: 0) }
    #expect(journal.row.passed == false)
}

@Test func benchmarkStatisticsExcludeWarmupsFailuresAndOtherCases() throws {
    func row(role: String, value: Double, passed: Bool = true, id: String = "fake") -> BenchmarkIteration {
        var row = BenchmarkIteration(caseID: id, role: role, iteration: 0)
        row.passed = passed
        row.totalSeconds = value
        for index in 0..<8 { row.stages["step\(index)"] = value }
        return row
    }
    let summary = BenchmarkStatistics.measured([row(role: "warmup", value: 1000), row(role: "measured", value: 1),
        row(role: "measured", value: 3), row(role: "measured", value: 999, passed: false),
        row(role: "measured", value: 3000, id: "other")], caseID: "fake")
    #expect(summary["totalIncludingObservers"]?.count == 2)
    #expect(summary["totalIncludingObservers"]?.median == 2)
    #expect(summary["totalIncludingObservers"]?.mean == 2)
    #expect(summary["sampling"]?.mean == 16)
    #expect(BenchmarkStatistics.summarize([.nan]) == nil)
    #expect(BenchmarkStatistics.summarize([-1]) == nil)
}

@Test func benchmarkPlanAndReferenceRequirePinnedCompatibleWorkloads() throws {
    let fixture = try BenchmarkFixture()
    let hash = String(repeating: "a", count: 64)
    #expect(throws: AetherError.self) {
        try KreaBenchmarkPlan(schema: 1, bundleFingerprint: hash, warmupIterations: 0,
                              measuredIterations: 3, cases: [fixture.item]).validate()
    }
    #expect(throws: AetherError.self) {
        try KreaBenchmarkPlan(schema: 1, bundleFingerprint: hash, warmupIterations: 1,
                              measuredIterations: 3, cases: [fixture.item, fixture.item]).validate()
    }
    let data = try JSONSerialization.data(withJSONObject: ["case": "fake", "prompt": "test", "width": 16, "height": 16,
        "steps": 8, "arm": "device-encoded", "residency": "balanced", "noiseSHA256": hash,
        "runs": [["encodedTextSHA256": fixture.expected.conditioning, "stepLatentsSHA256": fixture.expected.steps,
                  "pixelsSHA256": fixture.expected.pixels, "pixelShape": fixture.expected.shape]]])
    #expect(try BenchmarkExpected.load(data, item: fixture.item, noiseSHA256: hash).pixels == fixture.expected.pixels)
    #expect(throws: AetherError.self) { try BenchmarkExpected.load(data, item: fixture.item, noiseSHA256: "changed") }
}

@Test func benchmarkPreparationFailurePersistsWithoutInferenceOrStatistics() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    await #expect(throws: (any Error).self) {
        try await KreaBenchmarkCommand.run(["benchmark-krea", "/missing/bundle", "/missing/plan", "/missing/source", root.path, "--prepare-only"])
    }
    let data = try Data(contentsOf: root.appendingPathComponent("report.json"))
    let report = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    #expect(report["stage"] as? String == "failed")
    #expect(report["benchmarkExecuted"] as? Bool == false)
    #expect((report["statistics"] as? [String: Any])?.isEmpty == true)
    await #expect(throws: AetherError.self) {
        try await KreaBenchmarkCommand.run(["benchmark-krea", "/missing/bundle", "/missing/plan", "/missing/source", root.path, "--prepare-only"])
    }
    #expect(try Data(contentsOf: root.appendingPathComponent("report.json")) == data)
}

@Test func benchmarkReadsProcessMemoryWithoutInitializingMetal() throws {
    let memory = try BenchmarkMemory.read(device: nil)
    #expect(memory.physicalFootprintBytes > 0)
    #expect(memory.residentBytes > 0)
    #expect(memory.processLifetimePeakResidentBytes >= memory.residentBytes)
    #expect(memory.processLifetimePeakFootprintBytes >= memory.physicalFootprintBytes)
    #expect(memory.metalAllocatedBytes == nil)
}

private actor BenchmarkFakeConditioner {
    let conditioning: KreaConditioning
    let failEncoding: Bool
    var encodes = 0
    var closes = 0
    init(_ conditioning: KreaConditioning, failEncoding: Bool = false) {
        self.conditioning = conditioning
        self.failEncoding = failEncoding
    }
    func encode() throws -> KreaConditioning {
        encodes += 1
        if failEncoding { throw AetherError.invalidTensor("fake encoder failed") }
        return conditioning
    }
    func unload() { closes += 1 }
    var operation: BenchmarkConditioner {
        BenchmarkConditioner(encode: { try await self.encode() }, unload: { await self.unload() })
    }
}

@Test func providedBenchmarkEncodesOnceAndClosesBeforeRepeatedGeneration() async throws {
    let fixture = try BenchmarkFixture()
    let encoder = BenchmarkFakeConditioner(fixture.conditioning)
    let setup = try await BenchmarkConditioningPreparation.prepare(caseID: "fake", expected: fixture.expected.conditioning) {
        await encoder.operation
    }
    #expect(setup.record.excludedFromIterationTimings)
    #expect(setup.record.conditioningSHA256 == fixture.expected.conditioning)
    #expect(setup.conditioning.tensor == fixture.conditioning.tensor)
    let pipeline = BenchmarkFakePipeline(fixture)
    let operation = BenchmarkOperation(generate: { progress in
        guard await encoder.closes == 1 else { throw AetherError.invalidRequest("encoder not closed before denoising") }
        return try await pipeline.generate(progress)
    }, unload: { await pipeline.unload() })
    var rows: [BenchmarkIteration] = []
    try await BenchmarkLoop.run(item: fixture.item, expected: fixture.expected, warmups: 3, measurements: 5,
        profile: .provided, operation: operation, readMemory: fakeMemory,
        onStart: { _, _ in }, onIteration: { rows.append($0) })
    #expect(await encoder.encodes == 1)
    #expect(await encoder.closes == 1)
    #expect(await pipeline.calls == 8)
    #expect(await pipeline.closes == 1)
    #expect(rows.allSatisfy { $0.profile == .provided && $0.passed && $0.pixelsSHA256 == fixture.expected.pixels })
    #expect(BenchmarkStatistics.measured(rows, caseID: "fake")["totalIncludingObservers"]?.count == 5)
}

@Test func providedConditioningSetupRejectsMismatchAndClosesOnEncoderFailure() async throws {
    let fixture = try BenchmarkFixture()
    let mismatch = BenchmarkFakeConditioner(fixture.conditioning)
    await #expect(throws: AetherError.self) {
        try await BenchmarkConditioningPreparation.prepare(caseID: "fake", expected: "wrong") {
            await mismatch.operation
        }
    }
    #expect(await mismatch.encodes == 1)
    #expect(await mismatch.closes == 1)
    let failure = BenchmarkFakeConditioner(fixture.conditioning, failEncoding: true)
    await #expect(throws: AetherError.self) {
        try await BenchmarkConditioningPreparation.prepare(caseID: "fake", expected: fixture.expected.conditioning) {
            await failure.operation
        }
    }
    #expect(await failure.closes == 1)
}

@Test func benchmarkProfilesPreserveOriginalPlansAndRejectUnknownProtocols() throws {
    let fixture = try BenchmarkFixture()
    let original = KreaBenchmarkPlan(schema: 1, bundleFingerprint: String(repeating: "a", count: 64),
                                     warmupIterations: 1, measuredIterations: 3, cases: [fixture.item])
    let data = try JSONEncoder().encode(original)
    var object = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    #expect(object["profile"] == nil)
    #expect(try JSONDecoder().decode(KreaBenchmarkPlan.self, from: data).effectiveProfile == .encoded)
    object["profile"] = "provided"
    #expect(try JSONDecoder().decode(KreaBenchmarkPlan.self, from: JSONSerialization.data(withJSONObject: object)).effectiveProfile == .provided)
    object["profile"] = "unknown"
    #expect(throws: DecodingError.self) {
        try JSONDecoder().decode(KreaBenchmarkPlan.self, from: JSONSerialization.data(withJSONObject: object))
    }
}

@Test func benchmarkStatisticsNeverCombineDifferentProfiles() {
    var encoded = BenchmarkIteration(caseID: "same", role: "measured", iteration: 0)
    encoded.profile = .encoded
    encoded.passed = true
    encoded.totalSeconds = 10
    var provided = encoded
    provided.profile = .provided
    provided.totalSeconds = 1
    #expect(BenchmarkStatistics.measured([encoded, provided], caseID: "same").isEmpty)
    #expect(BenchmarkStatistics.measured([encoded], caseID: "same")["totalIncludingObservers"]?.mean == 10)
    #expect(BenchmarkStatistics.measured([provided], caseID: "same")["totalIncludingObservers"]?.mean == 1)
}
