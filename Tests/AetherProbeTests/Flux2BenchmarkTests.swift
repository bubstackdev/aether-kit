import AetherFlux2
import AetherFlux2CoreAI
import AetherKit
import CryptoKit
import Foundation
import Testing
@testable import AetherProbe

private let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

@Test func checkedInFlux2BenchmarkPlansValidate() throws {
    for name in ["flux2-baseline.json", "flux2-provided.json"] {
        let plan = try JSONDecoder().decode(Flux2BenchmarkPlan.self, from: Data(contentsOf: repository.appendingPathComponent("Benchmarks/\(name)")))
        try plan.validate()
        #expect(plan.cases.count == 4 && plan.cases.contains { $0.fixture == "reference" }, "\(name)")
    }
}

@Test func flux2BenchmarkPlanRefusesIncompleteOrUnpinnedPlans() throws {
    let hash = String(repeating: "a", count: 64)
    func plan(schema: Int = 1, fingerprint: String = hash, reference: String? = "ref", warmups: Int = 2, measured: Int = 5,
              cases: [Flux2BenchmarkPlan.Case] = [.init(id: "a", fixture: "t2i", fixtureCase: "fox-512")]) -> Flux2BenchmarkPlan {
        Flux2BenchmarkPlan(schema: schema, bundleFingerprint: fingerprint, textToImageFixture: "t2i", referenceFixture: reference,
                           warmupIterations: warmups, measuredIterations: measured, profile: .encoded, residency: .balanced,
                           minimumProvidedSSIM: 0.94, cases: cases)
    }
    try plan().validate()
    #expect(throws: AetherError.self) { try plan(schema: 2).validate() }
    #expect(throws: AetherError.self) { try plan(fingerprint: "short").validate() }
    #expect(throws: AetherError.self) { try plan(warmups: 0).validate() }
    #expect(throws: AetherError.self) { try plan(measured: 2).validate() }
    #expect(throws: AetherError.self) { try plan(cases: []).validate() }
    #expect(throws: AetherError.self) { try plan(cases: [.init(id: "a", fixture: "t2i", fixtureCase: "x"), .init(id: "a", fixture: "t2i", fixtureCase: "y")]).validate() }
    #expect(throws: AetherError.self) { try plan(cases: [.init(id: "a", fixture: "other", fixtureCase: "x")]).validate() }
    // A reference-image case needs a pinned reference fixture.
    #expect(throws: AetherError.self) { try plan(reference: nil, cases: [.init(id: "r", fixture: "reference", fixtureCase: "x")]).validate() }
}

/// A Forge-format fixture with one 32×32 case: every file listed in `fixture.json` with its SHA-256.
private func writeFixture(id: String = "fixture1", references: Bool = false) throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("flux2-benchmark-fixture-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root.appendingPathComponent("cases/c"), withIntermediateDirectories: true)
    var hashes: [String: String] = [:]
    func put(_ name: String, _ data: Data) throws {
        try data.write(to: root.appendingPathComponent(name))
        hashes[name] = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    func floats(_ values: [Float]) -> Data { values.withUnsafeBytes { Data($0) } }
    try put("cases/c/ids.i32", [Int32](arrayLiteral: 1, 2, 3).withUnsafeBytes { Data($0) })
    try put("cases/c/text.f32", floats(Array(repeating: 0.5, count: 512 * 7680)))
    try put("cases/c/noise.f32", floats(Array(repeating: 0.25, count: 4 * 128)))
    try put("cases/c/image.f32", floats(Array(repeating: 0.5, count: 32 * 32 * 3)))
    var line: [String: Any] = ["id": "c", "prompt": "a fox", "width": 32, "height": 32, "seed": 7, "steps": 4,
                               "files": ["ids": "cases/c/ids.i32", "text": "cases/c/text.f32", "noise": "cases/c/noise.f32",
                                         "image": "cases/c/image.f32"]]
    if references {
        try put("cases/c/reference0.pixels.f32", floats(Array(repeating: 0, count: 3 * 32 * 16)))
        line["references"] = [["width": 16, "height": 32]]
        var files = line["files"] as! [String: Any]
        files["referencePixels"] = ["cases/c/reference0.pixels.f32"]
        line["files"] = files
    }
    try put("cases.jsonl", JSONSerialization.data(withJSONObject: line) + Data("\n".utf8))
    try JSONSerialization.data(withJSONObject: ["id": id, "sha256": hashes]).write(to: root.appendingPathComponent("fixture.json"))
    return root
}

@Test func flux2BenchmarkWorkloadsComeFromPinnedFixtureFiles() throws {
    let root = try writeFixture(references: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let workload = try Flux2BenchmarkWorkload.load(fixture: root, expectedID: "fixture1", caseID: "c", benchmarkID: "bench")
    #expect(workload.id == "bench" && workload.prompt == "a fox" && workload.imageTokens == 4 && workload.seed == 7)
    #expect(workload.tokenIDs == [1, 2, 3] && workload.noise.shape == [4, 128] && workload.text.shape == [512, 7680])
    #expect(workload.references.count == 1 && workload.references[0].width == 16 && workload.image.count == 32 * 32 * 3)
    // Another fixture id, a missing case, and a file that no longer matches its pinned hash are refused.
    #expect(throws: AetherError.self) { try Flux2BenchmarkWorkload.load(fixture: root, expectedID: "other", caseID: "c", benchmarkID: "b") }
    #expect(throws: AetherError.self) { try Flux2BenchmarkWorkload.load(fixture: root, expectedID: "fixture1", caseID: "missing", benchmarkID: "b") }
    try Data([0]).write(to: root.appendingPathComponent("cases/c/noise.f32"))
    #expect(throws: AetherError.self) { try Flux2BenchmarkWorkload.load(fixture: root, expectedID: "fixture1", caseID: "c", benchmarkID: "b") }
}

@Test func flux2BenchmarkJournalTimesStagesInOrderAndRequiresIdenticalPixels() throws {
    let memory = BenchmarkMemorySampler { try BenchmarkMemory.read(device: nil) }
    let conditioning = try Flux2Conditioning(tensor: Float16Tensor(shape: [512, 7680], values: Array(repeating: 0.5, count: 512 * 7680)))
    let latents = try FloatTensor(shape: [4, 128], values: Array(repeating: 0.1, count: 512))
    let image = try GeneratedImage(pixelsCHW: FloatTensor(shape: [3, 32, 32], values: Array(repeating: 0.25, count: 3 * 32 * 32)))
    func journal() -> Flux2BenchmarkJournal {
        Flux2BenchmarkJournal(caseID: "c", role: "measured", iteration: 0, profile: .encoded, steps: 2, imageTokens: 4,
                              referenceTokens: 0, memory: memory, started: .now)
    }
    let ordered = journal()
    try ordered.accept(.conditioningReady(conditioning))
    try ordered.accept(.denoisersReady(imageTokens: 4))
    try ordered.accept(.step(index: 0, total: 2, latents: latents))
    try ordered.accept(.step(index: 1, total: 2, latents: latents))
    try ordered.accept(.decoded)
    let pixels = try ordered.finish(image, ended: .now, total: 0.1, expectedPixels: nil).pixels
    #expect(ordered.row.passed && Set(ordered.row.stages.keys) == ["conditioning", "denoiserSetup", "step0", "step1", "decode", "completion"])
    #expect(ordered.row.stepLatentsSHA256.count == 2 && memory.snapshot.samples > 0)
    // A step before the denoisers, a wrong token count, and pixels that differ from the first iteration are refused.
    let early = journal()
    try early.accept(.conditioningReady(conditioning))
    #expect(throws: AetherError.self) { try early.accept(.step(index: 0, total: 2, latents: latents)) }
    let tokens = journal()
    try tokens.accept(.conditioningReady(conditioning))
    #expect(throws: AetherError.self) { try tokens.accept(.denoisersReady(imageTokens: 5)) }
    let repeated = journal()
    for event: Flux2GenerationEvent in [.conditioningReady(conditioning), .denoisersReady(imageTokens: 4),
                                        .step(index: 0, total: 2, latents: latents), .step(index: 1, total: 2, latents: latents), .decoded] {
        try repeated.accept(event)
    }
    #expect(throws: AetherError.self) { try repeated.finish(image, ended: .now, total: 0.1, expectedPixels: String(repeating: "0", count: 64)) }
    #expect(throws: AetherError.self) {
        try repeated.finish(image, ended: .now, total: 0.1, expectedPixels: pixels, expectedConditioning: String(repeating: "0", count: 64))
    }
    let conditioningHash = repeated.row.conditioningSHA256
    #expect(try repeated.finish(image, ended: .now, total: 0.1, expectedPixels: pixels, expectedConditioning: conditioningHash).pixels == pixels)
}

@Test func flux2BenchmarkStatisticsAndSSIM() throws {
    var row = BenchmarkIteration(caseID: "c", role: "measured", iteration: 0)
    row.passed = true; row.totalSeconds = 2; row.stages = ["step0": 0.5, "step1": 0.5, "decode": 0.25]
    var warm = BenchmarkIteration(caseID: "c", role: "warmup", iteration: 0)
    warm.passed = true; warm.totalSeconds = 9; warm.stages = row.stages
    let statistics = BenchmarkStatistics.flux2([row, row, warm], caseID: "c", steps: 2)
    #expect(statistics["sampling"]?.median == 1 && statistics["totalIncludingObservers"]?.count == 2)
    let image = (0..<(16 * 16 * 3)).map { Float($0 % 7) / 7 }
    #expect(abs(flux2SSIM(image, image, width: 16, height: 16) - 1) < 1e-9)
    #expect(flux2SSIM(image, image.map { 1 - $0 }, width: 16, height: 16) < 0.5)
    #expect(flux2SSIM(image, image, width: 4, height: 4) == -1)
}

@Test func flux2BenchmarkTakesTheGPULeaseOnlyToRun() {
    let base = ["benchmark-flux2", "B", "T", "-", "P", "S", "O"]
    #expect(ProbeGPULease.usesGPU(base + ["--run"]))
    #expect(!ProbeGPULease.usesGPU(base + ["--prepare-only"]))
    #expect(!ProbeGPULease.usesGPU(Array(base.dropLast()) + ["--run"]))
}

@Test func flux2BenchmarkCommandRefusesMalformedArguments() async {
    await #expect(throws: AetherError.self) { _ = try await Flux2BenchmarkCommand.run(["benchmark-flux2", "B", "--run"]) }
    #expect(await (try? Flux2BenchmarkCommand.run(["benchmark-krea"])) == false)
}
