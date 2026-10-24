import AetherKit
import AetherKrea
import CryptoKit
import Foundation
import Testing
@testable import AetherProbe

private func batchArguments(output: URL, seeds: String = "17,18,17") -> [String] {
    ["render-krea-batch", "/unused-bundle", output.path, "a fox", "16", "16", seeds]
}

private func batchTemporaryRoot() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

private actor BatchFake: BatchRenderer {
    enum Fault { case none, prepare, secondImage, wrongSize, save }
    let fault: Fault
    let output: URL
    var prepares = 0
    var seeds: [UInt64] = []
    var steps: [Int] = []
    var closes = 0
    init(_ fault: Fault = .none, output: URL) { self.fault = fault; self.output = output }
    func prepare() throws -> BatchPreparation {
        prepares += 1
        if fault == .prepare { throw AetherError.invalidModel("injected setup failure") }
        return BatchPreparation(bundleFingerprint: String(repeating: "a", count: 64),
            conditioningSHA256: String(repeating: "b", count: 64), conditioningByteCount: 61_440,
            conditioningRows: 1, wasTruncated: false)
    }
    func generate(_ configuration: KreaConfiguration) throws -> GeneratedImage {
        seeds.append(configuration.seed)
        steps.append(configuration.steps)
        if fault == .secondImage && seeds.count == 2 { throw AetherError.invalidTensor("injected generation failure") }
        if fault == .save {
            try FileManager.default.createDirectory(at: output.appendingPathComponent("001-seed-17.png"), withIntermediateDirectories: false)
        }
        let width = fault == .wrongSize ? 32 : configuration.width
        return try GeneratedImage(pixelsCHW: FloatTensor(shape: [3, configuration.height, width],
            values: Array(repeating: configuration.seed == 17 ? 0.25 : -0.25, count: 3 * configuration.height * width)))
    }
    func unload() { closes += 1 }
}

private func batchReport(_ output: URL) throws -> [String: Any] {
    try #require(JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("report.json"))) as? [String: Any])
}

@Test func batchArgumentsValidateBeforeAnyModelOrOutputWork() throws {
    let output = URL(fileURLWithPath: "/unused-output")
    let valid = try KreaBatchOptions(batchArguments(output: output, seeds: "0,18446744073709551615,0"))
    #expect(valid.seeds == [0, UInt64.max, 0])
    for invalid in ["", "-1", "+1", "1,", ",1", "1,,2", "1, 2", "1.5", "18446744073709551616", Array(repeating: "1", count: 65).joined(separator: ",")] {
        #expect(throws: AetherError.self) { try KreaBatchOptions(batchArguments(output: output, seeds: invalid)) }
    }
    for (index, value) in [(0, "unknown"), (3, " \n"), (4, "0"), (4, "17"), (5, "-16"), (4, "999999")] {
        var args = batchArguments(output: output); args[index] = value
        #expect(throws: AetherError.self) { try KreaBatchOptions(args) }
    }
    #expect(throws: AetherError.self) { try KreaBatchOptions(["render-krea-batch"]) }
    #expect(try KreaBatchOptions(batchArguments(output: output) + ["/shape.json"]).registration?.path == "/shape.json")
    #expect(valid.steps == 8)
    let stepped = try KreaBatchOptions(batchArguments(output: output) + ["/shape.json", "--steps=16"])
    #expect(stepped.steps == 16 && stepped.registration?.path == "/shape.json")
    #expect(try KreaBatchOptions(["render-krea-batch", "--steps=12"] + batchArguments(output: output).dropFirst()).steps == 12)
    for invalid in [["--steps=0"], ["--steps=51"], ["--steps="], ["--steps=+8"], ["--steps=8", "--steps=16"]] {
        #expect(throws: AetherError.self) { try KreaBatchOptions(batchArguments(output: output) + invalid) }
    }
}

@Test func batchStepsReachEveryConfigurationAndTheReport() async throws {
    let root = try batchTemporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let output = root.appendingPathComponent("batch")
    let fake = BatchFake(output: output)
    let report = try await KreaBatchCommand.execute(options: KreaBatchOptions(batchArguments(output: output) + ["--steps=16"])) { fake }
    #expect(await fake.steps == [16, 16, 16])
    #expect(report.steps == 16)
    #expect(try batchReport(output)["steps"] as? Int == 16)
}

@Test func batchPreparesOnceRendersOrderedSeedsAndSavesVerifiedFiles() async throws {
    let root = try batchTemporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let output = root.appendingPathComponent("batch")
    let fake = BatchFake(output: output)
    let report = try await KreaBatchCommand.execute(options: KreaBatchOptions(batchArguments(output: output))) { fake }
    #expect(report.completed && report.stage == "complete")
    #expect(await fake.prepares == 1)
    #expect(await fake.seeds == [17, 18, 17])
    #expect(await fake.closes == 1)
    #expect(report.images.map(\.png) == ["001-seed-17.png", "002-seed-18.png", "003-seed-17.png"])
    #expect(report.images.allSatisfy { $0.status == "complete" && $0.generationSeconds != nil })
    #expect(report.images[0].pixelsSHA256 == report.images[2].pixelsSHA256)
    #expect(report.images[0].pixelsSHA256 != report.images[1].pixelsSHA256)
    for row in report.images {
        let data = try Data(contentsOf: output.appendingPathComponent(row.png))
        #expect(Array(data.prefix(8)) == [137, 80, 78, 71, 13, 10, 26, 10])
        #expect(row.pngSHA256 == SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
    }
    #expect(try batchReport(output)["completed"] as? Bool == true)
    #expect(try FileManager.default.contentsOfDirectory(atPath: output.path).count == 4)
}

@Test func batchRejectsExistingOutputWithoutStartingRenderer() async throws {
    let root = try batchTemporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let sentinel = root.appendingPathComponent("keep.txt"); try Data("keep".utf8).write(to: sentinel)
    let fake = BatchFake(output: root)
    await #expect(throws: AetherError.self) {
        try await KreaBatchCommand.execute(options: KreaBatchOptions(batchArguments(output: root))) { fake }
    }
    #expect(await fake.prepares == 0)
    #expect(try String(contentsOf: sentinel, encoding: .utf8) == "keep")
    #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("report.json").path))
}

@Test func batchFailuresCloseRendererAndPreserveCompletedImages() async throws {
    for fault: BatchFake.Fault in [.prepare, .secondImage, .wrongSize, .save] {
        let root = try batchTemporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("batch"); let fake = BatchFake(fault, output: output)
        await #expect(throws: (any Error).self) {
            try await KreaBatchCommand.execute(options: KreaBatchOptions(batchArguments(output: output))) { fake }
        }
        #expect(await fake.closes == 1)
        let report = try batchReport(output)
        #expect(report["completed"] as? Bool == false)
        #expect(report["stage"] as? String == "failed")
        #expect(report["error"] as? String != nil)
        if fault == .secondImage {
            let images = try #require(report["images"] as? [[String: Any]])
            #expect(images.count == 2)
            #expect(images[0]["status"] as? String == "complete")
            #expect(images[1]["status"] as? String == "failed")
            #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("001-seed-17.png").path))
            #expect(await fake.seeds == [17, 18])
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: output.path).allSatisfy { !$0.contains(".tmp.png") })
    }
}

@Test func batchFactoryFailureStillWritesIncompleteReport() async throws {
    let root = try batchTemporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let output = root.appendingPathComponent("batch")
    await #expect(throws: AetherError.invalidModel("missing assets")) {
        try await KreaBatchCommand.execute(options: KreaBatchOptions(batchArguments(output: output))) {
            throw AetherError.invalidModel("missing assets")
        }
    }
    #expect(try batchReport(output)["stage"] as? String == "failed")
}

private actor BatchCancellationGate: BatchRenderer {
    var entered = false
    var released = false
    var closes = 0
    var waitingForStart: CheckedContinuation<Void, Never>?
    var waitingForRelease: CheckedContinuation<Void, Never>?
    func prepare() -> BatchPreparation {
        BatchPreparation(bundleFingerprint: "fake", conditioningSHA256: "fake", conditioningByteCount: 0,
                         conditioningRows: 1, wasTruncated: false)
    }
    func generate(_ configuration: KreaConfiguration) async throws -> GeneratedImage {
        await withCheckedContinuation { continuation in
            waitingForRelease = continuation
            entered = true
            waitingForStart?.resume(); waitingForStart = nil
        }
        return try GeneratedImage(pixelsCHW: FloatTensor(shape: [3, 16, 16], values: Array(repeating: 0, count: 768)))
    }
    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { waitingForStart = $0 }
    }
    func release() { released = true; waitingForRelease?.resume(); waitingForRelease = nil }
    func unload() { closes += 1; #expect(released) }
}

@Test(.timeLimit(.minutes(1))) func batchCancellationWaitsForGenerationThenClosesWithoutSaving() async throws {
    let root = try batchTemporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let output = root.appendingPathComponent("batch")
    let options = try KreaBatchOptions(batchArguments(output: output))
    let gate = BatchCancellationGate()
    let task = Task { try await KreaBatchCommand.execute(options: options) { gate } }
    await gate.waitUntilEntered()
    task.cancel()
    #expect(await gate.closes == 0)
    await gate.release()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await gate.closes == 1)
    let report = try batchReport(output)
    #expect(report["completed"] as? Bool == false)
    #expect(report["stage"] as? String == "failed")
    #expect(try FileManager.default.contentsOfDirectory(atPath: output.path) == ["report.json"])
}

@Test func batchReferencesParseInOrderAndAreBounded() throws {
    let output = URL(fileURLWithPath: "/unused-output")
    #expect(try KreaBatchOptions(batchArguments(output: output)).references.isEmpty)
    let refs = try KreaBatchOptions(batchArguments(output: output) + ["--ref=/a.png", "--steps=16", "--ref=/b.png"])
    #expect(refs.references.map(\.path) == ["/a.png", "/b.png"] && refs.steps == 16 && refs.registration == nil)
    let four = (1...4).map { "--ref=/\($0).png" }
    #expect(try KreaBatchOptions(batchArguments(output: output) + ["/shape.json"] + four).references.count == 4)
    for invalid in [four + ["--ref=/5.png"], ["--ref="]] {
        #expect(throws: AetherError.self) { try KreaBatchOptions(batchArguments(output: output) + invalid) }
    }
}

@Test func batchReportRecordsReferencePathsAndHashes() async throws {
    let root = try batchTemporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let picture = root.appendingPathComponent("face.png")
    try Data("not really a png".utf8).write(to: picture)
    let output = root.appendingPathComponent("batch")
    let report = try await KreaBatchCommand.execute(
        options: KreaBatchOptions(batchArguments(output: output) + ["--ref=\(picture.path)"])) { BatchFake(output: output) }
    #expect(report.referencePaths == [picture.path])
    #expect(report.referenceSHA256s == [SHA256.hash(data: Data("not really a png".utf8)).map { String(format: "%02x", $0) }.joined()])
    #expect(try batchReport(output)["referencePaths"] as? [String] == [picture.path])
}
