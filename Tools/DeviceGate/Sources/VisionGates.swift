import AetherCore
import AetherEmbeddings
import AetherHost
@testable import AetherRuntime
@testable import AetherVision
import CoreAI
import CoreGraphics
import CryptoKit
import Darwin
import Foundation

/// The M4 gates on the device (docs/DEVICE.md): each model's tensor parity (the reference input tensor straight in)
/// against the same CPU float32 references and bars as the Mac gates (`DepthEstimatorGateTests`,
/// `TextEmbedderGateTests`, `ImageTextEncoderGateTests`, `ObjectDetectorDenseGateTests`, `ObjectDetectorDetrGateTests`),
/// then the placement measurement: the same graphs specialized for the GPU and for the Neural Engine, each timed over a
/// sustained run with the task's CPU energy and GPU time read from `TASK_POWER_INFO_V2`.
///
/// The GPU run is the qualification (`checks`); the Neural Engine run is a measurement (`placement`) and never fails the
/// gate. The bundles are the Forge exports, copied to Documents/models/<id> unchanged, and loaded with the iOS variant
/// manifest `run-on-device.sh` stages at Documents/manifests/<id>.json. `@testable` reaches the drivers' internal
/// pieces the Mac gates use (`PreparedImage` from a tensor, the detector heads); the script builds with
/// `ENABLE_TESTABILITY=YES` for that.
extension Gate {
    static let m4Models = ["depth-anything-3-small", "embeddinggemma-300m", "clip-vit-b32", "yolox-s", "rf-detr-nano",
                           "rf-detr-seg-nano"]
    /// The sustained run per model and compute unit: at least this many inferences and this many seconds.
    static let benchIterations = 30, benchSeconds = 10.0

    /// What one model's run under one compute unit gives the harness.
    struct Workload {
        /// The parity results, against the Mac gate's bars.
        var checks: [Report.Check]
        /// One representative inference (the first case's tensor), for the sustained run.
        var once: () async throws -> Void
        var ledgerPeakMB: Double
        var unload: () async -> Void
    }

    static func m4() async throws {
        report.computeUnitKinds = ComputeUnitKind.availableKinds.map { "\($0)" }.sorted()
        await log("Core AI compute unit kinds: \(report.computeUnitKinds ?? [])")
        for id in m4Models {
            await step(id) { try await m4Model(id) }
        }
    }

    static func m4Model(_ id: String) async throws {
        let bundleURL = documents.appending(path: "models/\(id)")
        let manifestURL = documents.appending(path: "manifests/\(id).json")
        let manifest = try Manifest.decode(Data(contentsOf: manifestURL))
        check("\(id).manifest", manifest.variant.platform == .iOS && manifest.id == id,
              "variant.platform \(manifest.variant.platform.rawValue), id \(manifest.id), graphs "
                  + manifest.graphs.keys.sorted().map { "\($0): \(manifest.graphs[$0]!.compute.rawValue)" }.joined(separator: ", "))
        if let sums = try? Data(contentsOf: bundleURL.appending(path: "SHA256SUMS")) {
            report.bundles[id] = SHA256.hash(data: sums).map { String(format: "%02x", $0) }.joined()
        }
        let bundle = try localBundle(bundleURL, manifest: manifest)
        for unit in [Manifest.ComputeUnit.gpu, .neuralEngine] {
            var placed = manifest
            for role in placed.graphs.keys { placed.graphs[role]!.compute = unit }
            await place(id, unit, bundle: bundle, manifest: placed)
        }
    }

    /// Loads, checks parity, runs the sustained loop and unloads one model under one compute unit.
    static func place(_ id: String, _ unit: Manifest.ComputeUnit, bundle: InstalledBundle, manifest: Manifest) async {
        let tag = unit == .gpu ? "gpu" : "ane"
        var placement = Report.Placement(model: id, compute: unit.rawValue)
        let footprintBefore = physFootprint()
        let thermalStart = ProcessInfo.processInfo.thermalState
        do {
            let started = ContinuousClock.now
            let workload = try await load(id, bundle: bundle, manifest: manifest)
            // The first request specializes the graph (or reads the specialization cache) and runs the parity cases.
            placement.parityAndLoadSeconds = seconds(ContinuousClock.now - started)
            placement.withinBars = workload.checks.allSatisfy(\.passed)
            placement.failed = workload.checks.filter { !$0.passed }.map { "\($0.name): \($0.detail)" }
            placement.parity = workload.checks.map { "\($0.name): \($0.detail)" }
            if unit == .gpu {
                for c in workload.checks { check("\(id).\(tag).\(c.name)", c.passed, c.detail) }
            }

            // The sustained run.
            try await workload.once()   // warm
            var latencies: [Double] = []
            let energyBefore = power()
            let loopStart = ContinuousClock.now
            while latencies.count < benchIterations || seconds(ContinuousClock.now - loopStart) < benchSeconds {
                let t0 = ContinuousClock.now
                try await workload.once()
                latencies.append(seconds(ContinuousClock.now - t0))
            }
            let wall = seconds(ContinuousClock.now - loopStart)
            let energyAfter = power()
            latencies.sort()
            placement.iterations = latencies.count
            placement.medianMS = 1000 * latencies[latencies.count / 2]
            placement.p90MS = 1000 * latencies[min(latencies.count - 1, latencies.count * 9 / 10)]
            placement.minMS = 1000 * latencies[0]
            placement.loopSeconds = wall
            let n = Double(latencies.count)
            placement.cpuEnergyMJPerRun = Double(energyAfter.cpuEnergyNJ &- energyBefore.cpuEnergyNJ) / 1e6 / n
            placement.gpuMSPerRun = Double(energyAfter.gpuNS &- energyBefore.gpuNS) / 1e6 / n
            placement.cpuMSPerRun = Double(energyAfter.cpuNS &- energyBefore.cpuNS) / 1e6 / n
            placement.ledgerPeakMB = workload.ledgerPeakMB
            placement.footprintDeltaMB = Double(Int64(physFootprint()) - Int64(footprintBefore)) / 1_048_576
            placement.status = "ran"
            await workload.unload()
            if unit == .gpu {
                measure("\(id).gpu.medianMS", placement.medianMS!)
                measure("\(id).gpu.ledgerPeakMB", workload.ledgerPeakMB)
                measure("\(id).gpu.footprintDeltaMB", placement.footprintDeltaMB!)
                measure("\(id).gpu.firstRequestSeconds", placement.parityAndLoadSeconds!)
            }
        } catch {
            placement.status = "failed: \(error)"
            if unit == .gpu { check("\(id).gpu.completed", false, "\(error)") }
        }
        placement.thermalStart = "\(thermalStart)"
        placement.thermalEnd = "\(ProcessInfo.processInfo.thermalState)"
        placement.physFootprintPeakMB = Double(physFootprintPeak() >> 20)
        report.placement.append(placement)
        await log("\(id) \(tag): \(placement.status), within bars \(placement.withinBars.map { "\($0)" } ?? "-"), "
                  + "median \(placement.medianMS.map { String(format: "%.2f", $0) } ?? "-") ms, "
                  + "GPU \(placement.gpuMSPerRun.map { String(format: "%.2f", $0) } ?? "-") ms/run")
        save()
    }

    static func load(_ id: String, bundle: InstalledBundle, manifest: Manifest) async throws -> Workload {
        let reference = documents.appending(path: "references/\(id)")
        switch id {
        case "depth-anything-3-small": return try await depth(bundle, manifest, reference)
        case "embeddinggemma-300m": return try await embeddingGemma(bundle, manifest, reference)
        case "clip-vit-b32": return try await clip(bundle, manifest, reference)
        case "yolox-s": return try await yolox(bundle, manifest, reference)
        default: return try await rfDetr(bundle, manifest, reference)
        }
    }

    static func localBundle(_ url: URL, manifest: Manifest) throws -> InstalledBundle {
        let ref = ModelRef(
            catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: manifest.variant.platform.rawValue,
            driver: manifest.driver, assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
        return InstalledBundle(ref: ref, url: url, isStale: false)
    }

    static func ledgerPeakMB(_ ledger: FootprintLedger) -> Double {
        Double(ledger.snapshot.values.map(\.peakBytes).reduce(0, +) >> 20)
    }

    static func whole(_ shape: [Int]) -> ImageGeometry {
        let (h, w) = (shape[shape.count - 2], shape[shape.count - 1])
        return ImageGeometry(sourceWidth: w, sourceHeight: h, width: w, height: h, content: CGRect(x: 0, y: 0, width: w, height: h))
    }

    /// ‖a − b‖ / ‖b‖ and the largest absolute difference.
    static func relL2(_ a: [Float], _ b: [Float]) -> (relL2: Double, maxAbs: Double) {
        guard a.count == b.count else { return (.infinity, .infinity) }
        var diff = 0.0, norm = 0.0, worst = 0.0
        for (x, y) in zip(a, b) {
            let d = Double(x) - Double(y)
            diff += d * d
            norm += Double(y) * Double(y)
            worst = max(worst, abs(d))
        }
        let r = (diff / max(norm, .leastNonzeroMagnitude)).squareRoot()
        return (r.isNaN ? .infinity : r, worst.isNaN ? .infinity : worst)
    }

    static func ints(_ url: URL) throws -> [Int32] {
        try Data(contentsOf: url).withUnsafeBytes { raw in
            (0..<(raw.count / 4)).map { Int32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self)) }
        }
    }

    static func json<T: Decodable>(_ type: T.Type, _ url: URL) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: url))
    }

    struct Index: Decodable {
        struct Entry: Decodable { var name: String }
        var shapes: [String: [Int]]?
        var cases: [Entry]
    }

    // MARK: Depth Anything 3 small (bars: DepthEstimatorGateTests, graph parity)

    static func depth(_ bundle: InstalledBundle, _ manifest: Manifest, _ reference: URL) async throws -> Workload {
        let index = try json(Index.self, reference.appending(path: "references.json"))
        let estimator = try DepthEstimator(bundle: bundle, manifest: manifest, device: .current)
        let shape = index.shapes!["input"]!
        var checks: [Report.Check] = []
        var first: PreparedImage?
        for entry in index.cases {
            let directory = reference.appending(path: entry.name)
            let prepared = PreparedImage(shape: shape, values: .float32(try readF32(directory.appending(path: "input.f32"))),
                                         geometry: whole(shape))
            first = first ?? prepared
            let t0 = ContinuousClock.now
            let map = try await estimator.estimate(prepared, DepthRequest(resizeBack: false))
            let elapsed = seconds(ContinuousClock.now - t0)
            let d = relL2(map.values, try readF32(directory.appending(path: "depth.f32")))
            let k = relL2(map.confidence ?? [], try readF32(directory.appending(path: "confidence.f32")))
            checks.append(.init(name: "\(entry.name).depth", passed: d.relL2 <= 0.02,
                                detail: "rel L2 \(d.relL2) (bar 0.02), max |Δ| \(d.maxAbs), \(ms(elapsed))"))
            checks.append(.init(name: "\(entry.name).confidence", passed: k.relL2 <= 0.02,
                                detail: "rel L2 \(k.relL2) (bar 0.02), max |Δ| \(k.maxAbs)"))
            let repeated = try await estimator.estimate(prepared, DepthRequest(resizeBack: false))
            checks.append(.init(name: "\(entry.name).repeat", passed: repeated == map, detail: "a repeat is bit-identical"))
        }
        let input = first!
        return Workload(checks: checks, once: { _ = try await estimator.estimate(input, DepthRequest(resizeBack: false)) },
                        ledgerPeakMB: ledgerPeakMB(estimator.options.ledger), unload: { await estimator.unload() })
    }

    // MARK: EmbeddingGemma 300M (bars: TextEmbedderGateTests)

    struct EmbeddingIndex: Decodable {
        struct Entry: Decodable { var name: String }
        struct Similarity: Decodable {
            var queries: [String], documents: [String]
            var cosine: [String: [String: Double]]
            var expectedBest: [String: [String]]
        }
        var cases: [Entry]
        var similarity: Similarity
    }

    struct EmbeddingCase: Decodable { var name: String, role: String, text: String, keptTokens: Int, truncated: Bool }

    static func embeddingGemma(_ bundle: InstalledBundle, _ manifest: Manifest, _ reference: URL) async throws -> Workload {
        let index = try json(EmbeddingIndex.self, reference.appending(path: "references.json"))
        let embedder = try await TextEmbedder(bundle: bundle, manifest: manifest, device: .current)
        var checks: [Report.Check] = []
        var embeddings: [String: Embedding] = [:]
        var first: TokenizedText?
        for entry in index.cases {
            let directory = reference.appending(path: entry.name)
            let c = try json(EmbeddingCase.self, directory.appending(path: "case.json"))
            let role: EmbeddingRole = c.role == "query" ? .query : .document
            let tokenized = try embedder.tokenize(c.text, role: role)
            let ids = try ints(directory.appending(path: "input_ids.i32"))
            let mask = try ints(directory.appending(path: "attention_mask.i32"))
            let idsExact = tokenized.ids == ids && tokenized.attentionMask == mask
                && tokenized.tokenCount == c.keptTokens && tokenized.isTruncated == c.truncated
            checks.append(.init(name: "\(c.name).tokens", passed: idsExact, detail: "ids, mask and truncation equal the reference's"))
            first = first ?? tokenized
            let t0 = ContinuousClock.now
            let embedding = try await embedder.embed([tokenized])[0]
            let elapsed = seconds(ContinuousClock.now - t0)
            let expected = try readF32(directory.appending(path: "embedding.f32"))
            let cosine = embedding.similarity(Embedding(expected))
            let norm = embedding.vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
            checks.append(.init(name: "\(c.name).cosine", passed: cosine >= 0.999 && abs(norm - 1) < 1e-3,
                                detail: "cosine \(cosine) (bar 0.999), norm \(norm), \(ms(elapsed))"))
            let repeated = try await embedder.embed([tokenized])
            checks.append(.init(name: "\(c.name).repeat", passed: repeated == [embedding], detail: "a repeat is bit-identical"))
            embeddings[c.name] = embedding
        }
        let s = index.similarity
        for query in s.queries {
            guard let q = embeddings[query] else { continue }
            let scores = s.documents.map { embeddings[$0].map { q.similarity($0) } ?? -.infinity }
            let best = s.documents[scores.indices.max { scores[$0] < scores[$1] }!]
            let worst = zip(s.documents, scores).map { abs(Double($1) - (s.cosine[query]?[$0] ?? .nan)) }.max() ?? .nan
            checks.append(.init(name: "\(query).retrieval", passed: s.expectedBest[query]?.contains(best) == true && worst <= 0.005,
                                detail: "best \(best), max |Δ cosine| to the reference \(worst) (bar 0.005)"))
        }
        let tokens = first!
        return Workload(checks: checks, once: { _ = try await embedder.embed([tokens]) },
                        ledgerPeakMB: ledgerPeakMB(embedder.options.ledger), unload: { await embedder.unload() })
    }

    // MARK: CLIP ViT-B/32 (bars: ImageTextEncoderGateTests, both towers)

    struct ClipIndex: Decodable {
        struct Texts: Decodable { var strings: [String] }
        struct ZeroShot: Decodable { var labels: [String]; var prompt: String }
        struct Entry: Decodable { var name: String }
        var shapes: [String: [Int]]
        var texts: Texts
        var zeroShot: ZeroShot
        var cases: [Entry]
    }

    struct ClipCase: Decodable { struct ZeroShot: Decodable { var argmax: Int; var logits: [Float] }; var zeroShot: ZeroShot }

    static func clip(_ bundle: InstalledBundle, _ manifest: Manifest, _ reference: URL) async throws -> Workload {
        let index = try json(ClipIndex.self, reference.appending(path: "references.json"))
        let encoder = try ImageTextEncoder(bundle: bundle, manifest: manifest, device: .current)
        let dimension = 512, length = index.shapes["tokens"]![1]
        var checks: [Report.Check] = []
        func rows<T>(_ v: [T], _ width: Int) -> [[T]] { stride(from: 0, to: v.count, by: width).map { Array(v[$0..<($0 + width)]) } }
        func argmax(_ v: [Float]) -> Int { v.indices.max { v[$0] < v[$1] }! }

        let texts = index.texts.strings
        let expectedIDs = rows(try ints(reference.appending(path: "texts/tokens.i32")), length)
        let ids = try encoder.tokenize(texts)
        let exact = zip(ids, expectedIDs).filter { $0 == $1 }.count
        checks.append(.init(name: "tokenizer", passed: exact == texts.count && ids.count == texts.count,
                            detail: "\(exact)/\(texts.count) texts id-exact"))
        let expectedTexts = rows(try readF32(reference.appending(path: "texts/embedding.f32")), dimension).map(Embedding.init)
        let t0 = ContinuousClock.now
        let textEmbeddings = try await encoder.embed(texts: texts)
        let textElapsed = seconds(ContinuousClock.now - t0)
        let cosines = zip(textEmbeddings, expectedTexts).map { $0.similarity($1) }
        let minText = cosines.map { $0.isNaN ? -1 : $0 }.min() ?? -1
        checks.append(.init(name: "textTower", passed: minText >= 0.999 && cosines.count == texts.count,
                            detail: "\(texts.count) texts (buckets 8+8+1), cosine min \(minText) (bar 0.999), \(ms(textElapsed))"))
        let alone = try await encoder.embed(texts: [texts[1]])
        let bucketOne = alone[0].similarity(textEmbeddings[1])
        checks.append(.init(name: "textTower.bucket1", passed: bucketOne >= 0.999, detail: "bucket 1 vs its bucket-8 row: cosine \(bucketOne)"))

        let labels = index.zeroShot.labels
        let labelTexts = try await encoder.embed(texts: labels.map { index.zeroShot.prompt.replacingOccurrences(of: "{}", with: $0) })
        let shape = index.shapes["image"]!
        var first: PreparedImage?
        for entry in index.cases {
            let directory = reference.appending(path: "images/\(entry.name)")
            let c = try json(ClipCase.self, directory.appending(path: "case.json"))
            let prepared = PreparedImage(shape: shape, values: .float32(try readF32(directory.appending(path: "input.f32"))),
                                         geometry: whole(shape))
            first = first ?? prepared
            let i0 = ContinuousClock.now
            let embedding = try await encoder.embed(image: prepared)
            let elapsed = seconds(ContinuousClock.now - i0)
            let cosine = embedding.similarity(Embedding(try readF32(directory.appending(path: "embedding.f32"))))
            let logits = labelTexts.map { encoder.logitScale * embedding.similarity($0) }
            let worst = zip(logits, c.zeroShot.logits).map { abs($0 - $1) }.max()!
            checks.append(.init(name: "\(entry.name).image", passed: cosine >= 0.999 && argmax(logits) == c.zeroShot.argmax,
                                detail: "cosine \(cosine) (bar 0.999), zero-shot \(labels[argmax(logits)]) (reference "
                                    + "\(labels[c.zeroShot.argmax])), logits max |Δ| \(worst), \(ms(elapsed))"))
            let repeated = try await encoder.embed(image: prepared)
            checks.append(.init(name: "\(entry.name).repeat", passed: repeated == embedding, detail: "a repeat is bit-identical"))
        }
        let input = first!
        return Workload(checks: checks, once: { _ = try await encoder.embed(image: input) },
                        ledgerPeakMB: ledgerPeakMB(encoder.options.ledger), unload: { await encoder.unload() })
    }

    // MARK: YOLOX-S (bars: ObjectDetectorDenseGateTests, detection parity)

    struct YoloxCase: Decodable {
        struct Letterbox: Decodable { var source: [Int]; var resized: [Int] }
        struct Source: Decodable { var letterbox: Letterbox? }
        var source: Source
    }

    struct YoloxReference: Decodable { var score: Double; var box: [Double]; var cocoId: Int }

    /// Each reference detection, by descending score, paired with the unpaired same-class detection of highest IoU.
    static func matchDense(_ got: [Detection], _ want: [(label: Int, score: Double, box: CGRect)]) -> Report.Check {
        let sameClasses = got.map(\.label.id).sorted() == want.map(\.label).sorted()
        var used = Set<Int>(), matched = 0, minIoU = 1.0, maxScore = 0.0
        for w in want.sorted(by: { $0.score > $1.score }) {
            var best: (Int, Double)?
            for (j, g) in got.enumerated() where !used.contains(j) && g.label.id == w.label {
                let v = iou(g.box, w.box)
                if v > (best?.1 ?? -1) { best = (j, v) }
            }
            guard let (j, v) = best else { continue }
            used.insert(j)
            matched += 1
            minIoU = min(minIoU, v)
            maxScore = max(maxScore, abs(Double(got[j].score) - w.score))
        }
        let passed = sameClasses && matched == want.count && got.count == want.count && minIoU >= 0.95 && maxScore <= 0.01
        return .init(name: "", passed: passed, detail: "\(got.count) detected / \(want.count) reference, \(matched) matched, "
                     + "same class set \(sameClasses), min IoU \(minIoU) (bar 0.95), max |Δscore| \(maxScore) (bar 0.01)")
    }

    static func iou(_ a: CGRect, _ b: CGRect) -> Double {
        let i = a.intersection(b)
        guard !i.isNull else { return 0 }
        let inter = Double(i.width * i.height)
        let union = Double(a.width * a.height + b.width * b.height) - inter
        return union > 0 ? inter / union : 1
    }

    static func yolox(_ bundle: InstalledBundle, _ manifest: Manifest, _ reference: URL) async throws -> Workload {
        let index = try json(Index.self, reference.appending(path: "references.json"))
        let detector = try ObjectDetector(bundle: bundle, manifest: manifest, device: .current)
        let shape = index.shapes!["input"]!
        let (height, width) = (shape[2], shape[3])
        var checks: [Report.Check] = []
        var first: PreparedImage?
        for entry in index.cases {
            let directory = reference.appending(path: entry.name)
            let c = try json(YoloxCase.self, directory.appending(path: "case.json"))
            let want = try json([YoloxReference].self, directory.appending(path: "detections.json"))
            let source = c.source.letterbox?.source ?? [width, height], resized = c.source.letterbox?.resized ?? [width, height]
            let geometry = ImageGeometry(sourceWidth: source[0], sourceHeight: source[1], width: width, height: height,
                                         content: CGRect(x: 0, y: 0, width: resized[0], height: resized[1]))
            let boxes = DetectionBoxes(format: .cxcywh, space: .letterboxPixels, geometry: geometry)
            let prepared = PreparedImage(shape: shape, values: .float32(try readF32(directory.appending(path: "input.f32"))),
                                         geometry: geometry)
            first = first ?? prepared
            let t0 = ContinuousClock.now
            let got = try await detector.detect(prepared)
            let elapsed = seconds(ContinuousClock.now - t0)
            var m = matchDense(got, want.map { (label: $0.cocoId, score: $0.score, box: boxes.source($0.box[0], $0.box[1], $0.box[2], $0.box[3])) })
            m.name = "\(entry.name).detections"
            m.detail += ", \(ms(elapsed))"
            checks.append(m)
            let repeated = try await detector.detect(prepared)
            checks.append(.init(name: "\(entry.name).repeat", passed: repeated == got, detail: "a repeat is bit-identical"))
        }
        let input = first!
        return Workload(checks: checks, once: { _ = try await detector.detect(input) },
                        ledgerPeakMB: ledgerPeakMB(detector.options.ledger), unload: { await detector.unload() })
    }

    // MARK: RF-DETR nano and seg-nano (bars: ObjectDetectorDetrGateTests, detection parity)

    /// Detections matched class by class, greedily by box IoU (> 0.5); an unmatched one passes only within the score
    /// bar of the threshold.
    static func matchDetr(_ got: [Detection], _ want: [Detection], threshold: Float) -> Report.Check {
        let sameClasses = Set(got.map(\.label)) == Set(want.map(\.label))
        var pairs = 0, minIoU = 1.0, maxScore: Float = 0, minMask: Double?, hard: [String] = [], borderline = 0
        func unmatched(_ d: Detection, _ side: String) {
            if abs(d.score - threshold) <= 0.01 { borderline += 1 } else { hard.append("\(side) \(d.label.name) \(d.score)") }
        }
        for label in Set(got.map(\.label)).union(want.map(\.label)) {
            let g = got.filter { $0.label == label }, w = want.filter { $0.label == label }
            var candidates: [(Double, Int, Int)] = []
            for (i, a) in g.enumerated() { for (j, b) in w.enumerated() { candidates.append((iou(a.box, b.box), i, j)) } }
            var usedG = Set<Int>(), usedW = Set<Int>()
            for (score, i, j) in candidates.sorted(by: { $0.0 > $1.0 }) where score > 0.5 && !usedG.contains(i) && !usedW.contains(j) {
                usedG.insert(i)
                usedW.insert(j)
                pairs += 1
                minIoU = min(minIoU, score)
                maxScore = max(maxScore, abs(g[i].score - w[j].score))
                if let a = g[i].mask, let b = w[j].mask { minMask = min(minMask ?? 1, a.intersectionOverUnion(b)) }
            }
            for (k, d) in g.enumerated() where !usedG.contains(k) { unmatched(d, "got") }
            for (k, d) in w.enumerated() where !usedW.contains(k) { unmatched(d, "want") }
        }
        let passed = sameClasses && minIoU >= 0.95 && maxScore <= 0.01 && (minMask ?? 1) >= 0.95 && hard.isEmpty
        return .init(name: "", passed: passed, detail: "\(got.count) vs \(want.count) detections, classes "
                     + "\(sameClasses ? "same" : "DIFFER"), \(pairs) pairs, min IoU \(minIoU) (bar 0.95), max |Δscore| \(maxScore) "
                     + "(bar 0.01)" + (minMask.map { ", min mask IoU \($0) (bar 0.95)" } ?? "")
                     + ", unmatched \(hard) (+\(borderline) borderline)")
    }

    static func rfDetr(_ bundle: InstalledBundle, _ manifest: Manifest, _ reference: URL) async throws -> Workload {
        let index = try json(Index.self, reference.appending(path: "references.json"))
        let shapes = index.shapes!
        let detector = try ObjectDetector(bundle: bundle, manifest: manifest, device: .current)
        guard case .detrSigmoid(let post)? = manifest.postprocess,
              case .detr(let head) = try ObjectDetector.Configuration(manifest).head else {
            throw AetherError.manifestInvalid(field: "postprocess", reason: "not detrSigmoid")
        }
        let threshold = Float(post.defaults.scoreThreshold)
        let shape = shapes["input"]!
        let geometry = whole(shape)
        var checks: [Report.Check] = []
        var first: PreparedImage?
        for entry in index.cases {
            let directory = reference.appending(path: entry.name)
            var raw: [String: DetectorOutput] = [:]
            for role in ["dets", "labels", "masks"] where shapes[role] != nil {
                raw[role] = DetectorOutput(shape: shapes[role]!, values: try readF32(directory.appending(path: "\(role).f32")))
            }
            // The reference's raw outputs through the driver's own head (the Mac gate checks this is upstream's PostProcess).
            let want = head.detections(raw, geometry: geometry, scoreThreshold: post.defaults.scoreThreshold,
                                       maxDetections: post.defaults.maxDetections)
            let prepared = PreparedImage(shape: shape, values: .float32(try readF32(directory.appending(path: "input.f32"))),
                                         geometry: geometry)
            first = first ?? prepared
            let t0 = ContinuousClock.now
            let got = try await detector.detect(prepared)
            let elapsed = seconds(ContinuousClock.now - t0)
            var m = matchDetr(got, want, threshold: threshold)
            m.name = "\(entry.name).detections"
            m.detail += ", \(ms(elapsed))"
            checks.append(m)
            let repeated = try await detector.detect(prepared)
            checks.append(.init(name: "\(entry.name).repeat", passed: repeated == got, detail: "a repeat is bit-identical"))
        }
        let input = first!
        return Workload(checks: checks, once: { _ = try await detector.detect(input) },
                        ledgerPeakMB: ledgerPeakMB(detector.options.ledger), unload: { await detector.unload() })
    }

    // MARK: counters

    static func ms(_ s: Double) -> String { String(format: "%.1f ms", 1000 * s) }

    struct Power { var cpuEnergyNJ: UInt64; var gpuNS: UInt64; var cpuNS: UInt64 }

    /// `TASK_POWER_INFO_V2`: the CPU energy (`task_energy`, nJ) and the GPU time (`task_gpu_utilisation`, ns) billed to
    /// this process, and its CPU time (`getrusage`). Neural Engine work runs in `aned` and is billed to none of them.
    static func power() -> Power {
        var info = task_power_info_v2_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_power_info_v2_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_POWER_INFO_V2), $0, &count)
            }
        }
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        let cpu = UInt64(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1_000_000_000
            + UInt64(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) * 1_000
        guard kr == KERN_SUCCESS else { return Power(cpuEnergyNJ: 0, gpuNS: 0, cpuNS: cpu) }
        return Power(cpuEnergyNJ: info.task_energy, gpuNS: info.gpu_energy.task_gpu_utilisation, cpuNS: cpu)
    }

    /// The process's current physical footprint (`TASK_VM_INFO`'s `phys_footprint`).
    static func physFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
    }
}
