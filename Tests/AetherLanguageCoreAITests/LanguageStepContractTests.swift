import AetherKit
@testable import AetherLanguageCoreAI
import Testing

struct LanguageStepContractTests {
    private let running = (inputs: Set(["input_ids", "position_ids"]), outputs: Set(["logits"]), states: Set(["keyCache", "valueCache"]))
    private let bucketed = (inputs: Set(["input_ids", "cache_positions", "keyCache", "valueCache"]),
                            outputs: Set(["logits", "keyUpdate", "valueUpdate"]), states: Set<String>())

    @Test func signatureIdentifiesContract() {
        let detect = LanguageStepContract.detect
        #expect(detect(running.inputs, running.outputs, running.states, [1, -1, 151936]) == .runningPositions)
        #expect(detect(bucketed.inputs, bucketed.outputs, bucketed.states, [1, 1, 151936]) == .bucketedPositions)
        // Last-row logits with running positions (an engine `prefill` function), or all-row logits with
        // cache positions, are neither contract.
        #expect(detect(running.inputs, running.outputs, running.states, [1, 1, 151936]) == nil)
        #expect(detect(bucketed.inputs, bucketed.outputs, bucketed.states, [1, -1, 151936]) == nil)
        // Mixed signatures: cache positions over in-place states, or running positions without states.
        #expect(detect(["input_ids", "cache_positions"], ["logits"], running.states, [1, 1, 8]) == nil)
        #expect(detect(bucketed.inputs, ["logits"], bucketed.states, [1, 1, 8]) == nil)
        #expect(detect(bucketed.inputs, bucketed.outputs, running.states, [1, 1, 8]) == nil)
        #expect(detect(bucketed.inputs, bucketed.outputs, bucketed.states, [1, 1]) == nil)
        #expect(LanguageStepContract.runningPositions.cachesAreStates)
        #expect(!LanguageStepContract.bucketedPositions.cachesAreStates)
    }

    @Test func oneFunctionImageSignatureIsBucketedWithImageRows() {
        let detect = LanguageStepContract.detect
        let rope = bucketed.inputs.union(["rope_positions"])
        let oneFunction = rope.union(["image_embeds", "deepstack"])
        #expect(detect(rope, bucketed.outputs, bucketed.states, [1, 1, 151936]) == .bucketedPositions)
        #expect(detect(oneFunction, bucketed.outputs, bucketed.states, [1, 1, 151936]) == .bucketedPositions)
        #expect(LanguageStepContract.takesImageRows(oneFunction))
        #expect(!LanguageStepContract.takesImageRows(rope))
        // Image rows without rotary positions, or only half of them, are no contract.
        #expect(detect(bucketed.inputs.union(["image_embeds", "deepstack"]), bucketed.outputs, bucketed.states, [1, 1, 8]) == nil)
        #expect(detect(rope.union(["image_embeds"]), bucketed.outputs, bucketed.states, [1, 1, 8]) == nil)
        // The 0.0.2 image entry point stays recognised beside a `main` without image inputs.
        #expect(LanguageStepContract.isImageStep(inputNames: oneFunction, outputNames: bucketed.outputs, stateNames: []))
        #expect(!LanguageStepContract.isImageStep(inputNames: rope, outputNames: bucketed.outputs, stateNames: []))
    }

    @Test func updateRowsLandAtTheirPositionsInEveryRow() {
        // Two rows (layer/head), capacity 5, width 2. Rows 3 and 4 of each are written from a
        // [rows, count = 2, width] update; everything else keeps its value.
        var cache = (0..<20).map { Float16($0) }
        let update: [Float16] = [100, 101, 102, 103, 200, 201, 202, 203]
        update.withUnsafeBufferPointer { source in
            cache.withUnsafeMutableBufferPointer { target in
                LanguageCachePlan.writeUpdate(source.baseAddress!, rowStride: 4, tokenStride: 2, elementStride: 1,
                    into: target.baseAddress!, rows: 2, capacity: 5, width: 2, start: 3, count: 2)
            }
        }
        #expect(cache == ([0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 10, 11, 12, 13, 14, 15, 200, 201, 202, 203] as [Float16]))
        // A transposed update layout ([width, count] per row) gives the same cache.
        var strided = (0..<20).map { Float16($0) }
        let transposed: [Float16] = [100, 102, 101, 103, 200, 202, 201, 203]
        transposed.withUnsafeBufferPointer { source in
            strided.withUnsafeMutableBufferPointer { target in
                LanguageCachePlan.writeUpdate(source.baseAddress!, rowStride: 4, tokenStride: 1, elementStride: 2,
                    into: target.baseAddress!, rows: 2, capacity: 5, width: 2, start: 3, count: 2)
            }
        }
        #expect(strided == cache)
    }

    @Test func updateTensorPlacementHandlesNestedAndPerRowStrides() {
        // Update [layers 2, batch 1, heads 2, count 2, width 3] into a [4 rows, capacity 4, width 3] cache at start 1.
        let shape = [2, 1, 2, 2, 3]
        func value(_ l: Int, _ h: Int, _ t: Int, _ e: Int) -> Float { Float(1000 * l + 100 * h + 10 * t + e) }
        var expected = [Float](repeating: -1, count: 4 * 4 * 3)
        for l in 0..<2 { for h in 0..<2 { for t in 0..<2 { for e in 0..<3 {
            expected[((l * 2 + h) * 4 + 1 + t) * 3 + e] = value(l, h, t, e)
        } } } }
        func place(strides: [Int]) -> [Float] {
            var update = [Float](repeating: .nan, count: 24)
            for l in 0..<2 { for h in 0..<2 { for t in 0..<2 { for e in 0..<3 {
                update[l * strides[0] + h * strides[2] + t * strides[3] + e * strides[4]] = value(l, h, t, e)
            } } } }
            var cache = [Float](repeating: -1, count: 4 * 4 * 3)
            update.withUnsafeBufferPointer { u in
                cache.withUnsafeMutableBufferPointer { c in
                    LanguageCachePlan.writeUpdateTensor(u.baseAddress!, shape: shape, strides: strides, into: c.baseAddress!,
                                                        capacity: 4, start: 1)
                }
            }
            return cache
        }
        #expect(place(strides: [12, 12, 6, 3, 1]) == expected)     // contiguous: rows nest, one run of rows
        #expect(place(strides: [12, 12, 1, 6, 2]) == expected)     // heads innermost: rows don't nest, per-row path
        #expect(place(strides: [12, 12, 6, 1, 2]) == expected)     // transposed token/element layout
    }

    @Test func positionsAndLogitRowsFollowContract() {
        #expect(LanguageStepContract.runningPositions.positions(at: 3, count: 2) == [0, 1, 2, 3, 4])
        #expect(LanguageStepContract.bucketedPositions.positions(at: 3, count: 2) == [3, 4])
        #expect(LanguageStepContract.bucketedPositions.positions(at: 1000, count: 1) == [1000])
        #expect(LanguageStepContract.runningPositions.logitsRows(count: 7) == 7)
        #expect(LanguageStepContract.bucketedPositions.logitsRows(count: 7) == 1)
        #expect(LanguageStepContract.runningPositions.positionInputName == "position_ids")
        #expect(LanguageStepContract.bucketedPositions.positionInputName == "cache_positions")
    }

    @Test func runningPrefillKeepsUniformChunksWithinTheBound() {
        let running = LanguageStepContract.runningPositions
        let counts = { (count: Int, request: Int?, bound: Int?) in
            running.prefillChunks(promptCount: count, requestedChunk: request, maximumChunk: bound).map(\.count)
        }
        #expect(counts(10, nil, nil) == [10])
        #expect(counts(10, 4, nil) == [4, 4, 2])
        #expect(counts(3, 64, nil) == [3])
        // A declared export bound caps the whole-prompt default and larger requests.
        #expect(counts(1000, nil, 512) == [512, 488])
        #expect(running.prefillChunks(promptCount: 10, requestedChunk: 4, maximumChunk: nil).map(\.offset) == [0, 4, 8])
    }

    @Test func bucketedPrefillUsesOneWidthAndAnOverlappingTail() {
        let bucketed = LanguageStepContract.bucketedPositions
        let plan = { (count: Int, request: Int?, bound: Int?) in
            bucketed.prefillChunks(promptCount: count, requestedChunk: request, maximumChunk: bound).map { [$0.offset, $0.count] }
        }
        let c = LanguageStepContract.defaultBucketedPrefillChunk
        // A run shorter than the width is one chunk of its own length.
        #expect(plan(1, nil, 512) == [[0, 1]])
        #expect(plan(12, nil, 512) == [[0, 12]])
        #expect(plan(c, nil, 512) == [[0, c]])
        // The default width follows the prompt's length: 64 up to 256 tokens (the 0.0.2 plan), 256 above.
        #expect(c == 64 && LanguageStepContract.longPromptPrefillChunk == 256 && LanguageStepContract.longPromptThreshold == 256)
        #expect(plan(255, nil, 512) == [[0, 64], [64, 64], [128, 64], [191, 64]])
        #expect(plan(256, nil, 512) == [[0, 64], [64, 64], [128, 64], [192, 64]])
        #expect(plan(257, nil, 512) == [[0, 256], [1, 256]])
        #expect(plan(449, nil, 512) == [[0, 256], [193, 256]])
        #expect(plan(2048, nil, 512).map { $0[0] } == [0, 256, 512, 768, 1024, 1280, 1536, 1792])
        // A request's own width, and the export bound, still win.
        #expect(plan(449, 64, 512).count == 8)
        #expect(plan(300, nil, 128) == [[0, 128], [128, 128], [172, 128]])
        #expect(LanguageStepContract.bucketedPositions.defaultPrefillChunk(promptCount: 256) == 64)
        #expect(LanguageStepContract.bucketedPositions.defaultPrefillChunk(promptCount: 257) == 256)
        #expect(LanguageStepContract.runningPositions.defaultPrefillChunk(promptCount: 4096) == nil)
        // Multiples of the width tile exactly; otherwise the last chunk covers the last C tokens.
        #expect(plan(130, 64, 512) == [[0, 64], [64, 64], [66, 64]])
        #expect(plan(449, 256, 512) == [[0, 256], [193, 256]])
        #expect(plan(256, 128, 512) == [[0, 128], [128, 128]])
        // The bound caps the width.
        #expect(plan(1300, 4096, 512) == [[0, 512], [512, 512], [788, 512]])
        #expect(plan(300, nil, 48) == [[0, 48], [48, 48], [96, 48], [144, 48], [192, 48], [240, 48], [252, 48]])
        // Every prompt length up to 2,048: the chunks cover it, start in order within it, end at it, step back only
        // in the last chunk, and use one width (or one chunk of the run's length).
        for bound in [512, 128, 48] {
            for request in [nil, 1, 7, 64, 512] as [Int?] {
                for count in 1...2048 {
                    let width = min(request ?? (count > LanguageStepContract.longPromptThreshold ? 256 : c), bound)
                    let chunks = bucketed.prefillChunks(promptCount: count, requestedChunk: request, maximumChunk: bound)
                    #expect(chunks.first?.offset == 0)
                    #expect(chunks.last.map { $0.offset + $0.count } == count)
                    #expect(chunks.allSatisfy { $0.count == min(width, count) })
                    for (previous, next) in zip(chunks, chunks.dropFirst()) {
                        #expect(next.offset > previous.offset && next.offset <= previous.offset + previous.count)
                    }
                    #expect(chunks.dropLast().enumerated().allSatisfy { $0.element.offset == $0.offset * width })
                }
            }
        }
    }

    @Test func exportBoundsAreRequiredForBucketedAndEnforced() throws {
        let bounds = LanguageExportBounds(maximumPrefillChunk: 128, maximumCacheCapacity: 4096)
        #expect(try LanguageStepContract.bucketedPositions.admit(bounds: bounds, maximumContextLength: 4096) == 128)
        #expect(throws: LanguageGenerationError.contextLimit(limit: 4096)) {
            try LanguageStepContract.bucketedPositions.admit(bounds: bounds, maximumContextLength: 4097)
        }
        #expect(throws: LanguageGenerationError.invalid("A bucketed asset must declare its export bounds")) {
            try LanguageStepContract.bucketedPositions.admit(bounds: nil, maximumContextLength: 2048)
        }
        #expect(throws: LanguageGenerationError.invalid("Invalid export bounds")) {
            try LanguageStepContract.bucketedPositions.admit(bounds: LanguageExportBounds(maximumPrefillChunk: 0, maximumCacheCapacity: 4096),
                                                             maximumContextLength: 2048)
        }
        // Running assets keep working without bounds, and honour them when declared.
        #expect(try LanguageStepContract.runningPositions.admit(bounds: nil, maximumContextLength: 16384) == nil)
        #expect(throws: LanguageGenerationError.contextLimit(limit: 4096)) {
            try LanguageStepContract.runningPositions.admit(bounds: bounds, maximumContextLength: 8192)
        }
    }

    @Test func recyclingDefaultsFollowShapeGrowth() {
        #expect(LanguageStepContract.runningPositions.defaultMaximumSubmissionsPerModel == 128)
        #expect(LanguageStepContract.bucketedPositions.defaultMaximumSubmissionsPerModel == nil)
    }

    @Test func bucketedCapacityTracksTheRealLength() throws {
        #expect(try LanguageCachePlan.bucketedCapacity(needed: 1, maximum: 2048) == 256)
        #expect(try LanguageCachePlan.bucketedCapacity(needed: 256, maximum: 2048) == 256)
        #expect(try LanguageCachePlan.bucketedCapacity(needed: 257, maximum: 2048) == 512)
        #expect(try LanguageCachePlan.bucketedCapacity(needed: 1001, maximum: 2048) == 1024)
        #expect(try LanguageCachePlan.bucketedCapacity(needed: 1025, maximum: 2048) == 1280)
        #expect(try LanguageCachePlan.bucketedCapacity(needed: 2000, maximum: 2000) == 2000)   // the maximum caps the last step
        #expect(throws: LanguageGenerationError.contextLimit(limit: 2048)) { try LanguageCachePlan.bucketedCapacity(needed: 2049, maximum: 2048) }
    }

    /// Over a full 16K context, the bucketed shapes are bounded by the prefill sizes times the
    /// capacity steps, however many tokens are decoded; decoding adds exactly one shape per
    /// 256 tokens it crosses (the running ladder would instead attend up to twice the length).
    /// The prefill runs at the prompt's capacity, so its early chunks attend over rows not yet
    /// written (masked); decoding never attends over more than a granularity step of slack.
    @Test func bucketedShapeVocabularyIsBoundedAndGrowsWithLength() throws {
        let prompt = LanguageStepContract.bucketedPositions.prefillChunks(promptCount: 1000, requestedChunk: nil, maximumChunk: 512)
            .map { (position: $0.offset, count: $0.count) }
        let decode = (1000..<16384).map { (position: $0, count: 1) }
        var promptShapes = Set<[Int]>(), shapes = Set<[Int]>()
        // The request starts at its prefill's capacity.
        var capacity = try LanguageCachePlan.requestCapacity(allocated: 256, initial: 256, resets: true, prefillTokens: 1000,
                                                             maximum: 16384)
        var decodeShapes = 0
        var worstSlack = 0
        for (index, step) in (prompt + decode).enumerated() {
            // Caches only grow within a request.
            capacity = max(capacity, try LanguageCachePlan.bucketedCapacity(needed: step.position + step.count, maximum: 16384))
            if index < prompt.count { promptShapes.insert([step.count, capacity]) } else {
                worstSlack = max(worstSlack, capacity - (step.position + step.count))
            }
            if shapes.insert([step.count, capacity]).inserted && index >= prompt.count { decodeShapes += 1 }
        }
        #expect(decode.last.map { $0.position + $0.count } == 16384)
        #expect(decodeShapes == (16384 - 1024) / LanguageCachePlan.bucketedGranularity + 1)
        // One prefill width at one capacity: a single prefill shape.
        #expect(promptShapes == [[256, 1024]])
        #expect(shapes.count == decodeShapes + promptShapes.count)
        #expect(worstSlack < LanguageCachePlan.bucketedGranularity)
    }

    @Test func bucketedRequestsStartAtTheirPrefillCapacity() throws {
        let capacity = { (allocated: Int, initial: Int, resets: Bool, prefill: Int, maximum: Int) in
            try LanguageCachePlan.requestCapacity(allocated: allocated, initial: initial, resets: resets, prefillTokens: prefill,
                                                  maximum: maximum)
        }
        // A prompt within the initial capacity starts there, whatever the previous request left (the 0.0.2 plan).
        #expect(try capacity(2048, 256, true, 12, 4096) == 256)
        #expect(try capacity(256, 256, true, 256, 4096) == 256)
        // A longer prompt starts at its own rows rounded up to the granularity, never at a previous request's larger capacity.
        #expect(try capacity(256, 256, true, 257, 4096) == 512)
        #expect(try capacity(256, 256, true, 2048, 4096) == 2048)
        #expect(try capacity(4096, 256, true, 1000, 4096) == 1024)
        // A larger initial capacity is kept; the maximum caps the last step; a running session keeps its allocation.
        #expect(try capacity(1024, 1024, true, 300, 4096) == 1024)
        #expect(try capacity(256, 256, true, 2000, 2000) == 2000)
        #expect(try capacity(2048, 256, false, 12, 4096) == 2048)
        // No prefill tokens: the 0.0.2 start, the initial capacity (a two-function asset's plan).
        #expect(try capacity(2048, 256, true, 0, 4096) == 256)
        #expect(try capacity(2048, 128, true, 0, 4096) == 128)
        #expect(LanguageStepContract.bucketedPositions.fixedPrefillChunk == 64)
        #expect(LanguageStepContract.runningPositions.fixedPrefillChunk == nil)
        #expect(throws: LanguageGenerationError.contextLimit(limit: 2048)) { try capacity(256, 256, true, 2049, 2048) }
    }

    @Test func commitAdvancesOnlyAfterTheCacheWrite() async throws {
        actor Order { var events: [String] = []; func add(_ e: String) { events.append(e) } }
        let order = Order()
        let next = try await LanguageCachePlan.commit(position: 10, count: 4) { await order.add("write") }
        #expect(next == 14)
        #expect(await order.events == ["write"])
        struct WriteFailure: Error {}
        var materialized = 14
        await #expect(throws: WriteFailure.self) {
            materialized = try await LanguageCachePlan.commit(position: 14, count: 1) { throw WriteFailure() }
        }
        #expect(materialized == 14)
    }
}
