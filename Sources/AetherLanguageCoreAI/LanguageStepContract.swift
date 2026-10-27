import AetherKit

/// How a language asset's `main` function addresses its KV cache. Detected from the function
/// signature, so registered bundles need no manifest change. Pure rules shared by the native
/// session and CPU tests.
public enum LanguageStepContract: String, Sendable, Equatable {
    /// Inputs `input_ids` [1, q] and `position_ids` [1, total] = 0..<total; states `keyCache` and
    /// `valueCache`, written in place; output `logits` [1, q, vocab]. The 0.0.1 and Apple LLM export
    /// contract. Every decode step is a new input shape, so the session recycles native model
    /// ownership after a bounded number of submissions.
    case runningPositions = "running-positions"
    /// Inputs `input_ids` [1, q], `cache_positions` [1, q] = start..<start+q, and read-only
    /// `keyCache`/`valueCache`; outputs `logits` [1, 1, vocab] and the chunk's `keyUpdate` and
    /// `valueUpdate` rows [layers, 1, kv heads, q, head dim], which the session copies into its
    /// cache at start..<start+q after the native join. No state and no position-dependent shape,
    /// so input shapes depend only on (q, cache capacity): q stays in powers of two up to the
    /// declared `LanguageExportBounds.maximumPrefillChunk`, and the cache capacity grows in
    /// `LanguageCachePlan.bucketedGranularity` steps up to the context, which bounds native shape
    /// caches without recycling. Attention reads and concatenates every cache row each step, so
    /// per-step cost grows with the context faster than the running contract's.
    case bucketedPositions = "bucketed-positions"

    /// Checks the declared export bounds against the requested context before any native work:
    /// a bucketed asset must declare them, and no contract may register a context beyond the
    /// exported cache capacity. Returns the prefill chunk limit to plan with (nil: unbounded).
    func admit(bounds: LanguageExportBounds?, maximumContextLength: Int) throws -> Int? {
        if let bounds {
            guard bounds.isValid else { throw LanguageGenerationError.invalid("Invalid export bounds") }
            guard maximumContextLength <= bounds.maximumCacheCapacity else {
                throw LanguageGenerationError.contextLimit(limit: bounds.maximumCacheCapacity)
            }
            return bounds.maximumPrefillChunk
        }
        guard self == .runningPositions else {
            throw LanguageGenerationError.invalid("A bucketed asset must declare its export bounds")
        }
        return nil
    }

    /// The name of the position input that identifies this contract.
    public var positionInputName: String {
        switch self {
        case .runningPositions: "position_ids"
        case .bucketedPositions: "cache_positions"
        }
    }

    /// Whether the KV caches are Core AI states (written in place) or read-only inputs whose
    /// update rows come back as outputs.
    public var cachesAreStates: Bool { self == .runningPositions }

    /// Native recycling default: bounded for the running contract, none for the bucketed one.
    public var defaultMaximumSubmissionsPerModel: Int? {
        switch self {
        case .runningPositions: 128
        case .bucketedPositions: nil
        }
    }

    /// Identifies the contract from the `main` function's names and logits shape (`-1` for a
    /// dynamic dimension). Returns nil for anything else.
    static func detect(inputNames: Set<String>, outputNames: Set<String>, stateNames: Set<String>,
                       logitsShape: [Int]) -> Self? {
        guard logitsShape.count == 3, logitsShape[0] == 1, logitsShape[2] > 0 else { return nil }
        if inputNames == ["input_ids", "position_ids"], outputNames == ["logits"],
           stateNames == ["keyCache", "valueCache"], logitsShape[1] == -1 { return .runningPositions }
        let bucketedInputs: Set<String> = ["input_ids", "cache_positions", "keyCache", "valueCache"]
        if inputNames == bucketedInputs || inputNames == bucketedInputs.union(["rope_positions"]),
           outputNames == ["logits", "keyUpdate", "valueUpdate"], stateNames.isEmpty, logitsShape[1] == 1 {
            return .bucketedPositions
        }
        return nil
    }

    /// The image submission bound of an asset with image input (its `withImage` q bound), so an
    /// oversized image is refused before Core AI. Nil for assets without image input, and for an
    /// image asset that declares no bound: it still opens for text-only use, and its image prompts
    /// are refused while planning (``LanguagePromptPlan/requireImageSpans(within:)``).
    static func imageChunkLimit(acceptsImages: Bool, bounds: LanguageExportBounds?) -> Int? {
        guard acceptsImages, let limit = bounds?.maximumImageChunk, limit > 0 else { return nil }
        return limit
    }

    /// Whether a bucketed function also takes 3-axis rotary positions (`rope_positions` [3, q],
    /// Qwen3-VL's M-RoPE), separate from the cache positions that address rows and the mask.
    static func takesRopePositions(_ inputNames: Set<String>) -> Bool { inputNames.contains("rope_positions") }

    /// The image entry point of a bucketed asset: the step inputs plus image rows and DeepStack
    /// features, with the same outputs.
    static func isImageStep(inputNames: Set<String>, outputNames: Set<String>, stateNames: Set<String>) -> Bool {
        inputNames == ["input_ids", "cache_positions", "rope_positions", "image_embeds", "deepstack", "keyCache", "valueCache"]
            && outputNames == ["logits", "keyUpdate", "valueUpdate"] && stateNames.isEmpty
    }

    /// Position values for submitting `count` tokens at cache `position`.
    func positions(at position: Int, count: Int) -> [Int32] {
        switch self {
        case .runningPositions: (0..<(position + count)).map(Int32.init)
        case .bucketedPositions: (position..<(position + count)).map(Int32.init)
        }
    }

    /// Rows of logits one submission of `count` tokens returns.
    func logitsRows(count: Int) -> Int {
        switch self {
        case .runningPositions: count
        case .bucketedPositions: 1
        }
    }

    /// The bucketed contract's prefill chunk when the request names none (within the export bound).
    static let defaultBucketedPrefillChunk = 64

    /// Submissions for a fresh-cache prefill of a `promptCount`-token run, never above `maximumChunk`
    /// (the export bound, when declared). The running contract keeps 0.0.1 behaviour: uniform chunks
    /// of `requestedChunk` (whole prompt when nil) and a shorter tail. The bucketed contract submits
    /// one width C (`requestedChunk`, default ``defaultBucketedPrefillChunk``, within the bound) so a
    /// warm request switches shape only between C and the 1-token decode step: Core AI retains memory
    /// on every shape change. A run that is not a multiple of C ends with a C-token chunk over its
    /// last C tokens, recomputing up to C - 1 already written rows. That is the same computation:
    /// the graph's cache mask admits only rows before the chunk's first position
    /// (`key_positions < cache_positions[0]`), so the stepped-over rows are invisible to the chunk,
    /// are recomputed from the same prefix, and are overwritten with the chunk's own keys and
    /// values (equal up to floating-point rounding). A run shorter than C is one chunk of its own
    /// length.
    func prefillChunks(promptCount: Int, requestedChunk: Int?, maximumChunk: Int?) -> [LanguagePrefillChunk] {
        precondition(promptCount > 0 && (requestedChunk ?? 1) > 0 && (maximumChunk ?? 1) > 0)
        switch self {
        case .runningPositions:
            let width = min(requestedChunk ?? promptCount, promptCount, maximumChunk ?? .max)
            return stride(from: 0, to: promptCount, by: width).map { LanguagePrefillChunk(offset: $0, count: min(width, promptCount - $0)) }
        case .bucketedPositions:
            let width = min(requestedChunk ?? Self.defaultBucketedPrefillChunk, maximumChunk ?? .max)
            guard promptCount > width else { return [LanguagePrefillChunk(offset: 0, count: promptCount)] }
            var chunks = stride(from: 0, through: promptCount - width, by: width).map { LanguagePrefillChunk(offset: $0, count: width) }
            if promptCount % width != 0 { chunks.append(LanguagePrefillChunk(offset: promptCount - width, count: width)) }
            return chunks
        }
    }

    static func floorPowerOfTwo(_ value: Int) -> Int {
        precondition(value > 0)
        return 1 << (Int.bitWidth - 1 - value.leadingZeroBitCount)
    }
}

/// One prefill submission: `count` tokens at `offset` from the run's start. Bucketed offsets can step
/// back over rows already written (the run's last chunk).
struct LanguagePrefillChunk: Sendable, Equatable {
    let offset: Int
    let count: Int
}
