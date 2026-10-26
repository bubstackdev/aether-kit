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
        if inputNames == ["input_ids", "cache_positions", "keyCache", "valueCache"],
           outputNames == ["logits", "keyUpdate", "valueUpdate"], stateNames.isEmpty, logitsShape[1] == 1 {
            return .bucketedPositions
        }
        return nil
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

    /// Submission sizes for a fresh-cache prefill, never above `maximumChunk` (the export bound, when
    /// declared). The running contract keeps 0.0.1 behaviour: uniform chunks of `requestedChunk`
    /// (whole prompt when nil) and a shorter tail. The bucketed contract takes the largest power of
    /// two that fits both the remaining tokens and the chunk limit (`requestedChunk` and the bound,
    /// rounded down to a power of two), so submissions take at most log2(bound) + 1 sizes and none
    /// is padded.
    func prefillChunks(promptCount: Int, requestedChunk: Int?, maximumChunk: Int?) -> [Int] {
        precondition(promptCount > 0 && (requestedChunk ?? 1) > 0 && (maximumChunk ?? 1) > 0)
        switch self {
        case .runningPositions:
            let width = min(requestedChunk ?? promptCount, promptCount, maximumChunk ?? .max)
            return stride(from: 0, to: promptCount, by: width).map { min(width, promptCount - $0) }
        case .bucketedPositions:
            let bound = maximumChunk ?? promptCount
            let limit = Self.floorPowerOfTwo(min(requestedChunk ?? bound, bound))
            var chunks: [Int] = []
            var remaining = promptCount
            while remaining > 0 {
                let chunk = min(limit, Self.floorPowerOfTwo(remaining))
                chunks.append(chunk)
                remaining -= chunk
            }
            return chunks
        }
    }

    static func floorPowerOfTwo(_ value: Int) -> Int {
        precondition(value > 0)
        return 1 << (Int.bitWidth - 1 - value.leadingZeroBitCount)
    }
}
