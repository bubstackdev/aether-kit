/// Between-request accounting of the native memory a loaded model keeps after its requests.
///
/// With reshape-enabled specialization, Core AI keeps Metal allocations every time a function's
/// input shape changes (sized by the shape it changes to) until the model is released. Every
/// request changes shapes (prefill chunks, then 1-token steps), so a long-lived session grows by a
/// similar amount per request even when every shape is already warm. The session measures Metal's
/// allocated size across each request (from GPU admission to the end, net of its own KV cache
/// changes) and records it here; once the counted growth passes the limit, the next request first
/// releases and reloads the model, which returns every retained allocation.
///
/// Metal's allocated size is process-wide, so a window can also see other sessions' loads, unloads
/// and allocations. Sharing a ``GPUAdmission`` keeps other admitted GPU work out of the window,
/// but not other sessions' loading or releasing, so a request's growth is classified:
/// - **warming**: the request submitted a function or (count, capacity) shape for the first time
///   since the load. Its growth is the one-time weight copy or specialization, not retention.
/// - **foreign**: a net decrease (the session frees nothing of its own inside a window, so
///   someone else did), or growth far above this model's own retention: more than half the limit
///   and more than ``outlierFactor`` times the expected growth for the request's submitted tokens,
///   once ``minimumSamples`` counted requests have established the per-token rate. Not counted.
/// - **counted**: everything else, including growth before a rate is known (an early foreign
///   spike costs one unnecessary reload; ignoring real retention would let it grow unbounded).
struct LanguageMemoryBudget: Sendable, Equatable {
    enum Disposition: Sendable, Equatable { case warming, foreign, counted }

    /// nil never reloads.
    let limitBytes: Int?
    private(set) var retainedBytes = 0
    /// Growth set aside as another allocator's, since the session opened.
    private(set) var foreignRequests = 0
    private(set) var foreignBytes = 0
    /// Recent counted growth per submitted token, newest last.
    private var rates: [Double] = []

    static let outlierFactor = 4.0
    static let minimumSamples = 3
    static let rateHistory = 8

    init(limitBytes: Int?) { self.limitBytes = limitBytes }

    @discardableResult
    mutating func record(requestGrowthBytes growth: Int, submittedTokens: Int, warming: Bool) -> Disposition {
        guard !warming else { return .warming }
        if growth < 0 || isOutlier(growth, submittedTokens: submittedTokens) {
            foreignRequests += 1
            foreignBytes += abs(growth)
            return .foreign
        }
        retainedBytes += growth
        if submittedTokens > 0 {
            rates.append(Double(growth) / Double(submittedTokens))
            if rates.count > Self.rateHistory { rates.removeFirst() }
        }
        return .counted
    }

    private func isOutlier(_ growth: Int, submittedTokens: Int) -> Bool {
        guard let limitBytes, growth > limitBytes / 2, rates.count >= Self.minimumSamples, submittedTokens > 0 else { return false }
        let median = rates.sorted()[rates.count / 2]
        return Double(growth) > Self.outlierFactor * max(median, 0) * Double(submittedTokens)
    }

    /// The limit is passed: reload before the next request.
    var exhausted: Bool { limitBytes.map { retainedBytes > $0 } ?? false }

    /// After a release and reload: nothing is retained. The learned rate is the model's and stays.
    mutating func reset() { retainedBytes = 0 }
}
