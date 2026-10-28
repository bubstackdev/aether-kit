import Foundation
import MLX

/// Requests over one loaded Qwen3, one at a time: teacher-forced logits (the qualification protocol) and greedy decoding.
/// MLX arrays aren't Sendable; every request runs to completion on the caller's task.
public final class MLXQwen3Runner: @unchecked Sendable {
    public let model: Qwen3
    public var vocabularySize: Int { model.configuration.vocabularySize }

    public init(directory: URL, dtype: DType = .float16) throws {
        model = try Qwen3.load(directory: directory, dtype: dtype)
    }

    /// Last-row logits after the prompt's prefill, then after each continuation token (a fresh cache).
    public func teacherForcedLogits(prompt: [Int], continuation: [Int]) -> [[Float]] {
        let caches = model.caches()
        var rows: [[Float]] = []
        var logits = model.logits(MLXArray(prompt.map(Int32.init)).reshaped(1, -1), caches: caches, lastOnly: true)
        rows.append(logits.reshaped(-1).asType(.float32).asArray(Float.self))
        for token in continuation {
            logits = model.logits(MLXArray([Int32(token)]).reshaped(1, 1), caches: caches, lastOnly: true)
            rows.append(logits.reshaped(-1).asType(.float32).asArray(Float.self))
        }
        return rows
    }

    public struct Greedy: Sendable {
        public let tokens: [Int]
        public let timeToFirstTokenSeconds: Double
        public let totalSeconds: Double
    }

    /// Greedy decoding: the whole prompt in one prefill, then one token per step.
    public func greedy(prompt: [Int], maximumTokens: Int, stopTokenIDs: Set<Int>) -> Greedy {
        let start = ContinuousClock.now
        let caches = model.caches()
        var logits = model.logits(MLXArray(prompt.map(Int32.init)).reshaped(1, -1), caches: caches, lastOnly: true)
        var next = argMax(logits.reshaped(-1)).item(Int.self)
        let first = seconds(since: start)
        var tokens: [Int] = []
        while tokens.count < maximumTokens, !stopTokenIDs.contains(next) {
            tokens.append(next)
            guard tokens.count < maximumTokens else { break }
            logits = model.logits(MLXArray([Int32(next)]).reshaped(1, 1), caches: caches, lastOnly: true)
            next = argMax(logits.reshaped(-1)).item(Int.self)
        }
        return Greedy(tokens: tokens, timeToFirstTokenSeconds: first, totalSeconds: seconds(since: start))
    }

    private func seconds(since start: ContinuousClock.Instant) -> Double {
        let elapsed = ContinuousClock.now - start
        return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    }
}
