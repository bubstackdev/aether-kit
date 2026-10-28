import Foundation

/// Pure capacity and row-copy rules shared by the native adapter and CPU tests.
@available(macOS 27, iOS 27, *)
enum LanguageCachePlan {
    static func nextCapacity(current: Int, needed: Int, maximum: Int) throws -> Int {
        guard current > 0, needed >= 0, maximum >= current, needed <= maximum else {
            throw LanguageGenerationError.contextLimit(limit: maximum)
        }
        var capacity = current
        while capacity < needed {
            if capacity >= 4096 {
                // Large KV caches grow in 1024-token increments to bound the
                // transient old-plus-new allocation during a copy.
                capacity += min(1024, maximum - capacity)
            } else {
                capacity = capacity > maximum / 2 ? maximum : capacity * 2
            }
        }
        return capacity
    }
    /// Growth step of a bucketed cache. Bucketed attention reads (and concatenates) every allocated
    /// row each step, so the allocation tracks the real length in these steps instead of the
    /// doubling ladder. A strided view of a larger buffer is not used: Core AI copies non-contiguous
    /// inputs every step, which measured 3-8x slower decode.
    static let bucketedGranularity = 256

    /// Capacity for `needed` rows of a bucketed cache: the real length rounded up to the
    /// granularity, within the maximum.
    static func bucketedCapacity(needed: Int, maximum: Int) throws -> Int {
        guard needed > 0, needed <= maximum else { throw LanguageGenerationError.contextLimit(limit: maximum) }
        return min(maximum, (needed + bucketedGranularity - 1) / bucketedGranularity * bucketedGranularity)
    }

    /// Commits a joined step: the cache rows are written first and only then does the materialized
    /// length advance, so a failed write leaves the session at its previous, consistent position.
    static func commit(position: Int, count: Int, write: () async throws -> Void,
                       isolation: isolated (any Actor)? = #isolation) async throws -> Int {
        try await write()
        return position + count
    }

    /// Capacity a new request starts from: bucketed sessions return to the initial capacity
    /// (their per-step cost follows the allocation they attend over); running sessions keep theirs.
    static func requestCapacity(allocated: Int, initial: Int, resets: Bool) -> Int {
        resets ? min(allocated, initial) : allocated
    }

    /// Writes a bucketed step's `[layers, batch, heads, count, width]` update (any strides) into a
    /// contiguous `[layers·batch·heads, capacity, width]` cache at `start..<start + count`. Rows
    /// whose strides nest are copied as one run of rows; otherwise each (layer, batch, head) row
    /// is placed separately.
    static func writeUpdateTensor<Scalar>(_ update: UnsafePointer<Scalar>, shape: [Int], strides: [Int],
                                          into cache: UnsafeMutablePointer<Scalar>, capacity: Int, start: Int) {
        precondition(shape.count == 5 && strides.count == 5)
        let (layers, batch, heads, count, width) = (shape[0], shape[1], shape[2], shape[3], shape[4])
        if strides[0] == strides[1] * batch && strides[1] == strides[2] * heads {
            writeUpdate(update, rowStride: strides[2], tokenStride: strides[3], elementStride: strides[4],
                        into: cache, rows: layers * batch * heads, capacity: capacity, width: width, start: start, count: count)
            return
        }
        for l in 0..<layers { for b in 0..<batch { for h in 0..<heads {
            let row = (l * batch + b) * heads + h
            writeUpdate(update + l * strides[0] + b * strides[1] + h * strides[2], rowStride: 0,
                        tokenStride: strides[3], elementStride: strides[4], into: cache + row * capacity * width,
                        rows: 1, capacity: capacity, width: width, start: start, count: count)
        } } }
    }

    static func strides(_ shape: [Int]) -> [Int] {
        var strides = Array(repeating: 1, count: shape.count)
        for index in stride(from: shape.count - 2, through: 0, by: -1) { strides[index] = strides[index + 1] * shape[index + 1] }
        return strides
    }
    /// Copies a bucketed step's update rows into a contiguous `[rows, capacity, width]` cache at
    /// token positions `start..<start + count`. `update` is `[rows, count, width]` addressed by
    /// `rowStride`, `tokenStride` and `elementStride` (elements), so any output layout works.
    static func writeUpdate<Scalar>(_ update: UnsafePointer<Scalar>, rowStride: Int, tokenStride: Int, elementStride: Int,
                                    into cache: UnsafeMutablePointer<Scalar>, rows: Int, capacity: Int, width: Int,
                                    start: Int, count: Int) {
        precondition(rows > 0 && width > 0 && count > 0 && start >= 0 && start + count <= capacity)
        if elementStride == 1, tokenStride == width {
            // Each row's tokens are contiguous on both sides: one copy per row instead of one per token.
            for row in 0..<rows {
                (cache + (row * capacity + start) * width).update(from: update + row * rowStride, count: count * width)
            }
            return
        }
        for row in 0..<rows {
            for token in 0..<count {
                let source = update + row * rowStride + token * tokenStride
                let destination = cache + (row * capacity + start + token) * width
                if elementStride == 1 {
                    destination.update(from: source, count: width)
                } else {
                    for element in 0..<width { destination[element] = source[element * elementStride] }
                }
            }
        }
    }
    /// A grown cache in one pass: each row's first `validTokens` from `source`, zeros for the rest of its new capacity,
    /// and zeros past the last row (`destinationBytes`). Every byte is the same as zeroing the whole buffer and then
    /// copying the prefixes, without writing the copied bytes twice; large caches are split across cores by rows.
    static func relayout(from source: UnsafeRawPointer, to destination: UnsafeMutableRawPointer, destinationBytes: Int,
                         rows: Int, oldCapacity: Int, newCapacity: Int, validTokens: Int, tokenBytes: Int) {
        precondition(rows > 0 && tokenBytes > 0 && validTokens >= 0 && validTokens <= oldCapacity && oldCapacity <= newCapacity)
        let rowBytes = newCapacity * tokenBytes
        precondition(destinationBytes >= rows * rowBytes)
        nonisolated(unsafe) let source = source, destination = destination
        let blocks = rows * rowBytes >= 8 << 20 ? min(rows, ProcessInfo.processInfo.activeProcessorCount) : 1
        DispatchQueue.concurrentPerform(iterations: blocks) { block in
            for row in stride(from: block, to: rows, by: blocks) {
                let target = destination.advanced(by: row * rowBytes)
                target.copyMemory(from: source.advanced(by: row * oldCapacity * tokenBytes), byteCount: validTokens * tokenBytes)
                target.advanced(by: validTokens * tokenBytes).initializeMemory(as: UInt8.self, repeating: 0,
                                                                               count: (newCapacity - validTokens) * tokenBytes)
            }
        }
        destination.advanced(by: rows * rowBytes).initializeMemory(as: UInt8.self, repeating: 0, count: destinationBytes - rows * rowBytes)
    }
}
