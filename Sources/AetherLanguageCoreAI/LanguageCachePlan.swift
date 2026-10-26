import Foundation

/// Pure capacity and row-copy rules shared by the native adapter and CPU tests.
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
    static func strides(_ shape: [Int]) -> [Int] {
        var strides = Array(repeating: 1, count: shape.count)
        for index in stride(from: shape.count - 2, through: 0, by: -1) { strides[index] = strides[index + 1] * shape[index + 1] }
        return strides
    }
    static func copyPrefix(from source: UnsafeRawPointer, to destination: UnsafeMutableRawPointer,
                           rows: Int, oldCapacity: Int, newCapacity: Int, validTokens: Int, tokenBytes: Int) {
        precondition(rows > 0 && tokenBytes > 0 && validTokens >= 0 && validTokens <= oldCapacity && oldCapacity <= newCapacity)
        for row in 0..<rows {
            destination.advanced(by: row * newCapacity * tokenBytes).copyMemory(
                from: source.advanced(by: row * oldCapacity * tokenBytes), byteCount: validTokens * tokenBytes)
        }
    }
}
