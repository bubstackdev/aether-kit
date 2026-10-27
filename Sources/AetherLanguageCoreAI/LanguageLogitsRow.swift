import Accelerate

/// Reading a step's logits for the sampler. Pure, so CPU tests check it exactly.
enum LanguageLogitsRow {
    /// The last row of [1, rows, count] fp16 logits as Float. Exact: every fp16 value, NaN and the infinities included,
    /// is representable as a Float. A contiguous row converts with vImage into a new array instead of element by element
    /// (about 6 µs instead of 0.26 ms for a 151,936-row head).
    static func lastRow(_ pointer: UnsafePointer<Float16>, rows: Int, rowStride: Int, count: Int, stride: Int) -> [Float] {
        let start = pointer + (rows - 1) * rowStride
        guard stride == 1, count > 0 else { return (0..<count).map { Float(start[$0 * stride]) } }
        return [Float](unsafeUninitializedCapacity: count) { row, initialized in
            var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: start), height: 1,
                                       width: vImagePixelCount(count), rowBytes: count * MemoryLayout<Float16>.size)
            var destination = vImage_Buffer(data: row.baseAddress!, height: 1, width: vImagePixelCount(count),
                                            rowBytes: count * MemoryLayout<Float>.size)
            if vImageConvert_Planar16FtoPlanarF(&source, &destination, vImage_Flags(kvImageNoFlags)) != kvImageNoError {
                for i in 0..<count { row[i] = Float(start[i]) }
            }
            initialized = count
        }
    }
}
