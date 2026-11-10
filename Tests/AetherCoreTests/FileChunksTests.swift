import CryptoKit
import Darwin
import Foundation
import Testing
@testable import AetherCore

/// Hashing a large file must not hold its chunks: each read's autoreleased buffer is freed before the next read.
@Suite("File chunks")
struct FileChunksTests {
    /// The process's `phys_footprint`, the figure the system's memory limits act on.
    static func footprint() -> Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return result == KERN_SUCCESS ? Int64(info.phys_footprint) : 0
    }

    @Test func aLargeFileHashesInBoundedMemoryAndEveryByteOnce() throws {
        let url = FileManager.default.temporaryDirectory.appending(path: "aether-file-chunks-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        let block = Data((0 ..< (1 << 20)).map { UInt8(truncatingIfNeeded: $0 &* 31) })
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let writer = try FileHandle(forWritingTo: url)
        for _ in 0 ..< 256 { try writer.write(contentsOf: block) }     // 256 MiB
        try writer.close()

        var expected = SHA256()
        for _ in 0 ..< 256 { expected.update(data: block) }

        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let before = Self.footprint()
        var peak = before, bytes = 0, hasher = SHA256()
        try FileChunks.forEach(handle) { chunk in
            hasher.update(data: chunk)
            bytes += chunk.count
            peak = max(peak, Self.footprint())
        }
        #expect(bytes == 256 << 20)
        #expect(Data(hasher.finalize()) == Data(expected.finalize()))
        // Without the pools the 32 reads (8 MiB each) stay resident until the loop ends: about 256 MiB.
        #expect(peak - before < 64 << 20, "footprint grew \(peak - before) bytes")
    }
}
