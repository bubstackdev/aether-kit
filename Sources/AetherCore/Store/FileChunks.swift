import Foundation

/// A file read front to back in fixed chunks, each read and its use inside its own autorelease pool. On Apple platforms
/// `FileHandle.read(upToCount:)` returns autoreleased buffers, and a loop that never returns to a run loop keeps every
/// one of them until it ends: hashing a 94 GB bundle reached 13.5 GB resident before the process was killed.
enum FileChunks {
    static let size = 8 << 20

    /// Calls `body` with each chunk of `handle`, in order, until the end of the file. Errors from reading and from `body`
    /// are thrown as they are.
    static func forEach(_ handle: FileHandle, size: Int = size, _ body: (Data) throws -> Void) throws {
        while try autoreleasepool(invoking: { () throws -> Bool in
            guard let chunk = try handle.read(upToCount: size), !chunk.isEmpty else { return false }
            try body(chunk)
            return true
        }) {}
    }
}
