import AetherAssets
import Darwin
import Foundation
import Testing

private struct ExternalFixture {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    var file: URL { root.appendingPathComponent("source") }
    init() throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data(0..<16).write(to: file)
    }
    func cleanup() { try? FileManager.default.removeItem(at: root) }
}

@Test func externalLeasePinsOriginalBytesAcrossReplacementAndUnlink() throws {
    let fixture = try ExternalFixture(); defer { fixture.cleanup() }
    let lease = try ExternalFileLease(files: [fixture.file, fixture.file])
    let moved = fixture.root.appendingPathComponent("old")
    try FileManager.default.moveItem(at: fixture.file, to: moved)
    try Data(repeating: 99, count: 16).write(to: fixture.file)
    try FileManager.default.removeItem(at: moved)
    var bytes = Data(count: 4)
    try bytes.withUnsafeMutableBytes { try lease.read(into: $0, from: fixture.file, offset: 3) }
    #expect(bytes == Data(3..<7))
    #expect(try lease.byteCount(of: fixture.file) == 16)
}

@Test func externalLeaseChecksCoverageRangesAndInPlaceChanges() throws {
    let fixture = try ExternalFixture(); defer { fixture.cleanup() }
    let lease = try ExternalFileLease(files: [fixture.file])
    var bytes = Data(count: 4)
    #expect(throws: ExternalFileError.invalidRange) {
        try bytes.withUnsafeMutableBytes { try lease.read(into: $0, from: fixture.file, offset: UInt64.max) }
    }
    #expect(throws: ExternalFileError.notLeased) { try lease.byteCount(of: fixture.root.appendingPathComponent("other")) }
    let writer = try FileHandle(forWritingTo: fixture.file)
    try writer.truncate(atOffset: 1); try writer.close()
    #expect(throws: ExternalFileError.fileChanged) {
        try bytes.withUnsafeMutableBytes { try lease.read(into: $0, from: fixture.file, offset: 0) }
    }
}

@Test func externalLeaseReleasesSharedLocksAndRollsBackPartialAcquisition() throws {
    let fixture = try ExternalFixture(); defer { fixture.cleanup() }
    let fd = open(fixture.file.path, O_RDONLY)
    #expect(fd >= 0)
    defer { close(fd) }
    do {
        let lease = try ExternalFileLease(files: [fixture.file])
        #expect(flock(fd, LOCK_EX | LOCK_NB) == -1)
        #expect(errno == EWOULDBLOCK)
        withExtendedLifetime(lease) {}
    }
    #expect(flock(fd, LOCK_EX | LOCK_NB) == 0)
    flock(fd, LOCK_UN)
    #expect(throws: ExternalFileError.self) { try ExternalFileLease(files: [fixture.file, fixture.root]) }
    #expect(flock(fd, LOCK_EX | LOCK_NB) == 0)
    flock(fd, LOCK_UN)
}

@Test func externalLeaseAllowsSnapshotAliasesButRejectsUnsupportedSources() throws {
    let fixture = try ExternalFixture(); defer { fixture.cleanup() }
    let alias = fixture.root.appendingPathComponent("snapshot")
    try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: fixture.file)
    let lease = try ExternalFileLease(files: [alias])
    try FileManager.default.removeItem(at: alias)
    try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: fixture.root)
    var bytes = Data(count: 2)
    try bytes.withUnsafeMutableBytes { try lease.read(into: $0, from: alias, offset: 1) }
    #expect(bytes == Data([1, 2]))
    #expect(throws: ExternalFileError.invalidFile) { try ExternalFileLease(files: []) }
    #expect(throws: ExternalFileError.invalidFile) { try ExternalFileLease(files: [URL(string: "https://example.invalid/model")!]) }
}
