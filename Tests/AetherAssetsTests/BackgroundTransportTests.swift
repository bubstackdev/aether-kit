@testable import AetherAssets
import Foundation
import Testing

/// The background session itself needs an app (and, for suspension, a device); the opt-in qualification covers it.
/// These tests cover what a finished background task does with its bytes, which runs the same with or without a
/// waiting fetch.
struct BackgroundTransportTests {
    private let directory = FileManager.default.temporaryDirectory.appendingPathComponent("background-\(UUID().uuidString)")
    private let data = Data((0..<600).map { UInt8($0 % 241) })

    private func response(_ status: Int, _ headers: [String: String] = [:]) -> HTTPURLResponse {
        HTTPURLResponse(url: URL(string: "https://example.com/f")!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
    }

    private func downloaded(_ bytes: Data) throws -> URL {
        let url = directory.appendingPathComponent("tmp-\(UUID().uuidString)")
        try bytes.write(to: url)
        return url
    }

    @Test func recordsSurviveTheirTaskDescriptionAndStayHomeRelative() {
        let home = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Store/downloads/x/files/f.bin").path
        let record = BackgroundTaskRecord(destination: BackgroundTaskRecord.portablePath(home), offset: 5, byteCount: 9, path: "f.bin")
        #expect(record.destination == "Library/Store/downloads/x/files/f.bin")
        #expect(record.destinationURL.path == URL(fileURLWithPath: home).standardizedFileURL.path)
        #expect(BackgroundTaskRecord(encoded: record.encoded) == record)
        #expect(BackgroundTaskRecord(encoded: "not json") == nil)
        #expect(BackgroundTaskRecord.portablePath("/elsewhere/f.bin") == "/elsewhere/f.bin")
    }

    @Test func aMatchingPartialResponseIsAppended() throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let target = directory.appendingPathComponent("f.bin")
        try data.prefix(200).write(to: target)
        let record = BackgroundTaskRecord(destination: target.path, offset: 200, byteCount: 600, path: "f.bin")
        try record.apply(downloadedFile: downloaded(data.suffix(400)), response: response(206, ["Content-Range": "bytes 200-599/600"]))
        #expect(try Data(contentsOf: target) == data)
    }

    @Test func aFullResponseReplacesTheFile() throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let target = directory.appendingPathComponent("sub/f.bin")
        let record = BackgroundTaskRecord(destination: target.path, offset: 0, byteCount: 600, path: "f.bin")
        try record.apply(downloadedFile: downloaded(data), response: response(200))
        #expect(try Data(contentsOf: target) == data)
    }

    @Test func aMismatchedOrFailedResponseLeavesTheFileAlone() throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let target = directory.appendingPathComponent("f.bin")
        try data.prefix(200).write(to: target)
        let record = BackgroundTaskRecord(destination: target.path, offset: 200, byteCount: 600, path: "f.bin")
        #expect(throws: PreparationError.self) {
            try record.apply(downloadedFile: downloaded(data.suffix(300)), response: response(206, ["Content-Range": "bytes 300-599/600"]))
        }
        #expect(throws: PreparationError.http(status: 503, path: "f.bin")) {
            try record.apply(downloadedFile: downloaded(Data()), response: response(503))
        }
        #expect(try Data(contentsOf: target) == data.prefix(200))
    }
}
