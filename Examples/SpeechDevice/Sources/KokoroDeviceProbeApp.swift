import Foundation
import SwiftUI

/// Runs the Kokoro device probe once at launch, prints one `KOKORO_DEVICE_REPORT` JSON line to stdout
/// (captured by `xcrun devicectl device process launch --console`), saves it to Documents, then exits.
@main
struct KokoroDeviceProbeApp: App {
    @State private var status = "Running Kokoro device probe…"

    var body: some Scene {
        WindowGroup {
            Text(status)
                .padding()
                .task {
                    let report = await runKokoroDeviceProbe()
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = [.sortedKeys]
                    let data = (try? encoder.encode(report)) ?? Data("{}".utf8)
                    let line = "KOKORO_DEVICE_REPORT " + String(decoding: data, as: UTF8.self)
                    print(line)
                    fflush(stdout)
                    if let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                        try? data.write(to: documents.appendingPathComponent("kokoro-device-report.json"))
                    }
                    status = report.passed ? "Passed" : "Failed: \(report.failures.joined(separator: "; "))"
                    try? await Task.sleep(for: .seconds(1))
                    exit(report.passed ? 0 : 1)
                }
        }
    }
}
