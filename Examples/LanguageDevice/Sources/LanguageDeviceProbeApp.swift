import Foundation
import SwiftUI

/// Runs the language device probe once at launch, prints one `LANGUAGE_DEVICE_REPORT` JSON line to stdout
/// (captured by `xcrun devicectl device process launch --console`), saves it to Documents, then exits.
@main
struct LanguageDeviceProbeApp: App {
    @State private var status = "Running language device probe…"

    var body: some Scene {
        WindowGroup {
            Text(status)
                .padding()
                .task {
                    let report = await runLanguageDeviceProbe()
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = [.sortedKeys]
                    let data = (try? encoder.encode(report)) ?? Data("{}".utf8)
                    print("LANGUAGE_DEVICE_REPORT " + String(decoding: data, as: UTF8.self))
                    fflush(stdout)
                    if let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                        try? data.write(to: documents.appendingPathComponent("language-device-report.json"))
                    }
                    status = report.passed ? "Passed" : "Failed: \(report.failures.joined(separator: "; "))"
                    try? await Task.sleep(for: .seconds(1))
                    exit(report.passed ? 0 : 1)
                }
        }
    }
}
