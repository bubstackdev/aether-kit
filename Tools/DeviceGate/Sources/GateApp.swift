import SwiftUI

@main
struct GateApp: App {
    var body: some Scene { WindowGroup { GateView() } }
}

struct GateView: View {
    @State private var log = "Running the device gates…\n"
    var body: some View {
        ScrollView {
            Text(log).font(.system(.footnote, design: .monospaced)).frame(maxWidth: .infinity, alignment: .leading).padding()
        }
        .task {
            let report = await Gate.run { line in await MainActor.run { log += line + "\n" } }
            log += "\nWrote Documents/gate.json (\(report.checks.count) checks, \(report.passed ? "ALL PASSED" : "FAILURES"))\n"
        }
    }
}
