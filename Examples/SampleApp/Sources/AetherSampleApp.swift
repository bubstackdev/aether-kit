import AetherAssets
import AetherKit
import Foundation
import SampleAppCore
import SwiftUI

/// AetherKit's sample app (docs/sample-app.md). Launched with `-qualify PHASES` (comma-separated: fresh, prepare, run,
/// idle, delete) it runs those steps through the same controllers the screens use, writes
/// `Documents/sample-app-qualification-PHASES.json` in its container, and quits. `-idleSeconds N` sets the idle unload.
@main
struct AetherSampleApp: App {
    @State private var launch = Launch.make()

    var body: some Scene {
        WindowGroup {
            Group {
                switch launch {
                case let .ready(app): RootView(app: app)
                case let .failed(message): ContentUnavailableView("Couldn't start", systemImage: "exclamationmark.triangle", description: Text(message))
                }
            }
            .task { await launch.qualifyIfAsked() }
        }
        #if os(iOS)
        // Background downloads that finish while the app is suspended or not running are delivered here.
        .backgroundTask(.urlSession(ModelController.backgroundSessionIdentifier)) {
            await BackgroundTransport.handleEvents(identifier: ModelController.backgroundSessionIdentifier)
        }
        #endif
    }
}

@MainActor
enum Launch {
    case ready(SampleAppState)
    case failed(String)

    static let arguments = UserDefaults.standard

    static func make() -> Launch {
        let idle = arguments.object(forKey: "idleSeconds") == nil ? nil : arguments.double(forKey: "idleSeconds")
        let policy = idle.map { ModelLifecyclePolicy(idleUnloadAfter: .milliseconds(Int64($0 * 1000))) } ?? .platformDefault
        do {
            let app = try SampleAppState.live(lifecycle: policy)
            // In qualification mode the run starts readiness itself, after hooking its recorder.
            if arguments.string(forKey: "qualify") == nil { app.model.start() }
            return .ready(app)
        } catch {
            return .failed("\(error)")
        }
    }

    func qualifyIfAsked() async {
        guard case let .ready(app) = self, let list = Self.arguments.string(forKey: "qualify") else { return }
        let phases = list.split(separator: ",").compactMap { QualificationRun.Phase(rawValue: String($0)) }
        let idle = Self.arguments.object(forKey: "idleSeconds") == nil ? nil : Self.arguments.double(forKey: "idleSeconds")
        let run = QualificationRun(app: app, phases: phases, idleUnloadAfter: idle.map { .milliseconds(Int64($0 * 1000)) }) { line in
            print(line)
            fflush(stdout)
        }
        do {
            let record = await run.run(storeRoot: try ModelController.defaultStore().root)
            let documents = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let file = documents.appendingPathComponent("sample-app-qualification-\(list.replacingOccurrences(of: ",", with: "-")).json")
            try encoder.encode(record).write(to: file)
            print("QUALIFICATION_RECORD \(file.lastPathComponent)")
        } catch {
            print("QUALIFICATION_ERROR \(error)")
        }
        fflush(stdout)
        exit(0)
    }
}
