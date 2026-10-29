import AetherAssets
import AetherKit
import AetherLanguage
import Foundation
import UIKit

struct LifecycleResult: Codable {
    struct Point: Codable { let seconds: Double; let event: String; let footprintMiB: Double; let availableMiB: Double }

    var installationID = ""
    var idleUnloadAfterSeconds = 0
    var firstReplySeconds = 0.0
    var reloadReplySeconds = 0.0
    var ballastMiB = 0
    var dispatchLevelsSeen: [String] = []
    var uiKitMemoryWarnings = 0
    var releasedUnderPressure = false
    var points: [Point] = []
}

/// `probe.json` mode "lifecycle": managed lifecycle on the installed model of `Assets/entry.json`
/// (docs/model-lifecycle.md).
/// 1. A reply loads the model, the idle policy unloads it, and a second reply loads it again.
/// 2. With the model idle and loaded, the probe allocates ballast in 32 MiB steps until iOS reports memory pressure. It
///    stops before about 150 MiB remain.
/// 3. It records which notices arrived (Kit's dispatch source, UIKit's memory warning) and whether the model released.
/// 4. The ballast is freed.
/// The run's record, written from the main actor by the run and by its notice observers.
@MainActor
private final class LifecycleRecorder {
    var result: LifecycleResult
    let started = ContinuousClock.now
    init(_ result: LifecycleResult) { self.result = result }
    func point(_ event: String) {
        result.points.append(.init(seconds: (ContinuousClock.now - started).seconds, event: event,
                                   footprintMiB: lifecycleFootprintMiB(), availableMiB: Double(os_proc_available_memory()) / 1_048_576))
        print("LIFECYCLE_EVENT \(event)")
        fflush(stdout)
    }
}

@MainActor
func runLifecycle(assets: URL, store storeName: String, idleSeconds: Int) async throws -> LifecycleResult {
    let entry = try JSONDecoder().decode(ModelCatalogEntry.self, from: Data(contentsOf: assets.appendingPathComponent("entry.json")))
    let support = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    let store = try LocalAssetStore(root: support.appendingPathComponent(storeName))
    let recorder = LifecycleRecorder(LifecycleResult(installationID: try entry.installationID, idleUnloadAfterSeconds: idleSeconds))
    func point(_ event: String) { recorder.point(event) }

    // Watch both notice paths for the whole run.
    let levels = Task { @MainActor in
        for await level in MemoryPressure.levels { recorder.result.dispatchLevelsSeen.append("\(level)"); recorder.point("dispatch level \(level)") }
    }
    let warnings = NotificationCenter.default.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil,
                                                          queue: .main) { _ in
        MainActor.assumeIsolated { recorder.result.uiKitMemoryWarnings += 1; recorder.point("UIKit memory warning") }
    }
    defer { NotificationCenter.default.removeObserver(warnings); levels.cancel() }

    let model = AetherLanguageModel(installation: try await store.acquire(try entry.installationID), platform: .ios,
                                    lifecycle: ModelLifecyclePolicy(idleUnloadAfter: .seconds(idleSeconds)))
    guard model.availability == .available else { throw PreparationProbeError.unavailable("\(model.availability)") }
    let states = Task { @MainActor in for await state in model.lifecycleStates { recorder.point("state \(state)") } }
    defer { states.cancel() }
    let options = AetherGenerationOptions(samplingMode: .greedy, maximumResponseTokens: 16)
    func reply() async throws -> Double {
        let began = ContinuousClock.now
        _ = try await AetherLanguageSession(model: model).respond(to: "Name one primary color.", options: options)
        return (ContinuousClock.now - began).seconds
    }

    point("baseline")
    recorder.result.firstReplySeconds = try await reply()
    point("first reply done")
    while model.lifecycleState != .unloaded(.idle) { try await Task.sleep(for: .milliseconds(200)) }
    try await Task.sleep(for: .seconds(2))
    point("after idle unload")
    recorder.result.reloadReplySeconds = try await reply()
    point("second reply done")

    // Ballast until iOS reports pressure, the model releases, or little memory remains.
    var ballast: [UnsafeMutableRawPointer] = []
    let step = 32 << 20
    while os_proc_available_memory() > 150 << 20, model.lifecycleState == .loaded, recorder.result.uiKitMemoryWarnings == 0 {
        guard let chunk = malloc(step) else { break }
        memset(chunk, 1, step)  // touch every page, so it counts against the footprint
        ballast.append(chunk)
        recorder.result.ballastMiB += 32
        try await Task.sleep(for: .milliseconds(50))
    }
    point("ballast stopped at \(recorder.result.ballastMiB) MiB")
    try await Task.sleep(for: .seconds(3))  // let a notice and its release land
    recorder.result.releasedUnderPressure = model.lifecycleState == .unloaded(.memoryPressure)
    point("after pressure")
    for chunk in ballast { free(chunk) }
    ballast.removeAll()
    try await Task.sleep(for: .seconds(3))
    point("ballast freed")
    await model.unload()
    return recorder.result
}

private func lifecycleFootprintMiB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let status = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return status == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : 0
}
