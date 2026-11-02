import AetherAssets
import AetherKit
import AetherLanguage
import Darwin
import Foundation
import Metal

/// `qualify-lifecycle STORE INSTALLATION_ID IDLE_SECONDS`: managed lifecycle on a real language model
/// (docs/model-lifecycle.md):
/// 1. A reply loads the model.
/// 2. The idle policy unloads it, and a second reply loads it again.
/// 3. A simulated warning releases it.
/// 4. A simulated critical level refuses the next load.
///
/// Records Metal's allocated size and the physical footprint at each point. Pressure is simulated through the
/// monitor's delivery seam, because raising real system pressure on this Mac would disturb everything else running. A
/// GPU command.
@available(macOS 27, iOS 27, *)
enum LifecycleCommand {
    static let usage = "aether-probe qualify-lifecycle STORE_DIRECTORY INSTALLATION_ID IDLE_SECONDS"

    private struct Point: Encodable {
        let step: String
        let seconds: Double
        let state: String
        let metalMiB: Double
        let footprintMiB: Double
    }

    static func run(_ arguments: [String]) async throws -> Bool {
        guard arguments.first == "qualify-lifecycle" else { return false }
        guard arguments.count == 4, let idle = Int(arguments[3]), (1...600).contains(idle) else {
            throw AetherError.invalidRequest("usage:\n" + usage)
        }
        let store = try LocalAssetStore(root: URL(fileURLWithPath: arguments[1]))
        let device = MTLCreateSystemDefaultDevice()
        let start = ContinuousClock.now
        var points: [Point] = []
        func record(_ step: String, _ model: AetherLanguageModel?) {
            points.append(Point(step: step, seconds: (ContinuousClock.now - start) / .seconds(1),
                                state: model.map { "\($0.lifecycleState)" } ?? "-",
                                metalMiB: Double(device?.currentAllocatedSize ?? 0) / 1_048_576, footprintMiB: footprintMiB()))
        }
        record("baseline", nil)
        let model = AetherLanguageModel(installation: try await store.acquire(arguments[2]),
                                        lifecycle: ModelLifecyclePolicy(idleUnloadAfter: .seconds(idle)))
        guard model.availability == .available else { throw AetherError.invalidModel("unavailable: \(model.availability)") }
        let options = AetherGenerationOptions(samplingMode: .greedy, maximumResponseTokens: 16)
        func reply(_ step: String) async throws -> Double {
            let began = ContinuousClock.now
            _ = try await AetherLanguageSession(model: model).respond(to: "Name one primary color.", options: options)
            let seconds = (ContinuousClock.now - began) / .seconds(1)
            record(step, model)
            return seconds
        }
        func wait(for state: ModelLifecycleState, within limit: Duration) async throws {
            let reached = try await withThrowingTaskGroup(of: Bool.self) { group in
                group.addTask { for await current in model.lifecycleStates where current == state { return true }; return false }
                group.addTask { try await Task.sleep(for: limit); return false }
                let reached = try await group.next() ?? false
                group.cancelAll()
                return reached
            }
            guard reached else { throw AetherError.invalidRequest("lifecycle never reached \(state); it is \(model.lifecycleState)") }
        }

        let firstReply = try await reply("first reply (loads)")
        let idleStarted = ContinuousClock.now
        try await wait(for: .unloaded(.idle), within: .seconds(idle + 60))
        let idleSeconds = (ContinuousClock.now - idleStarted) / .seconds(1)
        try await Task.sleep(for: .seconds(2))  // the unload completes after the state turns
        record("idle unload", model)
        let reloadReply = try await reply("second reply (reloads)")

        MemoryPressureMonitor.shared.deliver(.warning)
        try await wait(for: .unloaded(.memoryPressure), within: .seconds(60))
        try await Task.sleep(for: .seconds(2))
        record("simulated warning", model)
        MemoryPressureMonitor.shared.deliver(.critical)
        var refused = "loaded"
        do { _ = try await reply("reply under simulated critical") } catch { refused = "\(error)" }
        record("critical refusal", model)
        MemoryPressureMonitor.shared.deliver(.normal)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        struct Report: Encodable {
            let installationID: String
            let idleUnloadAfterSeconds: Int
            let firstReplySeconds: Double
            let idleUnloadAfterReplySeconds: Double
            let reloadReplySeconds: Double
            let criticalRefusal: String
            let points: [Point]
        }
        FileHandle.standardOutput.write(try encoder.encode(Report(
            installationID: arguments[2], idleUnloadAfterSeconds: idle, firstReplySeconds: firstReply,
            idleUnloadAfterReplySeconds: idleSeconds, reloadReplySeconds: reloadReply, criticalRefusal: refused,
            points: points)) + Data("\n".utf8))
        await model.unload()
        return true
    }

    private static func footprintMiB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return result == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : 0
    }
}
