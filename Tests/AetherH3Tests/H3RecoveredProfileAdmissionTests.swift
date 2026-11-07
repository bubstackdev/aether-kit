import AetherH3
import AetherKit
import Foundation
import Testing

// Scope digests of the independently qualified recovery profile, whose full content digest is e32d9409….
// The native records cover first-only and first+last renders; last-only has the first-only shape (the gate records
// this equivalence). These CPU tests exercise the actual bundled admission, including its 18/24 GiB split.
private let recoveryScopes: [String: String] = [
    "960x544": "dced93f95fc3bde46679aa681332f8b441bc4ae9b49173900c7d676115b1269f",
    "544x960,704x704": "13342b3be921e29c144f87376b8f93e2ff2fbf38784d7b4e02cc2d4a10872eb7",
    "544x960,704x704,960x544": "5856eca27f590637c3f4ada3146178c6d89e7b41d67753677a63a31913670ebb",
    "1280x704,704x1280": "93462af9fe3d81688db1361363bac44e7260be45d288b8f911977046ea9f6d72"
]

private func recoveryAdmission(recipe: String, physicalMemory: UInt64 = 128 << 30) throws -> H3ResourceAdmission {
    try H3MeasuredResources.bundledAdmission(
        executionIdentity: "h3.profile/2:e32d9409a0c4f08f8fbdd32076e6626f8db707fad441f9a0743bdb93ce4cab0b:" + recipe,
        scopedIdentity: { canvases in
            let scope = recoveryScopes[canvases.sorted().joined(separator: ",")] ?? "unmeasured"
            return "h3.profile/3:" + scope + ":" + recipe
        }, deviceArchitecture: "h17c", coordinator: try ResourceCoordinator(capacityBytes: 24 << 30),
        physicalMemoryBytes: physicalMemory)
}

private func recoveryRequest(width: Int, height: Int, frames: Int, anchors: Int = 1) throws -> H3VideoRequest {
    let geometry = try H3Geometry(width: width, height: height, requestedFrames: frames)
    let frame = try H3Keyframe(width: width, height: height, rgb: Array(repeating: 120, count: width * height * 3))
    return try H3VideoRequest(prompt: "A neutral object fixture.", geometry: geometry, seed: 1043,
        firstFrame: anchors == 1 || anchors == 3 ? frame : nil, lastFrame: anchors == 2 || anchors == 3 ? frame : nil, audio: true)
}

@Test(arguments: ["turbo8", "turbo4"])
func recoveryVideoAdmissionCoversOnlyTheQualifiedModes(recipe: String) throws {
    let admission = try recoveryAdmission(recipe: recipe)
    for (width, height) in [(960, 544), (544, 960), (704, 704)] {
        for frames in [120, 240] {
            for anchors in 1...3 {
                let request = try recoveryRequest(width: width, height: height, frames: frames, anchors: anchors)
                #expect(admission.admits(request))
                let envelope = try admission.envelope(for: request)
                #expect(envelope.memoryBarBytes == (frames == 120 ? 18 << 30 : 24 << 30))
            }
        }
    }
    for (width, height) in [(1280, 704), (704, 1280)] {
        #expect(admission.admits(try recoveryRequest(width: width, height: height, frames: 120)) == (recipe == "turbo8"))
        #expect(!admission.admits(try recoveryRequest(width: width, height: height, frames: 240)))
    }
    // These records measure FL2VA, not text-only video, and do not lend their extended envelopes to a smaller host.
    #expect(!admission.admits(try recoveryRequest(width: 960, height: 544, frames: 120, anchors: 0)))
    let smallerHost = try recoveryAdmission(recipe: recipe, physicalMemory: 32 << 30)
    #expect(smallerHost.admits(try recoveryRequest(width: 960, height: 544, frames: 120)))
    #expect(!smallerHost.admits(try recoveryRequest(width: 960, height: 544, frames: 240)))
    #expect(!smallerHost.admits(try recoveryRequest(width: 1280, height: 704, frames: 120)))
}

@Test func recoveryEntriesRemainBoundToPassingNativeEvidence() throws {
    struct Record: Decodable {
        struct TaskEntry: Decodable { let entry: H3MeasuredResources }
        let passed: Bool
        let failures: [String]
        let pinsUnchanged: Bool
        let renders: [String: Render]
        let measuredEntriesByTask: [String: TaskEntry]
        struct Render: Decodable { let withinBar: Bool }
    }
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let entries = try H3MeasuredResources.bundled().filter { $0.evidence.contains("h3-atelier-recovery-") }
    #expect(entries.count == 6)
    var renderCount = 0
    for entry in entries {
        let record = try JSONDecoder().decode(Record.self, from: Data(contentsOf: root.appendingPathComponent(entry.evidence)))
        #expect(record.passed && record.pinsUnchanged && record.failures.isEmpty)
        #expect(record.measuredEntriesByTask["fl2va"]?.entry == entry)
        #expect(entry.deviceArchitecture == "h17c" && entry.tasks == [.fl2va])
        for canvas in entry.canvases {
            #expect(record.renders["fl2va-" + canvas]?.withinBar == true)
            #expect(record.renders["fl2va-firstlast-" + canvas]?.withinBar == true)
        }
        renderCount += record.renders.count
    }
    #expect(renderCount == 28)
    // A changed scope, recipe or device must never borrow this profile's evidence.
    for (recipe, device, changedScope) in [("turbo16", "h17c", false), ("turbo8", "other", false), ("turbo8", "h17c", true)] {
        #expect(throws: AetherError.self) {
            _ = try H3MeasuredResources.bundledAdmission(executionIdentity: "unmeasured",
                scopedIdentity: { canvases in
                    "h3.profile/3:" + (changedScope ? "changed" : recoveryScopes[canvases.sorted().joined(separator: ",")] ?? "unmeasured") + ":" + recipe
                }, deviceArchitecture: device, coordinator: ResourceCoordinator(capacityBytes: 24 << 30))
        }
    }
}
