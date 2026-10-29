import AetherKit
import AetherH3
import Foundation
@preconcurrency import CoreAI

@available(macOS 27, iOS 27, *)
extension H3ProfileEngine {
    /// The Core AI device architecture measured resource entries are keyed by.
    public static var deviceArchitecture: String { AIModel.deviceArchitectureName }

    /// Resource admission from the recorded memory measurements for this exact profile, recipe and device, one
    /// envelope per recorded entry. Stage reservations cover every measured stage peak. Each entry is held to its own
    /// bar: 18 GiB, or 24 GiB for an entry beyond 960x544's pixels or 124 frames, used only with at least 64 GiB of
    /// physical memory (a request only such an entry covers is refused on a smaller device, naming the requirement).
    /// Each request is held to one covering entry, the 18 GiB class first, then the cheapest. A run whose operation plus
    /// largest needed stage exceeds the coordinator's capacity, or whose largest stage exceeds what is still available
    /// once the operation is reserved, is refused before any stage loads; capacity taken by others mid-run can still
    /// refuse a later stage. The bars bind only sessions created with this admission. An unmeasured combination has
    /// no estimate and throws: only profile/device pairs recorded in `h3-measured-resources.json` are admitted (other
    /// Macs are refused, not estimated). Pass the result to `H3VideoSession(stages:resources:gpu:)` together with the
    /// host's `GPUAdmission` owner: the operation and every stage reservation are then taken only after the GPU grant.
    public static func measuredAdmission(profile: H3Profile, recipe: String = "turbo8", coordinator: ResourceCoordinator,
                                         maximumPromptUTF8Bytes: Int = 16_384,
                                         physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) throws -> H3ResourceAdmission {
        try H3MeasuredResources.bundledAdmission(executionIdentity: resourceIdentity(profile: profile, recipe: recipe),
                                                 deviceArchitecture: deviceArchitecture, coordinator: coordinator,
                                                 maximumPromptUTF8Bytes: maximumPromptUTF8Bytes, physicalMemoryBytes: physicalMemoryBytes)
    }
}
