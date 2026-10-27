import AetherKit
import AetherH3
@preconcurrency import CoreAI

extension H3ProfileEngine {
    /// The Core AI device architecture measured resource entries are keyed by.
    public static var deviceArchitecture: String { AIModel.deviceArchitectureName }

    /// Resource admission from the recorded memory measurement for this exact profile, recipe and device. Stage
    /// reservations cover every measured stage peak, and entries over the 18 GiB H3 bar are refused, so a run that
    /// will not fit the coordinator is refused before any stage loads. An unmeasured combination has no estimate
    /// and throws: only profile/device pairs recorded in `h3-measured-resources.json` are admitted (other Macs are
    /// refused, not estimated). Pass the result to `H3VideoSession(stages:resources:gpu:)` together with the host's
    /// `GPUAdmission` owner: the operation and every stage reservation are then taken only after the GPU grant.
    public static func measuredAdmission(profile: H3Profile, recipe: String = "turbo8", coordinator: ResourceCoordinator,
                                         maximumPromptUTF8Bytes: Int = 16_384) throws -> H3ResourceAdmission {
        try H3MeasuredResources.bundled(executionIdentity: resourceIdentity(profile: profile, recipe: recipe),
                                        deviceArchitecture: deviceArchitecture)
            .admission(coordinator: coordinator, maximumPromptUTF8Bytes: maximumPromptUTF8Bytes)
    }
}
