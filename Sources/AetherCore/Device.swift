public import Foundation

#if canImport(Darwin)
import Darwin
#endif

/// The device a model is resolved and installed for (DESIGN.md §5).
///
/// AetherCore cannot ask Core AI for the architecture name, so it is injected: AetherRuntime offers
/// `Device.current` built from `AIModel.deviceArchitectureName`, and tests construct a fake.
public struct Device: Sendable, Hashable {
    public enum Platform: String, Sendable, Hashable, Codable {
        case macOS = "macos"
        case iOS = "ios"

        /// The platform this binary runs on.
        public static var host: Platform {
            #if os(macOS)
            .macOS
            #else
            .iOS
            #endif
        }
    }

    public var platform: Platform
    /// Core AI's device architecture name, e.g. `h17c` (M5 Max) or `h18p`.
    public var architecture: String
    /// Installed RAM in bytes.
    public var physicalMemory: UInt64

    public init(platform: Platform, architecture: String, physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) {
        self.platform = platform
        self.architecture = architecture
        self.physicalMemory = physicalMemory
    }

    /// The host platform with an injected architecture name.
    public static func host(architecture: String) -> Device {
        Device(platform: .host, architecture: architecture)
    }

    /// Memory the process may still allocate, in bytes. On iOS this is `os_proc_available_memory()`; macOS has no
    /// per-process limit, so it reports installed RAM and the residency policy (M6) applies its own fraction.
    public static func availableMemory() -> UInt64 {
        #if os(iOS)
        UInt64(os_proc_available_memory())
        #else
        ProcessInfo.processInfo.physicalMemory
        #endif
    }

    /// Free space for important (user-initiated) data on the volume holding `url`, in bytes.
    public static func freeDiskSpace(at url: URL) throws(AetherError) -> Int64 {
        do {
            let values = try url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            guard let capacity = values.volumeAvailableCapacityForImportantUsage else {
                throw AetherError.runtime(underlying: CocoaError(.fileReadUnknown))
            }
            return capacity
        } catch let error as AetherError {
            throw error
        } catch {
            throw AetherError.runtime(underlying: error)
        }
    }
}
