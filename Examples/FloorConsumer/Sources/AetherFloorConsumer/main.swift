import AetherLanguage
import Foundation

// Prints the facade's availability for an installed bundle (the first argument), or for a path with no bundle. Below
// macOS 27 / iOS 27 the answer is `unsupportedOS` whatever the bundle, because the OS is checked first; on 27 a missing
// bundle reports `bundleMissing`. It loads nothing, so it needs no weights.
let bundle = URL(fileURLWithPath: CommandLine.arguments.dropFirst().first ?? "/nonexistent/language-bundle")
let model = AetherLanguageModel(bundle: bundle)
let availability = switch model.availability {
case .available: "available"
case .unavailable(.unsupportedOS): "unsupportedOS"
case .unavailable(.unsupportedPlatform): "unsupportedPlatform"
case .unavailable(.bundleMissing): "bundleMissing"
case .unavailable(.bundleInvalid(let reason)): "bundleInvalid(\(reason))"
case .unavailable(.unsupportedTemplate): "unsupportedTemplate"
// UnavailableReason is @nonexhaustive: later 0.0.x releases may add reasons.
case .unavailable(let reason): "\(reason)"
}
print("OS: \(ProcessInfo.processInfo.operatingSystemVersionString)")
print("AetherLanguage availability: \(availability)")
