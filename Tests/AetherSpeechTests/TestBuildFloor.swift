import AetherKit

// Tests build at OS 27 (`swift test --triple arm64-apple-macosx27.0`); without the triple this line fails the build
// with a message that names it.
private let testBuildFloor = RunTestsWithTheOS27Triple.self
