import AetherKrea

// A separate consumer package verifying the model module's exported API.
// This prepares inputs only; it does not load weights or generate an image.
let configuration = try KreaConfiguration(width: 768, height: 1344, seed: 71344)
let schedule = try KreaSchedule(steps: configuration.steps)
let previewNoise = try KreaNoise.generate(seed: configuration.seed, count: 8)
print("Krea: \(configuration.imageTokenCount) image tokens, \(schedule.timesteps.count) steps")
print("Noise bit patterns: \(previewNoise.map(\.bitPattern))")
