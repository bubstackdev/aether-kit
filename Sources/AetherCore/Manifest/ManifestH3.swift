import Foundation

// The `h3` block that `minimax-h3@1` owns (MANIFEST.md §5.10). The driver implements exactly the rules typed here; the
// keyframe and vision fields (FL2VA) are decoded loosely, since the driver doesn't present keyframes yet.

extension Manifest {
    /// `h3` (MANIFEST.md §5.10): MiniMax H3's qualified profiles, the conditioner's row source and rotary tables, the
    /// recipes (both scheduler arms and each step's host tables), and the video and audio decode contracts.
    public struct H3: Sendable, Hashable, Codable {
        /// A qualified canvas: its size, frame count and the packed rows they imply.
        public struct Profile: Sendable, Hashable, Codable {
            public var key: String
            public var width: Int
            public var height: Int
            public var frames: Int
            public var latentFrames: Int
            public var videoRows: Int
            public var keyframeRows: Int
            public var audioLatents: Int
            /// The graph role decoding this profile's audio (a fixed latent count).
            public var audioDecoder: String
            /// The graph role finishing this profile's FL2VA keyframe encoding.
            public var keyframeEncoder: String?
            /// The recipes the profile offers (`turbo8`, `turbo4`); absent, every recipe. A request for another is
            /// refused before anything runs.
            public var recipes: [String]?
            /// The memory bar the profile's measured envelope is held to, and the physical memory a device needs.
            public var memory: Memory?
            /// The profile's ``ShapeProfile`` entries, by graph role and input: the roles a render at this profile
            /// runs, and the shapes of the inputs that vary between its renders (the body's modulation table and the
            /// heads' timestep embedding, one per recipe cardinality). An input not listed is unconstrained, or fixed
            /// by its graph.
            public var shapes: [String: [String: [[Int]]]]?

            /// The memory class of a profile (MANIFEST.md §5.10).
            public struct Memory: Sendable, Hashable, Codable {
                /// The bar every phase's measured peak plus the sampling margin must stay within.
                public var barBytes: Int64
                /// The physical memory a device needs to run the profile (0: no minimum beyond the budget).
                public var minimumPhysicalMemoryBytes: Int64

                public init(barBytes: Int64, minimumPhysicalMemoryBytes: Int64) {
                    self.barBytes = barBytes
                    self.minimumPhysicalMemoryBytes = minimumPhysicalMemoryBytes
                }
            }
        }

        /// A measured memory envelope (MANIFEST.md §5.10): what the memory gate measured for one admission key on one
        /// Core AI architecture, written by `Forge/h3/record_envelopes_h3.py`. The driver admits only keys with an
        /// envelope for its export and device, and seeds its ledger from them.
        public struct Envelope: Sendable, Hashable, Codable {
            public struct Phase: Sendable, Hashable, Codable {
                public var peakBytes: Int64
                public var heldBytes: Int64
            }
            /// The admission key: a profile key (text to video), or `<profile>/fl2va1` (one keyframe) or
            /// `<profile>/fl2va2` (first and last).
            public var key: String
            /// The recipes measured.
            public var recipes: [String]
            /// The Core AI architecture measured on (`Device.architecture`).
            public var architecture: String
            /// The export measured: its ``Manifest/contentDigest`` (which leaves the envelopes out).
            public var export: String
            /// The sampling margin added to each peak against the bar.
            public var marginBytes: Int64
            public var perPrompt: Phase
            public var perStep: Phase
            public var perImage: Phase
            /// How many renders it covers, and their records.
            public var renders: Int?
            public var record: String?

            /// The profile the key admits.
            public var profile: String { String(key.split(separator: "/").first ?? "") }
            /// Keyframes the key admits: 0, 1 or 2.
            public var keyframeCount: Int {
                key.hasSuffix("/fl2va2") ? 2 : key.hasSuffix("/fl2va1") ? 1 : 0
            }
            /// The largest phase peak.
            public var stagePeakBytes: Int64 { max(perPrompt.peakBytes, perStep.peakBytes, perImage.peakBytes) }
        }

        public struct Limits: Sendable, Hashable, Codable {
            /// Packed rows the body specializes for, `[lower, upper]`.
            public var rows: [Int]
            /// Distinct row timesteps a step may have, `[lower, upper]`.
            public var cardinality: [Int]
            public var textTokens: [Int]
        }

        /// The Qwen3-VL conditioner: `layers` decoder-layer calls over the embedding rows of the prompt, tapped after
        /// layer `tap` (unnormalized), with the text rotary tables' first T rows.
        public struct Conditioner: Sendable, Hashable, Codable {
            public struct Embedding: Sendable, Hashable, Codable {
                public enum DType: String, Sendable, Hashable, Codable { case bfloat16 }
                public var dtype: DType
                public var file: AssetPath
                public var rows: Int
                public var width: Int
            }
            public struct RopeTables: Sendable, Hashable, Codable {
                /// `[rows, headDim]` bf16, row t the text position t.
                public var cos: AssetPath
                public var sin: AssetPath
                public var rows: Int
            }
            public var layers: Int
            public var tap: Int
            public var embedding: Embedding
            public var ropeTables: RopeTables
            public var rope: JSONValue?
        }

        /// One scheduler arm: the recipe's sigmas (`steps + 1`, ending in 0) and timesteps (`1 − σ`, one per step).
        public struct Arm: Sendable, Hashable, Codable {
            public var shift: Float
            public var sigmas: [Float]
            public var timesteps: [Float]
        }

        /// One step's host tables: its distinct row timesteps (ascending), the heads' timestep embedding
        /// `[k, 2688]` float32, and each block's AdaLN table `[6, 3k, 5376]` bf16 (`blocks` with `{block:02d}`).
        public struct Step: Sendable, Hashable, Codable {
            public var timesteps: [Float]
            public var temb: AssetPath
            public var blocks: String

            /// Block `index`'s table file.
            public func table(_ index: Int) throws(AetherError) -> AssetPath {
                let number = index < 10 ? "0\(index)" : "\(index)"
                return try AssetPath(parsing: blocks.replacingOccurrences(of: "{block:02d}", with: number))
            }
        }

        public struct Recipe: Sendable, Hashable, Codable {
            public var video: Arm
            public var audio: Arm
            public var steps: [Step]
        }

        public struct Video: Sendable, Hashable, Codable {
            /// The initial noise: Kit's seeded stream, the source `[24, T, H, W]` video tensor first (then packed),
            /// then the channel-major audio rows (0.0.x's order).
            public struct Noise: Sendable, Hashable, Codable {
                public enum Kind: String, Sendable, Hashable, Codable { case splitmix64BoxMuller }
                public enum Order: String, Sendable, Hashable, Codable { case sourceVideoThenAudio }
                public var kind: Kind
                public var order: Order
            }
            public enum Packing: String, Sendable, Hashable, Codable { case patch2x2ChannelMajor }
            public var tile: Int
            public var windowLatents: Int
            public var windowFrames: Int
            public var windowStride: Int
            public var packing: Packing
            public var rowChannels: Int
            public var noise: Noise
        }

        public struct Audio: Sendable, Hashable, Codable {
            public var sampleRate: Int
            public var hop: Int
            public var channels: Int
            public var rowChannels: Int
        }

        public var tasks: [String]
        public var profiles: [Profile]
        public var limits: Limits
        public var conditioner: Conditioner
        /// `"<recipe>/<task>"` (`turbo8/t2va`, …) → its schedule and tables.
        public var recipes: [String: Recipe]
        public var defaultRecipe: String
        public var video: Video
        public var audio: Audio
        public var platforms: [String]
        public var adapter: JSONValue?
        public var keyframe: JSONValue?
        public var vision: JSONValue?
        /// The measured envelopes (absent: none recorded).
        public var envelopes: [Envelope]?

        public func profile(_ key: String) -> Profile? { profiles.first { $0.key == key } }

        func validate(graphs: [String: Graph], assets: [String: Asset]) throws(AetherError) {
            try require(!profiles.isEmpty, "h3.profiles", "at least one qualified profile is required")
            for (i, p) in profiles.enumerated() {
                let field = "h3.profiles.\(i)"
                try require(p.key == "\(p.width)x\(p.height)x\(p.frames)", "\(field).key", "must be <width>x<height>x<frames>")
                try require(p.width > 256 && p.height > 256 && p.width % 32 == 0 && p.height % 32 == 0,
                            "\(field)", "sides must be multiples of 32 over 256 px")
                try require(p.frames >= 22 && (p.frames - 5) % 17 == 0, "\(field).frames", "must be 17n + 5, n ≥ 1")
                try require(p.latentFrames == 5 * ((p.frames - 5) / 17) + 2, "\(field).latentFrames", "must be 5n + 2")
                try require(p.videoRows == p.latentFrames * (p.height / 32) * (p.width / 32), "\(field).videoRows",
                            "must be latentFrames · H/32 · W/32")
                try require(graphs[p.audioDecoder] != nil, "\(field).audioDecoder", "names no graph")
                if let encoder = p.keyframeEncoder { try require(graphs[encoder] != nil, "\(field).keyframeEncoder", "names no graph") }
                for (j, recipe) in (p.recipes ?? []).enumerated() {
                    try require(recipes["\(recipe)/t2va"] != nil, "\(field).recipes.\(j)", "names no t2va recipe")
                }
                if let recipes = p.recipes { try require(!recipes.isEmpty && Set(recipes).count == recipes.count, "\(field).recipes", "must be distinct and non-empty") }
                if let memory = p.memory {
                    try require(memory.barBytes > 0 && memory.minimumPhysicalMemoryBytes >= 0, "\(field).memory",
                                "needs a positive bar and a non-negative physical memory")
                }
                for (role, inputs) in p.shapes ?? [:] {
                    try require(graphs[role] != nil, "\(field).shapes.\(role)", "names no graph")
                    for (input, shapes) in inputs {
                        try require(graphs[role]!.inputs[input] != nil, "\(field).shapes.\(role).\(input)", "names no input of the graph")
                        try require(!shapes.isEmpty && shapes.allSatisfy { !$0.isEmpty && $0.allSatisfy { $0 > 0 } },
                                    "\(field).shapes.\(role).\(input)", "must list positive shapes")
                    }
                }
                if let shapes = p.shapes {
                    for role in [p.audioDecoder] + (p.keyframeEncoder.map { [$0] } ?? []) {
                        try require(shapes[role] != nil, "\(field).shapes", "must name the profile's graph '\(role)'")
                    }
                }
            }
            try require(Set(profiles.map(\.key)).count == profiles.count, "h3.profiles", "keys must be distinct")
            var measured = Set<[String]>()
            for (i, e) in (envelopes ?? []).enumerated() {
                let field = "h3.envelopes.\(i)"
                guard let p = profile(e.profile) else { throw .manifestInvalid(field: "\(field).key", reason: "names no profile") }
                try require(e.key == p.key || ((e.key == p.key + "/fl2va1" || e.key == p.key + "/fl2va2") && p.keyframeEncoder != nil),
                            "\(field).key", "must be <profile>, or <profile>/fl2va1 or /fl2va2 for a keyframe profile")
                try require(measured.insert([e.key, e.architecture]).inserted, "\(field)", "repeats its key and architecture")
                try require(!e.recipes.isEmpty, "\(field).recipes", "must name the recipes measured")
                for (j, recipe) in e.recipes.enumerated() {
                    try require(recipes["\(recipe)/\(e.keyframeCount == 0 ? "t2va" : "fl2va")"] != nil && (p.recipes?.contains(recipe) ?? true),
                                "\(field).recipes.\(j)", "names no recipe of the profile for its task")
                }
                try require(!e.architecture.isEmpty, "\(field).architecture", "must not be empty")
                try require(e.export.count == 64 && e.export.allSatisfy { "0123456789abcdef".contains($0) }, "\(field).export",
                            "must be a measurement digest (64 lowercase hex digits)")
                try require(e.marginBytes >= 0, "\(field).marginBytes", "must not be negative")
                for (name, phase) in [("perPrompt", e.perPrompt), ("perStep", e.perStep), ("perImage", e.perImage)] {
                    try require(phase.heldBytes >= 0 && phase.heldBytes <= phase.peakBytes, "\(field).\(name)", "needs 0 ≤ held ≤ peak")
                }
                guard let memory = p.memory else {
                    throw .manifestInvalid(field: "\(field)", reason: "its profile has no memory bar")
                }
                try require(e.stagePeakBytes + e.marginBytes <= memory.barBytes, "\(field)",
                            "its peak plus the margin is over the profile's bar")
            }
            try require(limits.rows.count == 2 && limits.rows[0] >= 1 && limits.rows[0] <= limits.rows[1], "h3.limits.rows", "must be [lower, upper]")
            try require(limits.cardinality.count == 2 && limits.cardinality[0] >= 1 && limits.cardinality[0] <= limits.cardinality[1],
                        "h3.limits.cardinality", "must be [lower, upper]")
            try require(limits.textTokens.count == 2 && limits.textTokens[0] >= 1 && limits.textTokens[0] <= limits.textTokens[1],
                        "h3.limits.textTokens", "must be [lower, upper]")
            try require(conditioner.layers >= 1 && conditioner.tap == conditioner.layers, "h3.conditioner.tap",
                        "the conditioner taps after its last layer")
            try require(conditioner.ropeTables.rows >= limits.textTokens[1], "h3.conditioner.ropeTables.rows",
                        "must cover the text token limit")
            try require(conditioner.embedding.rows > 0 && conditioner.embedding.width > 0, "h3.conditioner.embedding", "must be non-empty")
            for (field, path) in [("embedding.file", conditioner.embedding.file), ("ropeTables.cos", conditioner.ropeTables.cos),
                                  ("ropeTables.sin", conditioner.ropeTables.sin)] {
                try require(assets[path.asset] != nil, "h3.conditioner.\(field)", "names asset '\(path.asset)', which is not in assets")
            }
            try require(recipes[defaultRecipe + "/t2va"] != nil, "h3.defaultRecipe", "names no t2va recipe")
            for key in recipes.keys.sorted() {
                let recipe = recipes[key]!, field = "h3.recipes.\(key)"
                for (name, arm) in [("video", recipe.video), ("audio", recipe.audio)] {
                    let f = "\(field).\(name)"
                    try require(arm.timesteps.count == recipe.steps.count && arm.sigmas.count == recipe.steps.count + 1, f,
                                "needs one timestep a step and steps + 1 sigmas")
                    try require(arm.sigmas.first == 1 && arm.sigmas.last == 0 && zip(arm.sigmas, arm.sigmas.dropFirst()).allSatisfy { $0 > $1 },
                                "\(f).sigmas", "must fall from 1 to 0")
                    try require(zip(arm.timesteps, arm.sigmas).allSatisfy { $0.bitPattern == (1 - $1).bitPattern }, "\(f).timesteps",
                                "must be 1 − σ in float32")
                }
                for (i, step) in recipe.steps.enumerated() {
                    let f = "\(field).steps.\(i)"
                    try require((limits.cardinality[0]...limits.cardinality[1]).contains(step.timesteps.count), "\(f).timesteps",
                                "the step's cardinality is outside h3.limits.cardinality")
                    try require(zip(step.timesteps, step.timesteps.dropFirst()).allSatisfy { $0 < $1 }, "\(f).timesteps", "must ascend")
                    try require(step.blocks.contains("{block:02d}"), "\(f).blocks", "must contain {block:02d}")
                    _ = try step.table(0)
                }
            }
            try require(video.tile == 256 && video.windowLatents == 7 && video.windowFrames == 22 && video.windowStride == 5,
                        "h3.video", "the decoder windows are 256 px tiles of 7 latent frames at stride 5 (22 frames)")
            try require(video.rowChannels == 96, "h3.video.rowChannels", "must be 96 (24 channels × 2 × 2)")
            try require(audio.channels == 2 && audio.rowChannels == 32 && audio.hop > 0 && audio.sampleRate > 0, "h3.audio",
                        "must be stereo rows of 32 with a positive hop and rate")
        }
    }
}
