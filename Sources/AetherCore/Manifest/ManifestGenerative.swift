// MANIFEST.md §5.2–§5.3: the generic blocks of the generative drivers. `schedule` (the denoising schedule and the host
// update), `latent` (the image-latent contract and its seeded noise) and per-graph `residency`. Unlike the loose
// driver blocks of §2.5–§2.8 these are fully typed: a value outside the closed vocabularies below is rejected at
// decode time, naming its field, so a driver never meets a schedule it would have to guess at. Driver-owned blocks
// (`krea2`) are not here; they arrive with their drivers.

extension Manifest {
    /// When a graph is loaded and released, relative to one generation: `prompt → steps → image`
    /// (MANIFEST.md §5.3). Absent: resident for the model's lifetime.
    public enum Residency: String, Sendable, Hashable, Codable, CaseIterable {
        /// Loaded to encode the prompt, then releasable (text encoders).
        case perPrompt
        /// Resident through the denoising steps (denoisers).
        case perStep
        /// Loaded to produce the output, then releasable (decoders).
        case perImage
    }

    /// `schedule` (MANIFEST.md §5.2): the step schedule, the denoiser's timestep input and the host update.
    public struct Schedule: Sendable, Hashable, Codable {
        /// `flowMatchEuler` (Krea, FLUX.2). Other solvers (SD-Turbo's `epsilonEuler`, H3's) are added with their
        /// drivers; until then a manifest naming one is rejected.
        public enum Kind: String, Sendable, Hashable, Codable {
            case flowMatchEuler
        }

        /// The base sigma grid. The only one so far: numpy's `linspace(1, 1/steps, steps)`, computed in Double and
        /// rounded to Float, the last value exactly `1/steps`.
        public enum Base: String, Sendable, Hashable, Codable {
            case linspace = "linspace(1, 1/steps, steps)"
        }

        /// What the denoiser's `sigma` input receives for step `i`.
        public enum Timestep: String, Sendable, Hashable, Codable {
            /// `Float(σ · trainTimesteps) / trainTimesteps`, in Float (Krea, FLUX.2).
            case sigmaRoundTrip = "sigma * trainTimesteps / trainTimesteps"
        }

        /// The host update after each denoiser call.
        public enum Update: String, Sendable, Hashable, Codable {
            /// `x + (σ[i+1] − σ[i]) · v` in Float, the product rounded before the add (no fused multiply-add).
            case eulerVelocity = "x + (sigma[i+1] - sigma[i]) * velocity"
        }

        /// The flux pipelines' empirical μ rule (MANIFEST.md §5.2 `shift.rule: empirical`): linear in the step count
        /// between a low and a high line in the image-token count; above `tokenThreshold` only the high line.
        public struct EmpiricalShift: Sendable, Hashable, Codable {
            public var lowSteps: Int
            public var lowSlope: Double
            public var lowIntercept: Double
            public var highSteps: Int
            public var highSlope: Double
            public var highIntercept: Double
            public var tokenThreshold: Int

            public init(lowSteps: Int, lowSlope: Double, lowIntercept: Double, highSteps: Int, highSlope: Double,
                        highIntercept: Double, tokenThreshold: Int) {
                self.lowSteps = lowSteps
                self.lowSlope = lowSlope
                self.lowIntercept = lowIntercept
                self.highSteps = highSteps
                self.highSlope = highSlope
                self.highIntercept = highIntercept
                self.tokenThreshold = tokenThreshold
            }

            /// μ for `imageTokens` generated tokens and `steps` steps, in Double in the reference's operation order.
            public func mu(imageTokens n: Int, steps s: Int) -> Double {
                let high = highSlope * Double(n) + highIntercept
                if n > tokenThreshold { return high }
                let low = lowSlope * Double(n) + lowIntercept
                let slope = (high - low) / Double(highSteps - lowSteps)
                let intercept = high - Double(highSteps) * slope
                return slope * Double(s) + intercept
            }
        }

        /// `{rule: exponential, mu} | {rule: empirical, empirical: {…}} | {rule: none}`. The shifted sigma is
        /// `e / (e + (1/σ − 1))` in Float, `e = Float(exp(μ))`.
        public struct Shift: Sendable, Hashable, Codable {
            public enum Rule: String, Sendable, Hashable, Codable {
                /// A fixed μ (Krea 2 Turbo, `is_distilled`: 1.15).
                case exponential
                /// μ from the image-token count and the step count.
                case empirical
                /// The base grid unshifted.
                case none
            }

            public var rule: Rule
            public var mu: Double?
            public var empirical: EmpiricalShift?

            public init(rule: Rule, mu: Double? = nil, empirical: EmpiricalShift? = nil) {
                self.rule = rule
                self.mu = mu
                self.empirical = empirical
            }

            func validate(at prefix: String) throws(AetherError) {
                switch rule {
                case .exponential:
                    try require(mu?.isFinite == true, "\(prefix).mu", "exponential needs a finite mu")
                    try require(empirical == nil, "\(prefix).empirical", "only for rule empirical")
                case .empirical:
                    guard let e = empirical else {
                        throw .manifestInvalid(field: "\(prefix).empirical", reason: "rule empirical needs its coefficients")
                    }
                    try require(mu == nil, "\(prefix).mu", "only for rule exponential")
                    try require(e.lowSteps >= 1 && e.highSteps > e.lowSteps, "\(prefix).empirical",
                                "needs 1 <= lowSteps < highSteps")
                    try require(e.tokenThreshold >= 1, "\(prefix).empirical.tokenThreshold", "must be >= 1")
                    try require([e.lowSlope, e.lowIntercept, e.highSlope, e.highIntercept].allSatisfy(\.isFinite),
                                "\(prefix).empirical", "coefficients must be finite")
                case .none:
                    try require(mu == nil && empirical == nil, prefix, "rule none takes neither mu nor empirical")
                }
            }
        }

        public struct Sigmas: Sendable, Hashable, Codable {
            public var base: Base
            public var shift: Shift
            /// The value appended after the last step. Only 0 is defined.
            public var terminal: Double

            public init(base: Base = .linspace, shift: Shift, terminal: Double = 0) {
                self.base = base
                self.shift = shift
                self.terminal = terminal
            }
        }

        /// `{kind: none}` (distilled models) or `{kind: cfg, scale}`.
        public struct Guidance: Sendable, Hashable, Codable {
            public enum Kind: String, Sendable, Hashable, Codable { case none, cfg }
            public var kind: Kind
            public var scale: Double?

            public init(kind: Kind, scale: Double? = nil) {
                self.kind = kind
                self.scale = scale
            }
        }

        public var kind: Kind
        /// The default step count.
        public var steps: Int
        /// The largest step count a request may ask for. Absent: `steps`.
        public var maxSteps: Int?
        public var trainTimesteps: Int
        public var sigmas: Sigmas
        public var timestep: Timestep
        public var update: Update
        public var guidance: Guidance

        public init(kind: Kind = .flowMatchEuler, steps: Int, maxSteps: Int? = nil, trainTimesteps: Int = 1000,
                    sigmas: Sigmas, timestep: Timestep = .sigmaRoundTrip, update: Update = .eulerVelocity,
                    guidance: Guidance = Guidance(kind: .none)) {
            self.kind = kind
            self.steps = steps
            self.maxSteps = maxSteps
            self.trainTimesteps = trainTimesteps
            self.sigmas = sigmas
            self.timestep = timestep
            self.update = update
            self.guidance = guidance
        }

        /// The step-count bound requests are checked against.
        public var stepLimit: Int { maxSteps ?? steps }

        func validate() throws(AetherError) {
            try require(steps >= 1, "schedule.steps", "must be >= 1")
            if let maxSteps { try require(maxSteps >= steps, "schedule.maxSteps", "must be >= steps") }
            try require(trainTimesteps >= 1, "schedule.trainTimesteps", "must be >= 1")
            try require(sigmas.terminal == 0, "schedule.sigmas.terminal", "only 0 is defined")
            try sigmas.shift.validate(at: "schedule.sigmas.shift")
            switch guidance.kind {
            case .none:
                try require(guidance.scale == nil, "schedule.guidance.scale", "guidance none takes no scale")
            case .cfg:
                try require(guidance.scale.map { $0.isFinite && $0 > 0 } == true, "schedule.guidance.scale",
                            "cfg needs a finite scale > 0")
            }
        }
    }

    /// `latent` (MANIFEST.md §5.2): the image-latent contract a driver packs, noises and unpacks.
    public struct Latent: Sendable, Hashable, Codable {
        public enum Packing: String, Sendable, Hashable, Codable {
            /// 2×2 patches, channel-major: packed channel `c·4 + 2·dy + dx`.
            case patch2x2ChannelMajor
        }

        /// `{kind: splitmix64BoxMuller, order: packedToken, dtype: float32}`: the seeded stream, read in the order the
        /// denoiser's latents are laid out.
        public struct Noise: Sendable, Hashable, Codable {
            public enum Kind: String, Sendable, Hashable, Codable {
                /// SplitMix64 with double-precision Box–Muller, prefix-stable; the 0.0.x stream.
                case splitmix64BoxMuller
            }
            public enum Order: String, Sendable, Hashable, Codable {
                /// Value `i` is element `i` of the packed `[tokens, channels]` latents.
                case packedToken
            }

            public var kind: Kind
            public var order: Order
            public var dtype: ScalarType

            public init(kind: Kind = .splitmix64BoxMuller, order: Order = .packedToken, dtype: ScalarType = .float32) {
                self.kind = kind
                self.order = order
                self.dtype = dtype
            }
        }

        /// Per-token channels the denoiser sees.
        public var channels: Int
        /// Image pixels per latent token along each side; also the multiple width and height must be.
        public var pixelsPerToken: Int
        public var packing: Packing
        /// Descriptive: the denoiser's and the decoder's latent layouts, e.g. `tokens[rows*columns, 64]`.
        public var layout: String
        public var decoderLayout: String
        public var noise: Noise
        /// The largest `rows · columns` the export supports.
        public var maxImageTokens: Int?

        public init(channels: Int, pixelsPerToken: Int, packing: Packing = .patch2x2ChannelMajor, layout: String,
                    decoderLayout: String, noise: Noise = Noise(), maxImageTokens: Int? = nil) {
            self.channels = channels
            self.pixelsPerToken = pixelsPerToken
            self.packing = packing
            self.layout = layout
            self.decoderLayout = decoderLayout
            self.noise = noise
            self.maxImageTokens = maxImageTokens
        }

        func validate() throws(AetherError) {
            try require(channels >= 1, "latent.channels", "must be >= 1")
            if packing == .patch2x2ChannelMajor {
                try require(channels % 4 == 0, "latent.channels", "patch2x2ChannelMajor packs 4 values per VAE channel")
                try require(pixelsPerToken >= 2 && pixelsPerToken % 2 == 0, "latent.pixelsPerToken",
                            "must be an even number >= 2 for 2x2 patches")
            } else {
                try require(pixelsPerToken >= 1, "latent.pixelsPerToken", "must be >= 1")
            }
            try require(!layout.isEmpty, "latent.layout", "must not be empty")
            try require(!decoderLayout.isEmpty, "latent.decoderLayout", "must not be empty")
            try require(noise.dtype == .float32, "latent.noise.dtype", "the seeded stream is float32")
            if let maxImageTokens { try require(maxImageTokens >= 1, "latent.maxImageTokens", "must be >= 1") }
        }

        /// The token grid for an image of `width` × `height` pixels: each side a positive multiple of
        /// `pixelsPerToken`, and at most `maxImageTokens` tokens. Errors name the request, not the manifest.
        public func tokenGrid(width: Int, height: Int) throws(AetherError) -> (rows: Int, columns: Int) {
            guard width > 0, height > 0, width % pixelsPerToken == 0, height % pixelsPerToken == 0 else {
                throw .generation(.invalidParameters(
                    "width and height must be positive multiples of \(pixelsPerToken), not \(width)x\(height)"))
            }
            let rows = height / pixelsPerToken, columns = width / pixelsPerToken
            let (tokens, overflow) = rows.multipliedReportingOverflow(by: columns)
            if overflow || maxImageTokens.map({ tokens > $0 }) == true {
                throw .generation(.invalidParameters(
                    "\(width)x\(height) is \(rows)x\(columns) latent tokens; this model allows at most \(maxImageTokens ?? Int.max)"))
            }
            return (rows, columns)
        }
    }
}
