import AetherKit
import AetherCoreAI
import AetherKrea
import AetherKreaTokenizers
import CoreAI
import Foundation

@available(macOS 27, iOS 27, *)
@main
struct AetherProbe {
    static func main() async {
        let arguments = Array(CommandLine.arguments.dropFirst())
        // exit() skips deinit, so the lease is released explicitly on both paths.
        var lease: SharedGPULease?
        do {
            lease = try await ProbeGPULease.acquire(for: arguments)
            try await run(arguments)
            lease?.release()
        } catch {
            lease?.release()
            FileHandle.standardError.write(Data("aether-probe: \(error)\n".utf8))
            exit(1)
        }
    }

    static func run(_ arguments: [String]) async throws {
        guard let command = arguments.first, command != "--help" else {
            print("""
            Usage:
              aether-probe h3-profile-dump PROFILE.json PROMPT.txt WIDTH HEIGHT FRAMES SEED RECIPE NEW_DIRECTORY [FIRST_FRAME.png]
              aether-probe h3-profile-run PROFILE.json DUMP REFERENCES|- NEW_DIRECTORY --graphs|--render [--no-audio]
              aether-probe h3-profile-decode PROFILE.json DUMP RENDER NEW_DIRECTORY [DECODER.aimodel SHA256]
              aether-probe h3-memory-render PROFILE.json DUMP NEW_DIRECTORY [--spike-capture=GiB]
              aether-probe qualify-h3-session-lifecycle PROFILE.json DUMP NEW_DIRECTORY [--scenarios=NAME,...]
              aether-probe qualify-h3-denoising BUNDLE PHASES CPU_RECORD CHAIN MODULATION_DIRECTORY REFERENCES|- NEW_OUTPUT_DIRECTORY --prepare-only|--capture|--run
              aether-probe qualify-h3-phases BUNDLE.json PHASES.json CPU_CHECKS.json NEW_OUTPUT_DIRECTORY --prepare-only|--run
              aether-probe qualify-h3 BUNDLE.json PHASES.json NEW_OUTPUT_DIRECTORY --prepare-only|--run
              aether-probe qualify-h3-lifecycle BUNDLE.json PHASES.json NEW_OUTPUT_DIRECTORY --prepare-only|--run
              aether-probe qualify-h3-chain BUNDLE.json PHASES.json CHAIN.json NEW_OUTPUT_DIRECTORY --prepare-only|--capture|--run
              aether-probe doctor
              aether-probe inspect MODEL.aimodel [automatic|cpuOnly|preferGPU]
              aether-probe smoke MODEL.aimodel [automatic|cpuOnly|preferGPU]
              aether-probe prepare-krea BUNDLE PROMPT [WIDTH HEIGHT]
              aether-probe verify-krea BUNDLE
              aether-probe render-krea BUNDLE OUTPUT.png PROMPT WIDTH HEIGHT SEED [SHAPE.json]
              aether-probe render-krea-batch BUNDLE NEW_OUTPUT_DIRECTORY PROMPT WIDTH HEIGHT SEED[,SEED...] [SHAPE.json]
              aether-probe parity-krea BUNDLE LAB FIXTURES SHAPE.json OUTPUT_DIRECTORY provided|encoded
              aether-probe benchmark-krea BUNDLE PLAN.json SOURCE_ROOT OUTPUT_DIRECTORY --prepare-only|--run
              aether-probe benchmark-flux2 BUNDLE T2I_FIXTURE REFERENCE_FIXTURE|- PLAN.json SOURCE_ROOT OUTPUT_DIRECTORY --prepare-only|--run
            \(CatalogEntryCommand.usage.split(separator: "\n").map { "  " + $0 }.joined(separator: "\n"))
            \(PrepareModelCommand.usage.split(separator: "\n").map { "  " + $0 }.joined(separator: "\n"))

            smoke expects Forge's float32 affine fixture: y = x * 2 + 1, shape [1, 4].
            inspect specializes/loads the model; it is not a static file inspection.
            prepare-krea reads local assets/tokenizer and prepares inputs without native inference.
            verify-krea hashes all six model streams and can read tens of GB.
            catalog-entry prints the pinned catalog entry an app prepares the bundle from (docs/model-preparation.md).
            prepare-model qualifies preparation over HTTPS (HF_TOKEN for a gated repository); reply-installed opens the
            installation through the language facade and is a GPU command.

            GPU commands (inspect, smoke, render-krea, render-krea-batch, parity-krea, h3-profile-run,
            h3-profile-decode, h3-memory-render, h3-memory-encoder, qualify-h3-session-lifecycle, and the other
            qualify-h3*, benchmark-krea and benchmark-flux2 modes other than --prepare-only) take the
            shared GPU lease /tmp/atelier-gpu.lock themselves, waiting while Atelier or another tool holds it.
            Malformed commands print their usage error without waiting.
              AETHER_GPU_LEASE=off              do not take the lease (special cases only)
              AETHER_GPU_LEASE_TIMEOUT=SECONDS  stop waiting after SECONDS (at most one year); 0 fails fast
              AETHER_GPU_LEASE_PATH=/abs/dir    use another lease directory
              AETHER_GPU_LEASE_HOLDER=PID       an ancestor (such as a wrapper) holding the lease hands it to this
                                                command; without it, a lease an ancestor holds is waited for
            """)
            return
        }
        if try CatalogEntryCommand.run(arguments) { return }
        if try await PrepareModelCommand.run(arguments) { return }
        if try await H3ProfileCommand.run(arguments) { return }
        if try await H3MemoryDiagnosis.run(arguments) { return }
        if try await H3SessionLifecycleCommand.run(arguments) { return }
        if try await H3DenoisingCommand.run(arguments) { return }
        if try await H3PhaseCommand.run(arguments) { return }
        if try await H3QualificationCommand.run(arguments) { return }
        if try await H3LifecycleCommand.run(arguments) { return }
        if try await H3ChainCommand.run(arguments) { return }
        if try await KreaBenchmarkCommand.run(arguments) { return }
        if try await Flux2BenchmarkCommand.run(arguments) { return }
        if try await KreaBatchCommand.run(arguments) { return }
        if try await KreaCommands.run(arguments) { return }
        if command == "doctor", arguments.count == 1 {
            let report = [
                "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "architecture": AIModel.deviceArchitectureName,
                "computeUnits": ComputeUnitKind.availableKinds.map { String(describing: $0) }.sorted().joined(separator: ", "),
                "physicalMemoryBytes": String(ProcessInfo.processInfo.physicalMemory),
            ]
            try printJSON(report)
            return
        }
        if command == "verify-krea", arguments.count == 2 {
            let assets = try KreaAssets.open(at: URL(fileURLWithPath: arguments[1]))
            try assets.verify()
            try printJSON(["verified": "true", "fingerprint": assets.fingerprint])
            return
        }
        if command == "prepare-krea", arguments.count == 3 || arguments.count == 5 {
            let width = arguments.count == 5 ? Int(arguments[3]) : 1024
            let height = arguments.count == 5 ? Int(arguments[4]) : 1024
            guard let width, let height else { throw AetherError.invalidRequest("invalid image dimensions") }
            let configuration = try KreaConfiguration(width: width, height: height)
            let assets = try KreaAssets.open(at: URL(fileURLWithPath: arguments[1]))
            let tokenizer = try await KreaTokenizer.open(assets: assets)
            let prompt = try tokenizer.prepare(arguments[2])
            let rotary = try KreaRotary.image(configuration: configuration)
            try printJSON(KreaPreparationReport(
                fingerprint: assets.fingerprint, tokenIDs: prompt.tokenIDs,
                conditioningRows: prompt.conditioningRows, wasTruncated: prompt.wasTruncated,
                encoderInputShapes: try prompt.encoderInputs.mapValues(\.shape),
                rotaryShape: rotary.shape, imageTokens: configuration.imageTokenCount
            ))
            return
        }
        guard ["inspect", "smoke"].contains(command), (2...3).contains(arguments.count) else {
            throw AetherError.invalidRequest("unknown command or arguments; use --help")
        }
        let policyName = arguments.count == 3 ? arguments[2] : "automatic"
        guard let policy = ModelComputePolicy(rawValue: policyName) else {
            throw AetherError.invalidRequest("unknown compute policy '\(policyName)'")
        }
        let clock = ContinuousClock()
        let started = clock.now
        let session = try await ModelSession.open(at: URL(fileURLWithPath: arguments[1]), compute: policy)
        let loadDuration = started.duration(to: clock.now)
        do {
            if command == "inspect" {
                try printJSON(session.functions)
            } else {
                let input = try FloatTensor(shape: [1, 4], values: [-2, 0, 1, 3])
                var durations: [Double] = []
                var retained: FloatTensor?
                for _ in 0..<2 {
                    let start = clock.now
                    let output = try await session.run(inputs: ["x": input])
                    durations.append(seconds(start.duration(to: clock.now)))
                    guard output.count == 1, let value = output["y"],
                          value.shape == [1, 4], value.values == [-3, 1, 3, 7] else {
                        throw AetherError.invalidTensor("affine fixture output mismatch")
                    }
                    retained = value
                }
                await session.unload()
                guard let retained, retained.values == [-3, 1, 3, 7], await session.state == .closed else {
                    throw AetherError.invalidTensor("output ownership/unload check failed")
                }
                try printJSON(SmokeReport(
                    passed: true, computePolicy: policy.rawValue,
                    loadSeconds: seconds(loadDuration), runSeconds: durations,
                    output: retained.values
                ))
            }
            await session.unload()
        } catch {
            await session.unload()
            throw error
        }
    }

    static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    static func printJSON(_ value: some Encodable) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        print(String(decoding: try encoder.encode(value), as: UTF8.self))
    }

    struct SmokeReport: Encodable {
        let passed: Bool
        let computePolicy: String
        let loadSeconds: Double
        let runSeconds: [Double]
        let output: [Float]
    }

    struct KreaPreparationReport: Encodable {
        let fingerprint: String
        let tokenIDs: [Int32]
        let conditioningRows: Int
        let wasTruncated: Bool
        let encoderInputShapes: [String: [Int]]
        let rotaryShape: [Int]
        let imageTokens: Int
    }
}
