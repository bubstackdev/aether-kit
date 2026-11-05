import CoreAI
import Testing
@testable import AetherRuntime

@Suite("GraphOptions")
struct GraphOptionsTests {
    @Test func comeFromTheManifestAndNeverDefault() throws {
        let graph = try #require(try Fixtures.manifest("qwen3-0.6b").graphs["decoder"])
        let options = try GraphOptions(manifest: graph, at: "graphs.decoder", cachePolicy: .persistent, cache: .default)
        #expect(options.computeUnit == .gpu && options.expectFrequentReshapes)
        let specialization = options.specializationOptions
        #expect(specialization.preferredComputeUnitKind == .gpu)
        #expect(specialization.expectFrequentReshapes)
        #expect(specialization != .default)
        #expect(options.aiCachePolicy == .persistent)

        var bare = graph
        bare.specialization = nil
        expectField("graphs.decoder.specialization") {
            _ = try GraphOptions(manifest: bare, at: "graphs.decoder", cachePolicy: .default, cache: .default)
        }
    }

    @Test(arguments: Manifest.ComputeUnit.allCases)
    func everyComputeUnitMaps(_ unit: Manifest.ComputeUnit) {
        let options = GraphOptions(computeUnit: unit, expectFrequentReshapes: false, cachePolicy: .default, cache: .default)
        let expected: ComputeUnitKind = switch unit {
        case .cpu: .cpu
        case .gpu: .gpu
        case .neuralEngine: .neuralEngine
        }
        #expect(options.specializationOptions.preferredComputeUnitKind == expected)
        #expect(!options.specializationOptions.expectFrequentReshapes)
    }
}
