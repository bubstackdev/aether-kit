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
    }

    @Test func reshapeHintDefaultsFromTheSignature() throws {
        var graph = try #require(try Fixtures.manifest("qwen3.5-0.8b").graphs["decoder"])
        graph.expectFrequentReshapes = nil
        expectField("graphs.decoder.expectFrequentReshapes") {
            _ = try GraphOptions(manifest: graph, at: "graphs.decoder", cachePolicy: .default, cache: .default)
        }
        let signature = try #require(try Fixtures.signatures("qwen3.5-0.8b")["main"])
        #expect(try GraphOptions(manifest: graph, at: "g", signature: signature, cachePolicy: .default, cache: .default)
            .expectFrequentReshapes)
        var static_ = signature
        static_.states = static_.states.filter { !$0.isDynamic }
        static_.inputs = static_.inputs.map { TensorSpec(name: $0.name, scalarType: $0.scalarType, shape: [1, 1]) }
        #expect(!(try GraphOptions(manifest: graph, at: "g", signature: static_, cachePolicy: .default, cache: .default)
            .expectFrequentReshapes))
        graph.compute = .neuralEngine
        #expect(!(try GraphOptions(manifest: graph, at: "g", cachePolicy: .default, cache: .default).expectFrequentReshapes))
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
