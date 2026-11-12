import AetherCore
import AetherHost
import AetherRuntime
import Foundation
import Testing
@testable import AetherImage

@Suite("Krea 2 tiled VAE")
struct Krea2VAETilingTests {
    @Test func validatesPixelAlignmentAndOverlap() throws {
        for (size, overlap) in [(0, 0), (16, 0), (33, 16), (64, -16), (64, 64), (64, 17), (Int.max, 0)] {
            #expect(throws: AetherError.self) { try Krea2VAETiling(tileSize: size, overlap: overlap) }
        }
        _ = try Krea2VAETiling(tileSize: 32, overlap: 0)
        _ = try Krea2VAETiling(tileSize: 512, overlap: 256)
    }

    @Test func edgesKeepOneShapeAndCoverRectangles() throws {
        let plan = try Krea2VAETiling(tileSize: 512, overlap: 128)
        #expect(plan.axis(1024).starts == [0, 384, 512])
        #expect(plan.axis(1008).starts == [0, 384, 496])
        #expect(plan.axis(384).size == 384 && plan.axis(384).starts == [0])
        for extent in [32, 384, 512, 528, 768, 1008, 1024, 2048] {
            let axis = plan.axis(extent)
            var weights = [Float](repeating: 0, count: extent)
            for (i, start) in axis.starts.enumerated() {
                #expect(start % 16 == 0 && start + axis.size <= extent)
                for j in 0..<axis.size { weights[start + j] += axis.weights[i][j] }
            }
            #expect(weights.allSatisfy { $0 > 0 && $0.isFinite })
        }
    }

    @Test(arguments: [(48, 32), (80, 48), (32, 32), (16, 64)])
    func blendPreservesChannelsAndGlobalCoordinates(size: (Int, Int)) throws {
        let (width, height) = size
        let tiling = try Krea2VAETiling(tileSize: 32, overlap: 16)
        var result = Krea2VAETiling.Accumulator(width: width, height: height, tiling: tiling)
        for (iy, y) in result.y.starts.enumerated() {
            for (ix, x) in result.x.starts.enumerated() {
                let w = result.x.size, h = result.y.size
                let pixels = (0..<3 * w * h).map { i in
                    Float(i / (w * h)) + Float(x + i % w) / 1000 + Float(y + i / w % h) / 100
                }
                try result.add(pixels, row: iy, column: ix)
            }
        }
        let output = try result.finish()
        for i in output.indices {
            let expected = Float(i / (width * height)) + Float(i % width) / 1000 + Float(i / width % height) / 100
            #expect(abs(output[i] - expected) < 0.000002)
        }
    }

    @Test func tiledPipelineMatchesLocalDecoderAndReusesBuffers() async throws {
        let calls = Krea2Tests.Calls()
        let (pipeline, _) = try Krea2Tests.pipeline(calls: calls, wrap: { role, graph in
            role == "vaeDecoder" ? Krea2Tests.MemoryChecked(graph, role: role, checks: Krea2Tests.Calls()) : graph
        })
        let request = ImageRequest(prompt: "a cat", seed: 7, steps: 2, width: 80, height: 48)
        let full = try await pipeline.generate(request)
        try await pipeline.setDecoderTiling(tileSize: 32, overlap: 16)
        #expect(await pipeline.residentRoles.isEmpty, "changing decode shape releases old specializations")
        let tiled = try await pipeline.generate(request)
        #expect(zip(full.planarRGB, tiled.planarRGB).allSatisfy { abs($0 - $1) < 0.000002 })
        let resident = await pipeline.residentRoles
        try await pipeline.setDecoderTiling(tileSize: 32, overlap: 16)
        #expect(await pipeline.residentRoles == resident, "setting the same plan keeps warm graphs")
        #expect(try await pipeline.generate(request) == tiled)
        #expect(pipeline.options.ledger.snapshot.keys.contains { $0.profile.contains("vae-tiles-32-16") })
        try await pipeline.setDecoderTiling(tileSize: nil)
        #expect(try await pipeline.generate(request) == full)
        // One full decode, eight tiles per tiled render, then the restored full decode.
        #expect(calls.log.filter { $0 == "vae" }.count == 18)
        await pipeline.unload()
    }

    @Test func cancellationStopsBeforeTheNextTileAndReleasesGraphs() async throws {
        let calls = Krea2Tests.Calls(), filter = Krea2Tests.RecordingFilter()
        let (pipeline, _) = try Krea2Tests.pipeline(calls: calls, filter: filter, wrap: { role, graph in
            role == "vaeDecoder"
                ? LatentDiffusionTests.Tracked(graph, role: role, log: Krea2Tests.Calls(), afterRun: LatentDiffusionTests.cancelCurrentTask)
                : graph
        })
        try await pipeline.setDecoderTiling(tileSize: 32, overlap: 16)
        await #expect(throws: CancellationError.self) {
            try await pipeline.generate(.init(prompt: "a cat", steps: 1, width: 80, height: 48))
        }
        #expect(calls.log.filter { $0 == "vae" }.count == 1)
        #expect(filter.log == ["prompt a cat"])
        #expect(await pipeline.residentRoles.isEmpty)
    }
}
