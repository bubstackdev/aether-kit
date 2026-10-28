import AetherKit
import Foundation
import Synchronization
import Testing
@testable import AetherProbe

@Test func h3MemoryRenderOptionsParseTheOptionalSpikeCapture() throws {
    let plain = try H3MemoryDiagnosis.RenderOptions(["h3-memory-render", "/p.json", "/dump", "/out"])
    #expect(plain.spikeCapture == nil && plain.output.path == "/out")
    let spike = try H3MemoryDiagnosis.RenderOptions(["h3-memory-render", "/p.json", "/dump", "/out", "--spike-capture=12.5"])
    #expect(spike.spikeCapture == UInt64(12.5 * Double(1 << 30)))
    for bad in [["h3-memory-render", "/p.json", "/dump"], ["h3-memory-render", "/p.json", "/dump", "/out", "--spike-capture=0"],
                ["h3-memory-render", "/p.json", "/dump", "/out", "--spike-capture=x"], ["h3-memory-render", "/p.json", "/dump", "/out", "--other=1"],
                ["h3-memory-render", "/p.json", "/dump", "/out", "--spike-capture=12", "--spike-capture=13"]] {
        #expect(throws: AetherError.self) { _ = try H3MemoryDiagnosis.RenderOptions(bad) }
    }
}

@Test func h3SpikeTriggerFiresOnceOnlyForADenoisingSampleAboveTheThreshold() {
    let events = Mutex([H3SpikeTrigger.Event]())
    let trigger = H3SpikeTrigger(threshold: 100) { event in events.withLock { $0.append(event) } }
    func sample(_ t: Double, _ phase: String, _ metal: UInt64) -> H3MemoryTimeline.Sample {
        H3MemoryTimeline.Sample(t: t, phase: phase, metal: metal, footprint: 7, resident: 0)
    }
    trigger.observe(sample(0, "stage.encoding", 500))        // above, but not denoising
    trigger.observe(sample(1, "stage.denoising", 90))
    trigger.observe(sample(2, "stage.denoising", 100))       // at the threshold: not above
    #expect(!trigger.fired)
    trigger.observe(sample(3.5, "stage.denoising", 101))
    trigger.observe(sample(4, "stage.denoising.after-step1", 900))
    #expect(trigger.fired)
    #expect(events.withLock { $0 } == [H3SpikeTrigger.Event(phase: "stage.denoising", t: 3.5, metal: 101, footprint: 7, secondsIntoPhase: 2.5)])
}
