import AetherKit
import AetherH3CoreAI
import Testing
@testable import AetherProbe

@Test func denoisingLifecycleValidatesExactNativePrefixAndJoinedCancellation() throws {
    let keys=H3DenoisingLifecycleTrace.nativeSequence
    #expect(keys.count == 824)
    let full=keys.flatMap { [$0+":nativeSubmitted",$0+":nativeCompleted"] }+["slotReleased"]
    try H3DenoisingLifecycleTrace.validate(full,cancelled:false,closing:false)
    let prefix=Array(keys.prefix(433)).flatMap { [$0+":nativeSubmitted",$0+":nativeCompleted"] }
    let tail=["body.0.24:nativeSubmitted","cancellationRequested","unloadStarted","body.0.24:nativeCompleted","slotReleased","runCancelled","unloaded"]
    let valid=prefix+tail
    try H3DenoisingLifecycleTrace.validate(valid,cancelled:true,closing:true)
    try H3DenoisingLifecycleTrace.validate(valid.filter { !["unloadStarted","unloaded"].contains($0) },cancelled:true,closing:false)
    for invalid in [
        prefix+[tail[0],tail[1],tail[3],tail[2]]+Array(tail.suffix(3)),
        prefix+Array(tail.prefix(3))+["slotReleased",tail[3],"runCancelled","unloaded"],
        valid+["body.0.25:nativeSubmitted","body.0.25:nativeCompleted"],
        valid.filter { $0 != "time.0.-1:nativeCompleted" },
        valid+["runSucceeded"],
        valid+["slotReleased"],
    ] {
        #expect(throws:AetherError.self) { try H3DenoisingLifecycleTrace.validate(invalid,cancelled:true,closing:true) }
    }
    let trace=H3DenoisingLifecycleTrace()
    trace.reset { trace.append("cancelCallback") }
    trace.record(.init(stage:"body",step:0,block:23,event:.nativeSubmitted))
    trace.record(.init(stage:"body",step:0,block:24,event:.nativeSubmitted))
    trace.record(.init(stage:"body",step:0,block:24,event:.nativeCompleted))
    #expect(trace.events == ["body.0.23:nativeSubmitted","body.0.24:nativeSubmitted","cancellationRequested","cancelCallback","body.0.24:nativeCompleted"])
}
