import AetherKit
import AetherH3CoreAI
import Synchronization

/// Records native ownership boundaries without delaying a submitted GPU stream.
@available(macOS 27, iOS 27, *)
final class H3DenoisingLifecycleTrace: Sendable {
    private struct State {
        var events: [String] = []
        var cancel: (@Sendable () -> Void)?
    }
    private let state = Mutex(State())
    var events: [String] { state.withLock { $0.events } }
    func reset(cancel: (@Sendable () -> Void)? = nil) {
        state.withLock { $0 = State(cancel: cancel) }
    }
    func append(_ event:String) { state.withLock { $0.events.append(event) } }
    func record(_ event:H3DenoisingExecutionEvent) {
        let action=state.withLock { value -> (@Sendable () -> Void)? in
            let key=Self.key(event.stage,event.step,event.block)
            value.events.append(event.stage == "session" ? event.event.rawValue : key+":"+event.event.rawValue)
            guard key == "body.0.24",event.event == .nativeSubmitted,let action=value.cancel else { return nil }
            value.cancel=nil;value.events.append("cancellationRequested");return action
        }
        action?()
    }
    private static func key(_ stage:String,_ step:Int,_ block:Int?) -> String { "\(stage).\(step).\(block ?? -1)" }
    static var nativeSequence:[String] {
        var keys=(0..<8).map { key("time",$0,nil) }
        for block in 0..<50 { for step in 0..<8 { keys.append(key("modulation",step,block)) } }
        for step in 0..<8 {
            keys.append(key("prepare",step,nil))
            for block in 0..<50 { keys.append(key("body",step,block)) }
            keys.append(key("heads",step,nil))
        }
        return keys
    }
    static func validate(_ events:[String],cancelled:Bool,closing:Bool) throws {
        let keys=cancelled ? Array(nativeSequence.prefix(434)) : nativeSequence
        let native=events.filter { $0.contains(":") }
        guard native == keys.flatMap({ [$0+":nativeSubmitted",$0+":nativeCompleted"] }) else {
            throw AetherError.invalidModel("H3 trajectory native submission/completion prefix differs")
        }
        func position(_ name:String) throws -> Int {
            guard events.filter({ $0 == name }).count == 1,let i=events.firstIndex(of:name) else {
                throw AetherError.invalidModel("H3 trajectory lifecycle missing/duplicate \(name)")
            }
            return i
        }
        let release=try position("slotReleased"),complete=try position(keys.last!+":nativeCompleted")
        guard complete < release else { throw AetherError.invalidModel("H3 trajectory ownership released before join") }
        var controls=["slotReleased"]
        if cancelled {
            let request=try position("cancellationRequested"),submitted=try position("body.0.24:nativeSubmitted")
            let returned=try position("runCancelled")
            guard submitted < request,request < complete,release < returned else {
                throw AetherError.invalidModel("H3 trajectory cancellation was not joined before return")
            }
            controls += ["cancellationRequested","runCancelled"]
            if closing {
                let start=try position("unloadStarted"),end=try position("unloaded")
                guard request < start,start < complete,complete < end else {
                    throw AetherError.invalidModel("H3 trajectory concurrent unload missed pending GPU work")
                }
                controls += ["unloadStarted","unloaded"]
            }
        }
        guard events.filter({ !$0.contains(":") }).sorted() == controls.sorted() else {
            throw AetherError.invalidModel("H3 trajectory lifecycle has unexpected control events")
        }
    }
}
