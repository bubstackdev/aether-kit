import AetherAssets
import AetherKit
import AetherLanguage
import Foundation

/// The app's qualification mode (docs/sample-app.md): the same controller actions the buttons call, in order, with a
/// record of what happened. The app runs it when launched with `-qualify PHASES`.
public struct QualificationRecord: Codable, Sendable {
    public struct Transition: Codable, Sendable { public var seconds: Double; public var state: String }

    public var phases: [String] = []
    public var startedAt = ""
    public var platform = ""
    public var os = ""
    public var hardwareModel = ""
    public var physicalMemoryBytes: UInt64 = 0
    public var deviceMemoryLimitBytes: UInt64?
    /// macOS: whether the process runs in App Sandbox (it has a sandbox container). iOS apps always do.
    public var sandboxed = false
    /// Whether the model store is inside the app's own container (Library/Application Support/Models).
    public var storeInsideAppContainer = false
    public var repository = ""
    public var revision = ""
    public var subdirectory = ""
    public var installationID = ""
    public var entryFiles = 0
    public var entryBytes: UInt64 = 0
    public var idleUnloadAfterSeconds: Double?

    public var readiness: [Transition] = []
    public var lifecycle: [Transition] = []
    public var pressureLevels: [String] = []

    public var initialState = ""
    public var downloadSeconds: Double?
    public var verifiedReadySeconds: Double?

    public var chatPrompt = ""
    public var chatReply = ""
    public var chatFootnote = ""
    public var chatFirstTextSeconds: Double?
    public var chatSeconds: Double?

    public var structuredSentence = ""
    public var structuredJSON = ""
    public var structuredSeconds: Double?
    public var structuredError: String?

    public var idleUnloadSecondsAfterReply: Double?
    public var reloadReply = ""
    public var reloadSeconds: Double?

    public var stateAfterDelete = ""
    public var storeBytesAfterDelete: UInt64?

    public var failures: [String] = []
    public var passed = false
}

@MainActor
public final class QualificationRun {
    public enum Phase: String, CaseIterable, Sendable {
        /// Delete anything installed, so the run starts from nothing.
        case fresh
        /// Download, install, verify: until ready.
        case prepare
        /// A streamed chat reply, then the structured demo.
        case run
        /// Wait for the idle unload, then a reply that loads the model again.
        case idle
        /// Delete the installation.
        case delete
    }

    public static let chatPrompt = "In one sentence, what is a hash function?"
    public static let reloadPrompt = "Name one common hash function."

    let app: SampleAppState
    let phases: [Phase]
    let log: (String) -> Void
    let started = ContinuousClock.now
    public private(set) var record = QualificationRecord()

    public init(app: SampleAppState, phases: [Phase], idleUnloadAfter: Duration?, log: @escaping (String) -> Void) {
        self.app = app
        self.phases = phases
        self.log = log
        record.idleUnloadAfterSeconds = idleUnloadAfter.map(Self.seconds)
    }

    public func run(storeRoot: URL) async -> QualificationRecord {
        describeHost(storeRoot: storeRoot)
        app.model.stateDidChange = { [weak self] state in
            guard let self else { return }
            if record.readiness.last?.state != state {
                record.readiness.append(.init(seconds: now, state: state))
                log("READINESS \(state)")
            }
        }
        app.model.lifecycleDidChange = { [weak self] state in
            guard let self else { return }
            record.lifecycle.append(.init(seconds: now, state: "\(state)"))
            log("LIFECYCLE \(state)")
        }
        app.model.start()
        await waitFor("an initial readiness state", timeout: 120) { self.app.model.stateName != "starting" && !self.app.model.stateName.hasPrefix("verifying") }
        record.initialState = app.model.stateName
        do {
            for phase in phases {
                log("PHASE \(phase.rawValue)")
                switch phase {
                case .fresh: try await fresh()
                case .prepare: try await prepare()
                case .run: try await chatAndStructured()
                case .idle: try await idleAndReload()
                case .delete: try await delete(storeRoot: storeRoot)
                }
            }
        } catch {
            record.failures.append("\(error)")
        }
        record.pressureLevels = ["\(app.model.pressure)"]
        record.passed = record.failures.isEmpty
        log("QUALIFICATION \(record.passed ? "PASSED" : "FAILED") \(record.failures)")
        return record
    }

    struct Failure: Error, CustomStringConvertible { let description: String }

    private func fresh() async throws {
        if app.model.stateName == "ready" || app.model.display.actions.contains(.delete) {
            await app.model.perform(.delete)
            try await expect("notInstalled after deleting", timeout: 120) { self.app.model.stateName.hasPrefix("notInstalled(0/") }
        }
    }

    private func prepare() async throws {
        guard app.model.stateName != "ready" else { return }
        let begun = now
        await app.model.perform(.download)
        try await expect("verifying after the download", timeout: 3600) {
            self.app.model.stateName == "verifying" || self.app.model.stateName == "ready" || self.app.model.stateName.hasPrefix("failed")
        }
        record.downloadSeconds = now - begun
        try await expect("ready", timeout: 600) { self.app.model.stateName == "ready" && self.app.model.model != nil }
        record.verifiedReadySeconds = now - begun
    }

    private func chatAndStructured() async throws {
        try await expect("ready", timeout: 600) { self.app.chat.isAvailable && self.app.todo.isAvailable }
        record.chatPrompt = Self.chatPrompt
        let begun = now
        app.chat.send(Self.chatPrompt)
        await waitFor("the first text", timeout: 300) { !(self.app.chat.messages.last?.text.isEmpty ?? true) || !self.app.chat.isResponding }
        record.chatFirstTextSeconds = now - begun
        await app.chat.waitForReply()
        record.chatSeconds = now - begun
        let reply = app.chat.messages.last
        record.chatReply = reply?.text ?? ""
        record.chatFootnote = reply?.footnote ?? ""
        if record.chatReply.isEmpty || record.chatFootnote.hasPrefix("failed") || record.chatFootnote == "stopped" {
            throw Failure(description: "chat: \(record.chatFootnote)")
        }

        record.structuredSentence = TodoController.example
        let structuredBegun = now
        app.todo.run(TodoController.example)
        await app.todo.waitForResult()
        record.structuredSeconds = now - structuredBegun
        record.structuredJSON = app.todo.json ?? ""
        record.structuredError = app.todo.error
        if app.todo.item == nil { throw Failure(description: "structured: \(app.todo.error ?? "no item")") }
    }

    private func idleAndReload() async throws {
        guard let idle = record.idleUnloadAfterSeconds else { throw Failure(description: "idle: no idle policy") }
        let replied = now
        try await expect("unloaded(idle)", timeout: idle * 3 + 30) { self.app.model.lifecycleState == .unloaded(.idle) }
        record.idleUnloadSecondsAfterReply = now - replied
        let begun = now
        await app.chat.newChat()
        app.chat.send(Self.reloadPrompt)
        await app.chat.waitForReply()
        record.reloadSeconds = now - begun
        record.reloadReply = app.chat.messages.last?.text ?? ""
        if record.reloadReply.isEmpty { throw Failure(description: "reload: \(app.chat.messages.last?.footnote ?? "no reply")") }
    }

    private func delete(storeRoot: URL) async throws {
        await app.model.perform(.delete)
        if let error = app.model.actionError { throw Failure(description: "delete: \(error)") }
        try await expect("notInstalled after deleting", timeout: 120) { self.app.model.stateName.hasPrefix("notInstalled(0/") }
        record.stateAfterDelete = app.model.stateName
        record.storeBytesAfterDelete = Self.bytes(under: storeRoot)
    }

    private func describeHost(storeRoot: URL) {
        let entry = app.model.entry
        record.phases = phases.map(\.rawValue)
        record.startedAt = ISO8601DateFormatter().string(from: Date())
        record.platform = SampleCatalog.currentPlatform.rawValue
        record.os = ProcessInfo.processInfo.operatingSystemVersionString
        record.hardwareModel = Self.sysctlString(SampleCatalog.currentPlatform == .ios ? "hw.machine" : "hw.model")
        record.physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
        record.deviceMemoryLimitBytes = ModelReadiness.deviceMemoryLimit
        #if os(macOS)
        record.sandboxed = ProcessInfo.processInfo.environment["APP_SANDBOX_CONTAINER_ID"] != nil
        let container = FileManager.default.homeDirectoryForCurrentUser.standardizedFileURL.path
        record.storeInsideAppContainer = record.sandboxed && storeRoot.standardizedFileURL.path.hasPrefix(container + "/")
            && container.contains("/Library/Containers/")
        #else
        record.sandboxed = true
        record.storeInsideAppContainer = storeRoot.standardizedFileURL.path.hasPrefix(NSHomeDirectory())
        #endif
        if case let .huggingFace(repository, revision, subdirectory) = entry.source {
            record.repository = repository; record.revision = revision; record.subdirectory = subdirectory ?? ""
        }
        record.installationID = (try? entry.installationID) ?? ""
        record.entryFiles = entry.manifest.files.count
        record.entryBytes = entry.byteCount
    }

    private var now: Double { Self.seconds(ContinuousClock.now - started) }

    private func expect(_ what: String, timeout: Double, _ condition: @escaping @MainActor () -> Bool) async throws {
        guard await waitFor(what, timeout: timeout, condition) else {
            throw Failure(description: "timed out waiting for \(what); state \(app.model.stateName), \(app.model.display.detail)")
        }
    }

    @discardableResult
    private func waitFor(_ what: String, timeout: Double, _ condition: @escaping @MainActor () -> Bool) async -> Bool {
        let deadline = ContinuousClock.now + .seconds(timeout)
        while !condition() {
            if ContinuousClock.now > deadline { return false }
            try? await Task.sleep(for: .milliseconds(20))
        }
        return true
    }

    static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    static func sysctlString(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "" }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return "" }
        return String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    static func bytes(under root: URL) -> UInt64 {
        guard let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: UInt64 = 0
        for case let url as URL in files {
            total += UInt64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }
}
