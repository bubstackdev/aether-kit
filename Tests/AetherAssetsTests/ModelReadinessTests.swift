@testable import AetherAssets
import Foundation
import Testing

private typealias State = ModelReadiness.State

/// Waits until `readiness` reaches a state `done` accepts, returning every state seen (a slow reader may miss some
/// progress steps, never the latest state). Fails after 10 s.
private func settle(_ readiness: ModelReadiness, until done: @escaping @Sendable (State) -> Bool) async throws -> [State] {
    try await withThrowingTaskGroup(of: [State]?.self) { group in
        group.addTask {
            var seen: [State] = []
            for await state in readiness.states {
                seen.append(state)
                if done(state) { return seen }
            }
            return nil
        }
        group.addTask { try await Task.sleep(for: .seconds(10)); return nil }
        let first = try await group.next() ?? nil
        group.cancelAll()
        guard let first else { throw CancellationError() }
        return first
    }
}

private func isReady(_ state: State) -> Bool { if case .ready = state { true } else { false } }
private func isNotInstalled(_ state: State) -> Bool { if case .notInstalled = state { true } else { false } }
private func isFailed(_ state: State) -> Bool { if case .failed = state { true } else { false } }

private let mac = ModelReadiness.Device(platform: .macos, memoryLimit: { 1 << 30 })

private func readiness(_ fixture: PreparationFixture, _ transport: ScriptedTransport, entry: ModelCatalogEntry? = nil,
                       check: ModelRuntimeCheck = .none, device: ModelReadiness.Device = mac) throws -> ModelReadiness {
    ModelReadiness(entry: entry ?? fixture.entry, preparation: try fixture.preparation(transport), check: check, device: device)
}

private func entry(_ fixture: PreparationFixture, _ requirements: ModelRequirements) throws -> ModelCatalogEntry {
    try ModelCatalogEntry(manifest: fixture.entry.manifest, source: fixture.entry.source, requirements: requirements)
}

struct ModelReadinessTests {
    @Test func aModelThatIsNotInstalledWaitsForPrepareThenBecomesReady() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        let model = try readiness(fixture, transport)
        #expect(model.current == .notInstalled(bytesPresent: 0, bytesTotal: fixture.entry.byteCount))
        #expect(transport.calls.isEmpty)

        model.prepare()
        let seen = try await settle(model, until: isReady)
        guard case let .ready(lease) = seen.last else { Issue.record("not ready: \(seen)"); return }
        #expect(try lease.installationID == fixture.entry.installationID)
        #expect(seen.allSatisfy { if case .failed = $0 { false } else if case .unsupported = $0 { false } else { true } })
        model.prepare()  // ready: nothing to do
        #expect(model.current == .ready(lease))
    }

    @Test func anInstalledModelIsVerifiedOnCreation() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        _ = try await fixture.preparation(transport).install(fixture.entry)
        let calls = transport.calls.count
        let model = try readiness(fixture, transport)
        #expect(model.current == .verifying || isReady(model.current))
        _ = try await settle(model, until: isReady)
        #expect(transport.calls.count == calls)
    }

    @Test func aFailedDownloadIsRetryableAndResumesFromItsBytes() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        transport.on("model/main.mlirb", .cutAfter(128))
        let model = try readiness(fixture, transport)
        model.prepare()
        let failed = try await settle(model, until: isFailed).last
        guard case let .failed(failure) = failed else { Issue.record("expected a failure"); return }
        #expect(failure.isRetryable)

        // A fresh readiness sees the bytes the failed run left.
        let later = try readiness(fixture, transport)
        guard case let .notInstalled(present, total) = later.current else { Issue.record("\(later.current)"); return }
        #expect(present >= 128 && present < total)
        later.prepare()
        _ = try await settle(later, until: isReady)
        #expect(transport.calls.last { $0.path == "model/main.mlirb" }?.resume == true)
    }

    @Test func cancellingStopsThePreparationAndKeepsItsBytes() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        transport.on("model/main.mlirb", .waitForRelease)
        let model = try readiness(fixture, transport)
        model.prepare()
        _ = try await settle(model, until: { if case let .installing(progress) = $0 { progress.phase == .downloading && progress.bytesDone > 0 } else { false } })
        model.cancel()
        guard case let .notInstalled(present, _) = model.current else { Issue.record("\(model.current)"); return }
        #expect(present > 0)
        model.prepare()
        transport.release()
        _ = try await settle(model, until: isReady)
    }

    @Test func anotherPlatformIsUnsupportedWithoutDownloading() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        let model = try readiness(fixture, transport, entry: entry(fixture, ModelRequirements(platforms: [.ios])))
        #expect(model.current == .unsupported(.platform))
        model.prepare()
        #expect(model.current == .unsupported(.platform))
        #expect(transport.calls.isEmpty)
        let both = try readiness(fixture, transport, entry: entry(fixture, ModelRequirements(platforms: [.ios, .macos])))
        #expect(isNotInstalled(both.current))
    }

    @Test func aMemoryRequirementAboveTheDeviceLimitIsUnsupported() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        let small = ModelReadiness.Device(platform: .ios, memoryLimit: { 3_000 })
        let model = try readiness(fixture, transport, entry: entry(fixture, ModelRequirements(memoryBytes: 4_000)), device: small)
        #expect(model.current == .unsupported(.memory(requiredBytes: 4_000, availableBytes: 3_000)))
        model.prepare()
        #expect(transport.calls.isEmpty)
        let fits = try readiness(fixture, transport, entry: entry(fixture, ModelRequirements(memoryBytes: 3_000)), device: small)
        #expect(isNotInstalled(fits.current))
    }

    @Test func withoutAKnownMemoryLimitTheRequirementGoesUnchecked() throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let unknown = ModelReadiness.Device(platform: .ios, memoryLimit: { nil })
        let model = try readiness(fixture, ScriptedTransport(source: fixture.source),
                                  entry: entry(fixture, ModelRequirements(memoryBytes: 1 << 40)), device: unknown)
        #expect(isNotInstalled(model.current))
    }

    @Test func releasingAReadinessStopsItsRunAndEndsItsStreams() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        transport.on("model/main.mlirb", .waitForRelease)
        var model: ModelReadiness? = try readiness(fixture, transport)
        let stream = model!.states
        model!.prepare()
        _ = try await settle(model!, until: { if case let .installing(progress) = $0 { progress.phase == .downloading && progress.bytesDone > 0 } else { false } })
        model = nil
        // The stream ends, and the cancelled run releases the download lock: a new readiness prepares at once.
        let ended = Task { for await _ in stream {}; return true }
        let finished = try await withThrowingTaskGroup(of: Bool.self) { group in
            group.addTask { await ended.value }
            group.addTask { try await Task.sleep(for: .seconds(10)); return false }
            let first = try await group.next() ?? false
            group.cancelAll()
            return first
        }
        #expect(finished)
        let again = try readiness(fixture, transport)
        guard case let .notInstalled(present, _) = again.current else { Issue.record("\(again.current)"); return }
        #expect(present > 0)
        again.prepare()
        _ = try await settle(again, until: isReady)
    }

    @Test func prepareDuringUninstallWaitsForTheRemoval() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        let model = try readiness(fixture, transport)
        model.prepare()
        _ = try await settle(model, until: isReady)
        // An app that prepares whenever it sees notInstalled, racing the removal.
        let watcher = Task {
            for await state in model.states { if isNotInstalled(state) { model.prepare(); return } }
        }
        try await model.uninstall()
        await watcher.value
        _ = try await settle(model, until: isReady)
        #expect(transport.calls.filter { $0.path == "model/main.mlirb" }.count == 2)
    }

    @Test func anInstalledModelThatNoLongerFitsIsUnsupportedWithoutVerifying() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        _ = try await fixture.preparation(transport).install(fixture.entry)
        let small = ModelReadiness.Device(platform: .macos, memoryLimit: { 10 })
        let model = try readiness(fixture, transport, entry: entry(fixture, ModelRequirements(memoryBytes: 11)), device: small)
        #expect(model.current == .unsupported(.memory(requiredBytes: 11, availableBytes: 10)))
        // Not holding a lease: the store can remove it.
        try fixture.preparation(transport).store.remove(fixture.entry.installationID)
    }

    @Test func theProductCheckComesFirstAndJudgesTheInstallation() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        let oldOS = ModelRuntimeCheck(entry: { _ in .unsupported(.operatingSystem) }, installation: { _ in .runnable })
        let model = try readiness(fixture, transport, entry: entry(fixture, ModelRequirements(platforms: [.ios])), check: oldOS)
        #expect(model.current == .unsupported(.operatingSystem))

        let wrongFamily = ModelRuntimeCheck(entry: { _ in .invalid("not mine") }, installation: { _ in .runnable })
        let refused = try readiness(fixture, transport, check: wrongFamily)
        #expect(refused.current == .failed(.invalidModel("not mine")))
        #expect(!ModelReadiness.Failure.invalidModel("not mine").isRetryable)
        refused.prepare()
        #expect(refused.current == .failed(.invalidModel("not mine")))
        #expect(transport.calls.isEmpty)

        let brokenBundle = ModelRuntimeCheck(entry: { _ in .runnable }, installation: { _ in .invalid("no template") })
        let installed = try readiness(fixture, transport, check: brokenBundle)
        installed.prepare()
        #expect(try await settle(installed, until: isFailed).last == .failed(.invalidModel("no template")))
        // The same model would fail the same way: prepare doesn't verify it again.
        installed.prepare()
        #expect(installed.current == .failed(.invalidModel("no template")))
    }

    @Test func aCorruptInstallationFailsVerificationAndPrepareReplacesIt() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        let id = try await fixture.preparation(transport).install(fixture.entry)
        let file = fixture.storeRoot.appendingPathComponent("installed/\(id)/files/model/main.mlirb")
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
        var bytes = try Data(contentsOf: file)
        bytes[0] ^= 0xff
        try bytes.write(to: file)

        let model = try readiness(fixture, transport)
        guard case let .failed(failure) = try await settle(model, until: isFailed).last else { return }
        guard case .verification = failure else { Issue.record("\(failure)"); return }
        #expect(failure.isRetryable)
        model.prepare()
        _ = try await settle(model, until: isReady)
        #expect(transport.calls.contains { $0.path == "model/main.mlirb" && !$0.resume })
    }

    @Test func uninstallRemovesTheInstallationUnlessAModelStillHoldsIt() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        let model = try readiness(fixture, transport)
        model.prepare()
        do {
            guard case let .ready(lease) = try await settle(model, until: isReady).last else { return }
            // The app still holds the lease (as a model opened from it would): busy, and verified again.
            await #expect(throws: PreparationError.busy) { try await model.uninstall() }
            withExtendedLifetime(lease) {}
        }
        _ = try await settle(model, until: isReady)

        try await model.uninstall()
        #expect(model.current == .notInstalled(bytesPresent: 0, bytesTotal: fixture.entry.byteCount))
        #expect(try fixture.preparation(transport).status(fixture.entry) == .notInstalled(bytesTotal: fixture.entry.byteCount))
    }

    @Test func uninstallDiscardsPartialDownloads() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        transport.on("model/main.mlirb", .cutAfter(64))
        let model = try readiness(fixture, transport)
        model.prepare()
        _ = try await settle(model, until: isFailed)
        try await model.uninstall()
        #expect(model.current == .notInstalled(bytesPresent: 0, bytesTotal: fixture.entry.byteCount))
    }

    @Test func uninstallLeavesDownloadsAnotherRunHoldsAndStillSucceeds() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        let model = try readiness(fixture, transport)
        model.prepare()
        _ = try await settle(model, until: isReady)
        let store = try fixture.preparation(transport).store
        let held = try store.downloadLock(fixture.entry.installationID)  // as another process's run would
        try await model.uninstall()
        withExtendedLifetime(held) {}
        #expect(isNotInstalled(model.current))
        #expect(!store.isInstalled(try fixture.entry.installationID))
    }

    @Test func refreshFindsAnInstallationMadeElsewhere() async throws {
        let fixture = try PreparationFixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        let model = try readiness(fixture, transport)
        #expect(isNotInstalled(model.current))
        _ = try await fixture.preparation(transport).install(fixture.entry)
        model.refresh()
        _ = try await settle(model, until: isReady)
    }

    @Test func progressFractionsStayWithinTheirPhase() {
        typealias Progress = ModelReadiness.Progress
        #expect(Progress(phase: .downloading, bytesDone: 50, bytesTotal: 200).fraction == 0.25)
        #expect(Progress(phase: .checking, bytesDone: 0, bytesTotal: 0).fraction == 0)
        #expect(Progress(phase: .installing, bytesDone: 300, bytesTotal: 200).fraction == 1)
    }
}
