# Abandoned local asset stages

New imports name their staging directory `<installationID>.<UUID>` while holding the installation's exclusive advisory lock through publish or rollback. A process crash can leave this unfinished directory behind; the operating system closes its lock descriptor on process exit.

`LocalAssetStore.removeAbandonedStages()` explicitly reclaims recognizable staging directories. It acquires the corresponding installation lock without waiting before rechecking and removing each entry. Active imports and readers are skipped, so a busy stage can be retried later. No age threshold or process-ID guessing is involved. Published installations, external model files and specialization caches are outside this operation.

```swift
let store = try LocalAssetStore(root: privateAssetDirectory)
let removed = try store.removeAbandonedStages()
```

The synchronous call returns the number of removed stages. Hosts can invoke it on their own background worker at startup or during maintenance. It checks cancellation before scanning, between entries and before removal. Cancellation or another error may leave earlier entries removed; retry is safe. Recursive filesystem removal itself is not interruptible. The SDK does not automatically run maintenance or schedule retries.

Unknown names and the older UUID-only stage format are retained: these entries do not identify the lock needed to prove that their importer is inactive. Inspect and handle legacy entries separately while all users of that store are stopped. Recognized entries that are symlinks or non-directories are refused; symlink targets are never reclaimed. Lock files remain in place to preserve coordination with other processes. As with all store operations, the host must keep the root private; advisory locks do not defend against arbitrary software mutating the store directly.

CPU tests model the state left by a dead importer with unheld stage directories, then cover cleanup/retry, live import contention, reader protection, installed-file preservation, unknown/legacy entries, symlink refusal and pre-cancelled maintenance. These are synthetic lifecycle checks, not forced-process-kill or power-loss durability qualification.

## Executed checkpoint — 2026-10-20

The full CPU-only Release suite passed with 212 tests passing and 14 native/fixture tests skipped (226 registered). The independent Foundations consumer also passed. The [source-pinned record](evidence/asset-stage-recovery-2026-10-20.json) links hashes of the archived test, build and consumer logs. No model or GPU workload ran. Hosted CI still needs the required macOS 27 / Xcode 27.1 / Swift 6.4 toolchain.
