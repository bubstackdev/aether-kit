# Shared GPU lease and admission

AetherKit now owns one implementation of GPU coordination. It has two layers:

- **`SharedGPULease`** is a cross-process lease, wire-compatible with Atelier's `SharedGPULock`. Kit tools take it by default.
- **`GPUAdmission`** is an in-process admission object that app sessions can opt into. It can use the cross-process lease as a backend.

Both are in the `AetherKit` product. This work is part of the 0.0.2 roadmap item [Shared GPU lease and admission](roadmap-0.0.2.md#shared-gpu-lease-and-admission).

## Why `AetherKit`

The lease uses only Foundation, Darwin and Dispatch. The admission object belongs next to `ResourceCoordinator`, which also lives in `AetherKit`. Krea, H3, language, Core AI and the probe all depend on `AetherKit` already, so every session can accept the admission object without a new dependency. Atelier can depend on the small, Core AI-free `AetherKit` product instead of keeping its own copy.

`AetherCoreAI` would have forced Core AI linkage onto `AetherH3` and onto consumers that only coordinate. A separate target would have added a product without isolating anything: `AetherKit` has no dependencies.

## The lease protocol

The protocol is unchanged from Atelier and the lab scripts, so Atelier, `with-gpu-lock.sh`, `gpu.sh` and Kit tools all honor the same lease. The default path is `/tmp/atelier-gpu.lock`.

1. **Acquire.** `mkdir` the lease directory atomically. The holder writes its decimal pid, with no newline, to `pid.tmp.<pid>` inside the directory, then renames that file to `pid`. If `mkdir` fails with any error other than `EEXIST`, acquisition fails with `.unavailable`. It does not report the lease as held.
2. **Release.** Remove the directory only if `pid` still names this process. Release is idempotent, and it also runs on deinit.
3. **Stale reclaim.** A lease is stale only when its pid parses as a positive integer and `kill(pid, 0)` fails with `ESRCH`. Reclaim then runs Atelier's steps unchanged:
   1. Serialize on the sibling `<name>.reclaim` flock file. The file must be this user's single-link regular file with mode 0600; a 0644 file is tightened. Anything else makes the reclaimer fail safe.
   2. Move the lease aside atomically.
   3. Delete the moved copy only if it still names the same dead pid. Otherwise restore it with `renamex_np(RENAME_EXCL)`, so a lease created since is never replaced.

   A missing, empty or malformed pid is never stale. Neither is a live process that belongs to another user (`EPERM`).
4. **Signals.** Kit installs handlers for SIGINT, SIGTERM and SIGHUP only if they're still at their default disposition. On one of those signals, every lease the process holds is released before the signal is re-raised. A signal the process already handles or ignores is left alone: an owner with its own shutdown releases through it, and `nohup` stays in force. Such owners must install their handling before acquiring a lease.

Kit differs from Atelier's class in three ways. None of them changes the protocol:

- **One handler set per process.** Signals are process-wide, so one handler set covers every lease the process holds. Atelier installs handlers per lock: if a second lock is taken while the first holds the handlers, releasing the first leaves the second uncovered. A test covers this case.
- **Optional inheritance** (`Configuration.inheritsAncestorLease`). A lease whose pid names an ancestor process counts as held on the caller's behalf. Release leaves it alone. It's off by default, and tools turn it on.
- **Waiting acquisition.** `acquire(timeout:)` polls every 200 ms, with Swift-concurrency cancellation. Atelier only fails fast.

```swift
let lease = try await SharedGPULease.acquire(timeout: .seconds(30)) { holder in
    print("waiting for GPU lease held by \(holder.map(String.init) ?? "unknown")")
}
defer { lease.release() }
```

`SharedGPULease.tryAcquire(_:)` is the fail-fast form. It throws `SharedGPULeaseError.held(holder:)`.

## Kit tools take the lease by default

`aether-probe` takes the lease itself before any GPU command and holds it until the command returns. It releases the lease explicitly on the error path, because `exit` skips deinit.

GPU commands:

- `inspect`, `smoke`
- `render-krea`, `render-krea-batch`, `parity-krea`
- `h3-profile-run`, `h3-profile-decode`
- `benchmark-krea` and the `qualify-h3*` commands, in every mode except `--prepare-only`

`doctor`, `verify-krea`, `prepare-krea`, `h3-profile-dump` and `h3-weights-bench` are CPU-only and take nothing. While another process holds the lease, the probe waits and prints one line to standard error.

The two other tool paths:

- **`Scripts/benchmark-krea.sh`:** `run` relies on the probe's lease. The build step needs no lease.
- **Opt-in native suites (`AetherNativeTests`):** each enabled native test holds a grant from a test-local `GPUAdmission` whose backend is the lease. The tests take turns in-process and exclude other processes. The CPU suite never enables them.

These environment variables override the default, in both the probe and the native suites (`SharedGPULease.ToolPolicy`):

| Variable | Effect |
| --- | --- |
| `AETHER_GPU_LEASE=off` (or `0`) | Take no lease. Use this only when something else guarantees exclusivity. `on`, `1` or unset takes it. |
| `AETHER_GPU_LEASE_TIMEOUT=SECONDS` | Stop waiting after SECONDS. `0` fails fast. Unset waits indefinitely. |
| `AETHER_GPU_LEASE_PATH=/abs/dir` | Use another lease directory, for example a private one for experiments. |

Any other value is an error rather than a silent fallback. `Scripts/check-cpu.sh` clears every `AETHER_*` variable as before.

Tools always inherit a lease held by an ancestor process. Commands run under the development wrapper `~/.cache/aether-kit/with-gpu-lock.sh` therefore keep working: the wrapper's zsh holds the lease, and the probe or `swift test` running beneath it doesn't wait for its own parent. Kit tools no longer need the wrapper, but it remains harmless. It still serializes non-Kit GPU jobs, such as Python MPS exports, and it never reclaims.

## App sessions opt in

App sessions don't take the lease by default. A host creates one `GPUAdmission` and passes it explicitly to the sessions that should take turns:

```swift
let gpu = try GPUAdmission(maximumConcurrent: 1, waiting: .queue(timeout: .seconds(120)))
let krea = KreaGenerationSession(assets: assets, resources: kreaBudget, gpu: gpu.owner("krea"))
let video = H3VideoSession(stages: stages, gpu: gpu.owner("h3"))
let chat = try await LanguageGenerationSession.open(bundle: bundle, platform: .macos, gpu: gpu.owner("chat"))
```

What each session holds a grant for:

- **`KreaGenerationSession`:** each operation (prepare, generate, refine, image encode). The grant is taken before any model opens and released after the operation's native work has joined.
- **`H3VideoSession`:** each generation, from before its first stage until every stage is released.
- **`LanguageGenerationSession`:** each request, after validation, until its submissions have joined. Model loading isn't admitted.

Sessions created without `gpu:` behave as before. Admission composes with `ResourceCoordinator`. The coordinator answers whether memory is available, admission answers whose turn it is, and either can refuse.

- **Queue or fail fast.** `waiting: .queue(timeout:)` waits in a fair queue. A timeout throws `GPUAdmissionError.timedOut`. Cancelling the waiting task leaves the queue with `CancellationError`, and the session returns to `ready`. `.failFast` throws `GPUAdmissionError.busy` when every slot is in use or work is already queued ahead. Either mode can be set per request with `admit(owner:waiting:)`.
- **Owners and fairness.** Requests are FIFO within an owner and round-robin across owners. An owner arriving while another owner's burst runs gets the next turn. Owner names are labels, not credentials. `maximumOutstanding` and `maximumOutstandingPerOwner` bound the queue and throw `GPUAdmissionError.limit`.
- **Concurrency.** `maximumConcurrent` greater than 1 lets a host run models concurrently on purpose. There is no process-global default admission, because concurrency is a host decision.
- **Cross-process backend.** `GPUAdmission(crossProcessLease: SharedGPULease.Configuration())` holds the lease while any grant runs and releases it when the last grant ends. Only one request at a time waits on the lease; the others wait in the in-process queue. Lease failures surface as `SharedGPULeaseError`. The lease is polled, so there is no fairness between processes.

### Sandboxed apps

A sandboxed app gets a private temporary directory, so `/tmp/atelier-gpu.lock` inside the sandbox is not the path other processes use. There, the cross-process backend coordinates only the app with itself and excludes nothing outside it. Use in-process admission alone in sandboxed apps, and enable the lease backend only in unsandboxed hosts, such as developer tools, Atelier's unsandboxed host and command-line hosts. Ancestor inheritance relies on `sysctl(KERN_PROC_PID)`, which sandboxed or iOS processes may not be allowed to use; there it simply never matches.

## Reconciliation with `FairInferenceQueue`

`FairInferenceQueue` was found untracked in an older checkout. It is a bounded, transport-independent admission design for a host consumer: FIFO per owner, round-robin across owners, global and per-owner limits that include running work, and running tickets that keep their slot until the caller has drained native work. It doesn't own, start or cancel native work.

The design is adopted rather than duplicated:

- **Adopted as is.** `FairInferenceQueue` becomes a public `AetherKit` type with its original API, semantics and test cases. Its limits are now exposed read-only, and two configuration and multi-slot cases were added. Hosts that run their own coordination actor, such as a networked host serving several users, can keep using it directly.
- **Built on.** `GPUAdmission` is Kit's own consumer of that queue. It adds what the value type deliberately leaves out: suspension until a ticket starts, timeouts, cancellation of queued work, fail-fast admission, grant objects that release idempotently or on deinit, and the cross-process lease backend.
- **Not adopted.** The draft's description of owners as authenticated users belongs to that host. Kit treats owner names as labels.

Credit: the queue design and its scheduling tests come from that earlier admission draft (`docs/inference-admission.md` in the older checkout). The older checkout still has its untracked copy of the file. When it merges this branch, the tracked `Sources/AetherKit/FairInferenceQueue.swift` supersedes that copy, which should be removed rather than merged.

## Validation

The CPU tests use private temporary lease directories only and never touch `/tmp/atelier-gpu.lock`.

`SharedGPULeaseTests` mirrors every case in Atelier's `SharedGPULockTests`, plus Kit's additions. It uses a test-only helper process (`AetherGPULeaseHelper`) and a shell holder that follows the wrapper's protocol. Coverage:

- acquire and release, idempotence and deinit release
- live, launchd and parent holders, which are never reclaimed
- missing or malformed pids
- dead-pid reclaim
- the restore race, plus an outside remover that creates a newer lease mid-reclaim
- reclaim-file serialization and ownership
- release after the pid changed
- timeouts and cancellation
- contention with another Kit process in both directions
- ancestor inheritance, direct and through a shell, and a sibling, which is not inherited
- interop with an externally created mkdir+pid lease and with a wrapper-style holder, including SIGKILL reclaim
- SIGINT, SIGTERM and SIGHUP release, an owner's own handler, an ignored signal, and a handler installed after the lease
- signal release of a second lease after the first is released
- the tool environment

`GPUAdmissionTests` cover fair ordering, fail-fast refusal, timeout, cancellation before and during queueing, concurrent slots, limits, owner handles, and the cross-process backend: shared holding, an external holder, and cancelling the request that waits on the lease. Further tests cover the probe's command classification and overrides, and Krea and H3 sessions waiting for admission before any model or stage opens. The language session's wiring is compiled but needs a native model to exercise. The full weight-free suite (`sh Scripts/check-cpu.sh`) passes 279 tests, with 14 opt-in native or external tests skipped (293 registered); see the [CPU log](evidence/gpu-lease-cpu-tests-2026-10-26.log).

A recorded interop check ran Kit against Atelier's own `SharedGPULock` source, compiled verbatim into a small private-directory simulator, and against the wrapper's protocol. All 26 checks passed: the two sides exclude each other, each reclaims the other's SIGKILLed lease through the shared `.reclaim` file, and `aether-probe` waits for an Atelier holder and inherits a wrapper's lease. See the [evidence record](evidence/gpu-lease-interop-2026-10-26.json), [log](evidence/gpu-lease-interop-2026-10-26.log) and [driver script](evidence/gpu-lease-interop-check.zsh). Atelier's app itself was not run, and no GPU work ran.

## Limitations

- **Cooperative only.** Processes that ignore the lease aren't excluded. The lease is also polled, so there's no fairness between processes; a busy in-process queue that keeps re-acquiring can delay a polling outsider.
- **Pid reuse.** A stale lease whose pid has been reused by a live process looks live until that process exits. An inheriting tool could, in principle, mistake a reused pid that happens to be its ancestor for its holder.
- **Transient refusals.** A stale lease can briefly read as held. This happens when this process released the reclaim lock an instant earlier, and a child it is spawning still shares that descriptor between fork and exec. Waiting acquisition retries; the tests allow for it.
- **Shared `/tmp`.** Another user's live lease blocks everyone. Another user's dead lease can't be reclaimed across the sticky `/tmp` and is reported as held, as in Atelier.
- **Atelier hasn't adopted it.** Atelier still uses its own `SharedGPULock`. Adopting `AetherKit`'s lease there is Atelier's change to make.
