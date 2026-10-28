# Shared GPU lease and admission

AetherKit now owns one implementation of GPU coordination. It has two layers:

- **`SharedGPULease`** is a cross-process lease, wire-compatible with Atelier's `SharedGPULock`. Kit tools take it by default.
- **`GPUAdmission`** is an in-process admission object that app sessions can opt into. It can use the cross-process lease as a backend.

Both are in the `AetherKit` product. This work is part of the 0.0.2 roadmap item [Shared GPU lease and admission](roadmap-0.0.2.md#shared-gpu-lease-and-admission).

## Why `AetherKit`

The lease uses only Foundation, Darwin and Dispatch. The admission object belongs next to `ResourceCoordinator`, which also lives in `AetherKit`. Krea, FLUX.2, H3, language, Core AI and the probe all depend on `AetherKit` already, so every session can accept the admission object without a new dependency. Atelier can depend on the small, Core AI-free `AetherKit` product instead of keeping its own copy.

`AetherCoreAI` would have forced Core AI linkage onto `AetherH3` and onto consumers that only coordinate. A separate target would have added a product without isolating anything: `AetherKit` has no dependencies.

## The lease protocol

The protocol is unchanged from Atelier and the lab scripts, so Atelier, `with-gpu-lock.sh`, `gpu.sh` and Kit tools all honor the same lease. The default path is `/tmp/atelier-gpu.lock`.

1. **Acquire.** `mkdir` the lease directory atomically. The holder writes its decimal pid, with no newline, to `pid.tmp.<pid>` inside the directory, then renames that file to `pid`. If `mkdir` fails with any error other than `EEXIST`, acquisition fails with `.unavailable`. It does not report the lease as held.
2. **Release.** Only while `pid` still names this process, rename the directory aside under a unique name (`<name>.released-<UUID>`) and then delete the moved copy. If the moved copy names someone else (an outside remover deleted this lease and a new holder took the path), it is restored with `renamex_np(RENAME_EXCL)`. Deleting in place would remove `pid` before the directory, and a kill in between would leave a pid-less lease at the shared path; now an interrupted release leaves at most a harmless `.released-` directory. A failed pid write after `mkdir` is cleaned up the same way. Release is idempotent, and it also runs on deinit.
3. **Stale reclaim.** A lease is stale only when its pid parses as a positive integer and `kill(pid, 0)` fails with `ESRCH`. Reclaim then runs Atelier's steps unchanged:
   1. Serialize on the sibling `<name>.reclaim` flock file. The file must be this user's single-link regular file with mode 0600; a 0644 file is tightened. Anything else makes the reclaimer fail safe.
   2. Move the lease aside atomically.
   3. Delete the moved copy only if it still names the same dead pid. Otherwise restore it with `renamex_np(RENAME_EXCL)`, so a lease created since is never replaced.

   A missing, empty or malformed pid is never stale. Neither is a live process that belongs to another user (`EPERM`).
4. **Signals.** Kit installs handlers for SIGINT, SIGTERM and SIGHUP only if they're still at their default disposition when the first lease is acquired. On one of those signals, every lease the process holds is released before the signal is re-raised. A signal the process already handles or ignores is left alone: an owner with its own shutdown releases through it, and `nohup` stays in force. When the last lease is released, Kit restores `SIG_DFL` wherever the disposition is still `SIG_IGN`. It keeps a handler function installed meanwhile, but it can't tell its own `SIG_IGN` from a `SIG_IGN` installed later, such as the usual Swift pattern of `SIG_IGN` plus a `DispatchSource`. **Hosts must install their own signal handling before acquiring the first lease**; Kit then leaves that signal alone. Handling installed with `SIG_IGN` while a lease is held is reset to `SIG_DFL` when the last lease is released.

Kit differs from Atelier's class in these ways. None of them changes what other holders see:

- **One handler set per process.** Signals are process-wide, so one handler set covers every lease the process holds. Atelier installs handlers per lock: if a second lock is taken while the first holds the handlers, releasing the first leaves the second uncovered. A test covers this case.
- **Explicit hand-off** (`Configuration.handedOffBy`, `AETHER_GPU_LEASE_HOLDER`). A holder that runs a child as part of its own GPU job hands the lease over by setting `AETHER_GPU_LEASE_HOLDER=<holder pid>` in the child's environment (`SharedGPULease.handOffEnvironment` gives it). The child inherits only when the lease's pid equals that value and names a live ancestor. Being an ancestor alone is not enough: a long-lived holder doing its own GPU work that happens to spawn a Kit tool doesn't overlap with it, because the tool waits. Release leaves an inherited lease alone. `GPUAdmission` re-checks an inherited lease before every grant.
- **Release by rename**, as described above.
- **Waiting acquisition.** `acquire(timeout:onBusy:)` polls every 200 ms, with Swift-concurrency cancellation. Atelier only fails fast. `onBusy` reports a `SharedGPULease.Busy` value when the holder changes and again every minute while it persists: `.process(pid)`, `.ancestorWithoutHandOff(pid)` (with a hint to set the hand-off), or `.unidentified` for a pid-less lease.

```swift
let lease = try await SharedGPULease.acquire(timeout: .seconds(30)) { busy in
    print("waiting for the GPU lease: \(busy)")
}
defer { lease.release() }
```

`SharedGPULease.tryAcquire(_:)` is the fail-fast form. It throws `SharedGPULeaseError.held(holder:)`.

### Open item: pid-less leases

A lease directory without a readable `pid` is never reclaimed. Either a holder is between `mkdir` and its pid write, or a holder was killed in that window and abandoned it. Age-based reclaim would need every implementation (Atelier, the wrappers, Kit) to agree on a rule, which would change the shared protocol, so it isn't implemented. Kit no longer creates such leases on release. Waiting Kit tools report a pid-less lease explicitly, repeat the report every minute, and say how to clear it: if no GPU job is running, remove the directory by hand. The proposed wrapper logs it the same way.

### Proposal: declared residency and pid-less leases

Status: proposed, 2026-10-27, for Atelier to agree on; not implemented. One change to the shared protocol, covering two
gaps.

**Why.** The lease serializes GPU work, but a model stays loaded, with its wired pages and Metal allocations, after its
holder's turn ends. Two cooperating processes can therefore take turns correctly and still hold more memory together
than the host has. On Kit 0.0.2, Atelier's VL-4B describer (in its own helper process) beside a Krea 1024×1024 render
did that and both processes died inside MPSGraph at memorystatus level 44
([analysis](qwen3-vl-resource-admission.md#the-crash)). An in-process `ResourceCoordinator` cannot see the other
process.

**Declared residency.**
1. **The record.** While a process has models loaded, it keeps one file per process in a sibling directory,
   `<lease path>.residency/` (`/tmp/atelier-gpu.lock.residency/<pid>.json`): its pid, its process start time (so a reused
   pid is not mistaken for it), a label, and the bytes it has reserved for loaded models, which for Kit is its
   coordinator's reserved total. The file is written to a temporary name and renamed into place, updated when the total
   changes, and removed when the last model unloads. The lease directory itself is created and removed every turn, so
   the record cannot live inside it.
2. **The check.** Before loading a model, a process reads every record, ignores (and may delete) any whose pid is not
   alive with the recorded start time, adds its own reservation and the new model's estimate, and refuses or waits when
   the sum exceeds the host's agreed capacity (for example a fraction of physical memory, configured once per host).
3. **Under the lease.** The check and the record's update happen while the process holds the lease, so two processes
   never check and load at the same time. Kit's language sessions now load under their GPU grant, as Krea, FLUX and H3
   do, so a lease-backed `GPUAdmission` gives this ordering.
4. **Advisory.** A process that doesn't take part is invisible to it, as with the lease today. The capacity is an agreed
   number, not a measurement: it bounds what cooperating processes reserve, not what the host has free.

**Pid-less leases, in the same change.** The record's liveness rule (pid plus start time) and the change window make it
the moment to settle the [open item above](#open-item-pid-less-leases): for example, a lease directory with no readable
`pid` older than an agreed age (seconds, since the pid is written right after `mkdir`) may be reclaimed under the
existing `.reclaim` flock. Every implementation (Atelier's `SharedGPULock`, the wrappers and Kit) has to adopt the same
rule at once, which is why the two belong in one agreed change.

**What it needs from Atelier.** Agreement on the directory and record format, the capacity rule, the pid-less age, and
that `atelier-language` and Krea's process both write records. Kit would then implement it in `SharedGPULease` and have
`GPUAdmission` publish its sessions' `ResourceCoordinator` total.

## Kit tools take the lease by default

`aether-probe` takes the lease itself before any well-formed GPU command and holds it until the command returns. It releases the lease explicitly on the error path, because `exit` skips deinit. A malformed GPU command (wrong argument count, an unparsable number, an unknown mode or compute policy) takes no lease, so its usage error prints at once instead of after a wait.

GPU commands:

- `inspect`, `smoke`
- `render-krea`, `render-krea-batch`, `parity-krea`
- `h3-profile-run`, `h3-profile-decode`
- `benchmark-krea` and the `qualify-h3*` commands, in every mode except `--prepare-only`

`doctor`, `verify-krea`, `prepare-krea`, `h3-profile-dump` and `h3-weights-bench` are CPU-only and take nothing. While another process holds the lease, the probe waits and prints a line to standard error naming the holder, when it changes and every minute.

The two other tool paths:

- **`Scripts/benchmark-krea.sh`:** `run` relies on the probe's lease. The build step needs no lease.
- **Opt-in native suites (`AetherNativeTests`):** each enabled native test, including the FLUX.2 parity and lifecycle qualifications, holds a grant from a test-local `GPUAdmission` whose backend is the lease. The tests take turns in-process and exclude other processes. The CPU suite never enables them.

These environment variables override the default, in both the probe and the native suites (`SharedGPULease.ToolPolicy`):

| Variable | Effect |
| --- | --- |
| `AETHER_GPU_LEASE=off` (or `0`) | Take no lease. Use this only when something else guarantees exclusivity. `on`, `1` or unset takes it. |
| `AETHER_GPU_LEASE_TIMEOUT=SECONDS` | Stop waiting after SECONDS, from 0 to one year (31,536,000). `0` fails fast. Unset waits indefinitely. NaN, infinite, negative and larger values are rejected. |
| `AETHER_GPU_LEASE_PATH=/abs/dir` | Use another lease directory, for example a private one for experiments. |
| `AETHER_GPU_LEASE_HOLDER=PID` | Set by a holder for the child it runs as its GPU job: the child inherits that holder's lease (see above). |

Any other value is an error rather than a silent fallback. `Scripts/check-cpu.sh` clears every `AETHER_*` variable as before.

Tools inherit only an explicit hand-off. The development wrapper `~/.cache/aether-kit/with-gpu-lock.sh` predates it and doesn't set `AETHER_GPU_LEASE_HOLDER`, so under that wrapper a Kit tool waits for its own parent and says so (`held by ancestor process N, which did not hand it to this process`). Install the proposed replacement wrapper together with this change. The replacement exports the hand-off, installs its trap before taking the lease, releases by renaming the lease aside, and, when interrupted, stops the command's whole process tree before releasing, so no GPU work outlives the lease. Until it is installed, run Kit tools without the wrapper, which they no longer need, or set `AETHER_GPU_LEASE_HOLDER` by hand. The wrapper still serializes non-Kit GPU jobs, such as Python MPS exports, and it never reclaims.

## App sessions opt in

App sessions don't take the lease by default. A host creates one `GPUAdmission` and passes it explicitly to the sessions that should take turns:

```swift
let gpu = try GPUAdmission(maximumConcurrent: 1, waiting: .queue(timeout: .seconds(120)))
let krea = KreaGenerationSession(assets: assets, resources: kreaBudget, gpu: gpu.owner("krea"))
let video = H3VideoSession(stages: stages, gpu: gpu.owner("h3"))
let chat = try await LanguageGenerationSession.open(bundle: bundle, platform: .macos, gpu: gpu.owner("chat"))
let flux = Flux2GenerationSession(assets: fluxAssets, gpu: gpu.owner("flux"))
```

What each session holds a grant for:

- **`KreaGenerationSession` and `Flux2GenerationSession`:** each operation (prepare, generate, refine, image encode). The grant is taken before any model opens and released after the operation's native work has joined.
- **`H3VideoSession`:** each generation, from before its first stage until every stage is released.
- **`LanguageGenerationSession`:** its opening (the model's load and its functions' specialization), and each request, after validation, until its submissions have joined.
- **`VisionLanguageSession`:** the encoder's opening and the decoder's, each under its own grant; each image's encode; and each decoder request.

Sessions created without `gpu:` behave as before. Admission composes with `ResourceCoordinator`. The coordinator answers whether memory is available, admission answers whose turn it is, and either can refuse. With admission, a session takes its operation reservation only after its grant, so a request waiting for the GPU never holds capacity the running request needs. `unload()` withdraws an operation that is still waiting for admission; it ends with the session's closed error, so unload never waits behind the GPU queue. An operation that already holds its grant is waited for as before.

- **Queue or fail fast.** `waiting: .queue(timeout:)` waits in a fair queue. A timeout throws `GPUAdmissionError.timedOut`. Cancelling the waiting task leaves the queue with `CancellationError`, and the session returns to `ready`. `.failFast` throws `GPUAdmissionError.busy` when every slot is in use or work is already queued ahead. Either mode can be set per request with `admit(owner:waiting:)`.
- **Owners and fairness.** Requests are FIFO within an owner and round-robin across owners. An owner arriving while another owner's burst runs gets the next turn. Owner names are labels, not credentials. `maximumOutstanding` and `maximumOutstandingPerOwner` bound the queue and throw `GPUAdmissionError.limit`.
- **Concurrency.** `maximumConcurrent` greater than 1 lets a host run models concurrently on purpose. There is no process-global default admission, because concurrency is a host decision.
- **Cross-process backend.** `GPUAdmission(crossProcessLease: SharedGPULease.Configuration())` holds the lease while grants run. Only one request at a time acquires or releases the lease, outside the admission's lock; the others wait in the in-process queue. Lease failures surface as `SharedGPULeaseError`.
- **Turns between processes.** The lease is polled, so a busy in-process queue that re-acquired within microseconds of every release would starve other processes. Instead, an owned lease passes straight to the next queued request for up to `crossProcessTurn` (default 30 s). After that, the admission starts no new grant on that lease, even with free slots under `maximumConcurrent > 1`. Running grants drain, then the admission releases the lease and waits `crossProcessHandOffGap` (default 500 ms) before re-acquiring. The lease is therefore released within the turn plus the longest running grant, which gives any process polling at least that often (Kit tools poll every 200 ms) a window to take a turn. Pollers with longer intervals, such as the wrapper's 5 s, can still miss the window under continuous load, so starvation of those is possible. A fail-fast request doesn't wait out the gap. A bounded hand-off was chosen over documenting starvation alone because sustained in-process load, such as a language host admitting each request, is the realistic case and would otherwise lock out the Kit tools completely.

### Sandboxed apps

A sandboxed app gets a private temporary directory, so `/tmp/atelier-gpu.lock` inside the sandbox is not the path other processes use. There, the cross-process backend coordinates only the app with itself and excludes nothing outside it. Use in-process admission alone in sandboxed apps, and enable the lease backend only in unsandboxed hosts, such as developer tools, Atelier's unsandboxed host and command-line hosts. Hand-off inheritance relies on `sysctl(KERN_PROC_PID)` to confirm the holder is an ancestor, which sandboxed or iOS processes may not be allowed to use; there it simply never matches.

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
- timeouts and cancellation, and the busy reports (ancestor without hand-off, pid-less lease, repeated while it persists)
- contention with another Kit process in both directions
- explicit hand-off: inherited only from the named live ancestor, directly and through a shell; refused without it, with a wrong pid, or from a sibling; an inherited lease stops being valid once the lease names anyone else
- interop with an externally created mkdir+pid lease and with a wrapper-style holder, including SIGKILL reclaim
- SIGINT, SIGTERM and SIGHUP release, an owner's own handler, an ignored signal, and a handler installed after the lease
- signal release of a second lease after the first is released
- the tool environment, including the timeout bounds

`GPUAdmissionTests` cover fair ordering, fail-fast refusal, timeout, cancellation before and during queueing, concurrent slots, limits, owner handles, and the cross-process backend: shared holding, an external holder, cancelling the request that waits on the lease, re-checking an inherited lease before a grant, passing the lease straight on within a turn, and releasing it with a gap another process can use after the turn. With two slots and a steady stream of overlapping grants, the lease is released within the turn plus one grant's duration. They also cover withdrawing an admission before and during the wait and after the grant. Further tests cover the probe's command classification, malformed commands taking no lease, and the overrides. Krea, FLUX.2 and H3 sessions are tested to wait for admission before any model or stage opens, to hold no operation reservation while queued (with a shared coordinator and a shared admission), and to withdraw a queued operation on unload. The language session uses the same withdrawal; it needs a native model to exercise. The full weight-free suite (`sh Scripts/check-cpu.sh`) passes 316 tests, with 18 opt-in native or external tests skipped (334 registered), after rebasing onto the FLUX.2 merge; see the [CPU log](evidence/gpu-lease-cpu-tests-2026-10-26.log).

A recorded interop check ran Kit against Atelier's own `SharedGPULock` source, compiled verbatim into a small private-directory simulator, and against the wrapper's protocol. All 27 checks passed. The two sides exclude each other, and each reclaims the other's SIGKILLed lease through the shared `.reclaim` file. `aether-probe` waits for an Atelier holder. Under a wrapper-style parent, it waits without a hand-off and inherits the lease with one. See the [evidence record](evidence/gpu-lease-interop-2026-10-26.json), [log](evidence/gpu-lease-interop-2026-10-26.log) and [driver script](evidence/gpu-lease-interop-check.zsh). A separate check ran the [proposed wrapper](evidence/gpu-lease-with-gpu-lock-proposed.zsh) on private lease and lock paths: it hands its lease to the Kit helper and `aether-probe`, stops its command's whole process tree on SIGTERM before releasing, and logs and leaves alone a pid-less lease ([log](evidence/gpu-lease-wrapper-check-2026-10-26.log), [script](evidence/gpu-lease-wrapper-check.zsh)). Atelier's app itself was not run, and no GPU work ran.

## Limitations

- **Cooperative only.** Processes that ignore the lease aren't excluded. The lease is also polled: turns between processes are bounded as described above, but slow pollers can still starve under continuous load.
- **Inherited work after the holder dies.** A Kit tool that inherited a hand-off isn't stopped if its holder dies. The proposed wrapper stops its command first when it gets INT, TERM or HUP, but a SIGKILLed wrapper can't, and its lease then becomes stale and reclaimable while the command still runs.
- **Pid reuse.** A stale lease whose pid has been reused by a live process looks live until that process exits. An inheriting tool could, in principle, mistake a reused pid that happens to be its ancestor for its holder.
- **Transient refusals.** A stale lease can briefly read as held. This happens when this process released the reclaim lock an instant earlier, and a child it is spawning still shares that descriptor between fork and exec. Waiting acquisition retries; the tests allow for it.
- **Shared `/tmp`.** Another user's live lease blocks everyone. Another user's dead lease can't be reclaimed across the sticky `/tmp` and is reported as held, as in Atelier.
- **Atelier hasn't adopted it.** Atelier still uses its own `SharedGPULock`. Adopting `AetherKit`'s lease there is Atelier's change to make.
