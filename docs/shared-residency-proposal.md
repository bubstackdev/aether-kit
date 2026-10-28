# Proposal: shared declared residency and pid-less leases

Status: proposed, 2026-10-27, for Atelier and Kit to agree on. Nothing here is implemented. It changes the protocol
around the shared GPU lease ([GPU lease](gpu-lease.md)), so every implementation (Atelier's `SharedGPULock` and its
helpers, the development wrappers, Kit's `SharedGPULease`) adopts it together. This page is meant to be reviewed on its
own.

## The problem

The GPU lease (`/tmp/atelier-gpu.lock`) serializes GPU work. It does not account for memory: a model stays loaded,
with its wired pages and Metal allocations, after its process's turn on the GPU ends. Two cooperating processes can take
turns correctly and still hold more memory together than the Mac has. On Kit 0.0.2, Atelier's VL-4B describer, in its
own helper process, ran beside a Krea 1024×1024 render. Both processes died inside MPSGraph at memorystatus level 44
([analysis](qwen3-vl-resource-admission.md#the-crash)). Kit's in-process `ResourceCoordinator` cannot see another
process, and Atelier's helper gates its loads only on `kern.memorystatus_level`, which shows pressure after it has
built up.

## Constraints

- **No GPU grant required.** Atelier's `atelier-language` helper serves one request per process. It loads the writer
  or the describer without taking the GPU lease for the load. Today it is gated only by memory: it loads if
  `kern.memorystatus_level` stays at 25 or above, and assumes 9 GiB for the writer and 17 GiB for the describer
  (to be re-measured at its 0.0.3 bump). Declaring residency must therefore never depend on holding the lease.
- **Crash-safe.** A process may be killed at any point, including by the kernel under pressure. Whatever it leaves
  behind must be recognizable as stale by any other participant, without that process.
- **No pid reuse confusion.** macOS reuses pids, so liveness is checked by pid and process start time together.
- **Same user, unsandboxed.** Like the lease, this lives in `/tmp` and works only among unsandboxed processes of one
  user. A sandboxed app's temporary directory is private.

## Declared residency

### Where it lives

| Path | Kind | Purpose |
| --- | --- | --- |
| `/tmp/atelier-gpu.lock.residency/` | directory, mode 0700, owned by the user | one record per participating process |
| `/tmp/atelier-gpu.lock.residency.lock` | regular file, mode 0600, owned by the user, one link | the lock every read-check-write takes |

Both are siblings of the lease directory, named from the lease path (`<lease>.residency/`, `<lease>.residency.lock`), so
a private lease path for tests gets private residency too. The lease directory itself is created and removed every
turn, so records cannot live inside it. Whoever needs them first creates them (`mkdir`, `open(O_CREAT)` with the modes
above). A participant refuses to use either if it is a symlink, not owned by the user, or has wider permissions; a
wider lock file is tightened to 0600, as the lease's `.reclaim` file is today.

### The lock

Every read, check and write of the records happens under `flock(LOCK_EX)` on `<lease>.residency.lock`. It is held for
milliseconds: never across a model load, GPU work, or a wait for the GPU lease. A participant that needs both takes the
residency lock first, releases it, and only then waits for the lease. Neither lock is ever held while waiting for the
other, so they cannot deadlock.

### The record

One file per process: `<lease>.residency/<pid>.json`, written only by that process. It is UTF-8 JSON, one object:

```json
{
  "schema": 1,
  "pid": 41237,
  "processStart": 1793134512.481203,
  "label": "atelier-language describer",
  "reservedBytes": 18253611008,
  "updated": 1793134520.004117
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | integer | 1. Readers treat another value as a live record and count its bytes if `reservedBytes` is readable, but never delete it. |
| `pid` | integer | The writer's pid; must equal the file name's. |
| `processStart` | number | The writer's start time, seconds since 1970 with microseconds (`proc_pidinfo(PROC_PIDTBSDINFO)`: `pbi_start_tvsec` plus `pbi_start_tvusec`/10⁶). |
| `label` | string | For messages only. |
| `reservedBytes` | integer | Bytes the process has reserved for loaded models, **including a load in progress**. For Kit, its `ResourceCoordinator`'s reserved total; for `atelier-language`, its fixed 9 or 17 GiB while it holds a model. |
| `updated` | number | When the record was last written, seconds since 1970. Informational. |

A record is written to `<lease>.residency/.<pid>.tmp` and renamed over `<pid>.json`, so readers never see a partial
file.

### Capacity

All participants use one rule for the host's capacity, so they agree on the number: **physical memory (`hw.memsize`)
minus a reserve for the system and applications**, with the reserve agreed once (proposed: 16 GiB, which gives 48 GiB on
a 64 GB Mac). It bounds what cooperating processes declare; it is not a measurement of free memory, so a participant may
keep its own extra gate, such as Atelier's memorystatus check.

### Registering before a load

A process that is about to load a model, holding no GPU grant or holding one:
1. Take the residency lock.
2. Reclaim stale entries ([below](#reclaiming-stale-entries)).
3. Sum `reservedBytes` over every other record, add its own current reservation and the new model's estimate.
4. If the sum exceeds the capacity: release the lock, then either refuse or wait. To wait, poll from step 1 (every
   200 ms, as a waiting lease acquisition does) with a timeout.
5. Otherwise write its record with the new total, then release the lock.
6. Load the model.

The reservation is declared in step 5, before the load. A second process registering while the first is still
loading already counts it, so two processes cannot both fit into the same room.

### Refreshing and leaving

- **When the reservation changes** (a model unloaded, a load failed, another model loaded), the process takes the lock
  and rewrites its record with the new total, following the same steps as registering when the total grows.
- **When it reaches zero, or at exit,** the process takes the lock and deletes its record.
- **No heartbeat is needed.** Liveness comes from pid and start time, so a long-lived process with a model loaded need
  not rewrite its record to stay counted.
- **A one-request helper** registers once before its load and deletes its record after unloading, before exiting. If it
  is killed instead, its record is reclaimed as stale.

### Reclaiming stale entries

Any participant, under the residency lock, before it sums:
1. **Dead or replaced writers.** It reads each `<pid>.json`. The record is stale if `kill(pid, 0)` fails with `ESRCH`,
   or if the live process's start time differs from `processStart` by more than one millisecond (a reused pid). It
   deletes stale records. `EPERM` (another user's process) is not stale.
2. **Records without a usable pid.** A file in the directory whose name is not `<digits>.json`, or that does not parse,
   or whose `pid` is missing or differs from its name, carries no usable pid. It is deleted once its modification
   time is more than 10 seconds old. A writer only ever leaves such a file by dying mid-write, and a complete write
   takes milliseconds.
3. **Temporary files.** A `.<pid>.tmp` is deleted if its pid is dead or replaced, or once it is more than 10 seconds
   old.

Reclaiming needs no cooperation from the dead process, and a live writer's record is never deleted: it is named by a
live pid with a matching start time.

## Pid-less leases, in the same change

Today a lease directory without a readable `pid` is never reclaimed, because the protocol has no rule for it
([open item](gpu-lease.md#open-item-pid-less-leases)). The same liveness rule closes it:
- **`start` beside `pid`.** A holder writes its process start time (the `processStart` format above) to `start` in the
  lease directory, after `pid` and by the same temporary-then-rename step. Readers that predate the change ignore it.
- **Pid-less reclaim.** A lease directory with no readable `pid` whose modification time is more than 10 seconds old
  is stale. It is reclaimed with today's steps under the `.reclaim` flock: move aside, then delete only if the moved
  copy is still pid-less, otherwise restore it.
- **Replaced holders.** A lease whose `start` does not match the live process named by `pid` is stale too, so a
  reused pid no longer keeps a dead holder's lease.

## What each side would do

- **Atelier:**
  - `atelier-language` registers its 9 or 17 GiB before each load and deletes its record after unloading;
  - the Krea process registers its render's reservation the same way;
  - `SharedGPULock` writes `start` and adopts the pid-less rule.
- **Kit:**
  - an `AetherKit` implementation of the records;
  - an optional binding that publishes a `ResourceCoordinator`'s reserved total, so `LanguageResourceAdmission`,
    Krea's and FLUX.2's reservations register themselves;
  - `start` and pid-less reclaim in `SharedGPULease`;
  - the development wrappers likewise.
- **Both:** the paths, the record format, the capacity reserve and the 10-second age above, or agreed replacements.

## Worked example (64 GB Mac, proposed 48 GiB capacity)

| Already registered | Registering | Sum | Result |
| --- | --- | --- | --- |
| Krea 1024×1024, 32 GiB | writer helper, 9 GiB | 41 GiB | loads |
| Krea 1024×1024, 32 GiB | 0.0.2 describer helper, 17 GiB | 49 GiB | waits until Krea unloads, or refuses |
| Krea 1024×1024, 32 GiB | 0.0.3 describer, about 14 GiB (Kit's measured 13.8 GiB) | 46 GiB | loads |

The Krea figure is its manifest's floor for a balanced 1024×1024 render, standing in for the reservation it would
declare.
