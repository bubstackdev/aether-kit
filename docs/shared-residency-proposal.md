# Proposal: shared declared residency and pid-less leases

Status: a coordinated future change, not 0.0.3 work. Proposed 2026-10-27 and revised after Atelier's review the same day.
Nothing here is implemented. It changes the protocol around the shared GPU lease ([GPU lease](gpu-lease.md)), so it is
adopted only through an Atelier decision record and one coordinated release of Atelier's `SharedGPULock`, Kit's
`SharedGPULease` and the development wrappers. This page is meant to be reviewed on its own.

## Why

The GPU lease (`/tmp/atelier-gpu.lock`) serializes GPU work. It does not account for memory:
- **Models stay loaded.** A model keeps its wired pages and Metal allocations after its process's turn on the GPU ends.
- **CPU jobs take no lease.** Heavy CPU jobs, such as Kit's model exports and fixture regenerations, never take the
  lease.

So cooperating processes can take turns correctly and still hold more memory together than the Mac has.

On 2026-10-27 an Atelier memory test ran the Kit 0.0.2 VL-4B describer, in its own helper process, beside a Krea
1024×1024 render. Both processes died inside MPSGraph at memorystatus level 44
([analysis](qwen3-vl-resource-admission.md#the-crash)). During the same minutes, a Kit CPU export of the VL-4B text
tower peaked at 16.3 GB of RSS without any lease. The crash is therefore not a finding about the describer beside Krea:
a scheme that counted only Krea and the describer would not have prevented it, and the export had to be counted too.
Kit's in-process `ResourceCoordinator` sees no other process, and a live memorystatus check sees pressure only after it
has built up.

## Constraints

- **No GPU grant required.** Some participants hold memory without holding the lease: Atelier Studio's resident
  renderer pool between turns, the one-shot `atelier-language` helper, and Kit's CPU jobs. Declaring residency never
  depends on holding the lease.
- **Crash-safe.** A process may be killed or stopped at any point. What it leaves behind is recognizable as stale by any
  other participant, without its help.
- **No pid reuse confusion.** macOS reuses pids, so liveness is checked by pid and process start time together.
- **Same user, unsandboxed.** Like the lease, this lives in `/tmp` and works only among unsandboxed processes of one
  user.
- **An addition to live checks.** Non-participants are invisible to records, so each participant keeps its own live gate
  and ANDs the record check with it. Atelier's gates are a `kern.memorystatus_level` floor of 25 in LanguageMemory, and
  60 % or more before a render.

## Pid-less leases and exclusive ownership

Today a lease directory without a readable `pid` is never reclaimed ([open item](gpu-lease.md#open-item-pid-less-leases)).
An age-based reclaim alone would be unsafe:
1. A creates the lease directory, then stalls before writing `pid` (heavy load, `SIGSTOP`, a debugger).
2. A reclaimer R removes the pid-less directory once it passes the age.
3. B creates the directory and writes `pid` = B.
4. A resumes and writes its `pid` into B's directory. `Data.write(.atomic)` replaces the file.

Now both A and B believe they hold the lease. Had A's write failed instead, A's cleanup would have removed B's
directory. So every implementation adopts all of these rules in the same release:

1. **`pid` is created exclusively.** The holder writes its pid to a temporary file and moves it into place with
   `renamex_np(tmp, "pid", RENAME_EXCL)`. In a direct open that is `O_EXCL`; in the shell wrappers, `set -C`
   (noclobber). **Whoever creates `pid` owns the lease.**
   - A process whose exclusive create fails, because `pid` exists or the directory is gone, has lost.
   - A loser never removes the directory. It retries acquisition or reports busy.
   - In the race above, A's create fails against B's `pid`, and A retries.
2. **`start` before `pid`.** The holder writes `start` (its process start time, [format below](#start-time)) by
   temporary file and rename *before* it creates `pid`. A reader that sees `pid` then sees `start` too, except from an
   implementation that writes no `start`; for those, a reader checks the pid only.
3. **Age from birth time.** A lease directory with no readable `pid` is stale once its age exceeds **60 seconds**,
   measured from the directory's `st_birthtime`, not its modification time.
   - With rule 1 in place, the age no longer guards correctness. It only bounds how long a dead pid-less lease blocks
     everyone.
   - The directory is reclaimed with today's steps under the `.reclaim` flock: move it aside, and delete it only if the
     moved copy is still pid-less; otherwise restore it with `renamex_np(RENAME_EXCL)`.
4. **Replaced holders.** A lease is also stale when its `start` does not match, at whole seconds, the live process
   named by `pid`. A reused pid no longer keeps a dead holder's lease.

## Start time

Every start time here (the lease's `start`, a record's `processStart`) is **integer epoch seconds**: the process's
`pbi_start_tvsec` from `proc_pidinfo(pid, PROC_PIDTBSDINFO, …)`, compared at whole seconds.
- **Swift and C** read it with `proc_pidinfo`.
- **Python** reads it with `psutil.Process(pid).create_time()`, truncated to an integer.
- **The shell wrappers** write no `start`, because `ps -o lstart` is localized. For a lease without one, readers check
  the pid only.

## Declared residency

### Where it lives

| Path | Kind | Checks |
| --- | --- | --- |
| `<lease>.residency/` (`/tmp/atelier-gpu.lock.residency/`) | directory | mode 0700, owned by this user, not a symlink |
| `<lease>.residency.lock` | regular file | mode 0600, owned by this user, one link, not a symlink |

- **Beside the lease, not inside it.** Both are named from the lease path, so a private test lease gets private
  residency. The lease directory is created and removed every turn, so records cannot live inside it.
- **Created on first use**, with those modes, by whichever participant needs them first.
- **Checked before use.** A participant refuses either if a check fails. The one exception: a lock file with wider
  permissions is tightened to 0600, as the lease's `.reclaim` file is today.

### The lock

Every read, check and write of the records happens under `flock(LOCK_EX)` on `<lease>.residency.lock`.
- **Held for milliseconds.** It is never held across a load, a job, GPU work or a wait for the lease.
- **Lock order.** A participant that needs both takes the residency lock first and releases it, then waits for the
  lease. Neither lock is held while waiting for the other, so they cannot deadlock.

### Records

One file per participating process, `<lease>.residency/<pid>.json`, written only by that process:

```json
{
  "schema": 1,
  "pid": 41237,
  "processStart": 1793134512,
  "label": "Atelier Studio",
  "reservedBytes": 38654705664,
  "updated": 1793134520
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | integer | 1. A reader counts a record with another schema if `reservedBytes` is readable, and never deletes it. |
| `pid` | integer | The writer's pid; equals the file name's. |
| `processStart` | integer | The writer's start time ([format](#start-time)). |
| `label` | string | For messages only. |
| `reservedBytes` | integer | Everything the writer keeps loaded, plus anything it is about to load or run, plus the reservation of any child it admits on its behalf. |
| `updated` | integer | Epoch seconds of the last write; informational. |

- **Writing.** The writer writes `<pid>.json.tmp` and renames it over `<pid>.json`. Readers ignore every `*.tmp`.
- **Reading.** A reader opens each record with `O_NOFOLLOW`. On the open descriptor it checks that the file is regular,
  owned by this user, and has one link. A file that fails these checks is not read; it is reclaimed by age.

### What a reader counts

Declared bytes are a floor. For each live record, a reader counts **max(`reservedBytes`, measured)**, where measured
is `ri_phys_footprint` from `proc_pid_rusage(pid, RUSAGE_INFO_V4)`. For a same-user process this needs no privileges.
- **Why declare at all.** The footprint leaves out file-backed pages. A Core AI model's weights are mapped from its
  compiled file, so a describer wiring 16 GiB showed a footprint under 3 GiB.
- **Why measure as well.** The measurement catches a writer that uses more than it declared.

### Capacity

All participants use one rule: **a fraction of physical memory (`hw.memsize`)**. No fraction is proposed here. It will
be calibrated in a clean, announced measurement window that Kit and Atelier run together at Atelier's 0.0.3 bump, and
recorded in the decision record. It bounds what cooperating processes declare and use; it is not a measurement of free
memory, which is why the live gates stay.

### Registering before a load or a job

A participant about to load a model or start a memory-heavy job, whether or not it holds the lease:
1. Take the residency lock.
2. Reclaim stale entries ([below](#reclaiming-stale-entries)).
3. Add up max(declared, measured) over every other live record, then add its own current total and the new estimate.
4. If the sum exceeds the capacity, or its own live gate refuses, release the lock and refuse or wait. To wait, poll
   from step 1 every 200 ms with a timeout.
5. Otherwise write its record with the new total, then release the lock.
6. Load the model or start the job.

The reservation is declared before the load. A second participant registering while the first is still loading
already counts it, so two cannot both claim the same room.

### Refreshing and leaving

- **When the total changes** (a model unloaded, a load failed, a child finished), the participant takes the lock and
  rewrites its record. A growing total goes through registration.
- **At zero, or at exit,** it takes the lock and deletes its record.
- **No heartbeat.** Liveness comes from pid and start time, so a long-lived process need not rewrite its record to stay
  counted.

### Reclaiming stale entries

Any participant, under the residency lock, before it sums:
1. **Dead or replaced writers.** A record is stale if `kill(pid, 0)` fails with `ESRCH`, or if the live process's start
   time differs from `processStart` at whole seconds. Stale records are deleted. `EPERM` (another user's process) is
   not stale.
2. **Entries without a usable pid.** Such a file is deleted once its age from `st_birthtime` exceeds 60 seconds. It is
   one whose name is not `<digits>.json`, that fails the open checks, that does not parse, or whose `pid` is missing or
   differs from its name.
3. **Temporary files.** A `<pid>.json.tmp` is deleted if its pid is dead or replaced, or once its age from
   `st_birthtime` exceeds 60 seconds.

A live writer's record is never deleted, because it is named by a live pid with a matching start time.

## Who takes part

- **Atelier Studio** is the long-lived process. It keeps Krea resident in its renderer pool between turns. Studio
  admits and spawns the one-shot `atelier-language` helper, and it writes **one record**: the pool's resident bytes plus
  the reservation of any helper in flight. Today those reservations are 9 GiB for the writer and 17 GiB for the
  describer, to be re-measured at Atelier's 0.0.3 bump.
- **The `atelier-language` helper** writes no record. Studio's record covers it from before it is spawned until it
  exits.
- **Kit:**
  - **Library sessions** declare through a `ResourceCoordinator` bound to the records, when a host opts in. That covers
    `LanguageResourceAdmission` and Krea's and FLUX.2's reservations.
  - **`aether-probe`** declares for its runs.
  - **Heavy CPU jobs** (model exports, fixture regenerations, anything above a few GiB) declare their estimated peak
    before they start.
  - **`SharedGPULease`** adopts the exclusive `pid`, `start` and pid-less rules.
- **The development wrappers** adopt the exclusive `pid` (`set -C`) and pid-less rules, and write no `start`.
- **Agreed in the decision record:** the paths, the record format, the 60-second age, and the capacity fraction after
  the calibration window.

## For scale

These are the sums a capacity would be compared against on a 64 GB Mac, using today's floors and estimates. None of
them is a capacity:

| Already declared | Registering | Sum |
| --- | --- | --- |
| Krea 1024×1024, 32 GiB | writer helper, 9 GiB | 41 GiB |
| Krea 1024×1024, 32 GiB | 0.0.2 describer helper, 17 GiB | 49 GiB |
| Krea 1024×1024, 32 GiB | 0.0.3 describer, about 14 GiB | 46 GiB |
| Krea 32 GiB and 0.0.2 describer 17 GiB | Kit VL-4B text export, 16.3 GB peak | about 64 GiB |

The last row is the situation of the 2026-10-27 crash. The Krea figure is its manifest's floor for a balanced
1024×1024 render, standing in for what Studio's pool would declare.
