# Qwen3-VL resource admission (0.0.3 design)

Status: items 1 and 2 implemented, 2026-10-27, and the scratch figures measured; the guarded coexistence measurement
is ready and waits for a quiet host; the cross-process proposal waits for Atelier. This is the roadmap's safety item "Resource admission for the
Qwen3-VL encoder and decoder" ([roadmap](roadmap-0.0.3.md#safety-and-correctness)), moved ahead of other work by a
crash Atelier reported on Kit 0.0.2.

## The crash

Atelier ran a coexistence test (`atelier-17 D225 krea+language memory`, holding the GPU lease from 14:30:38 to
14:36:41 on 2026-10-27): the 0.0.2 VL-4B describer beside a Krea 1024×1024 render, at host memorystatus level 44. Both
processes died inside MPSGraph:
- **Krea** segfaulted while specializing a function: `CoreAIBackend.run` → `CoreAIFunctionCache.resolve` →
  `loadNativeFunction` → `MPSGraphDelegate.resolveCoreAI` → `-[MPSGraphExecutable specializedModuleWithDevice:…]` →
  `mlir::Operation::clone`, a read of address 0x2e (macOS 27.0.1, 26A434). A function's first use specializes it, so
  this was Krea loading a denoiser or decoder function lazily under memory pressure.
- **The VL process** failed with `allocateMTLBufferFromMTLHeap … exceeds heap total`.

The same test with the 4B writer instead of the describer ran clean twice. Two facts make the difference:
- **The 0.0.2 describer was twice the writer's size.** Its decoder wired two copies of the text weights (16.1 GiB of
  mapped-file pages) and reached about 23 GiB of Metal allocations, against 8.4 GiB for the writer; opening it read a
  22.5 GiB compiled file into the page cache ([evidence](evidence/qwen3-vl-memory-by-phase-2026-10-27.json)). Krea's own
  floor for a balanced 1024×1024 render is 32 GiB of RAM (`sizeFloors` in its manifest). On this 64 GB Mac the two
  together leave little for anything else.
- **Other host load overlapped.** From 14:32:16 to 14:34:05, inside that lease, this design's author ran the cleared
  VL-4B text export on the CPU, peaking at 16.3 GB of RSS. The GPU lease serializes GPU work, not CPU work, so the
  export was allowed, but it added to the pressure that reached level 44. The crash is therefore not reproduced
  evidence of the describer alone.

Neither failure is in Kit's code: both are Core AI and MPSGraph failing under memory
pressure, Krea's by a crash rather than an error. Kit's part is to keep sessions from being opened into that pressure
without the host knowing.

## What Kit has today

- **GPU admission** (`GPUAdmission`, optionally backed by the cross-process `SharedGPULease`) serializes *GPU work*:
  each request waits for a grant and holds it until its native work joins. It does not cover memory. A loaded model
  keeps its wired pages and Metal allocations between requests, after its grant is released, so two processes taking
  turns on the lease can still hold more memory together than the host has.
- **Resource admission** (`ResourceCoordinator`) is a fail-fast ledger of caller estimates, in one process. Krea
  (`KreaResourceAdmission`), FLUX.2 (`Flux2ResourceAdmission`), H3 and Kokoro take a coordinator and reserve each loaded
  model's residency (including its load peak) for as long as it is loaded, and each native call's scratch around the
  call. A load that does not fit is refused before any native work.
- **The language sessions take neither memory path.** `LanguageGenerationSession` and `VisionLanguageSession` take
  `gpu:` only, and **opening is not admitted at all**: the encoder's five grid functions and the decoder's function are
  loaded and specialized in `open`, outside any GPU grant. Krea and FLUX load inside an admitted operation. So a VL
  session opened while another holder renders specializes beside it, which is the pattern of Krea's crash.

## Proposal

### 1. Admit opening and reloads under the GPU grant

When a session has `gpu:`, `open` (and the decoder's memory-budget reload, which already does) takes a grant for the
loads and the first specialization: the encoder's grid functions, then the decoder's function. With a lease-backed
admission this keeps a VL open from specializing beside another process's render, as Krea's and FLUX's loads already
are. `unload()` keeps its rule of never waiting behind the GPU queue: a session still waiting for its open grant is
withdrawn.

Cost: an open waits for the GPU. With Krea rendering, that is the render's remaining time.

### 2. `LanguageResourceAdmission`, as Krea and FLUX have it

A public, opt-in profile against a host-selected `ResourceCoordinator`:

```swift
public struct LanguageModelResourceEstimate: Sendable {
    public init(modelSHA256: String, residentBytes: UInt64, runScratchBytes: UInt64) throws
}
public struct LanguageResourceAdmission: Sendable {
    public init(coordinator: ResourceCoordinator, models: [LanguageModelResourceEstimate],
                maximumContextLength: Int, maximumImageTokens: Int?) throws
}
```

- `LanguageGenerationSession.open(bundle:…, resources:)` and `VisionLanguageSession.open(bundle:…, resources:)` take it.
  Estimates are keyed by the registration's model hashes (`modelSHA256`, and `vision.modelSHA256` for the encoder), so
  other bytes are refused before loading, as Krea's and FLUX's are.
- **Residency** is reserved before each load and held while the model is loaded. The decoder's covers its weights,
  its KV caches at the registered context, and the memory budget's growth allowance, because the budget is what bounds
  retained memory. The encoder's covers its weights and its grid functions.
- **Scratch** is reserved around each request's native work (after the GPU grant, as Krea and FLUX order it).
- **Reloads** release the old model before loading the new one, so the decoder's residency carries over unchanged.
- **Shape limits** (`maximumContextLength`, `maximumImageTokens`) are checked while planning, so a request larger than
  the estimates cover is refused before native work.

The text writer gets the same profile through `LanguageGenerationSession`, so a host can budget every language session
the same way.

### 3. Reservations, from the one-function measurements

From the one-function bundles' memory by phase and budget runs
([evidence](evidence/qwen3-vl-one-function-2026-10-27.json)), MiB, rounded up:

| | VL-4B decoder | VL-4B encoder | VL-2B decoder | VL-2B encoder |
| --- | --- | --- | --- | --- |
| Weights wired, or Metal at first use | 8,443 | 992 open + 1,133 first grid | 3,578 | 872 open + 1,116 first grid |
| KV caches at the registered 4,096 tokens | 576 | – | 448 | – |
| Memory-budget allowance (default) | 2,048 | – | 2,048 | – |
| Largest growth of one warm request | about 500 | – | about 200 | – |
| **Resident estimate** | **11,600** | **2,200** | **6,300** | **2,000** |
| Largest sampled transient of a request (Metal / footprint) | 67 / 62 | (inside the image request) | 38 / 52 | (inside the image request) |
| **Scratch estimate** | **256** | **128** | **128** | **128** |

- **The encoder's further grids** add 0.8 GiB each to Metal's accounting but nothing resident (views of one copy), so
  the estimate counts only the first grid.
- **The budget allowance is a real bound.** The budget test kept the 4B within 13,311 MiB (warm 10,847 plus budget plus
  one request); the proposed 13,800 for a whole VL-4B session (decoder plus encoder) covers it.
- **Not reserved: the page cache.** Opening reads the compiled file (15 GiB for the one-function 4B) into clean,
  evictable pages. It pushes out other cache but cannot exhaust memory, so it is outside the ledger; the guide says so.
- **Scratch, measured.** `visionLanguageResourcePeaks` samples Metal's allocated size and the footprint every 50 ms
  on a separate thread, through open, a first and a warm text request, the first image request at each grid and a warm
  one ([evidence](evidence/qwen3-vl-resource-peaks-2026-10-27.json)). A request's peak above both its starting and its
  ending level was at most 67 MiB of Metal and 62 MiB of footprint on the 4B, and 38 and 52 MiB on the 2B. What a
  request leaves behind (retention, and an encoder grid's first use) persists and is covered by the residency and the
  budget allowance, not by scratch. The scratch estimates allow at least twice the largest sample, since 50 ms
  sampling can miss a shorter peak.
- **Metal's accounting is not residency.** Each further encoder grid adds 0.8 GiB to Metal's allocated size at its
  first image request (14.7 GiB on the 4B after all five) while adding nothing resident. The ledger reserves resident
  memory, so a host comparing it with Metal's figure sees the gap.
- **Checked natively.** The same test opens each bundle with the profile and GPU admission: the coordinator holds
  13,800 MiB (4B) and 8,300 MiB (2B) while the session is open and nothing after unload.
- **For comparison:** the 0.0.2 VL-4B would need about 20 GiB for the decoder alone. With Krea's 32 GiB floor, a
  48 GiB coordinator on this Mac refuses the 0.0.2 pair and admits the one-function describer beside Krea.

### 4. Across processes

`ResourceCoordinator` is per process, and Atelier runs each writer or describer request in its own helper process
(`atelier-language`), separate from Krea. So the in-process ledger does not prevent the case that crashed. Kit's part
here is a proposal for the shared lease protocol, written up with the pid-less lease item as one change for Atelier to
agree on: [declared residency](gpu-lease.md#proposal-declared-residency-and-pid-less-leases). In short:
- every cooperating process records the bytes it has reserved for loaded models in a per-process file beside
  `/tmp/atelier-gpu.lock`;
- before loading, it checks the host's total under a short lock of its own, not the lease, so a helper that holds no
  GPU grant (Atelier's `atelier-language`) takes part the same way;
- the reservation is declared before the load, so two processes checking at once can't both fit into the same room.

A host-pressure check (`kern.memorystatus_level` or a memory-pressure dispatch source) is no substitute: it is racy
and sees pressure only after it has built up. Not implemented.

## Reproduction

**The crash itself should not be reproduced on the shared Mac.** It needs memorystatus near 44, where the kernel starts
reclaiming from every process: other agents' jobs and Atelier Studio would be at risk, and both failures are inside
Apple frameworks, so a reproduction would show only that Core AI fails under pressure, which is already known.

**A guarded coexistence measurement is feasible and safe.** One test process, under the GPU lock, on a quiet host:
- open the one-function VL-4B session and a balanced Krea session with a shared `ResourceCoordinator`;
- alternate describer requests and a 1024×1024 Krea render;
- sample memorystatus level, Metal, wired mapped-file pages and the footprint every 50 ms;
- abort, releasing everything, if the level falls below 60 or free memory below 4 GiB.

It runs both models in one process and serializes their GPU work, so it measures the residency overlap that admission
must bound, not the cross-process race. It runs only the one-function describer: tripping the guard on purpose with
the 0.0.2 describer is not worth it on a shared Mac. It needs a quiet host, with no export or other heavy job alongside
(unlike the original run), and waits for a window from the lead. The guard acts between steps; a native call already
running finishes first.

## Implemented

- **Open admission.** With `gpu:`, `LanguageGenerationSession.open` and `VisionLanguageSession.open` load and
  specialize under a grant: the VL encoder under one, then the decoder under its own. Cancelling an opening that
  waits leaves the queue (the session does not exist yet, so there is nothing to unload).
- **`LanguageResourceAdmission`** on both sessions' bundle entry points (`resources:`). The raw entry points
  (`open(at:)`, `open(textModel:visionModel:…)`) take none, because they have no registered hash to key an estimate
  by. Residency is reserved after the open grant and before the load, released if the load fails, kept across
  memory-budget reloads, and released on unload. Scratch is reserved after each request's grant (and, in the VL
  session, after each encode's) and released when its native work joins. Image tokens are checked per image, before
  that image is encoded.
- **Tests.** CPU: the profile's validation and refusals, reservation accounting, the open admission's wait, release and
  cancellation, and the VL session's encode scratch, image-token refusal and residency release (with fakes). Native:
  `visionLanguageResourcePeaks` (above) and the guarded `visionLanguageBesideKreaGuarded`, not yet run.
