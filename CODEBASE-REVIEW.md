# Codebase review

Reviewed 2026-10-20 against base commit `4a74f99`, including the locally modified and untracked source present during this audit.

## Scope and methods

Reviewed the first-party Swift modules under `Sources`, corresponding tests and fixtures, seven scripts under `Scripts`, `Forge/export_smoke.py`, three benchmark plans, the local consumer example, package manifests and lockfiles, repository configuration, and relevant README, DocC and engineering documentation. No applicable `AGENTS.md`, CI workflow, or deployment configuration was found. Existing user changes were preserved; only this report was added.

Three read-only reviewers covered Krea/shared tensors, H3 contracts/native sessions, and the probe/scripts respectively. The primary reviewer inspected the shared CoreAI executor, package/example integration, test gates and resource ownership, and independently checked the reported controller issue. Candidates were checked against callers, tests and documented contracts. No application fixes were made.

Excluded generated build/dependency caches (`.build`), local model exports and runtime outputs (`.local`), Python bytecode, binary/OS metadata, and bulk archived `docs/evidence` records. Relevant evidence descriptions and the lifecycle handoff were consulted, but historical GPU results, external model assets and independent reference implementations were not requalified. The locally cached tokenizer loader was inspected to check its local-file contract; third-party dependencies did not receive a full implementation or vulnerability audit. iOS builds, native GPU inference, performance and long-running resource behavior were not independently exercised. This review does not establish exhaustive correctness or production readiness.

## Medium

### M1. Controller termination leaves the detached probe without its watchdog

**Locations:** [Scripts/qualify-h3-phases.py:84](Scripts/qualify-h3-phases.py#L84), cleanup at lines 92–102; [Scripts/qualify-h3-denoising.py:99](Scripts/qualify-h3-denoising.py#L99), cleanup at lines 107–117.

**Evidence and trigger:** Both wrappers start the probe with `start_new_session=True`, then rely on a Python `finally` block to terminate its process group and join it. Neither wrapper installs a SIGTERM or SIGHUP handler. Sending SIGTERM to the wrapper, or to its original process group through a supervisor, terminates Python without entering that cleanup. The child belongs to a different session/process group and survives. Ordinary timeout cleanup still works; this finding concerns interruption of the controller itself.

**Consequence:** A canceled qualification can continue consuming GPU time and memory after its controller exits. The wrapper's timeout no longer exists, so the probe can outlive the advertised budget; a hung probe could remain indefinitely. Its controller record stays unjoined, with no final interrupted status. This matters because the native CLI relies on external ownership for its timeout, and these wrappers are the documented bounded execution mechanism. A supervisor that independently terminates every descendant can mitigate the issue.

**Verification:** Independently extracted each wrapper's exact complete subprocess supervision block, supplied `/bin/sleep 20` as a harmless child and temporary output paths, then sent SIGTERM to the wrapper. Both reproductions returned controller exit `-15` while the detached child was still alive. Each dummy child process group was explicitly killed afterward. Survival is confirmed; continued GPU allocation is the consequence inferred for a real in-flight probe, not a native measurement from this audit. SIGHUP was established by inspection, not separately reproduced.

**Suggested fix:** Install SIGTERM/SIGHUP handling before spawning, and route interruption through a shared cleanup path that terminates the owned child group, escalates after a bounded grace period, joins the child and persists an interrupted result. Handle signals during spawn and initial bookkeeping as well as during `wait`. Add a small subprocess regression check using a harmless child to verify signal forwarding, joining and final status.

## Verification performed

- Offline Release build/test with Swift 6.4 and dependency auto-resolution disabled; all `AETHER_*` opt-ins were removed from the test environment. Swift Testing reported 119 cases: **112 passed, seven skipped**, with no failures. Skips covered four real-weight native tests and three external-fixture tests. The native SwiftPM build engine emitted its existing deprecation warning.
- Source inspection of tensor bounds, model descriptors, cancellation/unload ownership, Krea carry routing and asset selection, H3 numerical/layout and streamed-weight contracts, probe reporting, and script process management. No additional actionable findings were verified in these areas within the stated coverage.
- Two bounded, weight-free SIGTERM reproductions confirmed M1. No GPU workload, export, download or external service operation was performed.
