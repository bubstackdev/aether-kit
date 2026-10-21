# H3 lifecycle handoff — stopped at user request

Status: **stopped; overall lifecycle qualification incomplete**. On October 18, 2026 (America/Denver), the user requested: “Cancel current runs, capture a handoff doc, and spin down please.” Do not restart GPU work until the user asks to resume.

The owned native process group, PID 74347, received SIGTERM. Its controller joined the child with exit code -15 after 3,592.2917 seconds, without timeout. Source pins remained unchanged. The PID no longer exists, and the preparation, tests and native controller sessions have all exited. No additional run was launched. This termination was a user stop, not a failed numerical comparison. It does not establish a graceful native join at the final interruption point.

The immutable [interruption record](../evidence/h3-denoising-lifecycle-interruption-2026-10-18.json) links SHA-256-pinned copies of the raw report, five progress files, preparation/controller/provenance records, stderr, CPU test log and controller script. The native report retains `passed: false` and its last stage, `reopenedRecovery`; it was not rewritten to claim completion.

## Results preserved

| Role | Tensor hashes reproduced | CPU comparisons passed | State |
| --- | ---: | ---: | --- |
| Initial eight-step run | 864 | 464 | Complete |
| Retained cancellation at submitted step 0, block 24 | 435 | 410 | Complete |
| Full recovery on retained session | 864 | 464 | Complete |
| Cancellation with concurrent unload | 435 | 410 | Complete |
| Full recovery on reopened session | 631 | 431 | Interrupted; last delivered output `body.3.48.hidden` |

Step indices are zero-based: the reopened run completed three full steps and delivered block 48 of the fourth. Every persisted output hash matches the previously qualified capture, and every persisted CPU comparison passes. The two full runs each passed all 824 submitted/completed native pairs and all 864 qualified identities. Each cancellation passed the exact 434-submission prefix, joined submitted block 24 before releasing the operation slot, and submitted no later block. Retained cancellation returned the session to ready; concurrent unload started before the submitted work completed, waited for completion and left the session closed. Another request on that closed session was rejected.

CPU reference scope remains 448 complete tensors plus 16 sampled body checkpoints per complete trajectory, using actual captured native contexts. This is not independent full CPU-trajectory parity. Numerical limits remain 0.02 relative L2. Conditioning encoder, decoder/stitching, keyframes and media acceptance are still outstanding. This run was qualification, not a benchmark.

At the last report write, recorded peak physical footprint was 11,482,075,408 bytes (about 10.69 GiB), and sampled Metal peak was 12,455,460,864 bytes (about 11.60 GiB). These are the last persisted peaks, not a final whole-run memory report. The first session's five-second post-unload footprint was 1,176,291,392 bytes (about 1.10 GiB), resident memory 818,937,856 bytes, and Metal 720,896 bytes. The reopened session's final unload/settlement and the 512 MiB settled-growth comparison **were not reached**. The 20 GiB footprint/lifetime-peak and 18 GiB Metal limits were unchanged.

## Changes in this work

- `Sources/AetherProbe/H3DenoisingCommand.swift` adds explicit `--lifecycle`, per-role numerical recorders, retained and reopened recovery, submitted-work cancellation, concurrent unload, closed-session rejection and memory settlement checks. Existing capture-report decoding remains compatible through an optional lifecycle field.
- `Sources/AetherProbe/H3DenoisingLifecycleTrace.swift` records stage/step/block native events and validates exact submission/completion sequences, cancellation ordering and resource ownership without delaying the GPU observer.
- `Scripts/qualify-h3-denoising.py` adds `--lifecycle`, requiring matching CPU-only preparation and CPU references. Its predefined budget is 5,400 seconds and 3,340 native submissions: three full runs and two cancellation prefixes.
- `Tests/AetherProbeTests/H3DenoisingLifecycleTraceTests.swift` covers correct traces and rejects early release, late unload, missing completion, extra submissions and unexpected control events. Command tests reject native lifecycle execution without references before creating output.
- `docs/h3-integration.md` describes the repeatable lifecycle protocol. Its qualification result remains incomplete; the earlier eight-step numerical qualification remains valid.

Release validation passed **113 CPU tests** (117 registered; four native tests disabled). See `.local/h3-denoising-lifecycle-tests-02.log` and its archived copy. Python compilation and `git diff --check` passed. No native adapter, graph, weight, schedule or CPU-reference changes were made in this work. Compared with the prior successful denoising qualification, changed pins are only the CLI command, controller script, command tests and rebuilt executable; two new pins are the trace implementation and its tests. The current preparation/native provenance sets contain 169 identical pins.

## Workspace and evidence

Repository: `/Users/zakkeown/Code/bubstackdev/aether/aether-kit`.

- CPU preparation: `.local/h3-denoising-lifecycle-prepared-01/` and its sibling `-controller.json`, `-provenance.json`, `-stdout.json`, `-stderr.log`. Passed, CPU-only, joined, source pins unchanged; elapsed 86.2789 seconds.
- Interrupted native run: `.local/h3-denoising-lifecycle-native-01/` and the same sibling controller/provenance/stdout/stderr files. Progress is in one subdirectory per role. All files are retained.
- Controller console log: `.local/h3-denoising-lifecycle-controller-01.log`.
- Prior passing numerical gate: `.local/h3-denoising-native-01/` and [its qualification record](../evidence/h3-denoising-qualification-2026-10-18.json).
- CPU references: `.local/h3-denoising-cpu-references-01/references.json`.
- Original unqualified capture: `.local/h3-denoising-capture-02/`.
- `.local/finalize-h3-denoising-lifecycle.py` is an unexecuted helper for a **passing complete run**. It intentionally rejects this interrupted result. `.local/capture-h3-lifecycle-interruption.py` produced the interruption archive instead.

The checkout already contained extensive uncommitted Krea/H3 changes and evidence before this task. Preserve them; nothing was staged, committed or reverted. Do not change `.git`, Atelier, the prototype, exported graphs, source checkpoints or model weights. Models remain referenced from their existing locations; no weights were copied.

## Resume only when requested

The current CLI cannot resume a trajectory from the interrupted block. Repeat the bounded gate into new output directories after checking the user has released the GPU again. The preceding passed phases are useful evidence but do not make the overall lifecycle gate pass. A shorter follow-up would require an explicit harness change and fresh source-pinned preparation; do not silently skip phases or relax limits.

First run CPU validation if source changes are made:

```sh
source .local/native-assets.env
Scripts/check.sh > .local/h3-denoising-lifecycle-tests-03.log 2>&1
```

For a repeat of the current frozen harness, use the following arguments with a new preparation output, then a new native output:

```sh
python3 Scripts/qualify-h3-denoising.py \
  --bundle /private/tmp/aether-h3-qkv-ready/render/bundle.json \
  --phases /private/tmp/aether-h3-qkv-ready/phase/bundle.json \
  --cpu-record /private/tmp/aether-h3-qkv-phase-oracles/cpu-checks.json \
  --chain .local/h3-backbone-cpu-references-01/chain.json \
  --modulations /Users/zakkeown/Code/bubstackdev/aether/lab/aether-prototype/.local/h3-diagnostic/models \
  --references .local/h3-denoising-cpu-references-01/references.json \
  --output .local/h3-denoising-lifecycle-prepared-02 \
  --prepare-only
```

After that CPU-only controller passes and joins, repeat the command with `--output .local/h3-denoising-lifecycle-native-02 --prepared .local/h3-denoising-lifecycle-prepared-02 --lifecycle` in place of the final output/mode options. Native execution must remain owned by this wrapper; it enforces timeout, process-group cleanup, joining and source-pin checks. Existing outputs are rejected. Do not overwrite the interrupted attempt or publish it as successful.

A successful follow-up must finish all eight reopened recovery steps, unload the second session, pass the final settlement/growth gate, produce a complete passing report, join its controller, and preserve/verify its evidence. Only then update the lifecycle status. Real conditioning and decoders follow that gate. No work is scheduled or running now.
