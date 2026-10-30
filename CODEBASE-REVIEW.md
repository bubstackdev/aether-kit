# Codebase review

Reviewed on 2026-10-30 against commit `46d0e7d` (`Merge pull request #64 from bubstackdev/v0.0.3/ttft`). The checkout was clean at the start, with no nonignored untracked source files. This audit changes only this report.

**Results: 7 Medium and 6 Low findings. No Critical or High findings were verified.** Severity reflects this repository's local Apple SDK and experimental developer-tool context, including its documented use of trusted local model manifests. It does not assume a network-facing service or hostile remote model distribution.

## Medium

### M1. Valid Krea batch options bypass the shared GPU lease

**Locations:** `Sources/AetherProbe/ProbeGPULease.swift:21–23`; `Sources/AetherProbe/KreaBatchCommand.swift:24–42`; `Sources/AetherProbe/AetherProbe.swift:16–17`.

The lease classifier checks the raw argument count and fixed argument positions. The batch parser separately accepts `--steps=N` anywhere after the command and multiple `--ref=IMAGE` options, removing them before validating positional arguments. Consequently, a valid batch with a shape file plus `--steps=16`, with two reference flags, or with `--steps=16` immediately after the command is classified as CPU-only. `ProbeGPULease.acquire` then returns without acquiring a lease, and dispatch still runs the batch's GPU pipeline.

**Verification and consequence:** A temporary CPU harness using the current parser, configuration validator, and classifier accepted all three forms and printed `classified GPU=false`. No inference was needed. These ordinary supported options allow concurrent execution with another cooperating GPU user, defeating the CLI's stated exclusion policy and risking resource pressure or invalid benchmark timings. Actual GPU contention was not induced.

**Suggested fix:** Parse the command once and classify the parsed operation, or have the classifier validate `KreaBatchOptions` directly. Add classification tests for flags before positional arguments and for every supported shape/reference combination.

### M2. Numeric generation guides can terminate schema compilation

**Locations:** `Sources/AetherLanguage/LanguageJSONGrammar.swift:95–104,163–166`; `Sources/AetherLanguage/AetherGenerationSchema.swift:188–206`.

The grammar accepts finite numeric bounds and immediately converts rounded `Double` values to `Int64`; number bounds are first multiplied by one billion. Array counts similarly convert directly to `Int`. Finiteness and integrality do not establish representability. Public `.maximum(Int.max)` becomes an out-of-range rounded `Double`; a `Double` property with `.maximum(1e12)` exceeds `Int64` after scaling.

**Verification and consequence:** Both public-guide examples, compiled with the current pure schema/grammar sources in isolated subprocesses, exited with signal 5 and `Double value cannot be converted to Int64 ... greater than Int64.max`. An unsupported schema therefore crashes the embedding application instead of throwing `unsupportedSchema`. The two bound failures were reproduced; the corresponding count conversion has the same unchecked boundary by inspection.

**Suggested fix:** Validate rounded/scaled bounds and counts against representable values and the grammar's documented digit limits before conversion. Throw `unsupportedSchema` for unsupported ranges, and cover integer extrema, scaled numeric overflow, and excessive counts.

### M3. Requests queued behind unload retain the closed language backend

**Locations:** `Sources/AetherLanguage/LanguageModelResources.swift:64–69,81–88`; `Sources/AetherLanguageCoreAI/LanguageGenerationSession.swift:770`.

`withModel` resolves and retains the loaded model before acquiring its request queue. If request A is active, unload queues next, and request B follows, B captures A's backend and then waits behind unload. Unload clears the cache and closes that backend. B subsequently acquires its turn and executes with the retained, closed backend instead of loading the current model.

**Verification and consequence:** A deterministic CPU harness with the current resource actor and a fake backend that becomes closed on unload produced `later request received closed backend: true`. The harness synchronized the two queue positions before releasing A. Native unload is terminal, so the expected native consequence is a failed request after a concurrent unload. Existing facade fakes count unloads without enforcing closure, masking this ordering error.

**Suggested fix:** Acquire the request queue before resolving/loading the model, and release the queue on load failure as well as body completion. Add a test with A, unload, then B, asserting B uses a newly loaded backend.

### M4. Different enums with the same case name corrupt each other's schemas

**Locations:** `Sources/AetherLanguage/AetherGenerationSchema.swift:99,126–132`.

Union definitions are named only `Discriminated` plus the case name. The containing object's definition merge silently keeps the last value for duplicate names. Two distinct enum properties such as `First.value(Int)` and `Second.value(String)` therefore both reference `DiscriminatedValue`; the second enum replaces the first enum's payload schema.

**Verification and consequence:** Compiling a root object containing these two schemas rejected the correct first integer payload and accepted an incorrect first string payload. This uses ordinary distinct type names and needs no malformed schema. Structured generation enforces the wrong contract, potentially making successful Swift decoding impossible for the first property.

**Suggested fix:** Namespace case definitions by owning type and case, update references consistently, and detect incompatible definition-name collisions rather than silently overwriting them. Test two nested enums sharing a case name with different payloads.

### M5. Unicode escapes can complete the grammar but fail JSON decoding

**Locations:** `Sources/AetherLanguage/LanguageJSONMatcher.swift:139–145`; `Sources/AetherLanguage/LanguageJSONSchemaConstraint.swift:187`; `Sources/AetherLanguage/AetherLanguageSession.swift:62`.

The escaped-Unicode state accepts any four hexadecimal digits and returns directly to the string body. It does not track UTF-16 surrogate pairs. An isolated high surrogate such as `"\uD800"` is therefore accepted as a complete string, allowing the constraint to admit a stop token, although Swift's JSON decoder rejects it.

**Verification and consequence:** The current matcher accepted that complete value, while `JSONDecoder` rejected the same bytes with `Missing low code point in surrogate pair`. Structured output can reach an allowed completion and still fail decoding. This is a contract failure, not a demonstrated security exploit; it requires the model to choose the escaped surrogate sequence.

**Suggested fix:** Track the decoded UTF-16 escape value, require a high surrogate to be followed by a low-surrogate escape, and reject isolated low surrogates. Test valid pairs and invalid pairs across token boundaries.

### M6. Finite speech duration outputs can crash before frame limits apply

**Locations:** `Sources/AetherSpeech/KokoroSynthesis.swift:28–35`; `Sources/AetherSpeechCoreAI/KokoroSpeechSession.swift:431–435`.

`KokoroAlignment.frames` checks that duration sums and speed are finite, then converts rounded duration/speed values directly to `Int` and sums them with unchecked addition. A finite input may exceed `Int` or overflow during division or accumulation. Session synthesis passes model durations into this helper with `maximumFrames: .max`, checking its actual limits afterward.

**Verification and consequence:** Calling the current function with `[Float.greatestFiniteMagnitude]`, speed 1, and maximumFrames 4096 terminated an isolated CPU subprocess with signal 5 instead of throwing. Malformed or extreme finite model results, or direct public-helper input, can terminate the host. No ordinary qualified-model output was shown to trigger this.

**Suggested fix:** Validate divided/rounded durations before conversion, enforce a meaningful frame ceiling before allocating alignment buffers, and accumulate with checked arithmetic. Test representability, division overflow, and total overflow as well as NaN and modest frame-limit violations.

### M7. Interrupting the memory gate leaves its child workload running

**Location:** `Scripts/qualify-h3-memory.py:128–138`.

The gate launches each command with `start_new_session=True`, but its wait block only handles `TimeoutExpired`. Ctrl-C interrupts the controller's process group; the separate workload group does not receive that terminal signal. There is no `finally` cleanup to terminate and join the child when `KeyboardInterrupt` or another exception exits the controller.

**Verification and consequence:** Executing the current stdlib supervision block with a harmless five-second sleeper, then signaling the supervisor group, left the child alive after its supervisor exited. The reproduction explicitly cleaned up all spawned processes. On the actual render path, the inferred consequence is an unmonitored GPU job retaining its lease and continuing resource use after the user interrupts the gate. No GPU job was launched for this audit.

**Suggested fix:** Always terminate/kill and join the owned process group in `finally`, preserving normal successful completion, and persist an interrupted failure record. The other profile/trajectory controllers already provide a cleanup pattern to follow.

## Low

### L1. Decode-only H3 rejects valid first-frame renders

**Locations:** `Sources/AetherProbe/H3ProfileCommand.swift:293,398–409`; `Sources/AetherH3/H3VideoStitching.swift:231–233`; `Sources/AetherH3/H3VideoSession.swift:250–251`.

The render command saves the complete `final_video` tensor, including FL2VA keyframe anchor rows. Decode-only reads it unchanged and passes it to `videoWindowRows`, which requires exactly `targetVideoRows * 96` scalars. Normal generation explicitly drops the anchor prefix first. For 960×544×124, the persisted first-frame tensor has 19,380 rows while the decoder expects 18,870.

**Verification and consequence:** The writer, exact-size guard, and normal generation's prefix removal establish the mismatch without inference. `h3-profile-decode` fails before tile execution for a valid FL2VA render; T2VA lacks the prefix and is unaffected.

**Suggested fix:** Validate that dump and render metadata match, strip the declared condition rows, and test both T2VA and FL2VA decode preparation with CPU tensors.

### L2. Enum macro drops an explicit `value:` argument label

**Location:** `Sources/AetherLanguageMacrosPlugin/AetherGenerableMacro.swift:116,152`.

The macro records an absent associated-value label and an explicit label named `value` as the same JSON key. It then omits the Swift argument label whenever the sole key equals `value`. Consequently, `@AetherGenerable enum Labeled { case data(value: Int) }` expands to `.data(try container.decode(...))`.

**Verification and consequence:** Typechecking this declaration with the built project macro plugin fails with `missing argument label 'value:' in call`. A valid Swift enum cannot use the advertised macro.

**Suggested fix:** Retain the Swift argument label independently from its JSON field key. Test both `case data(Int)` and `case data(value: Int)`.

### L3. Speech compatibility checks allow a different phoneme mapping

**Locations:** `Sources/AetherSpeechCoreAI/KokoroSpeechSession.swift:421–429`; `Sources/AetherSpeech/KokoroPhonemes.swift:8–18,40–56`.

The session checks only whether each incoming ID belongs to its vocabulary's ID set. `KokoroPhonemes` can be constructed with another valid vocabulary that uses the same ID set for different symbols. With source `{a:1,b:2}` and target `{a:2,b:1}`, incoming phoneme `a` passes the guard as `[0,1,0]`, whereas the target encoding is `[0,2,0]`.

**Verification and consequence:** The current vocabulary/phoneme types and exact session predicate confirm acceptance of incorrect model input. Wrong speech is the inferred consequence; synthesis was not run. This requires mixing phonemes from different/custom vocabulary mappings, so standard use with one bundle is unaffected.

**Suggested fix:** Retokenize the retained phoneme string with the session vocabulary and compare IDs, or carry and validate a vocabulary fingerprint. Add a same-ID-set/different-mapping regression.

### L4. Memory-gate resume ignores changed workload arguments

**Locations:** `Scripts/qualify-h3-memory.py:76–80,114–118,157–164`.

The resume digest hashes files but excludes `--frames`, `--seed`, and `--recipe`. Controller record names include only task and canvas. Reusing the same work directory with unchanged files and different values for those arguments therefore reuses earlier passed dump/render stages. Timeline validation checks task and canvas, without comparing those changed requested values.

**Verification and consequence:** Inspection of digest construction, cache acceptance, and timeline validation confirms the omitted inputs. A user requesting a new workload can receive a successful gate based on the old one. The emitted entry retains the old actual execution identity and frame count; this is not evidence of falsely qualifying a new identity. No expensive gate was rerun to demonstrate the cache hit.

**Suggested fix:** Include normalized render arguments or complete commands in the resume identity, reject incompatible reuse, and compare resumed timeline/dump metadata with the requested recipe, seed, and snapped frame count.

### L5. H3 declaration validation admits values that later trap

**Locations:** `Sources/AetherH3/H3MeasuredResources.swift:47–60`; `Sources/AetherH3/H3ResourceAdmission.swift:18–25,40`; `Sources/AetherH3/H3Profile.swift:197–205`.

Canvas validation only checks digit syntax before force-unwrapping integer conversion or multiplying dimensions. For example, `999999999999999999999999999999x544` passes the measured-entry canvas syntax check, but accessing `maximumPixels` traps on `Int(nil)`. Representable sides can overflow the product. Measured peak-plus-margin-plus-operation sums and resource admission's operation-plus-stage sum also use unchecked addition despite accepting arbitrary positive `UInt64` inputs.

**Verification and consequence:** The accepted syntax, constructor guards, force unwraps, and arithmetic establish these failure paths by inspection. A malformed custom/local resource declaration can terminate the process instead of producing the documented validation/admission error. Supplied normal profiles were not shown to trigger this, and these manifests are trusted local data rather than a demonstrated remote attack surface.

**Suggested fix:** Parse canvas sides once with checked conversion, enforce geometry bounds, check products and byte sums, and throw typed errors. Include oversized decimal sides and overflowing sums in validation tests.

### L6. Missing static audio support is discovered after the render

**Locations:** `Sources/AetherH3CoreAI/H3ProfileStages.swift:518,536–551`; `Sources/AetherH3/H3VideoSession.swift:269–273`; `Sources/AetherH3/H3Profile.swift:50–53`.

Native stages select a fixed audio decoder only at final audio decoding, after conditioning, denoising, video decoding, and video-frame delivery. There is no geometry-dependent preflight of that decoder selection. A profile supporting static audio lengths 37 and 207 accepts ordinary geometry requested at 125 frames, which snaps to 141 frames and needs 235 audio latents per channel, provided its other shape limits permit that geometry. It then predictably fails at audio decoding.

**Verification and consequence:** The lookup order and failure are code-verified; the 37/207 decoder menu was confirmed from locally present profile metadata without loading weights. Full failure was not induced with inference. Unsupported requests can consume an entire render and emit all video frames before failing. Severity is Low because H3 is experimental and its documented qualified clip menu is narrow.

**Suggested fix:** Expose profile-specific geometry/support validation and invoke it before any stage loads, including whether requested audio has a matching fixed decoder. Test rejection using fake stages that assert no work or frame delivery begins.

## Scope, methods, and limitations

The review used three read-only component reviewers plus direct integration review. Candidates were checked against current source, relevant callers and tests, and duplicates or unsupported suspicions were removed. No application fixes, configuration changes, dependency changes, or committed test additions were made.

| Area | Coverage |
| --- | --- |
| Shared SDK and assets | Tensor/media values, resource coordinator, fair queue, GPU admission and interprocess lease, local installation/hash/path checks, external file leases, model execution and staged ownership; corresponding tests inspected. |
| Language and vision | Facade, model/session loading, stream cancellation and buffering, structured schemas/grammar/constraints/macros, FoundationModels adapter, sampling/decode, tokenizers, native resource/lifecycle integration, image preprocessing and vision prompting; targeted tests and Forge registration/export/staging reviewed. |
| Image and speech | Krea, FLUX.2 and Kokoro configurations, assets, schedules, references, tokenizers, contracts, generation lifecycles, budget/factory cleanup and matching tests; Forge FLUX.2 and Kokoro tooling inspected, with mathematical graph internals sampled. |
| H3 | Geometry/layout/schedules, tensor/file and profile contracts, admissions, stitching/session/tokenizer, graph and buffer ownership, block/chain/phase/denoising adapters, main export/oracle/qualification scripts and Probe integration; relevant tests/documentation inspected. |
| Tooling and distribution | Package manifests and lockfiles, CI/dependabot, build/floor/benchmark scripts, CLI dispatch and artifact handling, example consumer/device setup. Experimental Spikes and historical evidence/documentation were inventoried and selectively inspected, not reviewed line by line. |

Generated builds, `.git`, dependency checkouts, caches and virtual environments (`.build`, `.swiftpm`, `.local`, `.worktrees`, `Forge/.venv`, `Forge/h3-oracle/.venv`), binaries, model weights, raw tensors, and vendored experimental MLX code were excluded from source audit. Built modules/plugin executables were used only for verification, and a small amount of local H3 profile metadata was read to check supported shapes. Historical evidence was context, not fresh qualification. Dependencies were inspected through manifests/locks rather than audited internally; no online vulnerability advisory scan was performed.

No model inference, GPU qualification, checkpoint conversion, full model hashing, remote requests, or independent numerical oracle run was performed. iOS device behavior, macOS 14 launch, full floor builds, performance, and actual hardware resource peaks were not requalified. Coverage of experimental scripts and numerical graph implementations was selective; this report is not exhaustive.

## Verification performed

- `sh Scripts/check-cpu.sh --force-resolved-versions --skip-update` completed successfully: **527 individual tests passed; 52 native/external-fixture tests skipped; no failures**, across 18 test runs. Native/model opt-ins were cleared by the inspected wrapper. Existing passing tests do not cover the new reproductions above.
- `python3 -I Scripts/annotate-floor.py --check` passed: gated top-level declarations carry the Core AI floor attribute. This is a source check, not an old-OS launch test.
- Python AST parsing succeeded for all **88 tracked Python files**, without importing project modules or running their workloads.
- Temporary CPU-only Swift harnesses reproduced batch GPU misclassification, schema collisions, numeric-bound traps, Unicode escape acceptance/decoder disagreement, closed-backend queue ordering, the duration-conversion trap, and vocabulary-ID acceptance. The enum-label failure was reproduced using the built macro plugin. Harnesses used current source functions/types, with minimal fakes for unrelated boundaries; none changed repository sources.
- A harmless stdlib subprocess reproduction verified the interrupted controller's orphan-child behavior, with explicit process cleanup. Remaining H3 findings were established by checking current producers, consumers, guards, metadata, and cache-key construction rather than running GPU workloads.
- The pre-report Git status was clean. Final report/diff and commit scope are checked as the report is committed; this report is the only intended tracked change.
