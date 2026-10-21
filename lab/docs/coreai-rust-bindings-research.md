# Apple Core AI and direct Rust bindings

Research date: 2026-10-15  
Status: feasibility research and implementation proposal; no Rust binding has been implemented or benchmarked.

## Objective and conclusion

Build a Rust interface that calls Apple's native Core AI implementation directly. Use Apple's Python package as a reference for native entry points and behavior, without embedding Python, launching a Python worker, or adding a Swift wrapper to the application.

Apple does not need to publish a Rust package for a community binding to be technically possible. The unresolved issue is whether Rust can safely call the native interfaces needed for model loading, tensors, and inference. This is a technical feasibility conclusion, not a licensing or redistribution opinion.

The inspected Python runtime calls Core AI through Swift ABI symbols. It does not reveal a ready-made general C API to replace with Rust declarations. Direct Rust bindings are therefore plausible but unproven, and substantially more involved than translating Python source.

## Requirements established in the discussion

- The desired implementation calls the native backend from Rust.
- A Rust-to-Python bridge does not meet the requirement.
- A Rust-to-Swift wrapper was also rejected.
- Performance is the motivation: minimize host overhead, allocations, copies, and unnecessary synchronization.
- Existing Python and Swift APIs are research references rather than the proposed application runtime.
- Inference and graph authoring are separate scopes. Demonstrating inference does not establish a Rust replacement for the Python authoring and optimization toolchain.

The direct-binding design would still execute Apple's existing library code and may depend on Apple's Swift runtime libraries. Calling a framework implemented in Swift directly is different from adding a Swift wrapper of our own. Removing all Swift runtime dependencies would be a different and much larger requirement.

## Published language support

| Surface | Published support found | Meaning for Rust |
| --- | --- | --- |
| Core AI framework | Native Swift API for loading and running models | Provides public API semantics and SDK interfaces to study |
| Core AI Models | Python export recipes and primitives; Swift runtime utilities | Provides model assets, lifecycle examples, decode loops, and integration references |
| coreai-core | Python graph authoring and runtime APIs | Useful reference for the Python/native boundary |
| coreai-torch | Python conversion from PyTorch to Core AI | Conversion is separate from a runtime binding |
| Core AI Optimization | Python tooling for model compression | Optimization is separate from a runtime binding |
| Standalone C/C++, Rust, JavaScript, or other bindings | No official standalone package was found in the bounded search | This is not proof that no community project or lower-level interface exists |

Apple describes a workflow with Python for model preparation and Swift for application inference. Its Core AI framework supports control of memory, state, specialization, and execution. See [Apple's Core AI overview](https://developer.apple.com/core-ai/) and [Meet Core AI](https://developer.apple.com/videos/play/wwdc2026/324/).

The [Core AI Models repository](https://github.com/apple/coreai-models) publishes Swift helpers in its [swift directory](https://github.com/apple/coreai-models/tree/main/swift). Its requirements at the time of research list macOS/iOS 27.0+ and Xcode 27.0+ for running and application integration.

Apple's [coreai-core documentation](https://github.com/apple/coreai-torch/blob/main/docs/coreai-core/index.md) distinguishes **coreai.authoring**, for constructing models, from **coreai.runtime**, for executing existing .aimodel assets. No equivalent public Swift graph-authoring surface was established in this research.

## Local evidence

The local package was found in the existing uv cache. It was inspected statically; its Python extension was not imported or executed.

### Artifact identity

| Field | Value |
| --- | --- |
| Distribution | coreai-core |
| Inspected version | 1.0.0b2 |
| Interpreter/architecture | CPython 3.12, Darwin ARM64 |
| Package root | /Users/zakkeown/.cache/uv/archive-v0/LBs_td0hqU20OP4_/coreai |
| Inspected extension | runtime/_coreai_runtime_os.cpython-312-darwin.so |
| SHA-256 | da909ef79101acdbf53702abfdd52aad1b9b3bef38131666584653568fc5ed95 |
| REA evidence ID | ev_9871262a86bd2a92f6d946a5bed2a38af2c0f42c74911ae69dbb6e8953b0e71e |
| SDK | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk |

The full retained REA response is saved in [coreai-rust-bindings-evidence.json](coreai-rust-bindings-evidence.json). The companion [local inspection transcript](coreai-rust-bindings-local-inspection.txt) records package metadata, loader source, dependency/symbol output, and SDK interfaces.

### Python module selection

The package's runtime/__init__.py selects between two compiled modules:

- _coreai_runtime_os.cpython-312-darwin.so
- _coreai_runtime.cpython-312-darwin.so

The loader checks the operating system, whether installation came from a wheel, and the USE_OS_COREAI / USE_LOCAL_COREAI environment variables. On eligible macOS wheel installations it normally selects the OS-backed extension. The loader aliases modules to avoid loading two copies and corrupting Swift global state, according to its source comment.

This is an observation about the inspected package. The alternate extension was inventoried, but its native implementation was not analyzed.

### Native dependencies and exports

The OS-backed extension depends on:

- /System/Library/Frameworks/CoreAI.framework/Versions/A/CoreAI
- MetalPerformanceShadersGraph, CoreFoundation, CoreVideo, Foundation, libobjc, and Swift runtime/concurrency libraries

Its observed Core AI imports include Swift-mangled symbols for CoreAIAsset, CoreAICommon, CoreAIRuntime, and CoreAIDelegates. Examples include AIModelAsset initialization, InferenceFunction operations, NDArray-related values, and an asynchronous inference run entry point.

The recorded export list contains the Python module initializer **_PyInit__coreai_runtime_os**. It did not establish a general-purpose C callable inference API in this extension.

**Inference:** using the Python package as a reference can identify operations and native call targets, but replacing its Python-facing layer requires implementing the relevant native ABI and lifetime rules. Import names alone do not recover complete signatures or prove that a direct Rust call works.

### SDK inspection

The installed CoreAI.framework contains Swift module interfaces and a CoreAI.tbd linker stub. The stub reexports CoreAIDelegates and carries further Core AI subframework records and Swift symbol declarations.

A bounded textual inspection of quoted underscore-prefixed entries in the stub found Swift symbols rather than a general C API. That search is not an exhaustive inventory of every lower-level framework on the system. Subframework linkage restrictions and public versus internal interfaces must be assessed before choosing callable targets.

### Limits of the evidence

- The inspected package is 1.0.0b2. The [PyPI page](https://pypi.org/project/coreai-core/) listed 1.0.0b3, released September 24, 2026. The latter wheel was not inspected.
- An attempted download of the newer wheel was unsuccessful. Shell networking failed name resolution, and the web tool could not retrieve the wheel.
- REA warns that stripped or toolchain-hidden symbols may be absent and that normalized imports/exports may be incomplete.
- Static dependencies and imports establish linkage, not runtime behavior.
- No Rust program, FFI declaration, tensor operation, or inference call was tested.
- Authoring/compiler libraries were only inventoried at the filename level. Their callable interfaces remain unexamined.
- The native session was closed after inspection.

## Rust and the Swift ABI

Rust upstream exposes experimental work for a Swift calling convention under **abi_swift**, with **extern "Swift"** appearing in compiler sources. This is not a stable, complete Swift interoperability guarantee.

The [tracking issue](https://github.com/rust-lang/rust/issues/156481) discusses unresolved type-layout and argument-classification questions. The [unstable feature declaration](https://github.com/rust-lang/rust/blob/main/compiler/rustc_feature/src/unstable.rs) and [ABI documentation](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_abi/extern_abi/enum.ExternAbi.html) provide additional compiler context. Issue labels and source snapshots are not perfectly synchronized; the required toolchain must be tested directly.

A calling-convention feature by itself does not establish support for all of Core AI's types. A direct binding must account for:

- Argument and return-value lowering, including indirect results and special registers.
- Value size, alignment, stride, resilience, and metadata.
- Ownership transfers, reference counting, destruction, and borrowed lifetimes.
- Strings, URLs, dictionaries, enums, and error representations.
- Async contexts, continuations, cancellation, and completion lifetimes.
- Buffer layout, aliasing, alignment, and the lifetime of work still running on the GPU.
- OS/SDK compatibility and which symbols are supported for application use.

Swift's [calling convention documentation](https://github.com/swiftlang/swift/blob/main/docs/ABI/CallingConvention.rst) is a reference for this work. Merely applying an extern "C" declaration to a Swift-mangled symbol would not make an incompatible ABI safe.

## Approaches considered

| Approach | Technical assessment | Fit with the user's requirement |
| --- | --- | --- |
| Rust embeds Python and calls coreai.runtime | Reuses the existing API but retains interpreter and packaging dependencies | Rejected |
| Rust launches a Python service or worker | Adds process/IPC boundaries and data-transfer concerns | Rejected |
| Rust calls a small Swift library through a C ABI | A conventional native integration approach; not benchmarked here | Rejected because the user requested direct bindings |
| Rust calls a reusable existing C API | Potentially straightforward if suitable functions and supported contracts are found | Candidate only; no suitable interface was established in the inspected layer |
| Rust calls Core AI's Swift ABI directly | Fits the requested architecture, with significant ABI and lifecycle work | Primary research proposal; feasibility remains to be demonstrated |
| Rust calls a lower-level native C/C++ interface | Could be viable if inspection establishes suitable symbols, signatures, and supported access | Conditional alternative; no such API has been demonstrated |
| Rust reimplements the model runtime over Metal or other APIs | A separate backend implementation rather than Core AI bindings | Outside the present proposal |

The earlier suggestion of a Swift bridge was superseded by the user's direct-binding constraint. It should not return as an assumed implementation choice.

## Proposed direct binding architecture

Keep an unsafe low-level binding layer separate from a safe Rust API.

The low-level layer would declare only verified entry points and encapsulate native layouts, metadata, ownership, and invocation rules. It should initially target one known architecture and OS/SDK combination. Unverified layouts must not be exposed as safe Rust types.

The safe layer would provide model loading, named function lookup, typed tensor descriptors, buffer ownership, inference submission/completion, and errors. Its API must keep model and buffer resources alive until native execution has actually completed.

Use Apple's Python and Swift implementations as reference behavior for lifecycle and outputs. Do not assume that every Python wrapper operation has a directly reusable C ABI equivalent.

This architecture is a proposal. Crate boundaries, exact callable symbols, and supported tensor storage have not been selected.

## Proposed feasibility sequence

### 1. Establish a callable native entry point

Pin the package, SDK, OS, architecture, and Rust compiler versions. Inspect the current package as well as the cached older package. Map Python-facing operations to native symbols, recover the required ABI contract, and distinguish public API from internal implementation details.

Validate a minimal direct native call with the chosen Rust compiler. If the compiler lacks required lowering, record the exact missing capability before expanding the binding. LLVM or assembly workarounds, if needed, would be new engineering proposals rather than a proven path.

### 2. Load an existing model

Load a very small known .aimodel asset and enumerate or select its inference function. Handle both success and expected failure. Demonstrate correct cleanup through repeated load/drop cycles.

The application executable must not invoke Python or a custom Swift wrapper. A model prepared beforehand with existing tools is sufficient to prove runtime access; it does not prove Rust graph authoring.

### 3. Execute a deterministic tensor operation

Run a tiny model, such as a fixed-shape elementwise operation, and compare its output with a trusted result. Verify dtype, shape, strides, alignment, and buffer ownership.

Demonstrate that the Rust application can submit and receive tensors correctly before adding a language-model decode loop.

### 4. Establish buffer and completion safety

Investigate shared-memory or Metal-buffer paths, and measure whether hidden copies occur. Retain native objects and Rust allocations until completion. Exercise errors, repeated execution, and cancellation where supported.

A zero-copy path is an acceptance target, not an observed feature of the proposed Rust binding.

### 5. Measure host overhead and throughput

Benchmark identical model assets, compute choices, tensor shapes, precision, and warmup conditions. Separate cold loading/specialization, host submission latency, device execution, synchronization, and end-to-end throughput.

Compare against an existing native Swift caller and, if useful, the Python caller. The Swift baseline is a measurement reference, not part of the Rust application.

### 6. Expand only after the proof works

Add stateful execution, efficient command submission, persistent KV/cache buffers, and a Rust-controlled decode loop. Treat graph construction, model conversion, and optimization as a separate subsequent proposal.

## Performance expectations

A Rust binding can reduce interpreter involvement and give explicit control of allocations, scheduling, and buffer lifetimes. It does not inherently accelerate Apple's existing GPU or Neural Engine kernels.

The important costs are data movement, temporary allocations, submission frequency, synchronization, and workload granularity. A native FFI call by itself does not establish a material performance penalty; no bridge-versus-direct benchmark was performed. The direct-binding constraint remains the chosen design requirement regardless.

No numeric speedup is justified by the current research.

## Open decisions and unanswered questions

1. Can the chosen Rust toolchain express every ABI feature needed for a minimal Core AI inference path?
2. Which model-loading and inference functions are callable without a custom Swift wrapper?
3. Can opaque native values be managed safely without constructing complex Swift standard-library values in Rust?
4. Is there a simpler supported native entry point below the observed Swift surface?
5. How should asynchronous inference completion and errors map into Rust?
6. Which tensor storage forms permit shared buffers without extra copies?
7. Do 1.0.0b3 or later packages change the interface observed in 1.0.0b2?
8. Which APIs remain compatible across OS and SDK updates?
9. Is a future Rust authoring/compiler surface needed, or is the intended product solely a native inference runtime?
10. What licensing and redistribution terms apply to any Apple code or bundled libraries that a distributable crate would use? Package metadata alone does not settle this.

## Source register

Primary references consulted or identified during the conversation:

- [Apple Core AI overview](https://developer.apple.com/core-ai/) — Swift runtime, preparation workflow, memory and execution features.
- [Core AI framework documentation](https://developer.apple.com/documentation/CoreAI?language=swift) — public framework reference.
- [Meet Core AI, WWDC26](https://developer.apple.com/videos/play/wwdc2026/324/) — Python preparation and Swift integration workflow.
- [Integrate on-device AI models into your app using Core AI, WWDC26](https://developer.apple.com/videos/play/wwdc2026/326/) — app integration and compilation.
- [Core AI Models](https://github.com/apple/coreai-models) — Python recipes and Swift runtime utilities.
- [Core AI Models README](https://github.com/apple/coreai-models/blob/main/README.md) — repository organization and runtime requirements.
- [Core AI Models Swift package](https://github.com/apple/coreai-models/tree/main/swift) — native runtime reference.
- [Core AI PyTorch Extensions](https://github.com/apple/coreai-torch) — conversion tooling.
- [coreai-core API overview](https://github.com/apple/coreai-torch/blob/main/docs/coreai-core/index.md) — Python authoring/runtime split.
- [Core AI Optimization](https://github.com/apple/coreai-optimization) — model compression tooling.
- [coreai-core on PyPI](https://pypi.org/project/coreai-core/) — published distribution versions and wheel metadata.
- [Rust Swift ABI tracking issue](https://github.com/rust-lang/rust/issues/156481) — experimental interoperability status and unresolved questions.
- [Rust unstable feature declarations](https://github.com/rust-lang/rust/blob/main/compiler/rustc_feature/src/unstable.rs) — abi_swift feature gate.
- [Rust native ABI documentation](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_abi/extern_abi/enum.ExternAbi.html) — Swift calling convention entry.
- [Swift ABI calling convention](https://github.com/swiftlang/swift/blob/main/docs/ABI/CallingConvention.rst) — native ABI reference.
- [Swift C-compatible functions and enums proposal](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0495-cdecl.md) — context for the excluded C ABI wrapper approach.
- [PyO3 guide](https://pyo3.rs/) — context for the excluded Python embedding approach.

Searches for Rust bindings also surfaced projects for Apple's **Foundation Models** framework. Those are a different API and were not counted as Core AI bindings. Unrelated products named CoreAI were excluded.

Some guessed documentation URLs and a guessed apple/coreai repository URL could not be retrieved. Those failed fetches were not used as evidence that the corresponding API or repository cannot exist.

