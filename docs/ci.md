# GitHub CI

The [CI workflow](../.github/workflows/ci.yml) runs on pull requests targeting `main`, pushes to `main`, merge-queue checks, and manual dispatch. Branch pushes with an open PR do not create a second run. New commits cancel the previous run for that PR, and a new `main` push cancels an earlier `main` run that is still going: PRs must be up to date with `main` to merge, so every `main` commit is a tree its PR already passed, and the newest one is the one worth checking.

A pull request that changes only documentation (`docs/`, top-level Markdown, `LICENSE`, `NOTICE`), spikes (`Spikes/`) or Forge (`Forge/`) skips the Swift jobs: the package doesn't build those files and its weight-free tests don't read them. The **Change scope** job decides this from the PR's merge commit, and **CI passed** still reports, so no required check is left pending. Every other change, and every `main` push, merge-queue check and manual run, runs every job.

| Check | Runner | What it verifies |
| --- | --- | --- |
| Change scope | `ubuntu-24.04` | Whether the Swift jobs run (above). |
| Workflow validation | `ubuntu-24.04` | All workflows pass checksum-pinned actionlint 1.7.12, including its available shell checks. |
| Package floor (macOS 14) | `xcode-27` (Apple silicon), Xcode 27.1 | `Scripts/check-floor.sh --macos`: `annotate-floor.py --check`, every product at macOS 14, and `Examples/FloorConsumer` weak-linking CoreAI, which it runs and stages for the launch job. |
| Package floor (iOS 17) | `xcode-27`, Xcode 27.1 | `Scripts/check-floor.sh --ios`: every library at iOS 17, and the floor consumer's iOS 17 device build weak-linking CoreAI. |
| Release contracts and consumers | `xcode-27`, Xcode 27.1 | The complete weight-free Release suite built at macOS 27, then `Examples/Foundations` and `Examples/LocalPackage` as separate consumer packages. |
| Launch at the macOS 14 floor | `macos-14` (Apple silicon) | The floor consumer staged by the macOS floor job launches on macOS 14, which has no CoreAI, and prints `AetherLanguage availability: unsupportedOS` ([deployment floor](deployment-floor.md)). |
| **CI passed** | `ubuntu-24.04` | Change scope and workflow validation succeeded, and every Swift job succeeded, or, for a PR the change scope cleared, every Swift job was skipped. A failed or cancelled job, or a skip anywhere else, cannot produce a passing gate. |

The three `xcode-27` jobs run in parallel, each from a clean checkout, so a run takes about as long as its slowest job rather than the sum.

Each `xcode-27` job checks the host and SDK before building (`Scripts/check-ci-host.sh`). `Scripts/check-cpu.sh` clears inherited `AETHER_*` opt-ins, so external fixtures and model qualification cannot activate accidentally. Every Swift command uses its committed `Package.resolved` with `--force-resolved-versions`. Dependency repository caches reduce network work; compiler outputs and model assets are not cached. Every Swift job that runs builds from a clean checkout and executes its checks.

The macOS floor job uploads the staged floor consumer as a `floor-consumer-<run>-<attempt>` artifact (a tarball, which keeps the executable bit) for 3 days; the launch job downloads it.

Each `xcode-27` job uploads its toolchain information and logs, with xUnit results (`tests-swift-testing.xml`, when testing completes) from the contracts job, as a `ci-diagnostics-<job>-<run>-<attempt>` artifact for 14 days, including on failure. A successful test step must produce a nonempty XML report. Pipeline failures remain failures when output is captured with `tee`. Jobs have explicit timeouts. The workflow uses a read-only token, does not persist Git credentials, and pins official actions to full commit hashes. Dependabot proposes monthly action updates; the actionlint version and checksum are updated together in the workflow.

## Reproduce a run

Use Apple silicon with macOS 27, Xcode 27.1 and Swift 6.4:

```sh
export DEVELOPER_DIR=/Applications/Xcode_27.1.app/Contents/Developer
sh Scripts/check-floor.sh /tmp/aether-floor-stage   # both halves; then Scripts/launch-floor-consumer.sh /tmp/aether-floor-stage on macOS 14
sh Scripts/check-cpu.sh --force-resolved-versions --xunit-output /tmp/aether-tests.xml
swift run --package-path Examples/Foundations --force-resolved-versions -c release
swift run --package-path Examples/LocalPackage --force-resolved-versions -c release
```

To run manually on GitHub after the workflow lands on `main`:

```sh
gh workflow run ci.yml --ref main
gh run list --workflow ci.yml
gh run view <run-id> --log-failed
```

## Merge enforcement

The active **Protect main** repository ruleset requires pull requests and a successful **CI passed** check from the GitHub Actions app. PR branches must be up to date with `main` before merging. That stable check name avoids changing branch rules when individual jobs change.

Force pushes and deletion of `main` are blocked, and review conversations must be resolved before merging. Human approvals are optional (zero required); administrators have no bypass exemption. These rules became available when the repository moved to the public [`bubstackdev/aether-kit`](https://github.com/bubstackdev/aether-kit) repository on 2026-10-27. Inspect the active configuration in [repository rules](https://github.com/bubstackdev/aether-kit/settings/rules).

## Hardware qualification

Routine PR CI needs no self-hosted runner or connection to a developer computer: GitHub's [`xcode-27` image](https://github.com/actions/runner-images/blob/main/images/macos/xcode-27-arm64-Readme.md) supplies the supported OS and toolchain. The hosted runner's small memory budget is sufficient for weight-free contracts; these checks do not claim model parity, GPU performance, or device qualification.

Real-weight Krea, FLUX.2, language, speech and H3 qualification remains opt-in on an equipped Mac with local assets and the shared GPU lease. See the model guides and [GPU lease documentation](gpu-lease.md). If automated later, use a separate trusted, manually dispatched hardware workflow on a dedicated runner; do not route arbitrary PR code to a developer's logged-in machine. The old `AETHER_MAC_RUNNER` repository variable is no longer used.
