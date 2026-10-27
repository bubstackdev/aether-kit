# GitHub CI

The [CI workflow](../.github/workflows/ci.yml) runs on pull requests targeting `main`, pushes to `main`, merge-queue checks, and manual dispatch. Branch pushes with an open PR do not create a second run. New commits cancel the previous run for that PR; each `main` push gets its own run. Documentation changes run the same checks, so a required check cannot be left pending by path filters.

| Check | Runner | What it verifies |
| --- | --- | --- |
| Workflow validation | `ubuntu-24.04` | All workflows pass checksum-pinned actionlint 1.7.12, including its available shell checks. |
| Release contracts and consumers | `xcode-27` (Apple silicon), Xcode 27.1 | The complete weight-free Release suite, then `Examples/Foundations` and `Examples/LocalPackage` as separate consumer packages. |
| **CI passed** | `ubuntu-24.04` | Both preceding jobs succeeded. A failed, cancelled or skipped dependency cannot produce a passing gate. |

The SDK job checks the host and SDK before building. `Scripts/check-cpu.sh` clears inherited `AETHER_*` opt-ins, so external fixtures and model qualification cannot activate accidentally. Every Swift command uses its committed `Package.resolved` with `--force-resolved-versions`. Dependency repository caches reduce network work; compiler outputs and model assets are not cached. Every run still builds and executes its checks.

Toolchain information, the test log, xUnit results (`tests-swift-testing.xml`, when testing completes), and consumer logs are uploaded as a `ci-diagnostics-<run>-<attempt>` artifact for 14 days, including on failure. A successful test step must produce a nonempty XML report. Pipeline failures remain failures when output is captured with `tee`. Jobs have explicit timeouts. The workflow uses a read-only token, does not persist Git credentials, and pins official actions to full commit hashes. Dependabot proposes monthly action updates; the actionlint version and checksum are updated together in the workflow.

## Reproduce a run

Use Apple silicon with macOS 27, Xcode 27.1 and Swift 6.4:

```sh
export DEVELOPER_DIR=/Applications/Xcode_27.1.app/Contents/Developer
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

Use **CI passed** as the single required status check for `main`, with the branch required to be up to date (or a merge queue). That stable name avoids changing branch rules when individual jobs change. Keep pull requests required, and block force pushes and branch deletion.

As of setup on 2026-10-27, GitHub returns HTTP 403 for both rulesets and classic branch protection on this private repository: “Upgrade to GitHub Pro or make this repository public to enable this feature.” CI runs and reports results, but the current plan cannot enforce the merge gate. Once the repository owner enables a plan supporting private-repository branch protection, configure the rule above. Check results must be reviewed before merging in the meantime.

## Hardware qualification

Routine PR CI needs no self-hosted runner or connection to a developer computer: GitHub's [`xcode-27` image](https://github.com/actions/runner-images/blob/main/images/macos/xcode-27-arm64-Readme.md) supplies the supported OS and toolchain. The hosted runner's small memory budget is sufficient for weight-free contracts; these checks do not claim model parity, GPU performance, or device qualification.

Real-weight Krea, FLUX.2, language, speech and H3 qualification remains opt-in on an equipped Mac with local assets and the shared GPU lease. See the model guides and [GPU lease documentation](gpu-lease.md). If automated later, use a separate trusted, manually dispatched hardware workflow on a dedicated runner; do not route arbitrary PR code to a developer's logged-in machine. The old `AETHER_MAC_RUNNER` repository variable is no longer used.
