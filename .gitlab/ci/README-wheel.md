# Internal wheel CI

This directory owns NVIDIA GitLab scheduling and qualification for the public
wheel build implementation under `packaging/`. The OSS sanitizer removes the
entire `.gitlab` tree.

`packaging/variants.toml` contains the public runtime and build contract.
`wheel-qualification.toml` adds internal runner, image, SDK, and target details
keyed by `variant_id`. The CI entrypoint requires exact one-to-one coverage so a
public support row cannot be added without a qualification target.

`scripts/wheel_ci.py` only dispatches commands. The implementations are split
under `scripts/wheel_ci_lib/` by responsibility: generated matrices, shared
environment handling, Python provisioning, build jobs, and integration jobs.

Run CI commands through:

```bash
sh .gitlab/ci/scripts/wheel_ci.sh generate-ci
sh .gitlab/ci/scripts/wheel_ci.sh generate-ci --check
sh .gitlab/ci/scripts/wheel_ci.sh ci-precheck
```

The generated `wheel-generated.yml` contains the concrete build and integration
matrices. Do not edit it directly. Payload build jobs consume every public matrix
row joined with the internal qualification file. Rows with `ci_test_enabled =
true` install the assembled wheel in a fresh target environment and exercise
installed engine build and inference. The integration gate requires evidence for
every selected variant and its qualified Python ABIs before recording the exact
filename and SHA-256 digest of each tested wheel.

## Public wheel release

Release and rehearsal pipelines validate and stage the six final wheels formed
by CPython 3.10, 3.11, and 3.12 crossed with x86_64 and aarch64. Validation accepts
only the exact filename-to-SHA mapping approved by `wheel_integration_gate` and
stores `artifacts/release/release-manifest.json` for 90 days.

The protected release flow is:

1. `wheel_release_vault_smoke` proves that GitLab can resolve the two
   Artifactory fields from Vault without using their values.
2. `wheel_release_validate` binds the six wheels to the integration-qualified
   digests and current source revision.
3. `wheel_release_stage` automatically uploads a protected `release/*` pipeline,
   or an explicitly selected protected rehearsal pipeline, to
   `hw-tensorrt-edge-llm-pypi-local` and hash-checks all six indexed files.
4. A release maintainer starts `wheel_release_rehearse` on a release branch.
   The job deploys through the protected `wheel-publication-rehearsal`
   environment without a separate approval, asks Kitmaker to test the six
   staged wheels with `upload: false`, and retains its redacted terminal status.
5. After the release authorization gates are complete, a release maintainer
   approves `wheel_release_publish` in the protected `wheel-publication`
   environment.
6. Kitmaker publishes to Devzone and PyPI; CI polls the release, downloads all
   six wheels from both pypi.nvidia.com and pypi.org, and requires their
   SHA-256 digests to equal the integration-qualified wheels.

### Protected non-publishing rehearsal

The complete path through Kitmaker validation can run before merge without
making the production publication job available:

1. Create a branch under `rehearsal/*` with a unique PEP 440 development
   version, for example `0.10.0.dev2`, that is not already present in the
   Artifactory repository. Rehearsal manifests accept development versions,
   while the production publication path rejects them.
2. Protect the `rehearsal/*` branch pattern with the same push restrictions as
   `release/*`, and extend the Vault JWT role's protected-ref claim to that
   pattern. The Vault data policy does not change.
3. From **Build > Pipelines > New pipeline**, select the rehearsal branch and
   set `EDGELLM_WHEEL_RELEASE_REHEARSAL=true`. The pipeline must have source
   `web`; a branch push, schedule, or merge-request pipeline cannot activate the
   secret-bearing rehearsal path.
4. Wait for the six-wheel build, integration gate, release validation, Vault
   smoke test, and Artifactory byte-equivalence verification to pass.
   `wheel_release_rehearse` then runs automatically through the
   no-approval `wheel-publication-rehearsal` environment.
5. Retain `kitmaker-rehearsal-evidence.json` with the pipeline. Do not merge the
   rehearsal-only version commit into the feature branch.

Every rehearsal rule requires the explicit variable, a protected ref, and the
`rehearsal/*` namespace. `wheel_release_publish` accepts only protected
`release/*` branches plus `EDGELLM_RELEASE_AUTH_READY=true`, so it is absent
from the rehearsal pipeline. Kitmaker receives `upload: false` for all six
entries and cannot publish them.

A retry skips Artifactory upload only when all six existing filenames and
SHA-256 values match. Partial or mismatched releases fail closed. A partial
public release must be escalated to the Kitmaker administrator rather than
blindly resubmitted.

### GitLab-native Vault configuration

The jobs use GitLab `id_tokens` and native `secrets:vault` resolution. The Vault
authentication mount is fixed in CI as:

```text
VAULT_AUTH_PATH=jwt/nvidia/gitlab-master
```

A GitLab maintainer must add these non-secret values as protected project or
group variables:

- `VAULT_SERVER_URL`
- `VAULT_AUTH_ROLE`
- `VAULT_NAMESPACE`
- `KITMAKER_PROJECT_ID`
- `KITMAKER_PIC`

The ID-token audience is `VAULT_SERVER_URL`. Do not configure `VAULT_ADDR` for
this integration.

The Vault role must be bound to this GitLab project, protected refs, the
`release/*` and `rehearsal/*` patterns, and the configured audience. Its policy
needs read access
only to these KV-v2 values:

| Engine mount | Secret path | Field | CI variable |
| --- | --- | --- | --- |
| `edge-llm/kv` | `release/pypi` | `artifactory_username` | `ARTIFACTORY_USERNAME` |
| `edge-llm/kv` | `release/pypi` | `artifactory_token` | `ARTIFACTORY_TOKEN` |
| `edge-llm/kv` | `release/kitmaker` | `api_token` | `KITMAKER_API_TOKEN` |

All three native secret declarations use `file: false`, so their CI variables
contain raw values rather than temporary file paths. Jobs must not enable shell
tracing, echo these variables, put credentials in URLs or arguments, or archive
authentication files.

### Artifactory staging

Twine uploads to the local PyPI API endpoint:

```text
https://artifactory.nvidia.com/artifactory/api/pypi/hw-tensorrt-edge-llm-pypi-local
```

Byte-equivalence verification uses its simple index:

```text
https://artifactory.nvidia.com/artifactory/api/pypi/hw-tensorrt-edge-llm-pypi-local/simple
```

Twine receives credentials through `TWINE_USERNAME` and `TWINE_PASSWORD`.
Authenticated index reads and wheel downloads use an in-memory HTTP
Authorization header. Kitmaker receives canonical wheel URLs beneath:

```text
https://artifactory.nvidia.com/artifactory/hw-tensorrt-edge-llm-pypi-local/tensorrt-edgellm/<version>/
```

The Artifactory administrator must confirm that the Vault identity can create
and read packages without delete/overwrite privileges and that Kitmaker can
read these canonical URLs.

### Kitmaker onboarding

A Kitmaker owner must create or reserve the `tensorrt-edgellm` project, assign a
team PIC and backup owner, enable public wheel releases, generate a
project-scoped token, and write that token directly to
`edge-llm/kv/release/kitmaker` as `api_token`. Record only the project ID, PIC
alias, owners, publication policy, and token expiration date.

Protect `release/*`, `rehearsal/*`, `wheel-publication`, and
`wheel-publication-rehearsal`. Restrict `wheel-publication` deployment approval
to the release-maintainers group. Allow the same release maintainers to deploy
to `wheel-publication-rehearsal`, but configure no approval rule there because
that job always sends `upload: false`. Before production enablement, run
`wheel_release_rehearse` once with the staged wheel set. A
completed job confirms Kitmaker fetched and tested the six Artifactory URLs
without publishing them. Every request entry explicitly contains
`upload: false`; the deprecated `publish_to` field is omitted. CI stores the
redacted response in
`artifacts/release/kitmaker-rehearsal-evidence.json` for 90 days. The rehearsal
does not require or set the production readiness flag. After it succeeds, set
the protected variable:

```text
EDGELLM_RELEASE_AUTH_READY=true
```

### Release approval and operation

Before approving public publication, finish the normal release branch,
CHANGELOG, OSS QA, GitHub release, and documentation work. The approval record
must show OSRB, MVSB, OSS disclosure, documentation readiness, and no blocker
bugs. Supply that internal GitLab URL as the manual pipeline variable
`RELEASE_APPROVAL_REFERENCE`.

The public job sends one Kitmaker payload entry per wheel with
`job_type: wheel-release-job` and `upload: true`. Kitmaker's `publish_to` field
is deprecated and ignored, so CI does not send it. The job polls Kitmaker for
up to two hours and the public indexes for up to 30 minutes. Preserve the staged
manifest, redacted Kitmaker UUID/status, and public-index byte-equivalence
evidence for at least 90 days.
