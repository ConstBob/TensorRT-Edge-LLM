# Internal wheel CI

This directory owns NVIDIA GitLab scheduling and qualification for the public
wheel implementation under `packaging/`. The OSS sanitizer removes the complete
`.gitlab/` tree.

The canonical onboarding, identity, external-service, governance, and recovery
documentation is [`design/wheel_publishing/README.md`](../../design/wheel_publishing/README.md).
Do not duplicate those values here.

## Build and qualification

`packaging/variants.toml` defines the public runtime and build contract.
`wheel-qualification.toml` adds internal runner, image, SDK, and target details
by `variant_id`. CI requires one-to-one coverage so every public variant has a
qualification target.

`scripts/wheel_ci.py` dispatches implementations under `scripts/wheel_ci_lib/`.
Run it through the wrapper:

```bash
sh .gitlab/ci/scripts/wheel_ci.sh generate-ci
sh .gitlab/ci/scripts/wheel_ci.sh generate-ci --check
sh .gitlab/ci/scripts/wheel_ci.sh ci-precheck
```

The generated `wheel-generated.yml` contains the concrete build and integration
matrices; do not edit it directly. For each enabled x86_64 and aarch64 variant,
integration installs the assembled wheel in a fresh target environment, imports
it, builds an engine, and runs inference. `wheel_integration_gate` records the
exact qualified filename and SHA-256 digest for every selected Python ABI.

## Public wheel release

The release matrix is CPython 3.10, 3.11, and 3.12 crossed with x86_64 and
aarch64. Release jobs accept only the six filenames and digests approved by
`wheel_integration_gate`.

| Job | Behavior |
|---|---|
| `wheel_release_vault_smoke` | Proves GitLab OIDC and native Vault resolution of the two Artifactory fields without using or printing them. |
| `wheel_release_validate` | Binds the six wheels to the integration-qualified digests, source revision, and release-mode policy. |
| `wheel_release_stage` | Uploads to `hw-tensorrt-edge-llm-pypi-local` and downloads every indexed wheel for SHA-256 verification. |
| `wheel_release_rehearse` | Sends six Kitmaker entries with `upload: false` and retains redacted terminal evidence. |
| `wheel_release_publish` | Sends `upload: true`, polls Kitmaker, and verifies byte equivalence on `pypi.nvidia.com` and `pypi.org`. |

The current Kitmaker API derives production destinations from project `5413`.
CI intentionally omits the deprecated `publish_to` field. The standard
rehearsal is validation-only and does not publish to `test_pypi`.

### Activation rules

Secret-bearing release jobs run only for:

- a protected `release/*` branch; or
- a web-created pipeline on a protected `rehearsal/*` branch with
  `EDGELLM_WHEEL_RELEASE_REHEARSAL=true`.

Merge requests, ordinary pushes, schedules, and unprotected branches cannot
activate that path. Production additionally requires a protected release
branch, `EDGELLM_RELEASE_AUTH_READY=true`, a valid
`RELEASE_APPROVAL_REFERENCE`, a manual action, and approval through
`wheel-publication`.

### Non-publishing rehearsal

1. Create and protect a `rehearsal/*` branch with a unique PEP 440 development
   version.
2. Start a web pipeline with `EDGELLM_WHEEL_RELEASE_REHEARSAL=true`.
3. Wait for all six integration qualifications, validation, Vault smoke, and
   Artifactory byte-equivalence checks.
4. Confirm `wheel_release_rehearse` completes automatically through
   `wheel-publication-rehearsal` with `upload: false`.
5. Retain `kitmaker-rehearsal-evidence.json` and confirm
   `wheel_release_publish` is absent.
6. Do not merge the rehearsal-only version change.

### Production operation

1. Complete the authorization checklist in the canonical design.
2. Start the protected release pipeline with the approval-record URL in
   `RELEASE_APPROVAL_REFERENCE`.
3. Review the staged manifest and run the manual rehearsal.
4. Obtain the `wheel-publication` approval and run
   `wheel_release_publish`.
5. Preserve the staged manifest, redacted Kitmaker status, approval reference,
   and public-index verification evidence.

A retry skips Artifactory upload only when all six filenames and SHA-256 values
already match. Partial, unexpected, or byte-different releases fail closed.
Do not repair an immutable version or blindly resubmit a partially public
release from CI; follow the escalation procedure in the canonical design.

Release artifacts are retained for 90 days. Public-index verification can take
up to 30 minutes.
