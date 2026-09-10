<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NVIDIA PyPI wheel publication

> **NVIDIA internal only.** This is the canonical onboarding and operations
> document for TensorRT Edge-LLM wheel publication. It contains internal service
> names, identities, repository coordinates, and security controls. The
> repository's `DO_NOT_RELEASE` policy removes `design/` from the OSS release
> tree. Do not copy this document into public documentation or release notes.

| Field | Value |
|---|---|
| Status | Implemented and validated with a non-publishing rehearsal |
| Last validated | 2026-08-27 |
| Public package | `tensorrt-edgellm` |
| Kitmaker project | `tensorrt-edgellm` (`5413`) |

## 1. Purpose and release shape

The release flow takes six integration-qualified wheels from a protected GitLab
pipeline, stages them in NVIDIA cloud Artifactory, asks Kitmaker to publish
them, and verifies that the public indexes serve the exact tested bytes.

| CPU architecture | Python ABIs |
|---|---|
| `x86_64` | CPython 3.10, 3.11, and 3.12 |
| `aarch64` | CPython 3.10, 3.11, and 3.12 |

Artifactory is the durable staging source consumed by Kitmaker.
`nv-shared-pypi-local` is an internal-only path and is not part of this OSS
distribution flow. `.gitlab/ci/README-wheel.md` documents CI job operation and
links here for onboarding, ownership, and release policy.

## 2. Current ownership and configuration

### 2.1 Identity inventory

| Purpose | Current value |
|---|---|
| Release service account | `svc-edgellm-rel@nvidia.com` |
| Artifactory username | `svc-edgellm-rel` |
| Service-account primary owner | `jcalafato@nvidia.com` |
| Service-account contact group | `tensorrt-edge-llm-vault-admins` |
| Vault namespace-admin DL | `tensorrt-edge-llm-vault-admins` |
| Kitmaker owner and PIC | `jcalafato@nvidia.com` |
| GitLab protected-environment deployers | `jcalafato`, `mbreughe`, `luxiaoz`, `zhijial` |

The admin DL provides team continuity for the service account. Kitmaker
ownership is assigned to the owner and PIC listed above.

### 2.2 Credential inventory

Never record the credential values in GitLab, this document, job logs, shell
history, or pipeline artifacts.

| Credential | Vault location | Expiration |
|---|---|---|
| Artifactory username | `edge-llm/kv/release/pypi`, field `artifactory_username` | Not applicable |
| Artifactory identity token | `edge-llm/kv/release/pypi`, field `artifactory_token` | April 28, 2027 |
| Kitmaker project API token | `edge-llm/kv/release/kitmaker`, field `api_token` | August 23, 2028 |

### 2.3 Vault

Staging and production NVault namespaces were provisioned during onboarding.
Release CI uses production only; staging coordinates are intentionally not part
of the runtime configuration recorded here.

| Setting | Production value |
|---|---|
| `VAULT_SERVER_URL` | `https://prod.vault.nvidia.com` |
| `VAULT_NAMESPACE` | `hw-tensorrt-edge-llm` |
| `VAULT_AUTH_PATH` | `jwt/nvidia/gitlab-master` |
| `VAULT_AUTH_ROLE` | `edge-llm-release-gitlab` |
| ID-token audience | `$VAULT_SERVER_URL` |
| KV-v2 engine mount | `edge-llm/kv` |

Vault namespace administration is assigned to the dedicated admin DL rather
than a broad contributor group. The role is bound to the Edge-LLM GitLab
project, the configured audience,
protected refs, and the `release/*` and `rehearsal/*` branch patterns. Its policy
grants read access only to the fields in section 2.2. GitLab native Vault
resolution uses `VAULT_SERVER_URL`, not `VAULT_ADDR`, and all secret
declarations use `file: false`.

### 2.4 Artifactory

| Setting | Value |
|---|---|
| Repository type | AWS local PyPI repository in cloud Artifactory |
| Repository | `hw-tensorrt-edge-llm-pypi-local` |
| Permission target | `hw-tensorrt-edge-llm-pypi-permission` |
| Upload identity | `svc-edgellm-rel` (`svc-edgellm-rel@nvidia.com` in ITSS) |
| Kitmaker reader group | `sw-kitmaker-artifactory` |
| Twine upload | `https://artifactory.nvidia.com/artifactory/api/pypi/hw-tensorrt-edge-llm-pypi-local` |
| Pip simple index | `https://artifactory.nvidia.com/artifactory/api/pypi/hw-tensorrt-edge-llm-pypi-local/simple` |
| Kitmaker wheel prefix | `https://artifactory.nvidia.com/artifactory/hw-tensorrt-edge-llm-pypi-local/tensorrt-edgellm/<version>/` |

The service account was granted read and deploy/write access on the permission
target without delete or overwrite access. General cloud Artifactory access
alone does not grant upload rights. `sw-kitmaker-artifactory` has **Read** and
**Annotate**, allowing Kitmaker to fetch the canonical wheel URLs.

### 2.5 Kitmaker

Project `5413` is configured for package `tensorrt-edgellm`, with
`jcalafato@nvidia.com` as owner and PIC. Rehearsal/test and production
destinations are enabled.

The standard CI rehearsal sends `upload: false`. Kitmaker authenticates, reads,
and validates all six Artifactory URLs without publishing to the enabled
`test_pypi` destination or any public registry.

Production is configured project-side for `both_devzone_pypi`, which publishes
to both `pypi.nvidia.com` and `pypi.org`. CI sends `upload: true`; it does not
send the deprecated `publish_to` request field.

Kitmaker defines `upload_both` and `wheel-stub` as large-wheel mirroring
strategies. They are not Artifactory upload options or permanent repository
settings.

### 2.6 GitLab controls

Protected branches:

| Pattern | Push | Merge | Force push |
|---|---|---|---|
| `release/*` | No direct push | Developers and Maintainers | Disabled |
| `rehearsal/*` | No direct push | Developers and Maintainers | Disabled |

Protected environments:

| Environment | Allowed deployers | Approval |
|---|---|---|
| `wheel-publication` | `jcalafato`, `mbreughe`, `luxiaoz`, `zhijial` | One Maintainer approval |
| `wheel-publication-rehearsal` | Same four users | None; requests use `upload: false` |

These environments are configured under
**Settings > CI/CD > Protected environments** with the individual deployers
listed above.

Protected non-secret variables:

| Variable | Value |
|---|---|
| `VAULT_SERVER_URL` | `https://prod.vault.nvidia.com` |
| `VAULT_NAMESPACE` | `hw-tensorrt-edge-llm` |
| `VAULT_AUTH_ROLE` | `edge-llm-release-gitlab` |
| `KITMAKER_PROJECT_ID` | `5413` |
| `KITMAKER_PIC` | `jcalafato@nvidia.com` |

`VAULT_AUTH_PATH` is fixed in `.gitlab/ci/wheel-release-jobs.yml`.
`EDGELLM_RELEASE_AUTH_READY=true` is a protected production feature gate and
must remain unset outside an authorized release window.
`RELEASE_APPROVAL_REFERENCE` is supplied when starting the protected release
pipeline and must be an internal GitLab URL to the completed approval record.

## 3. Provisioned setup and dependency order

1. **Identity:** the team-owned service account, primary owner, contact group,
   and dedicated Vault admin DL were created in ITSS.
2. **Vault:** staging and production namespaces were provisioned,
   `jwt/nvidia/gitlab-master` was enabled, and the project-scoped JWT role was
   granted read-only access to the two release secret paths.
3. **Artifactory:** the AWS local PyPI repository and
   `hw-tensorrt-edge-llm-pypi-permission` were created.
4. **Repository access:** the service account received read and deploy/write
   access, and `sw-kitmaker-artifactory` received Read and Annotate.
5. **Kitmaker:** project `5413` was created with its owner/PIC, rehearsal and
   `both_devzone_pypi` production support were enabled, and a project-scoped
   token was generated.
6. **Secrets:** the Artifactory identity and token were written to
   `release/pypi`, and the Kitmaker token was written to `release/kitmaker`.
7. **GitLab OIDC:** the protected variables, native
   `id_tokens`/`secrets:vault` integration, branch patterns, and protected
   environments in section 2.6 were configured.
8. **Proof:** Twine upload, authenticated `/simple` reads, Kitmaker access
   through `sw-kitmaker-artifactory`, and the six-wheel `upload: false`
   rehearsal were validated.

The dependency chain is:

```text
ITSS ownership and admin DL
  -> service account
  -> Artifactory repository access and identity token
  -> Vault secret storage
  -> GitLab OIDC role and native secret resolution
  -> Kitmaker project, token, and Artifactory read access
  -> protected rehearsal
  -> approved production publication
```

## 4. CI release flow

Secret-bearing jobs run only for a protected `release/*` branch, or a web-created
pipeline on a protected `rehearsal/*` branch with
`EDGELLM_WHEEL_RELEASE_REHEARSAL=true`. Merge requests, ordinary pushes,
schedules, and unprotected branches cannot expose the secret-bearing release
path.

| Job | Purpose | Retained evidence |
|---|---|---|
| `wheel_integration_gate` | Records the exact six installed and inference-tested filenames and SHA-256 digests | Integration gate |
| `wheel_release_vault_smoke` | Proves GitLab OIDC and native resolution of both Artifactory fields without using or printing them | Job result |
| `wheel_release_validate` | Enforces the six-wheel matrix, release identity, source revision, qualified digests, and OSS text audit | `release-manifest.json` |
| `wheel_release_stage` | Uploads to Artifactory and downloads every indexed wheel to prove byte equivalence | `staged-release-manifest.json`, `artifactory-verification.json` |
| `wheel_release_rehearse` | Submits six Kitmaker entries with `upload: false` and polls their terminal state | `kitmaker-rehearsal-evidence.json` |
| `wheel_release_publish` | Submits `upload: true`, polls Kitmaker and both indexes, and verifies qualified digests | `publication-evidence.json` |

Final-wheel assembly already validates structure, tags, RECORD hashes, runtime
manifests, licenses, binary dependencies, glibc baselines, and size. Integration
installs each wheel in a fresh target environment, imports it, builds an engine,
and runs inference. Release validation preserves that evidence by requiring the
same filenames and SHA-256 digests at Artifactory and both public indexes.

The production job is available only when all of these conditions hold:

- protected `release/*` branch;
- protected `EDGELLM_RELEASE_AUTH_READY=true`;
- valid `RELEASE_APPROVAL_REFERENCE`;
- manual action by an allowed deployer; and
- approval through `wheel-publication`.

If both indexes already contain the exact release, the job records
`already_published`. If only one contains it, or any digest differs, the job
fails closed and requires escalation rather than resubmission.

## 5. Operating procedures

### 5.1 Non-publishing rehearsal

1. Create and protect `rehearsal/<name>` from the candidate implementation.
2. Use a unique PEP 440 development version that is absent from Artifactory.
3. Start a web pipeline with `EDGELLM_WHEEL_RELEASE_REHEARSAL=true`.
4. Confirm all six builds, integration tests, Vault smoke, validation, staging,
   and Artifactory byte-equivalence checks pass.
5. Confirm `wheel_release_rehearse` completes with `upload: false`, retains
   redacted evidence, and `wheel_release_publish` is absent.
6. Do not merge the rehearsal-only version change.

A non-publishing rehearsal validated this flow with `0.10.0.dev2`: all jobs
passed, Kitmaker read and validated all six wheels, and no registry was
modified.

### 5.2 Production authorization

Before publication, complete and record:

- protected `release/<version>` stabilization;
- `CHANGELOG.md`, release notes, and GitHub release staging;
- OSS QA;
- OSRB and MVSB approval;
- OSS disclosure review for documentation and release notes;
- **Documentation ready** and **No blocker bugs** sign-off;
- public-index project, file-size allowance, and large-wheel strategy; and
- final release-owner authorization.

The approval record must be an internal GitLab URL.

### 5.3 Production publication

1. Set protected `EDGELLM_RELEASE_AUTH_READY=true`.
2. Start the protected release pipeline with
   `RELEASE_APPROVAL_REFERENCE=<approval URL>`.
3. Wait for qualification, validation, Vault smoke, and Artifactory staging.
4. Run `wheel_release_rehearse` and inspect its redacted evidence.
5. Verify the version, revision, six filenames, and SHA-256 values.
6. Obtain the protected-environment approval and run
   `wheel_release_publish`.
7. Wait for Kitmaker and byte-equivalence verification on
   `pypi.nvidia.com` and `pypi.org`.
8. Retain the pipeline, approval record, and release evidence, then publish the
   matching public release notes and installation documentation.
9. Unset `EDGELLM_RELEASE_AUTH_READY` after the release window.

Public-index verification may take up to 30 minutes. SHA-256 equality proves the
published files are the wheels previously installed and used for engine build
and inference, so a weaker dependency-free reinstall adds no release evidence.

## 6. Security, governance, and recovery

- Secrets remain in Vault and are selected per job with GitLab OIDC.
- Jobs must not enable shell tracing, print secrets, place them in URLs or
  arguments, or archive authentication files.
- API evidence is redacted before persistence, and transfer operations require
  HTTPS, expected hosts, bounded responses, and exact digests.
- Wheel text is scanned for internal hosts and scratch paths before staging.
- This design and `.gitlab/` are removed by the OSS sanitizer. Run
  `python3 scripts/check_oss_release_sanitizer.py` before OSS publication.
- Public metadata, licenses, dependencies, documentation, and release notes
  remain subject to OSRB, MVSB, and OSS disclosure review.
- The flow collects no end-user data. Evidence contains release metadata,
  internal pipeline URLs, digests, and redacted service status only.
- GitLab retains release artifacts for 90 days. Preserve the pipeline and
  approval URLs for the longer applicable legal, security, or audit period.

| Failure | Required response |
|---|---|
| Vault OIDC or resolution fails | Verify production URL, namespace, audience, role claims, policy, and field spelling. Never substitute a GitLab-stored secret. |
| Artifactory contains a partial or mismatched version | Do not overwrite or delete from CI. Repair through the repository owner and restore the exact six-file state. |
| Kitmaker rehearsal fails | Keep `upload: false`, preserve redacted evidence, and escalate to the project owner. |
| Kitmaker publication partially updates the indexes | Do not resubmit blindly; escalate with the release UUID. |
| Public digest differs | Stop the release and treat it as an integrity incident. |

For credential rotation:

1. create a replacement with the same or narrower scope;
2. update only the relevant Vault field;
3. update the expiration in section 2.2;
4. run the Vault smoke and protected non-publishing rehearsal; and
5. revoke the old credential only after the rehearsal succeeds.

When ownership changes, update ITSS or Kitmaker first, then this inventory,
GitLab variables/environments as applicable, and the internal release checklist.
Run wheel CI commands through:

```bash
sh .gitlab/ci/scripts/wheel_ci.sh <command>
```

Calling `wheel_ci.py` directly does not establish the `packaging/` Python path.
