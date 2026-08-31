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

# NVIDIA PyPI Wheel Publication

> **NVIDIA internal only.** This document contains internal service names,
> repository coordinates, security controls, and operational procedures. The
> repository's `DO_NOT_RELEASE` policy removes the entire `design/` directory
> from the OSS release tree. Do not copy this document into public release
> notes, public documentation, or the GitHub release repository.

| Field | Value |
|---|---|
| Status | Implemented and validated with a non-publishing rehearsal |
| Tracking issue | GitLab issue 700 |
| Last validated | 2026-08-27 |
| Validation pipeline | [64935968](https://gitlab-master.nvidia.com/TensorRT/tensorrt-edge-llm/tensorrt-edge-llm/-/pipelines/64935968) |
| Public package name | `tensorrt-edgellm` |
| Kitmaker project ID | `5413` |
| Kitmaker PIC | `jcalafato@nvidia.com` |

## 1. Purpose and scope

The wheel release flow takes the six qualified TensorRT Edge-LLM wheels from a
protected GitLab release pipeline, stages them in NVIDIA Artifactory, asks
Kitmaker to publish them, and proves that both public indexes serve the exact
files that passed installation, engine-build, and inference qualification.

The six-wheel release matrix is:

| CPU architecture | Python ABIs |
|---|---|
| `x86_64` | CPython 3.10, 3.11, and 3.12 |
| `aarch64` | CPython 3.10, 3.11, and 3.12 |

This design covers service and account onboarding; GitLab, Vault, Artifactory,
and Kitmaker configuration; credential ownership; rehearsal; production
publication; verification; recovery; and release, legal, and privacy gates.

It does not replace the normal Edge-LLM release checklist. Branch stabilization,
CHANGELOG updates, OSS QA, GitHub release preparation, disclosure review, and
release sign-off remain prerequisites for public publication.

## 2. Architecture

```text
payload builds and target qualification
                |
                v
assemble 6 complete wheels ----> wheel_integration_gate
                |                         |
                +------------+------------+
                             v
                 wheel_release_validate
                             |
Vault OIDC smoke ------------+
                             v
                  wheel_release_stage
                   |              |
                   |              +--> Artifactory byte-equivalence check
                   v
          wheel_release_rehearse
              upload: false
                   |
        release authorization gates
                   |
                   v
           wheel_release_publish
               upload: true
                   |
                   v
               Kitmaker
                   |
          +--------+--------+
          v                 v
  pypi.nvidia.com        pypi.org
          |                 |
          +--------+--------+
                   v
      qualified-wheel byte equivalence
```

Artifactory is the durable staging source consumed by Kitmaker. The internal
`nv-shared-pypi-local` repository is not part of this OSS distribution flow.

## 3. External services and ownership

### 3.1 GitLab

The implementation lives in the internal TensorRT Edge-LLM GitLab project.
The OSS sanitizer removes `.gitlab/`, this `design/` directory, and the other
paths in `DO_NOT_RELEASE` before the public GitHub tree is staged.

The following branch protections are configured:

| Pattern | Push | Merge | Force push |
|---|---|---|---|
| `release/*` | No direct push | Developers and Maintainers | Disabled |
| `rehearsal/*` | No direct push | Developers and Maintainers | Disabled |

The following protected environments control Kitmaker use:

| Environment | Allowed deployers | Approval |
|---|---|---|
| `wheel-publication` | Joshua Calafato, Maximilien Breughe, Luxiao Zheng, Zhijia Liu | One Maintainer approval |
| `wheel-publication-rehearsal` | Same four maintainers | No approval; every request uses `upload: false` |

The production environment should eventually use a release-maintainers group
instead of individual users when the owning group is available. Until then,
changes to the four-user list require a GitLab Maintainer.

### 3.2 Vault

GitLab jobs authenticate to production Vault with a job-specific OIDC token.
No long-lived Vault token is stored in GitLab.

| Setting | Value |
|---|---|
| `VAULT_SERVER_URL` | `https://prod.vault.nvidia.com` |
| `VAULT_NAMESPACE` | `hw-tensorrt-edge-llm` |
| `VAULT_AUTH_PATH` | `jwt/nvidia/gitlab-master` |
| `VAULT_AUTH_ROLE` | `edge-llm-release-gitlab` |
| ID-token audience | `$VAULT_SERVER_URL` |

`VAULT_SERVER_URL`, `VAULT_NAMESPACE`, and `VAULT_AUTH_ROLE` are protected
GitLab variables. `VAULT_AUTH_PATH` is fixed in
`.gitlab/ci/wheel-release-jobs.yml`. GitLab native Vault resolution requires
`VAULT_SERVER_URL`; `VAULT_ADDR` is not used.

The Vault role is bound to this GitLab project, the configured audience, and
protected `release/*` and `rehearsal/*` refs. Its policy grants read access only
to these KV-v2 values:

| Engine mount | Secret path | Field | CI variable |
|---|---|---|---|
| `edge-llm/kv` | `release/pypi` | `artifactory_username` | `ARTIFACTORY_USERNAME` |
| `edge-llm/kv` | `release/pypi` | `artifactory_token` | `ARTIFACTORY_TOKEN` |
| `edge-llm/kv` | `release/kitmaker` | `api_token` | `KITMAKER_API_TOKEN` |

All three declarations use `file: false`. The job therefore receives the raw
secret value rather than a path to a temporary secret file.

### 3.3 Artifactory

The service identity associated with the Vault fields must have create and read
permission, but not delete or overwrite permission, on:

```text
hw-tensorrt-edge-llm-pypi-local
```

The endpoints are:

| Purpose | URL |
|---|---|
| Twine upload | `https://artifactory.nvidia.com/artifactory/api/pypi/hw-tensorrt-edge-llm-pypi-local` |
| Pip simple index | `https://artifactory.nvidia.com/artifactory/api/pypi/hw-tensorrt-edge-llm-pypi-local/simple` |
| Kitmaker wheel prefix | `https://artifactory.nvidia.com/artifactory/hw-tensorrt-edge-llm-pypi-local/tensorrt-edgellm/<version>/` |

Kitmaker must be able to read the canonical wheel URLs. Artifactory staging is
not public publication; it creates the stable input set for Kitmaker.

### 3.4 Kitmaker

Kitmaker onboarding required:

1. creating or reserving the `tensorrt-edgellm` project;
2. recording project ID `5413`;
3. assigning `jcalafato@nvidia.com` as the current PIC and identifying a backup
   owner;
4. enabling the project's public wheel-release capability;
5. generating a project-scoped API token; and
6. writing the token directly to Vault at
   `edge-llm/kv/release/kitmaker`, field `api_token`.

Only the project ID, PIC, owner names, publication policy, and token expiration
date belong in documentation. The API token must never appear in GitLab
variables, job logs, pipeline artifacts, shell history, or this document.

## 4. GitLab configuration

### 4.1 Protected non-secret variables

The current protected project variables are:

| Variable | Value | Purpose |
|---|---|---|
| `VAULT_SERVER_URL` | `https://prod.vault.nvidia.com` | Vault server and OIDC audience |
| `VAULT_NAMESPACE` | `hw-tensorrt-edge-llm` | Vault Enterprise namespace |
| `VAULT_AUTH_ROLE` | `edge-llm-release-gitlab` | GitLab JWT role |
| `KITMAKER_PROJECT_ID` | `5413` | Project-scoped release API path |
| `KITMAKER_PIC` | `jcalafato@nvidia.com` | PIC included in each wheel-release entry |

`EDGELLM_RELEASE_AUTH_READY=true` is the production publication feature gate.
Set it as a protected variable only after the non-publishing rehearsal and the
required release authorization reviews have succeeded. Removing or changing it
to any value other than `true` makes `wheel_release_publish` unavailable.

`RELEASE_APPROVAL_REFERENCE` is supplied when starting the protected release
pipeline. It must be an internal GitLab URL to the completed release approval
record.

### 4.2 Secret resolution

Each secret-bearing job requests:

```yaml
id_tokens:
  VAULT_ID_TOKEN:
    aud: $VAULT_SERVER_URL
```

The Artifactory and Kitmaker jobs request only the fields they need. The Vault
smoke job receives the two Artifactory fields and checks only that both are
non-empty. It does not make a network request with them or print them.

### 4.3 Pipeline activation rules

Secret-bearing staging jobs run only for:

- a protected `release/*` branch; or
- a web-created pipeline on a protected `rehearsal/*` branch with
  `EDGELLM_WHEEL_RELEASE_REHEARSAL=true`.

The production publish job additionally requires:

- a protected `release/*` branch;
- `EDGELLM_RELEASE_AUTH_READY=true`;
- a manual action;
- access to the protected `wheel-publication` environment; and
- a valid `RELEASE_APPROVAL_REFERENCE`.

A merge-request pipeline, branch push, schedule, unprotected branch, or
rehearsal branch cannot expose the production publication job.

## 5. Pipeline behavior

### 5.1 Build and qualification

The normal wheel pipeline builds the base Python wheel and all platform native
payloads, then assembles complete wheels for the six architecture/ABI pairs.
For every qualified Python ABI on x86_64 and aarch64, target integration creates
a fresh environment, installs the assembled wheel, imports its installed
packages, builds an engine, and runs inference. Only successful results enter
`wheel_integration_gate`, which records the exact tested filenames and SHA-256
digests.

### 5.2 `wheel_release_validate`

Validation accepts exactly six wheels and requires:

- CPython tags `cp310`, `cp311`, and `cp312` for both supported architectures;
- one version and one full Git source revision across the release;
- public wheel filename tags and native-manifest version consistency;
- native payload source revision matching `CI_COMMIT_SHA`;
- a SHA-256 match for every integration-qualified wheel; and
- no internal hostnames, scratch paths, or other forbidden public markers in
  audited text members.

Final-wheel assembly already validates wheel structure, tags, RECORD hashes,
runtime-manifest contents, licenses, and size before integration runs. Kitmaker
performs its own PyPI compatibility checks. Repeating those generic checks in
the release job would not strengthen the exact-digest qualification chain.

Production mode rejects development, local, and build-tagged filenames.
Rehearsal mode permits a unique PEP 440 development version but still enforces
the complete qualified matrix.

Output:

```text
artifacts/release/release-manifest.json
```

### 5.3 `wheel_release_vault_smoke`

The precheck proves that GitLab OIDC authentication and native Vault secret
resolution work for both Artifactory fields. A failure here occurs before any
staging or Kitmaker operation.

### 5.4 `wheel_release_stage`

The staging job:

1. loads the validated manifest;
2. constructs the six expected local wheel paths;
3. checks the Artifactory simple index;
4. skips upload only if all six filenames already exist with the expected
   SHA-256 values;
5. otherwise uploads all six through `python -m twine upload`;
6. waits for the index to expose the complete version;
7. downloads every indexed file and requires its SHA-256 digest to match the
   corresponding integration-qualified wheel.

Twine receives credentials through `TWINE_USERNAME` and `TWINE_PASSWORD`.
Authenticated reads use an in-memory HTTP Authorization header. Credentials are
never embedded in URLs.

Outputs:

```text
artifacts/release/staged-release-manifest.json
artifacts/release/artifactory-verification.json
```

### 5.5 `wheel_release_rehearse`

The rehearsal job sends one Kitmaker payload entry per staged wheel:

```json
{
  "pic": "jcalafato@nvidia.com",
  "job_type": "wheel-release-job",
  "url": "<canonical Artifactory wheel URL>",
  "upload": false
}
```

The request is sent to project `5413`. CI polls the returned release UUID for
up to two hours. Kitmaker authenticates, fetches all six Artifactory URLs, and
validates the release request without uploading to a public registry. The
deprecated `publish_to` field is intentionally omitted.

Output:

```text
artifacts/release/kitmaker-rehearsal-evidence.json
```

Credential-shaped response fields are redacted before evidence is written.

### 5.6 `wheel_release_publish`

The production job loads a manifest in `release` mode and validates the
approval-reference URL. Before creating a Kitmaker release, it checks both
public indexes:

- if both already contain the exact release, it records
  `already_published` and does not resubmit;
- if only one index contains the release, it fails and requires operator
  escalation; and
- if neither contains it, it submits the same six entries with `upload: true`.

CI polls Kitmaker for up to two hours, then polls each public index for up to
30 minutes. It downloads the exact six filenames and requires every SHA-256
digest to match the corresponding integration-qualified wheel. Unexpected,
partial, or byte-different same-version wheel sets fail closed.

SHA-256 equality establishes that the published wheels are byte-for-byte the
same artifacts that were installed, imported, and used for engine build and
inference before publication. Repeating a weaker dependency-free installation
after publication would add no coverage to that chain.

The indexes are:

```text
https://pypi.nvidia.com/simple/tensorrt-edgellm/
https://pypi.org/simple/tensorrt-edgellm/
```

Output:

```text
artifacts/release/publication-evidence.json
```

## 6. Setup procedure

### 6.1 Artifactory

1. Create `hw-tensorrt-edge-llm-pypi-local` as the local PyPI staging
   repository.
2. Create or select a release service identity.
3. Grant create and read access without delete or overwrite access.
4. Confirm Twine upload through the PyPI API endpoint.
5. Confirm authenticated index reads and wheel downloads through the `/simple`
   endpoint.
6. Confirm Kitmaker can read the canonical repository URLs.

### 6.2 Vault

1. Create the `hw-tensorrt-edge-llm` namespace if it does not exist.
2. Configure the `jwt/nvidia/gitlab-master` auth mount for the project.
3. Create role `edge-llm-release-gitlab` with the GitLab project, protected-ref,
   branch-pattern, and audience claims.
4. Grant read-only policy access to the two release secret paths.
5. Write `artifactory_username` and `artifactory_token` to
   `edge-llm/kv/release/pypi`.
6. Write the Kitmaker project token as `api_token` to
   `edge-llm/kv/release/kitmaker`.

### 6.3 Kitmaker

1. Create or reserve project `5413` for `tensorrt-edgellm`.
2. Assign the PIC and backup owner.
3. Enable public wheel release support.
4. Generate and vault the project-scoped API token.
5. Confirm the project can validate six Artifactory wheel URLs with
   `upload: false`.

### 6.4 GitLab

1. Add the five protected, non-secret project variables from section 4.1.
2. Protect `release/*` and `rehearsal/*`.
3. Protect `wheel-publication` with the release maintainers and a production
   approval rule.
4. Protect `wheel-publication-rehearsal` with the same deployers but no
   approval rule.
5. Keep `EDGELLM_RELEASE_AUTH_READY` unset until rehearsal and release
   authorization are complete.

## 7. Rehearsal procedure

Use this procedure to validate the entire external flow without public upload.

1. Create a branch under `rehearsal/*` from the candidate implementation.
2. Set a unique PEP 440 development version, such as `0.10.0.dev2`.
3. Protect the branch before running the pipeline.
4. In **Build > Pipelines > New pipeline**, select the rehearsal branch.
5. Add `EDGELLM_WHEEL_RELEASE_REHEARSAL=true`.
6. Start the pipeline from the web UI.
7. Confirm all six wheel builds and integration qualifications pass.
8. Confirm `wheel_release_vault_smoke`, `wheel_release_validate`, and
   `wheel_release_stage` pass.
9. Confirm `wheel_release_rehearse` finishes automatically and retains
   redacted Kitmaker evidence.
10. Confirm `wheel_release_publish` is absent.
11. Do not merge the rehearsal-only version commit into the feature or release
    branch.

Pipeline 64935968 exercised this procedure with version `0.10.0.dev2`. All 26
jobs passed. Kitmaker fetched and validated all six staged wheels with
`upload: false`; no public registry was modified.

## 8. Production release procedure

### 8.1 Release authorization

Before starting publication, complete and record:

- the protected `release/<version>` branch;
- `CHANGELOG.md` and release notes;
- OSS QA and the GitHub release staging PR;
- OSRB approval;
- MVSB approval;
- OSS disclosure review for documentation and release notes;
- documentation readiness;
- confirmation of no blocker bugs;
- public-index project and file-size settings; and
- final release-owner authorization.

The approval record must live at an internal GitLab URL.

### 8.2 Pipeline execution

1. Set the protected `EDGELLM_RELEASE_AUTH_READY=true` variable.
2. Start a pipeline for the protected `release/<version>` branch and provide
   `RELEASE_APPROVAL_REFERENCE=<internal GitLab approval URL>`.
3. Wait for build, integration, release validation, Vault smoke, and
   Artifactory staging.
4. Run the manual `wheel_release_rehearse` job and review its evidence.
5. Verify the staged manifest version, source revision, six filenames, and
   SHA-256 values.
6. Obtain the protected-environment approval for `wheel-publication`.
7. Run the manual `wheel_release_publish` job.
8. Wait for Kitmaker and both public-index verification checks.
9. Retain the pipeline and its release artifacts for the audit record.
10. Publish matching GitHub release notes and installation documentation.

After the release window, unset `EDGELLM_RELEASE_AUTH_READY` unless another
authorized release is immediately in progress.

## 9. Evidence and retention

| Artifact | Produced by | Evidence |
|---|---|---|
| `release-manifest.json` | validate | Qualified six-wheel filename, digest, version, revision, and pipeline identity |
| `staged-release-manifest.json` | stage | Canonical Artifactory URL for each wheel |
| `artifactory-verification.json` | stage | Staged-manifest digest and Artifactory byte-equivalence status |
| `kitmaker-rehearsal-evidence.json` | rehearse | Redacted Kitmaker terminal status with `upload: false` |
| `publication-evidence.json` | publish | Approval reference, staged-manifest digest, Kitmaker result, and both byte-identical public indexes |

GitLab retains release artifacts for 90 days. The release owner should preserve
the pipeline URL and approval record with the release checklist for the longer
of the required legal, security, and release-audit retention periods.

## 10. Security, privacy, legal, and disclosure controls

- Secrets live only in Vault and are selected per job using GitLab OIDC.
- Secret values are not passed on command lines, embedded in URLs, printed, or
  archived.
- API response fields that resemble tokens, passwords, authorization headers,
  or secrets are redacted before persistence.
- HTTP operations require HTTPS, validate the expected hostname, reject URL
  credentials, and bound response sizes.
- Staged and public files are verified against integration-qualified SHA-256
  digests.
- Wheel text members are scanned for internal hosts and scratch paths before
  staging.
- This design and all GitLab release infrastructure are removed by the OSS
  sanitizer. Run `python3 scripts/check_oss_release_sanitizer.py` before an OSS
  release.
- Public documentation and release notes require OSS disclosure review even
  though this internal design does not ship.
- Public package metadata, classifiers, license files, dependency declarations,
  and release notes remain part of OSRB/MVSB and OSS disclosure review.
- The flow does not collect end-user data. Pipeline evidence contains release
  metadata, internal pipeline URLs, artifact digests, and redacted service
  status only.
- Do not place personal passwords or personal API tokens in Vault. Use the
  release service identities and project-scoped credentials.

## 11. Failure handling

| Failure | Required response |
|---|---|
| Vault OIDC or secret resolution fails | Verify server URL, namespace, role audience, project/ref claims, and secret-field spelling. Do not substitute a GitLab-stored secret. |
| Artifactory contains a partial version | Do not overwrite or delete from CI. Repair through the repository owner, then rerun after the exact six-file state is restored. |
| Existing filename has a different digest | Treat as an immutable-release violation and escalate to the Artifactory owner. |
| Kitmaker rehearsal fails | Review redacted evidence and ask the Kitmaker project owner to inspect server-side details. Keep `upload: false`. |
| Kitmaker publication fails before either public index updates | Escalate with the release UUID before retrying. |
| Only one public index contains the release | Do not blindly resubmit. Escalate to Kitmaker because publication may be partial. |
| Public index is delayed | Allow the configured 30-minute polling window before escalation. |
| Public digest differs from the staged manifest | Stop the release and escalate as an integrity incident. |

Run wheel CI commands through the wrapper:

```bash
sh .gitlab/ci/scripts/wheel_ci.sh <command>
```

Calling `wheel_ci.py` directly does not establish the `packaging/` Python path
and can fail to import `wheellib`.

## 12. Credential rotation and ownership changes

When rotating Artifactory or Kitmaker credentials:

1. create the replacement credential with the same or narrower scope;
2. update only the corresponding Vault field;
3. run a protected Vault smoke and non-publishing rehearsal;
4. revoke the old credential after the rehearsal succeeds; and
5. record owner and expiration changes in the internal release checklist.

When the PIC changes, update the protected `KITMAKER_PIC` variable and the
Kitmaker project ownership record. When the release-maintainers group becomes
available, replace individual protected-environment deployers with that group
and retain the one-approval production rule.
