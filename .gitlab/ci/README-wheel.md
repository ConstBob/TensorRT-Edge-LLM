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
matrices. Do not edit it directly. Payload build jobs consume the public matrix
joined with the internal qualification file; target jobs install the assembled
wheel in a fresh environment and exercise installed engine build and inference.
The integration gate requires evidence for every public variant and qualified
Python ABI before the pipeline can succeed.
