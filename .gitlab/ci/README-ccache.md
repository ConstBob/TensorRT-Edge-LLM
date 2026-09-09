<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Edge-LLM compiler cache policy

Selected x86 native, wheel, and Sonar jobs use a disposable local ccache primary
and additive file backends under
`/scratch.edge_llm_cache/gitlab-ci-cache`. Merge requests write only
`mr/<project-id>/<mr-iid>/ccache/v1` and read the applicable trusted main or
`release/<series>` shard. Protected main and release jobs are the only trusted
writers. Ccache validates compiler content, preprocessed source, headers, flags,
architecture options, and environment inputs for every translation unit.
Linking, device linking, archives, packaging, and tests still run normally.

The local primary is capped at 2 GB. Each MR compiler overlay and each trusted
compiler/target shard is trimmed to 5 GiB by `prune_ci_ccache`; the MR cache is
also removed by the L0 environment stop job and has the existing 30-day
backstop. Release cache retirement is a team lifecycle decision. Remote board,
cross-build, and local Spark backends remain disabled because the experiment
measured 15-36% regressions on those paths.

The experiment measured a 53.6% best whole-job reduction and 48-51% reductions
for two selected x86 wheel jobs. Normal compile-heavy MRs are expected to save
15-40% of whole-job time. Test- or link-heavy jobs can be flat even with high
cacheable-call hit rates. The shared backend is a performance optimization, not
a security boundary; filesystem permissions must deny merge-request runners
writes to protected-cache namespaces.
