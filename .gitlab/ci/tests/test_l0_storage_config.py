# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import string
from pathlib import Path

import yaml


class GitLabCILoader(yaml.SafeLoader):
    pass


GitLabCILoader.add_constructor(
    "!reference",
    lambda loader, node: loader.construct_sequence(node),
)

CI_DIRECTORY = Path(__file__).parents[1]
ROOT_CONFIG = yaml.load(
    (Path(__file__).parents[3] / ".gitlab-ci.yml").read_text(),
    Loader=GitLabCILoader)
TEMPLATES = yaml.load((CI_DIRECTORY / "templates.yml").read_text(),
                      Loader=GitLabCILoader)
PRECHECK_JOBS = yaml.load((CI_DIRECTORY / "precheck-jobs.yml").read_text(),
                          Loader=GitLabCILoader)
SETUP_JOBS = yaml.safe_load((CI_DIRECTORY / "setup-jobs.yml").read_text())
L0_JOBS = yaml.safe_load((CI_DIRECTORY / "l0-jobs.yml").read_text())
L1_JOBS = yaml.safe_load((CI_DIRECTORY / "l1-jobs.yml").read_text())
CUTEDSL_JOBS = yaml.load((CI_DIRECTORY / "cutedsl-jobs.yml").read_text(),
                         Loader=GitLabCILoader)
INHERITABLE_CONFIGS = TEMPLATES | PRECHECK_JOBS

STORAGE_EXEMPT_JOBS = {
    "quantization_sanity",
    "l0_cross_build_jp6_cuda12.6",
    "l0_cross_build_d6l_cuda11.4",
    "l0_cross_build_d7l_cuda12.8",
}
L0_STORAGE_GATE = "init_l0_storage"
EXPORT_PRODUCER_LANES = {
    "l0_checkpoint_export": "checkpoint_export/onnx",
    "l0_checkpoint_export_ampere": "checkpoint_export_ampere/onnx",
}
EXPECTED_EXPORT_JOBS = {
    "l0_checkpoint_export": {
        "l0_checkpoint_export",
        "l0_rtx5080",
        "l0_b100",
        "l0_rtx5090",
        "l0_nemo_eval_rtx5090",
        "l0_drive_thor_1",
        "l0_drive_thor_1_trt11",
        "l0_drive_thor_2",
        "l0_drive_thor_2_trt11",
        "l0_jedha",
    },
    "l0_checkpoint_export_ampere": {
        "l0_checkpoint_export_ampere",
        "l0_a30",
        "l0_nemo_eval_a30",
        "l0_a30_trt11",
        "l0_rtx3090",
        "l0_jetson_orin",
    },
}


def _extends(config):
    parents = config.get("extends", [])
    return [parents] if isinstance(parents, str) else list(parents)


def _needed_jobs(config):
    return {
        need if isinstance(need, str) else need["job"]
        for need in config.get("needs", [])
    }


def _artifact_source_jobs(config):
    return {
        need if isinstance(need, str) else need["job"]
        for need in config.get("needs", [])
        if isinstance(need, str) or need.get("artifacts", True)
    }


def _upstream_jobs(job_name, jobs):
    pending = list(_needed_jobs(jobs[job_name]))
    upstream = set()
    while pending:
        name = pending.pop()
        if name in upstream:
            continue
        upstream.add(name)
        if name in jobs:
            pending.extend(_needed_jobs(jobs[name]))
    return upstream


def _visible_l0_jobs():
    return {
        name: config
        for name, config in L0_JOBS.items() if isinstance(config, dict)
        and not name.startswith(".") and config.get("stage") == "l0_test"
    }


def _configuration_hierarchy(config):
    seen = set()

    def visit(current):
        yield current
        for parent_name in _extends(current):
            if parent_name in seen:
                continue
            seen.add(parent_name)
            parent = INHERITABLE_CONFIGS.get(parent_name)
            if isinstance(parent, dict):
                yield from visit(parent)

    return list(visit(config))


def _rules(config):
    declared = [
        item["rules"] for item in _configuration_hierarchy(config)
        if "rules" in item
    ]
    assert len(declared) == 1
    return declared[0]


def _declared_environments(config):
    return [
        item["environment"] for item in _configuration_hierarchy(config)
        if "environment" in item
    ]


def _export_lane(config):
    lanes = {
        item["variables"]["L0_ONNX_CACHE_RELATIVE_PATH"]
        for item in _configuration_hierarchy(config)
        if "L0_ONNX_CACHE_RELATIVE_PATH" in item.get("variables", {})
    }
    assert len(lanes) <= 1
    return next(iter(lanes), None)


def _l0_lifecycle_jobs():
    return {
        name: config
        for name, config in SETUP_JOBS.items() if isinstance(config, dict)
        and config.get("environment", {}).get("action") in ("start", "stop")
    }


def test_storage_rules_select_context():
    lifecycle_jobs = _l0_lifecycle_jobs()
    start = next(config for config in lifecycle_jobs.values()
                 if config["environment"]["action"] == "start")
    rules = {rule["if"]: rule for rule in _rules(start)}
    mr_variables = rules['$CI_PIPELINE_SOURCE == "merge_request_event"'][
        "variables"]
    expected_mr_variables = {
        "L0_STORAGE_KIND": "mr",
        "L0_ENVIRONMENT_NAME": "l0/mr_${CI_MERGE_REQUEST_IID}",
        "L0_STORAGE_RESOURCE_GROUP": "l0-mr-${CI_MERGE_REQUEST_IID}",
        "L0_ONNX_REUSE": "1",
    }
    assert expected_mr_variables.items() <= mr_variables.items()

    stability_variables = rules[
        '$CI_PIPELINE_SOURCE == "schedule" && $L0_STABILITY == "true"'][
            "variables"]
    expected_stability_variables = {
        "L0_STORAGE_KIND": "stability",
        "L0_ENVIRONMENT_NAME": "l0/stability_${CI_PIPELINE_ID}",
        "L0_STORAGE_RESOURCE_GROUP": "l0-stability-${CI_PIPELINE_ID}",
        "L0_ONNX_REUSE": "0",
    }
    assert expected_stability_variables.items() <= stability_variables.items()


def test_l0_scratch_jobs_reach_one_non_artifact_storage_gate():
    jobs = _visible_l0_jobs()
    assert STORAGE_EXEMPT_JOBS <= set(jobs)
    scratch_jobs = set(jobs) - STORAGE_EXEMPT_JOBS
    graph = SETUP_JOBS | jobs

    for name, config in jobs.items():
        if name in STORAGE_EXEMPT_JOBS:
            assert L0_STORAGE_GATE not in _upstream_jobs(name, graph)
        else:
            assert L0_STORAGE_GATE in _upstream_jobs(name, graph)
        assert not _declared_environments(config)

    storage_roots = {
        name
        for name in scratch_jobs
        if not (_needed_jobs(jobs[name]) & scratch_jobs)
    }
    direct_gate_dependents = {
        name
        for name in scratch_jobs if L0_STORAGE_GATE in _needed_jobs(jobs[name])
    }
    assert direct_gate_dependents == storage_roots
    assert all(L0_STORAGE_GATE not in _artifact_source_jobs(jobs[name])
               for name in storage_roots)


def test_l0_builder_receives_cutedsl_artifacts():
    config = _visible_l0_jobs()["l0_builder_a30"]

    assert "build_cutedsl_docker_matrix" in _needed_jobs(config)
    assert "build_cutedsl_docker_matrix" in _artifact_source_jobs(config)


def test_cache_lanes_match_their_producer_dependencies():
    jobs = _visible_l0_jobs()
    producer_lanes = {
        producer: _export_lane(jobs[producer])
        for producer in EXPORT_PRODUCER_LANES
    }
    assert producer_lanes == EXPORT_PRODUCER_LANES

    actual_export_jobs = {
        producer: {
            name
            for name, config in jobs.items() if _export_lane(config) == lane
        }
        for producer, lane in producer_lanes.items()
    }
    assert actual_export_jobs == EXPECTED_EXPORT_JOBS

    producer_by_lane = {
        lane: producer
        for producer, lane in producer_lanes.items()
    }
    for name, config in jobs.items():
        lane = _export_lane(config)
        needed_producers = _needed_jobs(config) & set(producer_lanes)
        if lane is None:
            assert not needed_producers
            continue

        assert lane in producer_by_lane
        producer = producer_by_lane[lane]
        if name == producer:
            assert not needed_producers
        else:
            assert needed_producers == {producer}


def test_export_writers_serialize_by_storage_context_and_lane():

    def resolved_group(producer, mr_iid, pipeline_id):
        template = string.Template(L0_JOBS[producer]["resource_group"])
        return template.substitute(
            L0_STORAGE_RESOURCE_GROUP=f"l0-mr-{mr_iid}",
            CI_MERGE_REQUEST_IID=mr_iid,
            CI_PIPELINE_ID=pipeline_id,
        )

    mr_42_groups = {
        producer: resolved_group(producer, "42", "100")
        for producer in EXPORT_PRODUCER_LANES
    }
    assert len(set(mr_42_groups.values())) == len(EXPORT_PRODUCER_LANES)

    for producer, group in mr_42_groups.items():
        assert resolved_group(producer, "42", "101") == group
        assert resolved_group(producer, "43", "100") != group


def test_l0_storage_lifecycle_is_single_and_serialized():
    lifecycle_jobs = _l0_lifecycle_jobs()
    assert len(lifecycle_jobs) == 2
    starts = [(name, config) for name, config in lifecycle_jobs.items()
              if config["environment"]["action"] == "start"]
    stops = [(name, config) for name, config in lifecycle_jobs.items()
             if config["environment"]["action"] == "stop"]
    assert len(starts) == 1
    assert len(stops) == 1

    _, start = starts[0]
    stop_name, stop = stops[0]
    assert start["environment"]["name"] == stop["environment"]["name"]
    assert start["environment"]["on_stop"] == stop_name
    assert start["resource_group"] == stop["resource_group"]
    assert stop["interruptible"] is False

    assert _rules(start) == _rules(stop)
    assert stop["when"] == "manual"
    assert stop["allow_failure"] is True


def test_only_superseded_mr_work_is_automatically_cancelled():
    workflow = ROOT_CONFIG["workflow"]
    mr_rule = next(
        rule for rule in workflow["rules"]
        if rule.get("if") == '$CI_PIPELINE_SOURCE == "merge_request_event"')

    assert ROOT_CONFIG["default"]["interruptible"] is True
    assert workflow["auto_cancel"]["on_new_commit"] == "none"
    assert mr_rule["auto_cancel"]["on_new_commit"] == "interruptible"

    l0_finalizers = [
        config for config in _l0_lifecycle_jobs().values()
        if config["environment"]["action"] == "stop"
    ]
    l1_finalizers = [
        config for config in L1_JOBS.values() if isinstance(config, dict)
        and config.get("environment", {}).get("action") == "stop"
    ]
    cutedsl_finalizers = [
        config for config in CUTEDSL_JOBS.values() if isinstance(config, dict)
        and config.get("stage") == ".post" and config.get("when") == "always"
    ]
    assert len(l0_finalizers) == 1
    assert l1_finalizers
    assert cutedsl_finalizers
    assert all(config["interruptible"] is False for config in l0_finalizers +
               l1_finalizers + cutedsl_finalizers)


def test_ci_script_tests_are_blocking_for_every_supported_pipeline():
    config = PRECHECK_JOBS["ci_script_tests"]
    expected_conditions = {
        '$CI_PIPELINE_SOURCE == "merge_request_event"',
        '$CI_PIPELINE_SOURCE == "schedule" && $L0_STABILITY == "true"',
        '$L1 == "true"',
        '$CI_COMMIT_BRANCH == "main"',
        '$CI_COMMIT_BRANCH =~ /^release/',
        "$GITHUB_PR && $GITHUB_REPO",
    }

    assert config["stage"] == "precheck"
    assert not config.get("allow_failure", False)
    assert {rule["if"] for rule in _rules(config)} == expected_conditions
