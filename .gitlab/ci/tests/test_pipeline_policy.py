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

import pathlib

import yaml


class GitLabCILoader(yaml.SafeLoader):
    pass


GitLabCILoader.add_constructor(
    "!reference",
    lambda loader, node: loader.construct_sequence(node),
)

PROJECT_ROOT = pathlib.Path(__file__).parents[3]
CI_DIRECTORY = pathlib.Path(__file__).parents[1]


def _load_config(path):
    return yaml.load(path.read_text(), Loader=GitLabCILoader)


ROOT_CONFIG = _load_config(PROJECT_ROOT / ".gitlab-ci.yml")
TEMPLATES = _load_config(CI_DIRECTORY / "templates.yml")
PRECHECK_JOBS = _load_config(CI_DIRECTORY / "precheck-jobs.yml")
SETUP_JOBS = _load_config(CI_DIRECTORY / "setup-jobs.yml")
L0_JOBS = _load_config(CI_DIRECTORY / "l0-jobs.yml")
L1_JOBS = _load_config(CI_DIRECTORY / "l1-jobs.yml")
CUTEDSL_JOBS = _load_config(CI_DIRECTORY / "cutedsl-jobs.yml")
WHEEL_JOBS = _load_config(CI_DIRECTORY / "wheel-jobs.yml")
SONAR_JOBS = _load_config(CI_DIRECTORY / "sonar-jobs.yml")
LOCAL_BOARD_JOBS = _load_config(CI_DIRECTORY / "local-board-jobs.yml")
EXPERIMENTAL_DOCKER_JOBS = _load_config(CI_DIRECTORY /
                                        "experimental-docker-jobs.yml")

MR_WORK_CONFIGS = {
    "setup": SETUP_JOBS,
    "l0": L0_JOBS,
    "cutedsl": CUTEDSL_JOBS,
    "wheel": WHEEL_JOBS,
    "sonar": SONAR_JOBS,
    "local_board": LOCAL_BOARD_JOBS,
    "experimental_docker": EXPERIMENTAL_DOCKER_JOBS,
}
INHERITABLE_CONFIGS = {}
for config in (TEMPLATES, PRECHECK_JOBS, *MR_WORK_CONFIGS.values()):
    INHERITABLE_CONFIGS.update(config)

SKIP_SIGNALS = ("CI_MERGE_REQUEST_LABELS", "CI_MERGE_REQUEST_TITLE")
COMMIT_MESSAGE_SIGNAL = "CI_COMMIT_MESSAGE"


def _extends(config):
    parents = config.get("extends", [])
    return [parents] if isinstance(parents, str) else list(parents)


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


def _resolved_value(config, key, default=None):
    return next((item[key]
                 for item in _configuration_hierarchy(config) if key in item),
                default)


def _rules(config):
    return _resolved_value(config, "rules", [])


def _visible_jobs(config):
    return {
        name: job
        for name, job in config.items()
        if isinstance(job, dict) and not name.startswith(".")
    }


def _is_mr_rule(rule):
    condition = rule.get("if", "")
    return ("CI_PIPELINE_SOURCE" in condition
            and "merge_request_event" in condition)


def _mr_include_indices(rules):
    return [
        index for index, rule in enumerate(rules)
        if _is_mr_rule(rule) and rule.get("when", "on_success") != "never"
    ]


def _signal_rules(rules, signal):
    return [
        rule for rule in rules
        if signal in rule.get("if", "") and "skip tests" in rule["if"].lower()
    ]


def _workflow_signal_variables(signal):
    variables = {}
    for rule in _signal_rules(ROOT_CONFIG["workflow"]["rules"], signal):
        variables.update(rule.get("variables", {}))
    return variables


def _rule_skips_signal(rule, signal, workflow_variables):
    if rule.get("when") != "never":
        return False

    condition = rule.get("if", "")
    if signal in condition and "skip tests" in condition.lower():
        return True
    return any(f"${name}" in condition and str(value) in condition
               for name, value in workflow_variables.items())


def _skip_indices(rules, signal):
    workflow_variables = _workflow_signal_variables(signal)
    return [
        index for index, rule in enumerate(rules)
        if _rule_skips_signal(rule, signal, workflow_variables)
    ]


def _mr_work_jobs():
    return {
        f"{source}:{name}": job
        for source, config in MR_WORK_CONFIGS.items()
        for name, job in _visible_jobs(config).items()
        if _mr_include_indices(_rules(job))
    }


def test_skip_requests_keep_mr_pipeline_and_blocking_prechecks():
    workflow_rules = ROOT_CONFIG["workflow"]["rules"]
    ordinary_mr_rules = [
        rule for rule in workflow_rules
        if _is_mr_rule(rule) and not any(signal in rule.get("if", "")
                                         for signal in SKIP_SIGNALS)
        and rule.get("when", "on_success") != "never"
    ]
    assert ordinary_mr_rules

    precheck_jobs = _visible_jobs(PRECHECK_JOBS)
    assert precheck_jobs
    for signal in SKIP_SIGNALS:
        assert all(
            rule.get("when", "on_success") != "never"
            for rule in _signal_rules(workflow_rules, signal))
        for name, job in precheck_jobs.items():
            rules = _rules(job)
            mr_indices = _mr_include_indices(rules)
            skip_indices = _skip_indices(rules, signal)
            assert _resolved_value(job, "stage") == "precheck", name
            assert not _resolved_value(job, "allow_failure", False), name
            assert mr_indices, name
            assert not skip_indices or min(mr_indices) < min(
                skip_indices), name


def test_skip_requests_omit_expensive_mr_work():
    jobs = _mr_work_jobs()
    assert jobs
    for signal in SKIP_SIGNALS:
        for name, job in jobs.items():
            rules = _rules(job)
            skip_indices = _skip_indices(rules, signal)
            assert skip_indices, name
            assert min(skip_indices) < min(_mr_include_indices(rules)), name


def test_commit_message_does_not_skip_mr_work():
    workflow_rules = ROOT_CONFIG["workflow"]["rules"]
    assert not _signal_rules(workflow_rules, COMMIT_MESSAGE_SIGNAL)

    jobs = _visible_jobs(PRECHECK_JOBS) | _mr_work_jobs()
    for name, job in jobs.items():
        assert not _signal_rules(_rules(job), COMMIT_MESSAGE_SIGNAL), name


def test_only_superseded_mr_work_is_automatically_cancelled():
    workflow = ROOT_CONFIG["workflow"]
    mr_rules = [
        rule for rule in workflow["rules"]
        if _is_mr_rule(rule) and rule.get("when", "on_success") != "never"
    ]

    assert ROOT_CONFIG["default"]["interruptible"] is True
    assert workflow["auto_cancel"]["on_new_commit"] == "none"
    assert mr_rules
    assert all(rule["auto_cancel"]["on_new_commit"] == "interruptible"
               for rule in mr_rules)


def test_environment_finalizers_are_non_interruptible():
    l0_finalizers = [
        job for job in _visible_jobs(SETUP_JOBS).values()
        if job.get("environment", {}).get("action") == "stop"
    ]
    l1_finalizers = [
        job for job in _visible_jobs(L1_JOBS).values()
        if job.get("environment", {}).get("action") == "stop"
    ]
    assert len(l0_finalizers) == 1
    assert l1_finalizers
    assert all(job["interruptible"] is False
               for job in l0_finalizers + l1_finalizers)


def test_ci_script_tests_are_blocking_for_every_supported_pipeline():
    job = PRECHECK_JOBS["ci_script_tests"]
    expected_conditions = {
        '$CI_PIPELINE_SOURCE == "merge_request_event"',
        '$CI_PIPELINE_SOURCE == "schedule" && $L0_STABILITY == "true"',
        '$L1 == "true"',
        '$CI_COMMIT_BRANCH == "main"',
        '$CI_COMMIT_BRANCH =~ /^release/',
        "$GITHUB_PR && $GITHUB_REPO",
    }

    assert job["stage"] == "precheck"
    assert not job.get("allow_failure", False)
    assert {rule["if"] for rule in _rules(job)} == expected_conditions
