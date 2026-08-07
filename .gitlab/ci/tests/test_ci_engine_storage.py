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
"""Tests for the CI engine storage policy entry point."""

import importlib.util
import string
from pathlib import Path

import pytest
import yaml

SCRIPT_PATH = (Path(__file__).parents[1] / "scripts" / "ci_engine_storage.py")
MODULE_SPEC = importlib.util.spec_from_file_location("ci_engine_storage",
                                                     SCRIPT_PATH)
ci_engine_storage = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(ci_engine_storage)
REPO_ROOT = Path(__file__).parents[3]
CI_DIRECTORY = Path(__file__).parents[1]
ENGINE_BUILDING_TEMPLATES = frozenset((
    ".x86_test_template",
    ".nemo_eval_x86_template",
    ".device_test_template",
))
ENGINE_PATHS = {
    "L0": Path("L0/MR_42/workspace/pipeline_100/job_200_l0-a30/engines"),
    "L1": Path("L1/100/l1_a30_llm/engines"),
}


class GitLabCILoader(yaml.SafeLoader):
    pass


GitLabCILoader.add_constructor(
    "!reference",
    lambda loader, node: loader.construct_sequence(node),
)


def _load_ci_config(path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=GitLabCILoader)


def _all_ci_config():
    config = {}
    for path in [REPO_ROOT / ".gitlab-ci.yml", *CI_DIRECTORY.glob("*.yml")]:
        config.update(_load_ci_config(path))
    return config


def _extends(config):
    parents = config.get("extends", [])
    return [parents] if isinstance(parents, str) else list(parents)


def _inherited_names(name, configs):
    pending = [name]
    inherited = set()
    while pending:
        current_name = pending.pop()
        if current_name in inherited:
            continue
        inherited.add(current_name)
        current = configs.get(current_name, {})
        if isinstance(current, dict):
            pending.extend(_extends(current))
    return inherited


def _engine_building_jobs(jobs, configs):
    return [
        name for name, config in jobs.items()
        if isinstance(config, dict) and not name.startswith(".")
        and not ENGINE_BUILDING_TEMPLATES.isdisjoint(
            _inherited_names(name, configs))
    ]


def _effective_mapping(name, key, configs):
    effective = {}
    config = configs[name]
    for parent in _extends(config):
        if parent in configs:
            effective.update(_effective_mapping(parent, key, configs))
    effective.update(config.get(key, {}))
    return effective


def _effective_sequence(name, key, configs):
    value = None
    config = configs[name]
    for parent in _extends(config):
        if parent in configs:
            parent_value = _effective_sequence(parent, key, configs)
            if parent_value is not None:
                value = parent_value
    return config.get(key, value)


def _resolve_references(items, configs):
    resolved = []
    for item in items:
        if (isinstance(item, list) and len(item) == 2 and item[0] in configs):
            referenced = configs[item[0]][item[1]]
            resolved.extend(_resolve_references(referenced, configs))
        else:
            resolved.append(item)
    return resolved


def _expand_variables(variables):
    expanded = {
        **variables,
        "EDGE_LLM_CACHE_DIR": "/scratch.edge_llm_cache",
        "CI_PIPELINE_ID": "100",
    }
    for _ in range(len(expanded)):
        previous = dict(expanded)
        expanded = {
            name:
            string.Template(value).safe_substitute(previous) if isinstance(
                value, str) else value
            for name, value in previous.items()
        }
        if expanded == previous:
            break
    return expanded


@pytest.fixture
def engine_environment(request, tmp_path, monkeypatch):
    scratch_root = tmp_path / "scratch.edge_llm_cache"
    namespace = getattr(request, "param", "L0")
    engine_dir = scratch_root / ENGINE_PATHS[namespace]
    engine_dir.mkdir(parents=True)
    (engine_dir / "model.engine").write_bytes(b"engine")
    monkeypatch.setenv("EDGE_LLM_CACHE_DIR", str(scratch_root))
    monkeypatch.setenv("ENGINE_DIR", str(engine_dir))
    monkeypatch.delenv("CI_COMMIT_MESSAGE", raising=False)
    return engine_dir


@pytest.mark.parametrize("engine_environment", ("L0", "L1"), indirect=True)
def test_default_policy_removes_engine_directory(engine_environment):
    assert ci_engine_storage.main([]) == 0

    assert not engine_environment.exists()


def test_explicit_retention_preserves_engine_directory(engine_environment,
                                                       monkeypatch):
    monkeypatch.setenv("CI_COMMIT_MESSAGE",
                       "test: Reproduce failure [retain engines]")

    assert ci_engine_storage.main([]) == 0

    assert (engine_environment / "model.engine").is_file()


def test_cleanup_refuses_directory_outside_scratch(engine_environment,
                                                   monkeypatch, tmp_path):
    outside_engine_dir = tmp_path / "unmanaged" / "job" / "engines"
    outside_engine_dir.mkdir(parents=True)
    (outside_engine_dir / "keep.engine").write_bytes(b"engine")
    monkeypatch.setenv("ENGINE_DIR", str(outside_engine_dir))

    assert ci_engine_storage.main([]) == 1

    assert (outside_engine_dir / "keep.engine").is_file()


def test_cleanup_refuses_broad_directory_inside_scratch(tmp_path, monkeypatch):
    scratch_root = tmp_path / "scratch.edge_llm_cache"
    mr_root = scratch_root / "L0" / "MR_42"
    protected_file = mr_root / "workspace" / "pipeline_100" / "result.log"
    protected_file.parent.mkdir(parents=True)
    protected_file.write_text("keep", encoding="utf-8")
    monkeypatch.setenv("EDGE_LLM_CACHE_DIR", str(scratch_root))
    monkeypatch.setenv("ENGINE_DIR", str(mr_root))

    assert ci_engine_storage.main([]) == 1

    assert protected_file.is_file()


def test_engine_building_jobs_run_cleanup_policy():
    configs = _all_ci_config()
    default_after_script = configs["default"]["after_script"]

    for path in (CI_DIRECTORY / "l0-jobs.yml", CI_DIRECTORY / "l1-jobs.yml"):
        jobs = _load_ci_config(path)
        engine_jobs = _engine_building_jobs(jobs, configs)
        assert engine_jobs
        for name in engine_jobs:
            after_script = (_effective_sequence(name, "after_script", configs)
                            or default_after_script)
            commands = _resolve_references(after_script, configs)
            assert any("ci_engine_storage.py" in command
                       for command in commands), name


def test_l1_engine_building_jobs_use_scratch_storage():
    configs = _all_ci_config()
    l1_jobs = _load_ci_config(CI_DIRECTORY / "l1-jobs.yml")
    scratch_root = Path("/scratch.edge_llm_cache/L1/100")
    engine_jobs = _engine_building_jobs(l1_jobs, configs)
    assert engine_jobs

    for name in engine_jobs:
        variables = _expand_variables(
            _effective_mapping(name, "variables", configs))
        engine_dir = variables.get("ENGINE_DIR")
        assert engine_dir is not None, name
        assert "$" not in engine_dir, name
        assert Path(engine_dir).is_relative_to(scratch_root), name
