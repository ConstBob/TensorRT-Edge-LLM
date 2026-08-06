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

import configparser
import pathlib
import re

G_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_ci_github_dependencies_have_internal_mirrors():
    github_url_pattern = (
        r"(?:https?://github\.com/|ssh://git@github\.com/|git@github\.com:)"
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?")
    gitmodules = configparser.ConfigParser(interpolation=None)
    gitmodules.read(G_REPO_ROOT / ".gitmodules", encoding="utf-8")
    submodule_urls = {
        gitmodules[section]["url"]
        for section in gitmodules.sections()
        if section.startswith("submodule ")
    }
    github_dependencies = {
        url
        for url in submodule_urls if re.fullmatch(github_url_pattern, url)
    }

    ci_yaml_paths = [G_REPO_ROOT / ".gitlab-ci.yml"]
    ci_yaml_paths.extend((G_REPO_ROOT / ".gitlab" / "ci").rglob("*.yml"))
    ci_yaml_paths.extend((G_REPO_ROOT / ".gitlab" / "ci").rglob("*.yaml"))
    ci_yaml = "\n".join(
        path.read_text(encoding="utf-8") for path in ci_yaml_paths)
    internal_rewrite_prefixes = {
        public
        for mirror, public in re.findall(
            r'url\."([^"]+)"\.insteadOf\s+"([^"]+)"',
            ci_yaml,
        ) if mirror.startswith("https://gitlab-master.nvidia.com/")
    }

    # GitHub PR jobs clone the source under test, not a CI dependency.
    dependency_source_paths = [
        path for path in ci_yaml_paths if path.name != ".github-pr-jobs.yml"
    ]
    dependency_source_paths.extend(
        (G_REPO_ROOT / ".gitlab" / "ci" / "scripts").rglob("*.sh"))
    dependency_source_paths.extend(
        (G_REPO_ROOT / ".gitlab" / "ci" / "scripts").rglob("*.py"))
    dependency_source_paths.append(G_REPO_ROOT / "experimental" / "docker" /
                                   "build.sh")
    dependency_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in dependency_source_paths)
    dependency_commands = dependency_sources.replace("\\\n", " ")
    github_dependencies.update(
        re.findall(rf"\bgit\s+clone\b[^\n]*?({github_url_pattern})",
                   dependency_commands))

    unmapped_urls = {
        url
        for url in github_dependencies if not any(
            url.startswith(prefix) for prefix in internal_rewrite_prefixes)
    }
    assert not unmapped_urls, (
        "GitHub dependencies missing an internal insteadOf mapping: " +
        ", ".join(sorted(unmapped_urls)))
