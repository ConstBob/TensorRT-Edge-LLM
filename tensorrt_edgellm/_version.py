# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Custom version management for TensorRT Edge-LLM.

This module allows manual control of the base version while automatically
appending development information from git. It provides version strings
in PEP 440 compliant format for development releases.
"""

import subprocess
from datetime import datetime
from typing import Optional, Tuple

# Manual base version - change this when you want to bump the version
BASE_VERSION: str = "0.4.0.0"


def get_git_info() -> Tuple[Optional[str], str, str]:
    """
    Get git commit information for development version.
    
    Returns:
        Tuple containing:
        - commit_hash: Short git commit hash or None if git not available
        - commit_count: Number of commits since last tag as string
        - date_str: Current date in YYYYMMDD format
    """
    try:
        # Get git commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True).strip()

        # Get number of commits since last tag
        try:
            commit_count = subprocess.check_output(
                ["git", "rev-list", "--count", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True).strip()
        except subprocess.CalledProcessError:
            commit_count = "0"

        # Get current date
        date_str = datetime.now().strftime("%Y%m%d")

        return commit_hash, commit_count, date_str
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, "0", datetime.now().strftime("%Y%m%d")


def get_version() -> str:
    """
    Generate version string with base version and development info.
    
    Returns:
        PEP 440 compliant version string in format:
        - Development: {base_version}.dev{commit_count}+{commit_hash}.d{date}
        - Fallback: {base_version}.dev0+d{date}
    """
    commit_hash, commit_count, date_str = get_git_info()

    if commit_hash:
        # Development version: 0.4.0.0.dev{commit_count}+{commit_hash}.d{date}
        return f"{BASE_VERSION}.dev{commit_count}+{commit_hash}.d{date_str}"
    else:
        # Fallback version: 0.4.0.0.dev0+d{date}
        return f"{BASE_VERSION}.dev0+d{date_str}"


# Set the version
__version__: str = get_version()
