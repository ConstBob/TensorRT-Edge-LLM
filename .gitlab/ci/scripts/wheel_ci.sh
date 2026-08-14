#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$(python3 -c 'import sys; print(sys.version_info[:2] < (3, 11))')" = "True" ]; then
    python3 -m pip install 'tomli==2.2.1'
fi

exec python3 packaging/wheel_cli.py "$@"
