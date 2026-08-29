#!/bin/sh
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

set -u

if cp -a /opt/sonar-scanner ./sonar-scanner \
    && "$JAVA_HOME/bin/jlink" \
        --add-modules ALL-MODULE-PATH \
        --no-header-files \
        --no-man-pages \
        --output ./sonar-jre \
    && env \
        JAVA_HOME="$CI_PROJECT_DIR/sonar-jre" \
        PATH="$CI_PROJECT_DIR/sonar-jre/bin:$PATH" \
        ./sonar-scanner/bin/sonar-scanner --version; then
    printf 'ready\n' > sonar-scanner.status
else
    printf 'unavailable\n' > sonar-scanner.status
    echo "WARNING: SonarScanner could not be prepared"
fi
