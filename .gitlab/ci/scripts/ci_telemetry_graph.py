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

import html
import json
import typing

SVG_WIDTH = 1200
CHART_ROW_HEIGHT = 34
CHART_LEFT = 360
CHART_RIGHT = 110
MAX_CHART_ROWS = 15


def _number(value: typing.Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _duration(value: typing.Any) -> str:
    seconds = _number(value)
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def _svg_text(value: typing.Any) -> str:
    return html.escape(str(value), quote=True)


def render_svg(summary: typing.Mapping[str, typing.Any]) -> str:
    """Render a dependency-free SVG summary for GitLab artifact preview."""
    rows = list(summary.get("top_latest_success_jobs_by_execution_seconds",
                            []))
    if not rows:
        rows = list(summary.get("top_jobs_by_execution_seconds", []))
    rows = rows[:MAX_CHART_ROWS]
    height = 300 + max(1, len(rows)) * CHART_ROW_HEIGHT
    chart_width = SVG_WIDTH - CHART_LEFT - CHART_RIGHT
    maximum = max((_number(row.get("execution_seconds")) for row in rows),
                  default=1.0)
    if maximum <= 0:
        maximum = 1.0

    cards = [
        ("Total consumed",
         _duration(summary.get("total_consumed_execution_seconds"))),
        ("Latest success",
         _duration(summary.get("latest_success_execution_seconds"))),
        ("Retry cost", _duration(summary.get("retry_execution_seconds"))),
        ("Runner queue", _duration(summary.get("queue_seconds_total"))),
        ("Phase coverage",
         f"{_number(summary.get('phase_unique_coverage_percent')):.1f}%"),
    ]
    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{height}" '
        f'viewBox="0 0 {SVG_WIDTH} {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Edge LLM CI telemetry</title>',
        '<desc id="description">Pipeline execution summary and longest latest-success jobs.</desc>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172b4d}.title{font-size:28px;font-weight:700}'
        '.subtitle{font-size:14px;fill:#5e6c84}.card-label{font-size:13px;fill:#5e6c84}'
        '.card-value{font-size:22px;font-weight:700}.job{font-size:13px}.value{font-size:12px;fill:#44546f}'
        '.bar{fill:#1f75cb}.grid{stroke:#dfe1e6;stroke-width:1}</style>',
        '<text id="title-text" class="title" x="32" y="42">Edge LLM CI telemetry</text>',
        '<text class="subtitle" x="32" y="66">Execution excludes runner queue and pipeline scheduling wait.</text>',
    ]
    card_width = (SVG_WIDTH - 64 - 4 * 16) / 5
    for index, (label, value) in enumerate(cards):
        x = 32 + index * (card_width + 16)
        output.extend([
            f'<rect x="{x:.1f}" y="88" width="{card_width:.1f}" height="86" rx="8" fill="#ffffff" stroke="#dfe1e6"/>',
            f'<text class="card-label" x="{x + 14:.1f}" y="116">{_svg_text(label)}</text>',
            f'<text class="card-value" x="{x + 14:.1f}" y="150">{_svg_text(value)}</text>',
        ])

    output.append(
        '<text class="title" x="32" y="222" font-size="20">Longest latest-success jobs</text>'
    )
    chart_top = 246
    if not rows:
        output.append(
            '<text class="subtitle" x="32" y="276">No completed job durations were available.</text>'
        )
    for index, row in enumerate(rows):
        y = chart_top + index * CHART_ROW_HEIGHT
        label = str(row.get("job_name", ""))
        attempt_count = int(_number(row.get("attempt_count")))
        if attempt_count > 1:
            label += f" (attempt {row.get('attempt_number')}/{attempt_count})"
        value = _number(row.get("execution_seconds"))
        width = chart_width * value / maximum
        output.extend([
            f'<line class="grid" x1="{CHART_LEFT}" y1="{y + 19}" x2="{SVG_WIDTH - CHART_RIGHT}" y2="{y + 19}"/>',
            f'<text class="job" x="32" y="{y + 15}">{_svg_text(label[:62])}</text>',
            f'<rect class="bar" x="{CHART_LEFT}" y="{y + 3}" width="{width:.1f}" height="18" rx="3"/>',
            f'<text class="value" x="{CHART_LEFT + width + 8:.1f}" y="{y + 17}">{_svg_text(_duration(value))}</text>',
        ])
    output.append("</svg>\n")
    return "\n".join(output)


def render_html(summary: typing.Mapping[str, typing.Any], svg: str) -> str:
    """Render a self-contained static report with links to detailed files."""
    embedded_summary = html.escape(json.dumps(summary, sort_keys=True),
                                   quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Edge LLM CI telemetry</title>
  <style>
    body {{ background:#f8fafc; color:#172b4d; font-family:Arial,sans-serif; margin:0; padding:24px; }}
    main {{ margin:auto; max-width:1200px; }}
    .chart {{ background:white; border:1px solid #dfe1e6; border-radius:10px; overflow:auto; }}
    .links {{ display:flex; flex-wrap:wrap; gap:12px; margin:18px 0; }}
    a {{ color:#0b66c3; }}
    .note {{ color:#5e6c84; font-size:14px; }}
  </style>
</head>
<body>
<main>
  <div class="chart">{svg}</div>
  <nav class="links" aria-label="Detailed telemetry reports">
    <a href="summary.md">Markdown summary</a>
    <a href="summary.json">JSON summary</a>
    <a href="jobs.csv">Job attempts CSV</a>
    <a href="phases.csv">Phases CSV</a>
    <a href="cache_events.csv">Cache events CSV</a>
    <a href="test_functions.csv">Test functions CSV</a>
    <a href="metrics.txt">OpenMetrics</a>
  </nav>
  <p class="note">Summed execution is consumed runner compute, not parallel pipeline wall time. Queue is reported separately.</p>
  <details><summary>Embedded machine-readable summary</summary><pre>{embedded_summary}</pre></details>
</main>
</body>
</html>
"""


def render_metrics(summary: typing.Mapping[str, typing.Any]) -> str:
    """Render stable pipeline-level OpenMetrics for GitLab's metrics widget."""
    gauges = (
        ("edgellm_ci_execution_seconds_total",
         "Total execution across all job attempts.",
         summary.get("total_consumed_execution_seconds")),
        ("edgellm_ci_latest_success_execution_seconds",
         "Execution of the latest successful attempt per logical job.",
         summary.get("latest_success_execution_seconds")),
        ("edgellm_ci_retry_execution_seconds",
         "Execution consumed by superseded job attempts.",
         summary.get("retry_execution_seconds")),
        ("edgellm_ci_queue_seconds_total",
         "GitLab runner queue time across job attempts.",
         summary.get("queue_seconds_total")),
        ("edgellm_ci_phase_unique_seconds_total",
         "Union of instrumented phase intervals within jobs.",
         summary.get("phase_unique_seconds_total")),
        ("edgellm_ci_phase_coverage_percent",
         "Unique phase interval coverage as a percent of execution.",
         summary.get("phase_unique_coverage_percent")),
        ("edgellm_ci_job_attempts",
         "Number of GitLab job attempts in the report.",
         summary.get("job_count")),
        ("edgellm_ci_cache_events", "Number of validated cache events.",
         summary.get("cache_event_count")),
        ("edgellm_ci_cache_hits", "Number of cache hit decisions.",
         summary.get("cache_hit_count")),
        ("edgellm_ci_cache_misses", "Number of cache miss decisions.",
         summary.get("cache_miss_count")),
    )
    lines: typing.List[str] = []
    for name, help_text, value in gauges:
        lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} gauge",
                      f"{name} {_number(value):.6f}"))
    return "\n".join(lines) + "\n"
