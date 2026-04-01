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

"""AOT-compile CuTe DSL kernels (FMHA + GDN) into a single static library for CMake linking.

Usage (run from the repo root):
  python kernelSrcs/build_cutedsl.py                    # build all kernels supported by this GPU
  python kernelSrcs/build_cutedsl.py --kernels gdn      # build a specific group only
  python kernelSrcs/build_cutedsl.py --gpu_arch sm_87   # override SM detection (rarely needed)

The GPU SM is auto-detected via cupy / nvidia-smi and used to filter which kernel variants
are compiled.  All kernel scripts are invoked without --gpu_arch (device-native JIT), which
works uniformly on Linux and QNX.

Output (under {output_dir}/{arch}/):
  libcutedsl_{arch}.a   — merged static archive: kernel objects + DSL runtime
  include/cutedsl_all.h — umbrella header (#includes every variant header)
  metadata.json         — groups / variants list consumed by cmake/CuteDsl.cmake
"""

import argparse
import concurrent.futures
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_DEFAULT_OUTPUT_DIR = (_SCRIPT_DIR / "../cpp/kernels/cuteDSLArtifact").resolve()
_CUTLASS_DSL_VERSION = "4.4.1"
_CUPY_VERSIONS = {12: ("cupy-cuda12x", "12.3.0"), 13: ("cupy-cuda13x", "13.6.0")}

# Common flag sets for FMHA variants
_LLM = ["--is_causal", "--is_persistent", "--export_only", "--bottom_right_align"]
_LLM_FP8 = _LLM + ["--in_dtype", "Float8E4M3FN"]
_VIT = ["--is_persistent", "--export_only", "--vit_mode"]


@dataclass
class KernelVariant:
    """One compilable kernel variant in the CuTe DSL registry.

    Attributes:
        name:          Unique identifier — used as --file_name / --function_prefix.
        group:         Logical group ("gdn" or "fmha"). cmake sets CUTE_DSL_<GROUP>_ENABLED.
        supported_sms: Explicit SM whitelist. With --kernels ALL, only variants whose
                       supported_sms contains the detected/requested SM are compiled.
        script:        Kernel script path relative to kernelSrcs/.
        script_args:   Args forwarded verbatim after --output_dir/--file_name/--function_prefix.
                       GDN variants MUST include "--export_only" here.

    """
    name: str
    group: str
    supported_sms: list[int]
    script: str
    script_args: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Kernel registry — add new groups/variants here.
#
# GDN: Gated Delta Net (Ampere SM80+, arch-polymorphic).
#
# FMHA: Fused Multi-Head Attention (Blackwell SM100/SM101). The fmha.py script
#       is hardcoded to SM100 Blackwell instructions (TMEM, Blackwell MMA).
#
# Neither group receives --gpu_arch from the build script; they all compile
# device-native, which works uniformly across all platforms.
# ---------------------------------------------------------------------------
KERNEL_VARIANTS = [
    # --- GDN group ---
    KernelVariant(
        name="gdn_decode",
        group="gdn",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="gdn_cutedsl/gdn_decode.py",
        script_args=["--export_only"],
    ),
    KernelVariant(
        name="gdn_prefill",
        group="gdn",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="gdn_cutedsl/gdn_prefill.py",
        script_args=["--export_only"],
    ),
    # --- FMHA group ---
    KernelVariant(
        name="fmha_d64",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"] + _LLM,
    ),
    KernelVariant(
        name="fmha_d128",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"] + _LLM,
    ),
    KernelVariant(
        name="fmha_d64_sw",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"]
                    + _LLM + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d128_sw",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"]
                    + _LLM + ["--window_size", "4096,-1"],
    ),
    # LLM FP8 input → FP16 output
    KernelVariant(
        name="fmha_d64_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"] + _LLM_FP8,
    ),
    KernelVariant(
        name="fmha_d128_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"] + _LLM_FP8,
    ),
    KernelVariant(
        name="fmha_d64_sw_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"]
                    + _LLM_FP8 + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d128_sw_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"]
                    + _LLM_FP8 + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="vit_fmha_d64",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,14,64"] + _VIT,
    ),
    KernelVariant(
        name="vit_fmha_d72",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,72", "--k_shape", "1,1024,14,72"] + _VIT,
    ),
    KernelVariant(
        name="vit_fmha_d80",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,80", "--k_shape", "1,1024,14,80"] + _VIT,
    ),
    KernelVariant(
        name="vit_fmha_d128",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,14,128"] + _VIT,
    ),
]

# All known group names (set for O(1) membership check — no manual maintenance needed).
_ALL_GROUPS: set[str] = {v.group for v in KERNEL_VARIANTS}


# ---------------------------------------------------------------------------
# Variant selection
# ---------------------------------------------------------------------------

def _parse_sm(gpu_arch_str):
    """Parse SM number from "sm_87" → 87, or raise ValueError."""
    s = gpu_arch_str.strip().lower()
    if s.startswith("sm_"):
        s = s[3:]
    try:
        sm = int(s)
    except ValueError:
        raise ValueError(
            f"Invalid --gpu_arch {gpu_arch_str!r}. Expected format: sm_87, sm_100, etc."
        )
    if sm <= 0:
        raise ValueError(
            f"Invalid --gpu_arch {gpu_arch_str!r}: SM number must be positive (got {sm})."
        )
    return sm


def detect_gpu_sm() -> int:
    """Auto-detect the current GPU SM.

    Returns the SM as an integer, e.g. 87 for SM87, 100 for SM100, 110 for SM110.

    Detection order:
      1. cupy.cuda.Device — works on all platforms (Linux, QNX, etc.) since cupy is
         already a required dependency.  compute_capability returns e.g. "87", "100".
      2. nvidia-smi --query-gpu=compute_cap — fallback for environments where cupy
         is not yet importable at this point in the script (rare).

    Raises RuntimeError if both methods fail; caller should re-run with --gpu_arch.
    """
    # 1. Try cupy first — platform-agnostic, already a required dep.
    try:
        import cupy  # noqa: PLC0415
        cap = cupy.cuda.Device(0).compute_capability  # e.g. "87", "100", "110"
        sm = int(cap)
        if sm > 0:
            return sm
    except Exception:
        pass

    # 2. Fall back to nvidia-smi (Linux/x86; not available on QNX).
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Could not detect GPU SM: cupy unavailable and nvidia-smi not found. "
            "Pass --gpu_arch explicitly (e.g. --gpu_arch sm_87)."
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed: {result.stderr.strip() or result.stdout.strip()}. "
            "Pass --gpu_arch explicitly to override."
        )
    # compute_cap format from nvidia-smi is "8.7" → 87, "10.0" → 100.
    line = result.stdout.strip().splitlines()[0].strip()
    parts = line.split(".")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise RuntimeError(
            f"Unexpected nvidia-smi compute_cap format: {line!r}. "
            "Pass --gpu_arch explicitly to override."
        )
    return int(parts[0]) * 10 + int(parts[1])


def select_variants(sm: int, kernels_arg: str):
    """Return the list of KernelVariants to compile for the given SM.

    sm:
      Integer SM number (e.g. 87, 100, 110) — used for supported_sms filtering.

    kernels_arg:
      "ALL"         — compile variants whose supported_sms contains the SM.
      "gdn"/"fmha"  — compile all variants in that group; error if SM is incompatible.
      "gdn,fmha"    — compile all variants in the listed groups.
    """
    groups_requested = kernels_arg.strip().upper()

    if groups_requested == "ALL":
        selected = [v for v in KERNEL_VARIANTS if sm in v.supported_sms]
        if not selected:
            print(f"WARNING: No CuTe DSL variants support SM{sm}. "
                  f"Check supported_sms in KERNEL_VARIANTS.")
        return selected

    # Parse explicit group list: "fmha,gdn" or "gdn".
    tokens = [t.strip().lower() for t in kernels_arg.split(",")]
    unknown = [t for t in tokens if t not in _ALL_GROUPS]
    if unknown:
        raise ValueError(
            f"Unknown kernel group(s): {unknown}. "
            f"Valid groups: {sorted(_ALL_GROUPS)}"
        )

    selected = [v for v in KERNEL_VARIANTS if v.group in tokens]

    # Explicit group + incompatible SM → error (almost certainly a user mistake).
    unsupported = [v for v in selected if sm not in v.supported_sms]
    if unsupported:
        names = ", ".join(v.name for v in unsupported)
        raise ValueError(
            f"SM{sm} is not in supported_sms for: {names}.\n"
            f"Use --kernels ALL to auto-filter by SM, or check supported_sms in KERNEL_VARIANTS."
        )

    return selected


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def detect_arch(override=None):
    if override:
        m = override.lower().replace("-", "_")
        if m in ("x86_64", "amd64"):
            return "x86_64"
        if m in ("aarch64", "arm64"):
            return "aarch64"
        raise ValueError(f"Unsupported --arch: {override!r}. Use 'x86_64' or 'aarch64'.")
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "aarch64"
    raise RuntimeError(
        f"Unsupported architecture: {platform.machine()!r}. Use --arch to override."
    )


def _nvcc_version():
    """Return CUDA version string (e.g. "12.6.0") or None.

    Detection order:
      1. nvcc on PATH
      2. /usr/local/cuda/bin/nvcc  (common on Jetson / embedded devices)
      3. cupy.cuda.runtime          (works on QNX and any platform with cupy)
    """
    # 1 & 2: try nvcc
    for nvcc in ("nvcc", "/usr/local/cuda/bin/nvcc"):
        try:
            out = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT, text=True)
            for token in out.split():
                if token.startswith("V") and token[1:2].isdigit():
                    return token[1:].split(",")[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # 3: cupy runtime API — runtimeGetVersion() returns e.g. 12060 for 12.6.0
    try:
        import cupy  # noqa: PLC0415
        v = cupy.cuda.runtime.runtimeGetVersion()   # e.g. 12060
        major, rest = divmod(v, 1000)
        minor, patch = divmod(rest, 10)
        return f"{major}.{minor}.{patch}"
    except Exception:
        pass

    return None


def check_dependencies():
    errors = []

    # nvidia-cutlass-dsl
    try:
        ver = importlib.metadata.version("nvidia-cutlass-dsl")
        if ver != _CUTLASS_DSL_VERSION:
            errors.append(
                f"nvidia-cutlass-dsl: found {ver}, need {_CUTLASS_DSL_VERSION}\n"
                f"  Fix: pip install nvidia-cutlass-dsl=={_CUTLASS_DSL_VERSION}"
            )
            lib_dir = None
        else:
            spec = importlib.util.find_spec("nvidia_cutlass_dsl")
            pkg_dir = (
                Path(next(iter(spec.submodule_search_locations)))
                if spec.submodule_search_locations
                else Path(spec.origin).parent
            )
            lib_dir = pkg_dir / "lib"
    except importlib.metadata.PackageNotFoundError:
        errors.append(
            f"nvidia-cutlass-dsl not found.\n"
            f"  Fix: pip install nvidia-cutlass-dsl=={_CUTLASS_DSL_VERSION}"
        )
        lib_dir, ver = None, "unknown"

    # cupy
    cuda_ver = _nvcc_version()
    if cuda_ver:
        major = int(cuda_ver.split(".")[0])
        if major in _CUPY_VERSIONS:
            cupy_pkg, cupy_req = _CUPY_VERSIONS[major]
            try:
                found = importlib.metadata.version(cupy_pkg)
                if found != cupy_req:
                    errors.append(
                        f"cupy: found {cupy_pkg}=={found}, need {cupy_req}\n"
                        f"  Fix: pip install {cupy_pkg}=={cupy_req}"
                    )
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"cupy not found.\n  Fix: pip install {cupy_pkg}=={cupy_req}")
        else:
            errors.append(f"Unsupported CUDA major version {major} for cupy.")
    else:
        errors.append("Could not detect CUDA version (is nvcc on PATH?).")

    if not shutil.which("ar"):
        errors.append("'ar' not found on PATH. Install binutils.")

    if errors:
        print("Dependency check failed:\n" + "\n".join(f"  • {e}" for e in errors))
        sys.exit(1)

    assert ver is not None and lib_dir is not None and cuda_ver is not None
    print(f"  nvidia-cutlass-dsl=={ver} ✓  CUDA {cuda_ver} ✓  ar ✓")
    return ver, lib_dir, cuda_ver


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def _compile_one(variant, staging_dir, verbose):
    """Invoke a kernel script to AOT-compile one variant into .o + .h.

    Returns (name, ok, elapsed_secs, error_msg).
    """
    cmd = [sys.executable, str(_SCRIPT_DIR / variant.script)]
    cmd += ["--output_dir", str(staging_dir),
            "--file_name", variant.name,
            "--function_prefix", variant.name]
    cmd += variant.script_args

    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(_SCRIPT_DIR), capture_output=not verbose, text=True)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        # Show the head (traceback / first error) rather than the tail, which is typically more diagnostic.
        return variant.name, False, elapsed, (result.stderr or result.stdout or "")[:4000]
    obj = staging_dir / f"{variant.name}.o"
    hdr = staging_dir / f"{variant.name}.h"
    if not obj.exists() or not hdr.exists():
        return variant.name, False, elapsed, f"{obj.name} / {hdr.name} not found after successful exit"
    return variant.name, True, elapsed, ""


def compile_variants(variants, staging_dirs, jobs, verbose):
    """Compile all selected variants in parallel via a process pool.

    staging_dirs: dict mapping variant.name → Path of its dedicated staging dir.
    """
    print(f"\nCompiling {len(variants)} kernel variant(s) (jobs={jobs})...")
    failures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_compile_one, v, staging_dirs[v.name], verbose): v
            for v in variants
        }
        for future in concurrent.futures.as_completed(futures):
            name, ok, elapsed, msg = future.result()
            print(f"  {'✓' if ok else '✗'} {name:<25} ({elapsed:.1f}s)")
            if not ok:
                failures.append((name, msg))

    if failures:
        for name, msg in failures:
            print(f"\n  [{name}]\n{msg}")
        sys.exit(1)


def _check_obj_name_collision(kernel_objs, runtime_objs):
    kernel_names = {f.name for f in kernel_objs}
    runtime_names = {f.name for f in runtime_objs}
    collision = kernel_names & runtime_names
    if collision:
        raise RuntimeError(
            f"Object name collision between kernel and runtime archives: {collision}\n"
            "Rename the affected kernel variant(s) in KERNEL_VARIANTS to resolve."
        )


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------

def build(args):
    arch = detect_arch(args.arch)
    output_dir = Path(args.output_dir) / arch

    # Resolve SM: explicit override or auto-detect from the running GPU.
    if args.gpu_arch:
        sm = _parse_sm(args.gpu_arch)
        sm_source = "--gpu_arch override"
    else:
        sm = detect_gpu_sm()
        sm_source = "auto-detected"

    print(f"Target arch : {arch}")
    print(f"GPU SM      : SM{sm} ({sm_source})")
    print(f"Output dir  : {output_dir}")

    variants = select_variants(sm, args.kernels)
    if not variants:
        print("No variants selected — nothing to build.")
        return

    groups_selected = sorted({v.group for v in variants})
    print(f"Groups      : {groups_selected}")
    print(f"Variants    : {[v.name for v in variants]}")

    # Check dependencies before cleaning — so a failed dep check doesn't
    # silently destroy a previously good build.
    print("\nChecking dependencies...")
    dsl_ver, lib_dir, cuda_ver = check_dependencies()

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    # Per-variant staging dirs prevent .o / .h filename collisions when multiple
    # variants share the same underlying script (e.g. all fmha.py invocations).
    root_staging = Path(tempfile.mkdtemp(prefix="cutedsl_build_"))
    try:
        staging_dirs = {}
        for v in variants:
            d = root_staging / v.name
            d.mkdir()
            staging_dirs[v.name] = d

        compile_variants(variants, staging_dirs, args.jobs, args.verbose)

        # Collect all .o files from per-variant staging dirs.
        kernel_obj_files = [staging_dirs[v.name] / f"{v.name}.o" for v in variants]

        # Extract the DSL runtime static lib into a separate dir to avoid collisions.
        runtime = lib_dir / "libcuda_dialect_runtime_static.a"
        if not runtime.exists():
            raise FileNotFoundError(f"{runtime} not found. Verify nvidia-cutlass-dsl installation.")
        runtime_obj_dir = root_staging / "runtime_objs"
        runtime_obj_dir.mkdir()
        subprocess.run(["ar", "x", str(runtime)], cwd=str(runtime_obj_dir), check=True)
        runtime_objs = sorted(runtime_obj_dir.glob("*.o"))

        _check_obj_name_collision(kernel_obj_files, runtime_objs)

        # Pack everything into one archive.
        output_dir.mkdir(parents=True, exist_ok=True)
        lib_path = output_dir / f"libcutedsl_{arch}.a"
        subprocess.run(
            ["ar", "rcs", str(lib_path)]
            + [str(o) for o in kernel_obj_files]
            + [str(o) for o in runtime_objs],
            check=True,
        )
        print(f"\n  Created {lib_path.name} ({lib_path.stat().st_size // 1024} KB)")

        # Copy per-variant headers and write umbrella header.
        inc_dir = output_dir / "include"
        inc_dir.mkdir(exist_ok=True)
        for v in variants:
            shutil.copy2(staging_dirs[v.name] / f"{v.name}.h", inc_dir)
        umbrella = inc_dir / "cutedsl_all.h"
        umbrella.write_text(
            "#pragma once\n"
            "// Auto-generated by build_cutedsl.py -- do not edit\n"
            + "".join(f'#include "{v.name}.h"\n' for v in variants)
        )

        # Write build provenance + metadata for cmake consumption.
        (output_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "arch": arch,
                    "gpu_arch": f"sm_{sm}",
                    "cuda_version": cuda_ver,
                    "cutlass_dsl_version": dsl_ver,
                    "build_date": datetime.now(timezone.utc).isoformat(),
                    "groups": groups_selected,
                    "variants": [v.name for v in variants],
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        shutil.rmtree(root_staging, ignore_errors=True)

    print(f"\nDone. Artifacts written to: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--gpu_arch",
        default=None,
        help="Override target GPU SM (e.g. sm_87, sm_100). "
             "Default: auto-detect via cupy / nvidia-smi. "
             "Used only for variant filtering — never forwarded to kernel scripts.",
    )
    p.add_argument(
        "--kernels",
        default="ALL",
        help="Which kernels to build: ALL (default), a group name (fmha | gdn), "
             "or a comma-separated list of group names (fmha,gdn).",
    )
    p.add_argument(
        "--output_dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help=f"Root output dir (artifacts go into {{output_dir}}/{{arch}}/). "
             f"Default: {_DEFAULT_OUTPUT_DIR}",
    )
    p.add_argument(
        "--arch",
        default=None,
        help="Host/target CPU architecture: x86_64 or aarch64 (default: auto-detected).",
    )
    p.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Parallel compile jobs (use -j 1 if GPU memory is limited). Default: 4.",
    )
    p.add_argument("--verbose", action="store_true", help="Show per-variant kernel script output.")
    p.add_argument("--clean", action="store_true", help="Remove output arch dir before building.")
    build(p.parse_args())


if __name__ == "__main__":
    main()
