# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# ---------------------------------------------------------------------------
# CuTe DSL unified kernel library
#
# Prebuilt artifacts are generated offline by: python
# kernelSrcs/build_cutedsl.py --gpu_arch <sm_NN> and committed to the repository
# under: cpp/kernels/cuteDSLArtifact/{arch}/
#
# No Python, CUTLASS DSL, or GPU is needed at CMake build time.
#
# ENABLE_CUTE_DSL cache variable controls which kernel groups are linked: OFF —
# disable entirely (default) ALL              — enable all groups found in
# metadata.json fmha             — enable only the FMHA group gdn              —
# enable only the GDN group fmha;gdn         — semicolon-separated list of
# groups (CMake list syntax)
#
# Usage: include(cmake/CuteDsl.cmake) cute_dsl_setup( TARGETS      target1
# target2 ...   # compile definitions + include path only LINK_TARGETS target3
# target4 ...   # compile definitions + include path + link )
#
# Per-group compile definitions set on each target: CUTE_DSL_FMHA_ENABLED  — set
# when the fmha group is active CUTE_DSL_GDN_ENABLED   — set when the gdn group
# is active
# ---------------------------------------------------------------------------

set(ENABLE_CUTE_DSL
    "OFF"
    CACHE
      STRING
      "CuTe DSL kernels: OFF, ALL, or semicolon-separated group list (fmha;gdn)"
)

# Include guard — safe to include from multiple CMakeLists.txt directories.
if(DEFINED _CUTE_DSL_CMAKE_INCLUDED)
  return()
endif()
set(_CUTE_DSL_CMAKE_INCLUDED TRUE)

# ---------------------------------------------------------------------------
# cute_dsl_setup()
#
# TARGETS      — targets that need compile definitions + include path
# LINK_TARGETS — targets that additionally link libcutedsl_<arch>.a
# ---------------------------------------------------------------------------
function(cute_dsl_setup)
  cmake_parse_arguments(ARG "" "" "TARGETS;LINK_TARGETS" ${ARGN})

  string(TOUPPER "${ENABLE_CUTE_DSL}" _cute_dsl_norm)

  if(_cute_dsl_norm STREQUAL "OFF")
    return()
  endif()

  # Guard against accidental empty-string assignment (e.g. -DENABLE_CUTE_DSL=).
  if(_cute_dsl_norm STREQUAL "")
    message(
      FATAL_ERROR
        "ENABLE_CUTE_DSL is set to an empty string.\n"
        "Set it to OFF, ALL, or a semicolon-separated group list (e.g. fmha;gdn)."
    )
  endif()

  # Detect host/target CPU architecture.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_arch "aarch64")
  else()
    set(_arch "x86_64")
  endif()

  # Unified artifact directory (produced by kernelSrcs/build_cutedsl.py).
  set(_artifact_dir "${CMAKE_SOURCE_DIR}/cpp/kernels/cuteDSLArtifact/${_arch}")
  set(_static_lib "${_artifact_dir}/libcutedsl_${_arch}.a")
  set(_inc_dir "${_artifact_dir}/include")
  set(_metadata "${_artifact_dir}/metadata.json")

  # Validate artifacts exist.
  if(NOT EXISTS "${_static_lib}")
    message(
      FATAL_ERROR
        "Prebuilt CuTe DSL library not found:\n"
        "  ${_static_lib}\n"
        "Generate it with:\n"
        "  python kernelSrcs/build_cutedsl.py --gpu_arch <sm_NN> --arch ${_arch}\n"
        "then commit the resulting ${_arch}/ directory under "
        "cpp/kernels/cuteDSLArtifact/.")
  endif()

  if(NOT EXISTS "${_metadata}")
    message(FATAL_ERROR "metadata.json not found in ${_artifact_dir}/\n"
                        "Re-run build_cutedsl.py to regenerate artifacts.")
  endif()

  if(NOT EXISTS "${_inc_dir}/cutedsl_all.h")
    message(
      FATAL_ERROR "Umbrella header cutedsl_all.h not found in ${_inc_dir}/\n"
                  "Re-run build_cutedsl.py to regenerate artifacts.")
  endif()

  # Parse the "groups" array from metadata.json. metadata.json example: {
  # "groups": ["gdn", "fmha"], "variants": [...] } Requires CMake >= 3.19 for
  # string(JSON ...).
  file(READ "${_metadata}" _meta_json)
  string(JSON _n_groups LENGTH "${_meta_json}" "groups")

  if(_n_groups EQUAL 0)
    message(
      WARNING
        "CuTe DSL: metadata.json has empty 'groups' array in ${_artifact_dir}/. "
        "Re-run build_cutedsl.py to regenerate artifacts.")
    return()
  endif()

  math(EXPR _last_idx "${_n_groups} - 1")

  # Determine which groups to activate based on ENABLE_CUTE_DSL.
  set(_active_groups)
  foreach(_i RANGE ${_last_idx})
    string(JSON _g GET "${_meta_json}" "groups" ${_i})
    if(_cute_dsl_norm STREQUAL "ALL")
      list(APPEND _active_groups "${_g}")
    else()
      # _cute_dsl_norm is a semicolon-separated list (CMake list), e.g.
      # "FMHA;GDN".
      string(TOUPPER "${_g}" _g_upper)
      if("${_g_upper}" IN_LIST _cute_dsl_norm)
        list(APPEND _active_groups "${_g}")
      endif()
    endif()
  endforeach()

  if(NOT _active_groups)
    message(
      WARNING
        "CuTe DSL: ENABLE_CUTE_DSL='${ENABLE_CUTE_DSL}' matched no groups in "
        "${_metadata} (available: ${_meta_json}). Nothing will be linked.")
    return()
  endif()

  # Shim / --wrap branches follow the toolkit version the project uses:
  # CUDA_CTK_VERSION (see root CMakeLists.txt). Fall back to
  # CMAKE_CUDA_COMPILER_VERSION only if CTK is unset.
  if(DEFINED CUDA_CTK_VERSION AND NOT CUDA_CTK_VERSION STREQUAL "")
    set(_cute_dsl_cuda_ver "${CUDA_CTK_VERSION}")
  elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION
         AND NOT CMAKE_CUDA_COMPILER_VERSION STREQUAL "")
    set(_cute_dsl_cuda_ver "${CMAKE_CUDA_COMPILER_VERSION}")
  else()
    set(_cute_dsl_cuda_ver "")
  endif()

  if(NOT _cute_dsl_cuda_ver STREQUAL "" AND _cute_dsl_cuda_ver VERSION_LESS
                                            12.0)
    message(
      FATAL_ERROR
        "CuTe DSL requires CUDA Toolkit 12.0+ (detected ${_cute_dsl_cuda_ver}). "
        "Use -DENABLE_CUTE_DSL=OFF or set -DCUDA_CTK_VERSION to a supported toolkit."
    )
  endif()

  # Shim: cudaLibrary* → cu* when libcudart omits exports (e.g. some 12.0–12.6
  # embedded). From CUDA 12.8 onward, cuda_runtime_api.h declares these APIs
  # with runtime types; compiling the weak shim conflicts with those
  # declarations. Use INTERFACE only (no .c) for 12.8+.
  set(_cutedsl_cudart_shim_src
      "${CMAKE_SOURCE_DIR}/cpp/kernels/gdnKernels/cutedsl_cuda_runtime_library_shim.c"
  )
  if(NOT TARGET trt_edgellm_cutedsl_cudart_shim)
    if(NOT _cute_dsl_cuda_ver STREQUAL "" AND _cute_dsl_cuda_ver
                                              VERSION_GREATER_EQUAL 12.8)
      add_library(trt_edgellm_cutedsl_cudart_shim INTERFACE)
    else()
      if(NOT EXISTS "${_cutedsl_cudart_shim_src}")
        message(
          FATAL_ERROR
            "CuTe DSL libcudart shim source missing:\n  ${_cutedsl_cudart_shim_src}\n"
            "It must be committed with the repository (not generated).")
      endif()
      add_library(trt_edgellm_cutedsl_cudart_shim STATIC
                  "${_cutedsl_cudart_shim_src}")
      target_include_directories(trt_edgellm_cutedsl_cudart_shim
                                 PRIVATE ${CUDA_INCLUDE_DIR})
      # 12.0–12.6: AOT uses cudaKernel_t; shim needs
      # CUTEDSL_WRAP_LAUNCH_KERNEL_EX.
      if(NOT _cute_dsl_cuda_ver STREQUAL ""
         AND _cute_dsl_cuda_ver VERSION_GREATER_EQUAL 12.0
         AND _cute_dsl_cuda_ver VERSION_LESS 12.8)
        target_compile_definitions(trt_edgellm_cutedsl_cudart_shim
                                   PRIVATE CUTEDSL_WRAP_LAUNCH_KERNEL_EX)
      endif()
    endif()
  endif()

  # Parse the "variants" array from metadata.json to determine which kernel
  # variants are present (used for fine-grained per-variant compile defines).
  set(_variants)
  string(JSON _n_variants ERROR_VARIABLE _json_err LENGTH "${_meta_json}"
                                                          "variants")
  if(NOT _json_err AND NOT _n_variants EQUAL 0)
    math(EXPR _last_var_idx "${_n_variants} - 1")
    foreach(_vi RANGE ${_last_var_idx})
      string(
        JSON
        _vname
        ERROR_VARIABLE
        _verr
        GET
        "${_meta_json}"
        "variants"
        ${_vi})
      if(NOT _verr AND _vname)
        list(APPEND _variants "${_vname}")
      endif()
    endforeach()
  endif()

  # Apply compile definitions and include path to all targets.
  foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
    target_include_directories(${_tgt} SYSTEM PRIVATE "${_inc_dir}")
    foreach(_g ${_active_groups})
      string(TOUPPER "${_g}" _gu)
      target_compile_definitions(${_tgt} PRIVATE "CUTE_DSL_${_gu}_ENABLED")
    endforeach()
  endforeach()

  # Check for Blackwell GDN variant specifically and set a clean define.
  list(FIND _variants "gdn_prefill_blackwell" _bw_idx)
  if(NOT ${_bw_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt}
                                 PRIVATE "CUTE_DSL_GDN_BLACKWELL_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: Blackwell GDN prefill variant found — CUTE_DSL_GDN_BLACKWELL_ENABLED set"
    )
  endif()

  # Check for Blackwell SSD variant and set a clean define.
  list(FIND _variants "ssd_prefill_blackwell_d64_n128" _ssd_bw_idx)
  if(NOT ${_ssd_bw_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt}
                                 PRIVATE "CUTE_DSL_SSD_BLACKWELL_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: Blackwell SSD prefill variant found — CUTE_DSL_SSD_BLACKWELL_ENABLED set"
    )
  endif()

  # Link libcuda again after the .a (--as-needed can drop an earlier libcuda).
  foreach(_tgt ${ARG_LINK_TARGETS})
    target_link_libraries(${_tgt} PRIVATE "${_static_lib}"
                                          trt_edgellm_cutedsl_cudart_shim)
    if(CUDA_DRIVER_LIB AND NOT CUDA_DRIVER_LIB MATCHES "-NOTFOUND$")
      target_link_libraries(${_tgt} PRIVATE "${CUDA_DRIVER_LIB}")
    endif()
    # CUDA < 12.8: wrap _cudaLaunchKernelEx (cudaKernel_t → CUfunction, e.g.
    # JetPack 6).
    if(NOT _cute_dsl_cuda_ver STREQUAL "" AND _cute_dsl_cuda_ver VERSION_LESS
                                              12.8)
      target_link_options(${_tgt} PRIVATE "-Wl,--wrap=_cudaLaunchKernelEx")
    endif()
  endforeach()

  message(
    STATUS
      "CuTe DSL: arch=${_arch}  groups=[${_active_groups}]  lib=${_static_lib}")
endfunction()
