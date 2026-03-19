# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# CuTe DSL FMHA: compile kernelSrcs/fmha_cutedsl_blackwell/fmha.py during build
# to generate fmha.h / fmha.o for the current GPU architecture.
#
# Usage: include(cmake/CuteDslFMHA.cmake) then call cute_dsl_fmha_setup(TARGETS
# target1 target2 ... PLUGIN_TARGET plugin_target)

# ---------------------------------------------------------------------------
# cute_dsl_fmha_setup()
#
# Generates FMHA kernel objects at build time and wires them into the given
# targets.
#
# TARGETS       — libraries that compile cuteDslFMHARunner.cpp (get
# CUTE_DSL_FMHA_ENABLED define + dependency on codegen) PLUGIN_TARGET — the
# shared-library target that links the .o files + cute_dsl_runtime
# ---------------------------------------------------------------------------
function(cute_dsl_fmha_setup)
  cmake_parse_arguments(ARG "" "PLUGIN_TARGET" "TARGETS" ${ARGN})

  # Prefer an activated virtual environment ($VIRTUAL_ENV or $CONDA_PREFIX).
  set(Python3_FIND_VIRTUALENV FIRST)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  # ---------- dependency management -----------------------------------------
  # Must run first: ensures packages are installed and propagates
  # CUTE_DSL_PYTHON and CUTLASS_DSL_LIB_DIR into this scope.
  _cute_dsl_fmha_ensure_dependencies()

  set(CUTE_DSL_FMHA_OUTPUT_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/contextAttentionKernels/cuteDSLArtifact"
  )

  # Remove stale artifacts before (re-)generating so platform-specific binaries
  # from a previous build (e.g. different arch or CUDA version) are not mixed
  # in.
  message(
    STATUS "Removing stale CuTe DSL artifacts at ${CUTE_DSL_FMHA_OUTPUT_DIR}")
  file(REMOVE_RECURSE "${CUTE_DSL_FMHA_OUTPUT_DIR}")
  file(MAKE_DIRECTORY "${CUTE_DSL_FMHA_OUTPUT_DIR}")

  set(CUTE_DSL_FMHA_SCRIPT
      "${CMAKE_SOURCE_DIR}/kernelSrcs/fmha_cutedsl_blackwell/fmha.py")
  set(CUTE_DSL_FMHA_DEPENDS
      ${CUTE_DSL_FMHA_SCRIPT}
      ${CMAKE_SOURCE_DIR}/kernelSrcs/fmha_cutedsl_blackwell/fmha_helpers.py)
  set(CUTE_DSL_FMHA_LLM_FLAGS
      --is_causal --is_persistent --export_only --bottom_right_align
      --output_dir ${CUTE_DSL_FMHA_OUTPUT_DIR})
  set(CUTE_DSL_FMHA_VIT_FLAGS --is_persistent --export_only --vit_mode
                              --output_dir ${CUTE_DSL_FMHA_OUTPUT_DIR})

  # ---------- kernel variant definitions ------------------------------------
  set(_ALL_ARTIFACTS "")

  # Helper macro to avoid repeating the add_custom_command boilerplate for each
  # head-dim / sliding-window variant.
  macro(_add_fmha_variant NAME Q_SHAPE K_SHAPE BASE_FLAGS)
    set(_OBJ "${CUTE_DSL_FMHA_OUTPUT_DIR}/${NAME}.o")
    set(_HDR "${CUTE_DSL_FMHA_OUTPUT_DIR}/${NAME}.h")
    list(APPEND _ALL_ARTIFACTS ${_OBJ} ${_HDR})

    # Collect any extra flags (e.g. --window_size) passed after the four
    # positional arguments.
    set(_EXTRA_FLAGS ${ARGN})

    add_custom_command(
      OUTPUT ${_OBJ} ${_HDR}
      COMMAND ${CMAKE_COMMAND} -E remove -f ${_OBJ} ${_HDR}
      COMMAND
        ${CUTE_DSL_PYTHON} ${CUTE_DSL_FMHA_SCRIPT} ${${BASE_FLAGS}} --q_shape
        ${Q_SHAPE} --k_shape ${K_SHAPE} --file_name ${NAME} --function_prefix
        ${NAME} ${_EXTRA_FLAGS}
      DEPENDS ${CUTE_DSL_FMHA_DEPENDS}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/kernelSrcs/fmha_cutedsl_blackwell
      COMMENT "Compiling CuTe DSL FMHA kernel (${NAME})...")

    # Store OBJ path for linking (macro shares caller's scope)
    set(CUTE_DSL_FMHA_${NAME}_OBJ ${_OBJ})
  endmacro()

  # LLM variants: causal, bottom-right-aligned, with KV cache
  _add_fmha_variant(fmha_d64 1,1024,14,64 1,1024,1,64 CUTE_DSL_FMHA_LLM_FLAGS)
  _add_fmha_variant(fmha_d128 1,1024,14,128 1,1024,1,128
                    CUTE_DSL_FMHA_LLM_FLAGS)
  _add_fmha_variant(fmha_d64_sw 1,1024,14,64 1,1024,1,64
                    CUTE_DSL_FMHA_LLM_FLAGS --window_size 4096,-1)
  _add_fmha_variant(fmha_d128_sw 1,1024,14,128 1,1024,1,128
                    CUTE_DSL_FMHA_LLM_FLAGS --window_size 4096,-1)

  # ViT variants: non-causal, packed varlen, separate Q/K/V
  _add_fmha_variant(vit_fmha_d64 1,1024,14,64 1,1024,14,64
                    CUTE_DSL_FMHA_VIT_FLAGS)
  _add_fmha_variant(vit_fmha_d72 1,1024,14,72 1,1024,14,72
                    CUTE_DSL_FMHA_VIT_FLAGS)
  _add_fmha_variant(vit_fmha_d80 1,1024,14,80 1,1024,14,80
                    CUTE_DSL_FMHA_VIT_FLAGS)
  _add_fmha_variant(vit_fmha_d128 1,1024,14,128 1,1024,14,128
                    CUTE_DSL_FMHA_VIT_FLAGS)

  add_custom_target(cute_dsl_fmha_gen ALL DEPENDS ${_ALL_ARTIFACTS})

  # ---------- find cute_dsl_runtime library ---------------------------------
  find_library(CUDA_DIALECT_RUNTIME_LIB cute_dsl_runtime
               HINTS ${CUTLASS_DSL_LIB_DIR})

  if(NOT CUDA_DIALECT_RUNTIME_LIB)
    message(
      FATAL_ERROR
        "cute_dsl_runtime library not found in ${CUTLASS_DSL_LIB_DIR}. "
        "Please verify nvidia-cutlass-dsl installation.")
  else()
    message(STATUS "Found cute_dsl_runtime: ${CUDA_DIALECT_RUNTIME_LIB}")
  endif()

  # ---------- wire into build targets ---------------------------------------
  # All TARGETS get the compile definition + dependency on codegen.
  foreach(_tgt ${ARG_TARGETS})
    add_dependencies(${_tgt} cute_dsl_fmha_gen)
    target_compile_definitions(${_tgt} PRIVATE CUTE_DSL_FMHA_ENABLED)
  endforeach()

  # PLUGIN_TARGET additionally links the generated .o files + runtime lib.
  if(ARG_PLUGIN_TARGET)
    add_dependencies(${ARG_PLUGIN_TARGET} cute_dsl_fmha_gen)
    target_compile_definitions(${ARG_PLUGIN_TARGET}
                               PRIVATE CUTE_DSL_FMHA_ENABLED)
    target_link_libraries(
      ${ARG_PLUGIN_TARGET}
      ${CUTE_DSL_FMHA_fmha_d64_OBJ}
      ${CUTE_DSL_FMHA_fmha_d128_OBJ}
      ${CUTE_DSL_FMHA_fmha_d64_sw_OBJ}
      ${CUTE_DSL_FMHA_fmha_d128_sw_OBJ}
      ${CUTE_DSL_FMHA_vit_fmha_d64_OBJ}
      ${CUTE_DSL_FMHA_vit_fmha_d72_OBJ}
      ${CUTE_DSL_FMHA_vit_fmha_d80_OBJ}
      ${CUTE_DSL_FMHA_vit_fmha_d128_OBJ}
      ${CUDA_DIALECT_RUNTIME_LIB})
  endif()

  message(
    STATUS "CuTe DSL FMHA: ENABLED (artifacts will be generated at build time)")
endfunction()

# ---------------------------------------------------------------------------
# _cute_dsl_fmha_ensure_dependencies  (internal)
#
# Ensures the pinned versions of cupy and nvidia-cutlass-dsl are installed,
# respecting an activated virtual environment if present, then propagates
# CUTE_DSL_PYTHON and CUTLASS_DSL_LIB_DIR to parent scope.
# ---------------------------------------------------------------------------
function(_cute_dsl_fmha_ensure_dependencies)
  set(CUTLASS_DSL_REQUIRED_VERSION "4.4.1")
  set(_python ${Python3_EXECUTABLE})

  string(REGEX MATCH "^([0-9]+)" _cuda_major_ver "${CUDA_CTK_VERSION}")
  if(_cuda_major_ver EQUAL 12)
    set(_cupy_package "cupy-cuda12x==12.3.0")
    set(_cupy_required_version "12.3.0")
  elseif(_cuda_major_ver EQUAL 13)
    set(_cupy_package "cupy-cuda13x==13.6.0")
    set(_cupy_required_version "13.6.0")
  else()
    message(
      FATAL_ERROR "Unsupported CUDA major version for cupy: ${_cuda_major_ver}")
  endif()

  # -- cupy ------------------------------------------------------------------
  execute_process(
    COMMAND ${_python} -c "import cupy; print(cupy.__version__)"
    OUTPUT_VARIABLE _cupy_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _cupy_result
    ERROR_QUIET)

  set(_cupy_ok FALSE)
  if(_cupy_result EQUAL 0 AND _cupy_version VERSION_EQUAL
                              _cupy_required_version)
    set(_cupy_ok TRUE)
    message(
      STATUS "Found cupy==${_cupy_version} (for CuTe DSL kernel compilation)")
  elseif(_cupy_result EQUAL 0)
    message(
      STATUS
        "cupy version mismatch: found ${_cupy_version}, need ${_cupy_required_version}"
    )
  else()
    message(STATUS "cupy not found; will install")
  endif()

  # -- nvidia-cutlass-dsl ----------------------------------------------------
  set(_CUTLASS_DSL_FIND_CMD
      "import importlib.util, os, sys; spec = importlib.util.find_spec('nvidia_cutlass_dsl'); sys.exit(1) if spec is None else None; p = spec.submodule_search_locations[0] if spec.submodule_search_locations else os.path.dirname(spec.origin); print(os.path.join(p, 'lib'))"
  )
  set(_CUTLASS_DSL_VERSION_CMD
      "from importlib.metadata import version; v = version('nvidia-cutlass-dsl'); print(v)"
  )

  execute_process(
    COMMAND ${_python} -c "${_CUTLASS_DSL_FIND_CMD}"
    OUTPUT_VARIABLE _lib_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _cutlass_dsl_result
    ERROR_QUIET)

  set(_cutlass_dsl_version_ok FALSE)
  if(_cutlass_dsl_result EQUAL 0)
    execute_process(
      COMMAND ${_python} -c "${_CUTLASS_DSL_VERSION_CMD}"
      OUTPUT_VARIABLE _cutlass_dsl_version
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _ver_result
      ERROR_QUIET)
    if(_ver_result EQUAL 0 AND _cutlass_dsl_version VERSION_EQUAL
                               ${CUTLASS_DSL_REQUIRED_VERSION})
      set(_cutlass_dsl_version_ok TRUE)
      message(
        STATUS
          "Found nvidia-cutlass-dsl==${_cutlass_dsl_version} (for CuTe DSL kernel compilation)"
      )
    else()
      message(
        STATUS
          "nvidia-cutlass-dsl version mismatch: found ${_cutlass_dsl_version}, need ${CUTLASS_DSL_REQUIRED_VERSION}"
      )
    endif()
  endif()

  # Install / upgrade if either package is missing or wrong version.
  if(NOT _cutlass_dsl_version_ok OR NOT _cupy_ok)
    message(
      STATUS
        "Installing nvidia-cutlass-dsl==${CUTLASS_DSL_REQUIRED_VERSION} and ${_cupy_package} ..."
    )
    # Warn if installing into the system Python (no active venv).
    if(NOT DEFINED ENV{VIRTUAL_ENV} AND NOT DEFINED ENV{CONDA_PREFIX})
      message(
        WARNING
          "No active virtual environment detected ($VIRTUAL_ENV / $CONDA_PREFIX). "
          "Installing CuTe DSL dependencies into the system Python. "
          "Consider activating a venv to avoid polluting the system Python.")
    endif()
    execute_process(
      COMMAND
        ${_python} -m pip install --break-system-packages
        nvidia-cutlass-dsl==${CUTLASS_DSL_REQUIRED_VERSION} ${_cupy_package}
      RESULT_VARIABLE _pip_install_result)
    if(NOT _pip_install_result EQUAL 0)
      message(
        FATAL_ERROR
          "Failed to install nvidia-cutlass-dsl==${CUTLASS_DSL_REQUIRED_VERSION} "
          "and ${_cupy_package}.")
    endif()
    execute_process(
      COMMAND ${_python} -c "${_CUTLASS_DSL_FIND_CMD}"
      OUTPUT_VARIABLE _lib_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _cutlass_dsl_result)
    if(NOT _cutlass_dsl_result EQUAL 0)
      message(
        FATAL_ERROR "nvidia-cutlass-dsl installed but cannot be imported.")
    endif()
  endif()

  # Propagate to parent scope
  set(CUTE_DSL_PYTHON
      ${_python}
      PARENT_SCOPE)
  set(CUTLASS_DSL_LIB_DIR
      ${_lib_dir}
      PARENT_SCOPE)
endfunction()
