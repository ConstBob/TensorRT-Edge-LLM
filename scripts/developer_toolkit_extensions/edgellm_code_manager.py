# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Self-contained EdgeLLM CodeManager adapter."""

from __future__ import annotations

import dataclasses
import shlex
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from trt_dev_toolkit.code_manager import models as code_manager_models
from trt_dev_toolkit.code_manager.generators.artifact_generator import \
    SourceBuildArtifactGeneratorBase
from trt_dev_toolkit.code_manager.generators.artifact_generator_trt import \
    normalize_trt_container_branch
from trt_dev_toolkit.code_manager.generators.shared_build_helpers import \
    source_build_parallel_config
from trt_dev_toolkit.command_manager import command_manager
from trt_dev_toolkit.command_manager.data_structures import (command_spec,
                                                             output_mode)
from trt_dev_toolkit.command_manager.targets import target as command_target
from trt_dev_toolkit.constants import Arch
from trt_dev_toolkit.container_manager import container_manager
from trt_dev_toolkit.container_manager import types as container_types
from trt_dev_toolkit.container_manager.backends import (docker_backend,
                                                        git_trt_runc_backend)

from . import gpu
from .errors import OrchestrationError

EDGELLM_COMPONENT = "edgellm"
_CUTE_DSL_IMAGE = "tensorrt-edge-llm/cutedsl-kernel-builder:4.6.1"


class EdgeLlmArtifactGenerator(SourceBuildArtifactGeneratorBase):
    """EdgeLLM source-build generator with CuTeDSL artifact preparation."""

    component_id = EDGELLM_COMPONENT
    component_display_name = "EdgeLLM"

    def __init__(
        self,
        *,
        container_manager: container_manager.ContainerManager,
        default_exec_target: command_target.Target | None,
        gpu_selection: gpu.GPUSelection | None,
        validate_container: bool,
        keep_containers_running: bool,
    ) -> None:
        """Initialize the EdgeLLM artifact generator.

        Args:
            container_manager: DevToolkit container manager for build containers.
            default_exec_target: Local or remote build target.
            gpu_selection: Optional selected GPU used for build and CuTeDSL.
            validate_container: Whether reused containers are validated.
            keep_containers_running: Whether build containers remain running.

        Returns:
            None.
        """
        super().__init__(
            container_manager=container_manager,
            high_core_selector=None,
            default_exec_target=default_exec_target,
            validate_container=validate_container,
            keep_containers_running=keep_containers_running,
        )
        self._command_manager = command_manager.CommandManager()
        self._gpu_selection = gpu_selection

    @classmethod
    def register_with_code_manager(
        cls,
        code_manager: Any,
        *,
        container_manager: container_manager.ContainerManager,
        default_exec_target: command_target.Target | None,
        gpu_selection: gpu.GPUSelection | None,
        validate_container: bool = True,
        keep_containers_running: bool = False,
    ) -> None:
        """Register EdgeLLM source/pre-built artifact support with CodeManager.

        Args:
            code_manager: CodeManager instance to extend.
            container_manager: DevToolkit container manager for builds.
            default_exec_target: Local or remote build target.
            gpu_selection: Optional selected GPU used for build and CuTeDSL.
            validate_container: Whether reused containers are validated.
            keep_containers_running: Whether build containers remain running.

        Returns:
            None.
        """
        code_manager.register_component(
            edgellm_component_spec(),
            generator=cls(
                container_manager=container_manager,
                default_exec_target=default_exec_target,
                gpu_selection=gpu_selection,
                validate_container=validate_container,
                keep_containers_running=keep_containers_running,
            ),
            dependency_rules=edgellm_dependency_rules(),
        )

    def required_host_commands_for_build(self) -> tuple[str, ...]:
        """Return host commands required by EdgeLLM builds.

        Returns:
            Host tools required for source build and optional CuTeDSL image prep.
        """
        return ("docker", "git", "bash", "cmake", "make")

    def required_container_tools_for_build(self) -> tuple[str, ...]:
        """Return tools that must exist in EdgeLLM build containers.

        Returns:
            Container tools required for EdgeLLM CMake builds.
        """
        return ("bash", "cmake", "make")

    def generate_artifact(
        self,
        target: code_manager_models.ArtifactTarget,
        deps: Mapping[str, code_manager_models.ArtifactResult],
    ) -> code_manager_models.ArtifactResult:
        """Generate an EdgeLLM artifact.

        Args:
            target: EdgeLLM artifact request.
            deps: Dependency artifacts produced earlier in the CodeManager plan.

        Returns:
            DevToolkit artifact result.
        """
        logger = self._logger.bind(
            arch=str(
                getattr(target.platform.arch, "value", target.platform.arch)),
            mode=str(getattr(target.mode, "value", target.mode)),
            source=str(getattr(target.source, "value", target.source)),
        )
        source = code_manager_models.parse_artifact_source(target.source)
        if source is code_manager_models.ArtifactSource.BUILD:
            return self.generate_source_artifact(target, deps, logger=logger)
        if source is code_manager_models.ArtifactSource.PRE_BUILT:
            return self.generate_pre_built_artifact(target,
                                                    deps,
                                                    logger=logger)
        return code_manager_models.ArtifactResult(
            success=False,
            component=self.component_id,
            log_file=target.build.log_file,
            error_messages=[
                f"Unsupported source '{source.value}' for EdgeLLM generator"
            ],
        )

    def _prepare_target(
        self,
        target: code_manager_models.ArtifactTarget,
        deps: Mapping[str, code_manager_models.ArtifactResult],
    ) -> code_manager_models.ArtifactTarget:
        """Apply dependency paths and CuTeDSL-generated CMake values.

        Args:
            target: Source-build target.
            deps: Previously generated dependency artifacts.

        Returns:
            Prepared target used by the shared source-build flow.
        """
        prepared = target
        if not prepared.build.trt_package_dir:
            trt_artifact = deps.get("trt_artifact")
            if trt_artifact is not None and trt_artifact.output_dir:
                build = dataclasses.replace(
                    prepared.build, trt_package_dir=trt_artifact.output_dir)
                prepared = dataclasses.replace(prepared, build=build)
        return self._prepare_cute_dsl(prepared)

    def _validate_target(
        self,
        target: code_manager_models.ArtifactTarget,
        deps: Mapping[str, code_manager_models.ArtifactResult],
    ) -> None:
        """Validate prepared EdgeLLM target state.

        Args:
            target: Prepared target.
            deps: Previously generated dependency artifacts.

        Raises:
            ValueError: If no TensorRT package is available.
        """
        _ = deps
        if not target.build.trt_package_dir:
            raise ValueError(
                "EdgeLLM build requires target.build.trt_package_dir or an upstream TRT artifact"
            )

    def _start_build_message_suffix(
        self,
        target: code_manager_models.ArtifactTarget,
        deps: Mapping[str, code_manager_models.ArtifactResult],
    ) -> str:
        """Append TensorRT package information to the build-start log."""
        _ = deps
        return (f" trt_package_dir={target.build.trt_package_dir}"
                if target.build.trt_package_dir else "")

    def _additional_metadata(
        self,
        target: code_manager_models.ArtifactTarget,
        deps: Mapping[str, code_manager_models.ArtifactResult],
    ) -> dict[str, str]:
        """Expose resolved TensorRT package path in artifact metadata."""
        _ = deps
        return {"trt_package_dir": target.build.trt_package_dir or ""}

    def _resolve_build_dir(
        self,
        target: code_manager_models.ArtifactTarget,
    ) -> str:
        """Resolve the EdgeLLM build output directory."""
        if target.build.build_dir:
            return target.build.build_dir
        arch = str(getattr(target.platform.arch, "value",
                           target.platform.arch))
        mode = str(getattr(target.mode, "value", target.mode))
        return str(Path.cwd() / "build" / f"edgellm_{arch}_{mode}")

    def _default_container_name(
        self,
        target: code_manager_models.ArtifactTarget,
    ) -> str:
        """Return the default EdgeLLM build-container name."""
        arch = str(getattr(target.platform.arch, "value",
                           target.platform.arch))
        mode = str(getattr(target.mode, "value", target.mode))
        return f"cm-edgellm-{arch}-{mode}"

    def _container_pattern(
        self,
        target: code_manager_models.ArtifactTarget,
    ) -> container_types.ContainerPattern:
        """Build the TRT container pattern for EdgeLLM source builds."""
        return container_types.ContainerPattern(
            kind=container_types.ContainerKind.TRT,
            arch=target.platform.arch,
            branch=normalize_trt_container_branch(target.branch),
            cuda_version=target.platform.cuda_version,
            ubuntu_version=target.platform.ubuntu_version,
            trt_type=target.platform.trt_type,
        )

    def _container_extra_args(
        self,
        target: code_manager_models.ArtifactTarget,
        deps: Mapping[str, code_manager_models.ArtifactResult],
        *,
        descriptor: container_types.ContainerDescriptor,
    ) -> list[str] | None:
        """Return backend-specific launch arguments."""
        _ = deps
        if descriptor.backend is not container_types.ContainerBackendType.GIT_TRT_RUNC:
            return None
        args = self._no_nvidia_runtime_extra_args(
            target, list(target.build.git_trt_runc_args)) or []
        if self._gpu_selection is not None:
            args.extend(
                ("-e", f"CUDA_VISIBLE_DEVICES={self._gpu_selection.uuid}"))
        return args or None

    def _additional_mounts(
        self,
        target: code_manager_models.ArtifactTarget,
        deps: Mapping[str, code_manager_models.ArtifactResult],
        *,
        build_dir: str,
    ) -> list[container_types.MountSpec]:
        """Return EdgeLLM-specific mounts appended to the shared mount plan."""
        _ = build_dir
        mounts: list[container_types.MountSpec] = []
        if target.build.trt_package_dir:
            mounts.append(
                container_types.MountSpec(
                    host_path=target.build.trt_package_dir,
                    container_path=target.build.trt_package_dir,
                    read_only=False,
                ))
        for dep in deps.values():
            if dep.output_dir:
                mounts.append(
                    container_types.MountSpec(
                        host_path=dep.output_dir,
                        container_path=dep.output_dir,
                        read_only=True,
                    ))
        return mounts

    def _generate_build_script(
        self,
        target: code_manager_models.ArtifactTarget,
    ) -> str:
        """Render the EdgeLLM CMake build script."""
        if not target.build.repo_path:
            raise ValueError("EdgeLLM build requires target.build.repo_path")
        if not target.build.build_dir:
            raise ValueError("EdgeLLM build requires target.build.build_dir")
        if not target.build.trt_package_dir:
            raise ValueError(
                "EdgeLLM build requires target.build.trt_package_dir")

        arch = (target.platform.arch if isinstance(target.platform.arch, Arch)
                else Arch(str(target.platform.arch)))
        cuda_version = target.platform.cuda_version or arch.default_cuda_version
        configure_command = [
            "cmake",
            target.build.repo_path,
            f"-DTRT_PACKAGE_DIR={target.build.trt_package_dir}",
            f"-DCUDA_CTK_VERSION={cuda_version}",
            "-DBUILD_UNIT_TESTS=ON",
        ]
        configure_command.extend(target.build.cmake_args)

        parallel_setup_lines, parallel_flag = source_build_parallel_config(
            arch,
            parallel_jobs=target.build.parallel_jobs,
            flag_template='-j"${var_name}"',
        )
        build_items = [*(shlex.quote(item) for item in target.build.targets)]
        build_items.extend(
            shlex.quote(item) for item in target.build.make_args)
        build_command = f"make {parallel_flag}"
        if build_items:
            build_command = f"{build_command} {' '.join(build_items)}"

        lines = [
            f"mkdir -p {shlex.quote(target.build.build_dir)}",
            f"cd {shlex.quote(target.build.build_dir)}",
            shlex.join(configure_command),
        ]
        lines.extend(parallel_setup_lines)
        lines.append(build_command)
        return "\n".join(lines)

    def _prepare_cute_dsl(
        self,
        target: code_manager_models.ArtifactTarget,
    ) -> code_manager_models.ArtifactTarget:
        """Generate CuTeDSL artifacts required by the target CMake arguments."""
        groups = cute_dsl_groups(target)
        if not groups:
            return target
        if self._gpu_selection is None:
            raise ValueError(
                "CuTeDSL builds require an endpoint GPU selection")
        if not target.build.repo_path:
            raise ValueError("CuTeDSL builds require build.repo_path")
        if not target.platform.cuda_version:
            raise ValueError("CuTeDSL builds require platform.cuda_version")

        cuda_major = target.platform.cuda_version.split(".", 1)[0]
        if cuda_major not in {"12", "13"}:
            raise ValueError(
                f"CuTeDSL supports CUDA 12 or 13, got {target.platform.cuda_version}"
            )
        repo_path = Path(target.build.repo_path)
        artifact_root = repo_path / "cpp" / "kernels" / "cuteDSLArtifact"
        artifact_tag = cute_dsl_artifact_tag(
            target) or self._gpu_selection.sm_arch
        self._run_required_host_command(
            name="edgellm-cutedsl-source",
            argv=["git", "submodule", "update", "--init"],
            cwd=str(repo_path),
        )
        self._run_required_host_command(
            name="edgellm-cutedsl-artifact-directory",
            argv=["mkdir", "-p", str(artifact_root)],
            cwd=str(repo_path),
        )
        if not _has_cute_dsl_artifacts(artifact_root, artifact_tag,
                                       target.platform.arch):
            self._ensure_cute_dsl_builder_image(repo_path)
            self._generate_cute_dsl_artifacts(
                artifact_root=artifact_root,
                artifact_tag=artifact_tag,
                cuda_major=cuda_major,
                groups=groups,
                repo_path=repo_path,
                target=target,
            )

        cmake_args = [
            argument for argument in target.build.cmake_args
            if not argument.startswith("-DCUTE_DSL_ARTIFACT_TAG=")
        ]
        cmake_args.append(f"-DCUTE_DSL_ARTIFACT_TAG={artifact_tag}")
        return replace(target,
                       build=replace(target.build, cmake_args=cmake_args))

    def _ensure_cute_dsl_builder_image(self, repo_path: Path) -> None:
        """Build the CuTeDSL image when Docker cannot find it locally.

        DevToolkit ContainerManager launches containers, but it does not expose
        an image-build operation, so image construction remains a direct
        CommandManager host command in this EdgeLLM-specific adapter.
        """
        inspection = self._command_manager.run(
            self._default_exec_target,
            command_spec.CommandSpec(
                argv=["docker", "image", "inspect", _CUTE_DSL_IMAGE],
                cwd=str(repo_path),
                output_mode=output_mode.OutputMode.CAPTURE,
                operation_name="edgellm-cutedsl-image-inspect",
            ),
        )
        if inspection.success:
            return
        self._run_required_host_command(
            name="edgellm-cutedsl-image-build",
            argv=[
                "docker",
                "build",
                "--quiet",
                "-f",
                str(repo_path / "kernelSrcs" / "Dockerfile.cutedsl"),
                "--build-arg",
                "CUTE_DSL_BUILDER_VERSION=4.6.1",
                "-t",
                _CUTE_DSL_IMAGE,
                str(repo_path / "kernelSrcs"),
            ],
            cwd=str(repo_path),
        )

    def _generate_cute_dsl_artifacts(
        self,
        *,
        artifact_root: Path,
        artifact_tag: str,
        cuda_major: str,
        groups: tuple[str, ...],
        repo_path: Path,
        target: code_manager_models.ArtifactTarget,
    ) -> None:
        """Run the CuTeDSL builder through DevToolkit ContainerManager."""
        assert self._gpu_selection is not None
        manager = make_cute_dsl_container_manager(self._command_manager)
        name = f"edgellm-cutedsl-{artifact_tag}-{cuda_major}"
        if manager.exists(name, exec_target=self._default_exec_target):
            stale_handle = container_types.ContainerHandle(
                name=name,
                backend=container_types.ContainerBackendType.DOCKER,
                image=_CUTE_DSL_IMAGE,
            )
            manager.remove(stale_handle,
                           force=True,
                           exec_target=self._default_exec_target)
        log_file = artifact_root / f"cutedsl_{artifact_tag}_cu{cuda_major}.log"
        handle = manager.launch(
            descriptor=container_types.ContainerDescriptor(
                backend=container_types.ContainerBackendType.DOCKER,
                image_or_profile=_CUTE_DSL_IMAGE,
                kind=container_types.ContainerKind.TRT,
            ),
            name=name,
            mounts=[
                container_types.MountSpec(
                    host_path=str(repo_path / "kernelSrcs"),
                    container_path="/workspace/kernelSrcs",
                    read_only=True,
                ),
                container_types.MountSpec(
                    host_path=str(artifact_root),
                    container_path="/artifacts",
                    read_only=False,
                ),
            ],
            extra_args=[
                "--entrypoint",
                "sleep",
                "-e",
                f"CUDA_VISIBLE_DEVICES={self._gpu_selection.uuid}",
            ],
            exec_target=self._default_exec_target,
        )
        try:
            arch = _cute_dsl_target_arch(target.platform.arch)
            success = manager.exec_progress(
                handle,
                argv=[
                    f"/opt/cutedsl/venvs/cu{cuda_major}/bin/python",
                    "kernelSrcs/build_cutedsl.py",
                    "--kernels",
                    ",".join(groups),
                    "--gpu_arch",
                    artifact_tag,
                    "--arch",
                    arch,
                    "--cuda-version",
                    cuda_major,
                    "--output_dir",
                    "/artifacts",
                    "--jobs",
                    str(target.build.parallel_jobs or 1),
                ],
                cwd="/workspace",
                timeout_s=None,
                tee_file=str(log_file),
                exec_target=self._default_exec_target,
            )
        finally:
            manager.stop(handle, exec_target=self._default_exec_target)
            manager.remove(handle,
                           force=True,
                           exec_target=self._default_exec_target)
        if not success:
            raise OrchestrationError(
                f"CuTeDSL artifact generation failed; see {log_file}")

    def _run_required_host_command(self,
                                   *,
                                   name: str,
                                   argv: list[str],
                                   cwd: str | None = None) -> None:
        """Run a required host command with DevToolkit CommandManager."""
        result = self._command_manager.run(
            self._default_exec_target,
            command_spec.CommandSpec(
                argv=argv,
                cwd=cwd,
                output_mode=output_mode.OutputMode.PROGRESS,
                operation_name=name,
            ),
        )
        if not result.success:
            raise OrchestrationError(result.error_message or f"{name} failed")


def edgellm_component_spec() -> code_manager_models.ComponentSpec:
    """Return CodeManager metadata for the EdgeLLM component.

    Returns:
        Component metadata used for deterministic planning.
    """
    return code_manager_models.ComponentSpec(EDGELLM_COMPONENT, order=3)


def edgellm_dependency_rules(
) -> tuple[code_manager_models.ComponentDependencyRule, ...]:
    """Return implicit dependencies required by EdgeLLM source builds.

    Returns:
        Dependency rules that bind the latest prior TRT artifact when the caller
        did not provide ``BuildConfig.trt_package_dir`` explicitly.
    """
    return (code_manager_models.ComponentDependencyRule(
        consumer=EDGELLM_COMPONENT,
        provider=code_manager_models.TRT_COMPONENT,
        dep_name="trt_artifact",
        condition=lambda target: not target.build.trt_package_dir,
        target_updater=_bind_trt_package_dir,
    ), )


def _bind_trt_package_dir(
    target: code_manager_models.ArtifactTarget,
    provider_step: code_manager_models.PlanStep,
) -> code_manager_models.ArtifactTarget:
    """Inject TRT package dir from a prior TRT plan step."""
    if target.build.trt_package_dir or not provider_step.target.build.build_dir:
        return target
    build = dataclasses.replace(
        target.build, trt_package_dir=provider_step.target.build.build_dir)
    return dataclasses.replace(target, build=build)


def _has_cute_dsl_artifacts(
    artifact_root: Path,
    artifact_tag: str,
    arch: Arch | str,
) -> bool:
    """Return whether required generated CuteDSL files already exist.

    Args:
        artifact_root: Root cuteDSLArtifact directory.
        artifact_tag: GPU SM artifact tag, such as ``sm_110``.
        arch: DevToolkit target architecture.

    Returns:
        True when the generated aggregate header and static library exist.
    """
    target_arch = _cute_dsl_target_arch(arch)
    artifact_dir = artifact_root / target_arch / artifact_tag
    return (artifact_dir / "include" / "cutedsl_all.h").is_file() and (
        artifact_dir / f"libcutedsl_{target_arch}.a").is_file()


def _cute_dsl_target_arch(arch: Arch | str) -> str:
    """Return the architecture spelling accepted by build_cutedsl.py.

    Args:
        arch: DevToolkit target architecture.

    Returns:
        ``x86_64`` for host builds and ``aarch64`` for Linux/QNX edge targets.
    """
    resolved = arch if isinstance(arch, Arch) else Arch(str(arch))
    if resolved is Arch.X86_64:
        return "x86_64"
    return "aarch64"


def cute_dsl_artifact_tag(
    artifact_target: code_manager_models.ArtifactTarget, ) -> str | None:
    """Read an explicit CuTeDSL artifact tag from CMake arguments.

    Args:
        artifact_target: EdgeLLM target whose CMake arguments are inspected.

    Returns:
        The requested CuTeDSL artifact tag, or None when unspecified.
    """
    prefix = "-DCUTE_DSL_ARTIFACT_TAG="
    for argument in reversed(artifact_target.build.cmake_args):
        if argument.startswith(prefix):
            value = argument.removeprefix(prefix).strip()
            return value or None
    return None


def cute_dsl_groups(
    artifact_target: code_manager_models.ArtifactTarget, ) -> tuple[str, ...]:
    """Read the active CuTeDSL group selection from CMake arguments.

    Args:
        artifact_target: EdgeLLM target whose CMake arguments are inspected.

    Returns:
        Ordered CuTeDSL groups, or an empty tuple when disabled or unset.
    """
    prefix = "-DENABLE_CUTE_DSL="
    for argument in reversed(artifact_target.build.cmake_args):
        if argument.startswith(prefix):
            value = argument.removeprefix(prefix).strip()
            if not value or value.casefold() == "off":
                return ()
            return tuple(group for group in value.split(";") if group)
    return ()


def make_container_manager(
    command_manager_: command_manager.CommandManager,
    trt_repo_path: Path | None,
) -> container_manager.ContainerManager:
    """Create the container manager used for EdgeLLM source builds.

    Args:
        command_manager_: Shared command manager.
        trt_repo_path: Optional TensorRT repository used by GitTrt containers.

    Returns:
        DevToolkit container manager.
    """
    git_trt_backend = None
    if trt_repo_path:
        git_trt_backend = git_trt_runc_backend.GitTrtRuncBackend(
            command_manager_,
            config=git_trt_runc_backend.GitTrtRuncConfig(
                trt_repo=str(trt_repo_path),
                skip_branch_check=True,
            ),
        )
    return container_manager.ContainerManager(
        command_manager=command_manager_,
        git_trt_backend=git_trt_backend,
    )


def make_cute_dsl_container_manager(
    command_manager_: command_manager.CommandManager,
) -> container_manager.ContainerManager:
    """Create a Docker-backed manager for short-lived CuTeDSL builders.

    Args:
        command_manager_: Shared command manager.

    Returns:
        Container manager with target-local docker commands for generation.
    """
    backend = docker_backend.DockerBackend(
        command_manager_,
        config=docker_backend.DockerBackendConfig(
            default_command=["infinity"]),
    )
    return container_manager.ContainerManager(command_manager=command_manager_,
                                              docker_backend=backend)
