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
"""Validate and publish qualified Edge-LLM release wheels."""

from __future__ import annotations

import base64
import dataclasses
import html.parser
import http.client
import json
import os
import pathlib
import re
import ssl
import subprocess
import sys
import tempfile
import time
import typing
import urllib.parse
import zipfile

from wheel_ci_lib import common
from wheellib import config, oss, wheel_artifact

PACKAGE_NAME = wheel_artifact.PACKAGE_NAME
WheelIdentity = wheel_artifact.WheelIdentity
PYTHON_ABIS = tuple(common.ABI_INTERPRETERS)
CPU_ARCHITECTURES = tuple(wheel_artifact.PUBLIC_PLATFORM_TAGS)
EXPECTED_MATRIX = frozenset((python_abi, cpu_arch)
                            for python_abi in PYTHON_ABIS
                            for cpu_arch in CPU_ARCHITECTURES)
MANIFEST_SCHEMA_VERSION = 3
RELEASE_MODES = frozenset({"release", "rehearsal"})
KITMAKER_API_BASE = "https://kitmaker-portal.nvidia.com/api/v0"
ARTIFACTORY_PYPI_URL = ("https://artifactory.nvidia.com/artifactory/api/pypi/"
                        "hw-tensorrt-edge-llm-pypi-local")
NVIDIA_PYPI_SIMPLE_URL = "https://pypi.nvidia.com/simple/tensorrt-edgellm/"
PYPI_ORG_SIMPLE_URL = "https://pypi.org/simple/tensorrt-edgellm/"
PUBLIC_INDEX_URLS = (NVIDIA_PYPI_SIMPLE_URL, PYPI_ORG_SIMPLE_URL)
MAX_HTTP_RESPONSE_BYTES = 8 * 1024 * 1024
REDACTED = "<redacted>"


@dataclasses.dataclass(frozen=True)
class HttpResponse:
    """Bounded HTTP response used by the release client."""

    status: int
    headers: typing.Mapping[str, str]
    body: bytes


class ReleaseHttpClient(typing.Protocol):
    """Network operations required by release workflows."""

    def request(self,
                method: str,
                url: str,
                *,
                headers: typing.Optional[typing.Mapping[str, str]] = None,
                body: typing.Optional[bytes] = None) -> HttpResponse:
        """Send a bounded HTTP request."""

    def download(self, url: str, destination: pathlib.Path,
                 headers: typing.Mapping[str, str]) -> HttpResponse:
        """Download a response body to a file."""


class HttpsClient:
    """Strict HTTPS client with bounded requests and streaming downloads."""

    def __init__(self, timeout_seconds: int = 300) -> None:
        if timeout_seconds <= 0:
            raise ValueError("HTTP timeout must be positive.")
        self._timeout_seconds = timeout_seconds
        self._ssl_context = ssl.create_default_context()

    def _connection(
        self, url: str
    ) -> typing.Tuple[http.client.HTTPSConnection, urllib.parse.SplitResult]:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise RuntimeError(f"Release URL must use HTTPS: {url!r}.")
        if parsed.username or parsed.password or parsed.fragment:
            raise RuntimeError(
                f"Release URL contains forbidden fields: {url!r}.")
        port = parsed.port or 443
        connection = http.client.HTTPSConnection(
            parsed.hostname,
            port=port,
            timeout=self._timeout_seconds,
            context=self._ssl_context,
        )
        return connection, parsed

    @staticmethod
    def _target(parsed: urllib.parse.SplitResult) -> str:
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        return target

    @staticmethod
    def _headers(response: http.client.HTTPResponse) -> typing.Dict[str, str]:
        return {key.lower(): value for key, value in response.getheaders()}

    def request(self,
                method: str,
                url: str,
                *,
                headers: typing.Optional[typing.Mapping[str, str]] = None,
                body: typing.Optional[bytes] = None) -> HttpResponse:
        connection, parsed = self._connection(url)
        try:
            connection.request(method,
                               self._target(parsed),
                               body=body,
                               headers=dict(headers or {}))
            response = connection.getresponse()
            data = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
            if len(data) > MAX_HTTP_RESPONSE_BYTES:
                raise RuntimeError(
                    f"HTTP response from {url!r} exceeds the safety limit.")
            return HttpResponse(response.status, self._headers(response), data)
        finally:
            connection.close()

    def download(self, url: str, destination: pathlib.Path,
                 headers: typing.Mapping[str, str]) -> HttpResponse:
        connection, parsed = self._connection(url)
        try:
            connection.request("GET",
                               self._target(parsed),
                               headers=dict(headers))
            response = connection.getresponse()
            response_headers = self._headers(response)
            if response.status < 200 or response.status >= 300:
                data = response.read()
                return HttpResponse(response.status, response_headers, data)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as file:
                while chunk := response.read(8 * 1024 * 1024):
                    file.write(chunk)
            return HttpResponse(response.status, response_headers, b"")
        finally:
            connection.close()


class _SimpleIndexParser(html.parser.HTMLParser):

    def __init__(self) -> None:
        super().__init__()
        self.links: typing.List[str] = []

    def handle_starttag(
        self, tag: str,
        attributes: typing.List[typing.Tuple[str,
                                             typing.Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attributes:
            if key.lower() == "href" and value:
                self.links.append(value)


def parse_wheel_identity(filename: str,
                         *,
                         release_mode: str = "release") -> WheelIdentity:
    """Apply release policy to one shared final-wheel identity."""
    if release_mode not in RELEASE_MODES:
        raise RuntimeError(f"Unsupported release mode {release_mode!r}.")
    identity = wheel_artifact.parse_final_wheel_filename(filename)
    if identity.python_abi not in PYTHON_ABIS:
        raise RuntimeError(f"Unsupported Python ABI in {filename!r}.")
    if identity.cpu_arch not in CPU_ARCHITECTURES:
        raise RuntimeError(f"Unsupported CPU architecture in {filename!r}.")
    lowered_version = identity.version.lower()
    if "+" in identity.version:
        raise RuntimeError(
            f"Local versions cannot be released: {identity.version!r}.")
    if ".dev" in lowered_version and release_mode != "rehearsal":
        raise RuntimeError("Development versions require rehearsal mode: "
                           f"{identity.version!r}.")
    return identity


def _audit_release_provenance(wheel: pathlib.Path, identity: WheelIdentity,
                              source_revision: str) -> None:
    """Confirm the tested wheel belongs to this source release."""
    try:
        archive = zipfile.ZipFile(wheel)
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"Invalid wheel archive {wheel}.") from error
    with archive:
        names = set(archive.namelist())
        runtime_manifest = "tensorrt_edgellm/_native/variants.json"
        if runtime_manifest not in names:
            raise RuntimeError(f"{wheel.name} has no native runtime manifest.")
        try:
            variants = json.loads(archive.read(runtime_manifest))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"{wheel.name} has invalid native runtime metadata."
            ) from error
        if not isinstance(variants, dict):
            raise RuntimeError(
                f"{wheel.name} has non-object native runtime metadata.")
        if variants.get("package_version") != identity.version:
            raise RuntimeError(
                f"{wheel.name} native metadata has inconsistent version.")
        if variants.get("source_revision") != source_revision:
            raise RuntimeError(
                f"{wheel.name} was not built from {source_revision}.")
        oss.require_policy(variants)
        oss.audit_archive(archive)


def _qualified_wheel_digests(path: pathlib.Path,
                             release_mode: str) -> typing.Dict[str, str]:
    try:
        gate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read integration gate evidence {path}: {error}."
        ) from error
    if not isinstance(gate, dict):
        raise RuntimeError(f"Invalid integration gate evidence {path}.")
    schema_version = gate.get("schema_version")
    if schema_version != common.INTEGRATION_GATE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Integration gate evidence {path} uses schema version "
            f"{schema_version!r}; expected "
            f"{common.INTEGRATION_GATE_SCHEMA_VERSION}.")
    required = {"schema_version", "qualified_rows", "qualified_wheels"}
    if (set(gate) != required
            or not isinstance(gate.get("qualified_rows"), int)
            or gate["qualified_rows"] < 1
            or not isinstance(gate.get("qualified_wheels"), dict)):
        raise RuntimeError(f"Invalid integration gate evidence {path}.")
    qualified = {}
    for filename, digest in gate["qualified_wheels"].items():
        if (not isinstance(filename, str) or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise RuntimeError(f"Invalid qualified wheel in {path}.")
        parse_wheel_identity(filename, release_mode=release_mode)
        qualified[filename] = digest
    return qualified


def prepare_manifest(
        wheel_root: pathlib.Path,
        source_revision: str,
        integration_gate: pathlib.Path,
        output: pathlib.Path,
        *,
        release_mode: str = "release") -> typing.Dict[str, object]:
    """Validate the six release wheels and write their release manifest."""
    if release_mode not in RELEASE_MODES:
        raise RuntimeError(f"Unsupported release mode {release_mode!r}.")
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise RuntimeError("Source revision must be a full lowercase Git SHA.")
    qualified_wheels = _qualified_wheel_digests(integration_gate, release_mode)
    wheels = sorted(wheel_root.rglob("*.whl"))
    identities: typing.Dict[typing.Tuple[str, str], WheelIdentity] = {}
    artifacts: typing.List[typing.Dict[str, object]] = []
    versions: typing.Set[str] = set()
    for wheel in wheels:
        identity = parse_wheel_identity(wheel.name, release_mode=release_mode)
        key = (identity.python_abi, identity.cpu_arch)
        if key in identities:
            raise RuntimeError(f"Duplicate release wheel for {key}.")
        _audit_release_provenance(wheel, identity, source_revision)
        digest = config.sha256(wheel)
        if qualified_wheels.get(wheel.name) != digest:
            raise RuntimeError(
                f"Integration gate does not qualify {wheel.name}.")
        identities[key] = identity
        versions.add(identity.version)
        artifacts.append({
            "filename": identity.filename,
            "sha256": digest,
        })
    if set(identities) != EXPECTED_MATRIX:
        raise RuntimeError(
            "Release wheel matrix is incomplete; "
            f"missing={sorted(EXPECTED_MATRIX - set(identities))}, "
            f"extra={sorted(set(identities) - EXPECTED_MATRIX)}.")
    if len(versions) != 1:
        raise RuntimeError(
            f"Release wheels use inconsistent versions: {versions}.")
    filenames = {identity.filename for identity in identities.values()}
    if set(qualified_wheels) != filenames:
        raise RuntimeError(
            "Integration gate wheel set differs from the release matrix.")
    manifest: typing.Dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "package_name": PACKAGE_NAME,
        "package_version": versions.pop(),
        "release_mode": release_mode,
        "source_revision": source_revision,
        "pipeline": {
            "id": os.environ.get("CI_PIPELINE_ID", ""),
            "url": os.environ.get("CI_PIPELINE_URL", ""),
            "job_id": os.environ.get("CI_JOB_ID", ""),
        },
        "artifacts": sorted(artifacts, key=lambda item: str(item["filename"])),
    }
    config.write_json(output, manifest)
    return manifest


def load_manifest(path: pathlib.Path,
                  *,
                  required_mode: typing.Optional[str] = None
                  ) -> typing.Dict[str, object]:
    """Load and validate a release manifest written by prepare_manifest."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read release manifest {path}: {error}.") from error
    required = {
        "schema_version", "package_name", "package_version", "source_revision",
        "release_mode", "pipeline", "artifacts"
    }
    if (not isinstance(manifest, dict) or set(manifest) != required
            or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
            or manifest.get("package_name") != PACKAGE_NAME):
        raise RuntimeError(f"Invalid release manifest {path}.")
    release_mode = manifest.get("release_mode")
    if release_mode not in RELEASE_MODES:
        raise RuntimeError("Release manifest has an invalid release mode.")
    if required_mode is not None and release_mode != required_mode:
        raise RuntimeError(
            f"Release manifest mode {release_mode!r} is not {required_mode!r}."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts,
                      list) or len(artifacts) != len(EXPECTED_MATRIX):
        raise RuntimeError("Release manifest must contain exactly six wheels.")
    matrix = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise RuntimeError("Release manifest artifact must be an object.")
        required_artifact = {"filename", "sha256"}
        allowed_artifact = required_artifact | {"artifactory_url"}
        if not required_artifact.issubset(artifact) or not set(
                artifact).issubset(allowed_artifact):
            raise RuntimeError("Release manifest artifact has schema drift.")
        identity = parse_wheel_identity(str(artifact["filename"]),
                                        release_mode=str(release_mode))
        if identity.version != manifest["package_version"]:
            raise RuntimeError("Release manifest artifact version disagrees.")
        if not re.fullmatch(r"[0-9a-f]{64}", str(artifact["sha256"])):
            raise RuntimeError(
                "Release manifest artifact has an invalid digest.")
        matrix.add((identity.python_abi, identity.cpu_arch))
    if matrix != EXPECTED_MATRIX:
        raise RuntimeError("Release manifest wheel matrix is incomplete.")
    return manifest


def _validate_host(url: str, expected_host: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != expected_host
            or parsed.username or parsed.password or parsed.query
            or parsed.fragment):
        raise RuntimeError(f"URL must use https://{expected_host}: {url!r}.")


def _authorization(token: str) -> typing.Dict[str, str]:
    if not token or any(character.isspace() for character in token):
        raise RuntimeError("Release API token is empty or malformed.")
    return {"Authorization": f"Bearer {token}"}


def _basic_authorization(username: str,
                         password: str) -> typing.Dict[str, str]:
    if (not username or not password or ":" in username
            or any(character.isspace() for character in username)
            or any(character.isspace() for character in password)):
        raise RuntimeError("Artifactory credentials are empty or malformed.")
    value = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {value}"}


def _redact_text(value: str) -> str:
    value = re.sub(r"(?i)((?:basic|bearer)\s+)[^\s]+", rf"\g<1>{REDACTED}",
                   value)
    return re.sub(
        r"(?i)(token|secret|password)\s*[:=]\s*[^\s,;]+",
        rf"\g<1>={REDACTED}",
        value,
    )


def _sanitized_response_body(body: bytes) -> str:
    text = body.decode(errors="replace")[:1000]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _redact_text(text)
    return json.dumps(redact(parsed), sort_keys=True)


def _response_error(action: str, response: HttpResponse) -> RuntimeError:
    text = _sanitized_response_body(response.body)
    return RuntimeError(f"{action} failed with HTTP {response.status}: {text}")


def _artifactory_endpoints(repository_url: str,
                           version: str) -> typing.Tuple[str, str]:
    _validate_host(repository_url, "artifactory.nvidia.com")
    parsed = urllib.parse.urlsplit(repository_url)
    prefix = "/artifactory/api/pypi/"
    if not parsed.path.startswith(prefix):
        raise RuntimeError(
            "Artifactory PyPI URL must use the PyPI API endpoint.")
    repository = parsed.path.removeprefix(prefix).strip("/")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", repository):
        raise RuntimeError(
            "Artifactory PyPI URL has an invalid repository name.")
    origin = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, "", "", ""))
    simple_url = (f"{repository_url.rstrip('/')}/simple/"
                  f"{PACKAGE_NAME}/")
    download_base = (f"{origin}/artifactory/{repository}/{PACKAGE_NAME}/"
                     f"{urllib.parse.quote(version, safe='')}")
    return simple_url, download_base


def _local_release_wheels(
        manifest: typing.Mapping[str, object],
        wheel_root: pathlib.Path) -> typing.Dict[str, pathlib.Path]:
    wheels: typing.Dict[str, pathlib.Path] = {}
    for wheel in wheel_root.rglob("*.whl"):
        if wheel.name in wheels:
            raise RuntimeError(f"Duplicate local wheel {wheel.name!r}.")
        wheels[wheel.name] = wheel
    artifacts = typing.cast(typing.Sequence[typing.Mapping[str, object]],
                            manifest["artifacts"])
    expected = {str(artifact["filename"]) for artifact in artifacts}
    if set(wheels) != expected:
        raise RuntimeError(
            "Local wheel set differs from the release manifest.")
    for artifact in artifacts:
        filename = str(artifact["filename"])
        if config.sha256(wheels[filename]) != artifact["sha256"]:
            raise RuntimeError(
                f"Local wheel changed after validation: {filename}.")
    return wheels


def stage_artifacts(
    manifest_path: pathlib.Path,
    wheel_root: pathlib.Path,
    output: pathlib.Path,
    repository_url: str,
    username: str,
    password: str,
    client: ReleaseHttpClient,
    runner: typing.Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> typing.Dict[str, object]:
    """Upload an exact release to Artifactory PyPI and verify its index."""
    manifest = load_manifest(manifest_path)
    version = str(manifest["package_version"])
    simple_url, download_base = _artifactory_endpoints(repository_url, version)
    wheels = _local_release_wheels(manifest, wheel_root)
    artifacts = typing.cast(typing.List[typing.Dict[str, object]],
                            manifest["artifacts"])
    for artifact in artifacts:
        filename = str(artifact["filename"])
        artifact["artifactory_url"] = (
            f"{download_base}/{urllib.parse.quote(filename, safe='')}")

    headers = _basic_authorization(username, password)
    already_staged = verify_index(
        manifest,
        simple_url,
        client,
        expected_host="artifactory.nvidia.com",
        headers=headers,
        allow_absent=True,
    )
    if not already_staged:
        environment = os.environ.copy()
        environment.update({
            "TWINE_REPOSITORY_URL": repository_url,
            "TWINE_USERNAME": username,
            "TWINE_PASSWORD": password,
        })
        runner(
            [
                sys.executable,
                "-m",
                "twine",
                "upload",
                "--non-interactive",
                *[str(wheels[name]) for name in sorted(wheels)],
            ],
            check=True,
            env=environment,
        )
        verify_index(
            manifest,
            simple_url,
            client,
            expected_host="artifactory.nvidia.com",
            headers=headers,
            wait_seconds=300,
            poll_seconds=10,
            tolerate_partial=True,
        )
    config.write_json(output, manifest)
    return manifest


def kitmaker_payload(manifest: typing.Mapping[str, object], pic: str, *,
                     upload: bool) -> typing.Dict[str, object]:
    """Build a project-scoped Kitmaker wheel release payload."""
    if not pic or "@" not in pic:
        raise RuntimeError("Kitmaker PIC must be a valid team email alias.")
    if not isinstance(upload, bool):
        raise RuntimeError("Kitmaker upload mode must be an explicit boolean.")
    entries = []
    artifacts = typing.cast(typing.Sequence[typing.Mapping[str, object]],
                            manifest["artifacts"])
    for artifact in artifacts:
        url = artifact.get("artifactory_url")
        if not isinstance(url, str):
            raise RuntimeError("Kitmaker manifest has an unstaged wheel.")
        _validate_host(url, "artifactory.nvidia.com")
        entries.append({
            "pic": pic,
            "job_type": "wheel-release-job",
            "url": url,
            "upload": upload,
        })
    return {"project_name": PACKAGE_NAME, "payload": entries}


def _json_response(action: str,
                   response: HttpResponse) -> typing.Dict[str, object]:
    if response.status < 200 or response.status >= 300:
        raise _response_error(action, response)
    try:
        value = json.loads(response.body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{action} returned invalid JSON.") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{action} returned a non-object response.")
    return value


def redact(value: object) -> object:
    """Remove credential-shaped fields before logging API responses."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if any(marker in str(key).lower()
                   for marker in ("authorization", "password", "secret",
                                  "token")):
                result[key] = REDACTED
            else:
                result[key] = redact(item)
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def release_with_kitmaker(
        manifest: typing.Mapping[str, object],
        project_id: str,
        pic: str,
        token: str,
        client: ReleaseHttpClient,
        *,
        upload: bool,
        poll_seconds: int = 20,
        timeout_seconds: int = 7200,
        api_base: str = KITMAKER_API_BASE) -> typing.Dict[str, object]:
    """Submit one Kitmaker validation or release and poll its status."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", project_id):
        raise RuntimeError("Kitmaker project ID is empty or malformed.")
    if poll_seconds <= 0 or timeout_seconds <= 0:
        raise RuntimeError("Kitmaker polling intervals must be positive.")
    _validate_host(api_base, "kitmaker-portal.nvidia.com")
    payload = kitmaker_payload(manifest, pic, upload=upload)
    api_root = api_base.rstrip("/")
    endpoint = f"{api_root}/projects/{project_id}/releases"
    headers = _authorization(token)
    headers["Content-Type"] = "application/json"
    response = client.request("POST",
                              endpoint,
                              headers=headers,
                              body=json.dumps(payload).encode("utf-8"))
    created = _json_response("Create Kitmaker release", response)
    release_uuid = created.get("release_uuid")
    if not isinstance(release_uuid, str) or not release_uuid:
        raise RuntimeError("Kitmaker did not return a release UUID.")
    status_url = f"{api_root}/status/{release_uuid}"
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = client.request("GET",
                                  status_url,
                                  headers=_authorization(token))
        status_data = _json_response("Read Kitmaker release status", response)
        status = status_data.get("status")
        if status == "completed":
            return typing.cast(typing.Dict[str, object], redact(status_data))
        if status == "failed":
            sanitized = redact(status_data)
            raise RuntimeError(
                f"Kitmaker release {release_uuid} failed: {sanitized}.")
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Kitmaker release {release_uuid} did not finish within "
                f"{timeout_seconds} seconds.")
        time.sleep(poll_seconds)


def _index_links(
    index_url: str,
    client: ReleaseHttpClient,
    expected_host: str,
    headers: typing.Optional[typing.Mapping[str, str]] = None
) -> typing.Dict[str, str]:
    _validate_host(index_url, expected_host)
    response = client.request("GET", index_url, headers=headers or {})
    if response.status == 404:
        return {}
    if response.status < 200 or response.status >= 300:
        raise _response_error(f"Read package index {index_url}", response)
    parser = _SimpleIndexParser()
    parser.feed(response.body.decode("utf-8"))
    links = {}
    for href in parser.links:
        resolved = urllib.parse.urldefrag(urllib.parse.urljoin(
            index_url, href)).url
        parsed = urllib.parse.urlsplit(resolved)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query
                or (headers and parsed.hostname != expected_host)):
            raise RuntimeError(
                f"Package index returned an unsafe URL: {resolved!r}.")
        filename = pathlib.PurePosixPath(urllib.parse.unquote(
            parsed.path)).name
        if filename in links and links[filename] != resolved:
            raise RuntimeError(f"Package index repeats {filename!r}.")
        links[filename] = resolved
    return links


def verify_index(manifest: typing.Mapping[str, object],
                 index_url: str,
                 client: ReleaseHttpClient,
                 *,
                 expected_host: str,
                 headers: typing.Optional[typing.Mapping[str, str]] = None,
                 wait_seconds: int = 0,
                 poll_seconds: int = 20,
                 allow_absent: bool = False,
                 tolerate_partial: bool = False) -> bool:
    """Wait for and hash-verify an exact six-wheel release on an index."""
    if wait_seconds < 0 or poll_seconds <= 0:
        raise RuntimeError("Index polling intervals are invalid.")
    artifacts = typing.cast(typing.Sequence[typing.Mapping[str, object]],
                            manifest["artifacts"])
    expected = {str(artifact["filename"]): artifact for artifact in artifacts}
    version = str(manifest["package_version"])
    deadline = time.monotonic() + wait_seconds
    while True:
        links = _index_links(index_url, client, expected_host, headers)
        same_version = set()
        for filename in links:
            try:
                identity = wheel_artifact.parse_final_wheel_filename(
                    filename, allow_build_tag=True)
            except RuntimeError:
                continue
            if identity.version == version:
                same_version.add(filename)
        present = same_version & set(expected)
        unexpected = same_version - set(expected)
        if unexpected:
            raise RuntimeError(
                f"Package index contains unexpected wheels for {version}: "
                f"{sorted(unexpected)}.")
        if present == set(expected):
            break
        deadline_reached = time.monotonic() >= deadline
        if present and (deadline_reached or not tolerate_partial):
            raise RuntimeError(
                f"Package index contains a partial release: {sorted(present)}."
            )
        if deadline_reached:
            if allow_absent and not same_version:
                return False
            raise RuntimeError(
                f"Package index does not contain release {version}.")
        time.sleep(poll_seconds)
    with tempfile.TemporaryDirectory(prefix="edgellm-index-verify-") as root:
        directory = pathlib.Path(root)
        for filename, artifact in expected.items():
            destination = directory / filename
            response = client.download(links[filename], destination, headers
                                       or {})
            if response.status < 200 or response.status >= 300:
                raise _response_error(f"Download {filename}", response)
            if config.sha256(destination) != artifact["sha256"]:
                raise RuntimeError(
                    f"Package index checksum mismatch for {filename}.")
    return True


def ci_validate() -> None:
    """Validate release wheels produced by the current GitLab pipeline."""
    output = config.REPO_ROOT / "artifacts" / "release" / "release-manifest.json"
    rehearsal = (
        os.environ.get("CI_PIPELINE_SOURCE") == "web"
        and os.environ.get("CI_COMMIT_BRANCH", "").startswith("rehearsal/")
        and os.environ.get("CI_COMMIT_REF_PROTECTED") == "true"
        and os.environ.get("EDGELLM_WHEEL_RELEASE_REHEARSAL") == "true")
    prepare_manifest(
        config.REPO_ROOT / "dist",
        config.required_environment("CI_COMMIT_SHA"),
        config.REPO_ROOT / "artifacts" / "integration" / "gate.json",
        output,
        release_mode="rehearsal" if rehearsal else "release",
    )
    print(output)


def ci_stage() -> None:
    """Upload and verify the current release in Artifactory PyPI."""
    release_root = config.REPO_ROOT / "artifacts" / "release"
    staged_manifest = release_root / "staged-release-manifest.json"
    username = config.required_environment("ARTIFACTORY_USERNAME")
    password = config.required_environment("ARTIFACTORY_TOKEN")
    client = HttpsClient()
    manifest = stage_artifacts(
        release_root / "release-manifest.json",
        config.REPO_ROOT / "dist",
        staged_manifest,
        ARTIFACTORY_PYPI_URL,
        username,
        password,
        client,
    )
    config.write_json(
        release_root / "artifactory-verification.json", {
            "schema_version": 1,
            "package_version": manifest["package_version"],
            "repository_url": ARTIFACTORY_PYPI_URL,
            "staged_manifest_sha256": config.sha256(staged_manifest),
            "status": "byte-identical",
        })


def _release_with_configured_kitmaker(
        manifest: typing.Mapping[str, object], client: ReleaseHttpClient, *,
        upload: bool) -> typing.Dict[str, object]:
    return release_with_kitmaker(
        manifest,
        config.required_environment("KITMAKER_PROJECT_ID"),
        config.required_environment("KITMAKER_PIC"),
        config.required_environment("KITMAKER_API_TOKEN"),
        client,
        upload=upload,
    )


def ci_rehearse() -> None:
    """Run Kitmaker validation without publishing and retain evidence."""
    release_root = config.REPO_ROOT / "artifacts" / "release"
    staged_manifest = release_root / "staged-release-manifest.json"
    manifest = load_manifest(staged_manifest)
    result = _release_with_configured_kitmaker(
        manifest,
        HttpsClient(),
        upload=False,
    )
    config.write_json(
        release_root / "kitmaker-rehearsal-evidence.json", {
            "schema_version": 1,
            "operation": "validation",
            "upload": False,
            "staged_manifest_sha256": config.sha256(staged_manifest),
            "kitmaker": result,
        })


def ci_publish() -> None:
    """Release staged wheels with Kitmaker and verify both public indexes."""
    release_root = config.REPO_ROOT / "artifacts" / "release"
    staged_manifest = release_root / "staged-release-manifest.json"
    manifest = load_manifest(staged_manifest, required_mode="release")
    approval = config.required_environment("RELEASE_APPROVAL_REFERENCE")
    if not re.fullmatch(r"https://gitlab-master\.nvidia\.com/[^\s]+",
                        approval):
        raise RuntimeError(
            "RELEASE_APPROVAL_REFERENCE must be an internal GitLab URL.")
    client = HttpsClient()
    present = [
        verify_index(
            manifest,
            index_url,
            client,
            expected_host=typing.cast(
                str,
                urllib.parse.urlsplit(index_url).hostname),
            allow_absent=True,
        ) for index_url in PUBLIC_INDEX_URLS
    ]
    if all(present):
        result = {"status": "already_published"}
    elif any(present):
        raise RuntimeError(
            "The public indexes disagree about whether the release exists.")
    else:
        result = _release_with_configured_kitmaker(
            manifest,
            client,
            upload=True,
        )
        for index_url in PUBLIC_INDEX_URLS:
            verify_index(
                manifest,
                index_url,
                client,
                expected_host=typing.cast(
                    str,
                    urllib.parse.urlsplit(index_url).hostname),
                wait_seconds=1800,
                tolerate_partial=True,
            )
    config.write_json(
        release_root / "publication-evidence.json", {
            "schema_version":
            1,
            "approval_reference":
            approval,
            "staged_manifest_sha256":
            config.sha256(staged_manifest),
            "kitmaker":
            result,
            "public_indexes": [{
                "simple_url": url,
                "status": "byte-identical-to-qualified-wheels"
            } for url in PUBLIC_INDEX_URLS],
        })
