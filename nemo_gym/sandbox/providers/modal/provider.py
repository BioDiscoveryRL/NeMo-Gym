# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modal Sandboxes as NeMo Gym sandboxes.

One Modal Sandbox per Gym sandbox, kept alive with `sleep infinity` and driven with `exec`. Use it where Docker is
unavailable, such as inside Modal GPU containers. Images: a name listed under `images` resolves to a published Modal
image (`published: <name>`, no build at launch) or is built from a Dockerfile (`dockerfile: <path>`, relative to the
NeMo Gym checkout); any other name is pulled from its registry. Volumes in
`provider_options["volumes"]` use the Docker form `src:dst[:ro]`, with `src` naming a Modal Volume.
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)


LOGGER = logging.getLogger(__name__)

# Sandbox-runtime failure, not the command's exit code.
SANDBOX_RUNTIME_RETURN_CODE = 125
GYM_ROOT = Path(__file__).resolve().parents[4]


class ModalProvider:
    """Runs each sandbox as a Modal Sandbox; `block_network` (default on) gives it no network access."""

    name = "modal"

    def __init__(
        self,
        app_name: str = "nemo-gym-sandboxes",
        block_network: bool = True,
        images: dict[str, dict[str, str]] | None = None,
        default_timeout_s: float = 180.0,
        concurrency: int = 64,
    ) -> None:
        import modal

        self._modal = modal
        self._app_name = app_name
        self._block_network = block_network
        self._image_sources = images or {}
        self._images: dict[str, Any] = {}
        self._default_timeout_s = default_timeout_s
        self._semaphore = asyncio.Semaphore(concurrency)
        self._app: Any = None

    def _image(self, name: str) -> Any:
        if name not in self._images:
            source = self._image_sources.get(name)
            if source is None:
                self._images[name] = self._modal.Image.from_registry(name)
            elif "published" in source:
                self._images[name] = self._modal.Image.from_name(source["published"])
            else:
                dockerfile = Path(source["dockerfile"])
                dockerfile = dockerfile if dockerfile.is_absolute() else GYM_ROOT / dockerfile
                self._images[name] = self._modal.Image.from_dockerfile(dockerfile, context_dir=dockerfile.parent)
        return self._images[name]

    def _volumes(self, specs: list[str] | str | None) -> dict[str, Any]:
        mounts: dict[str, Any] = {}
        for spec in [specs] if isinstance(specs, str) else specs or []:
            parts = spec.split(":")
            if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] not in ("ro", "rw")):
                raise ValueError(f"Modal sandbox volume must be 'volume_name:/mount/path[:ro|rw]', got {spec!r}")
            volume = self._modal.Volume.from_name(parts[0])
            mounts[parts[1]] = volume.read_only() if len(parts) == 3 and parts[2] == "ro" else volume
        return mounts

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if not spec.image:
            raise ValueError("The Modal sandbox provider needs SandboxSpec.image")
        if self._app is None:
            self._app = await self._modal.App.lookup.aio(self._app_name, create_if_missing=True)
        resources = spec.resources if isinstance(spec.resources, SandboxResources) else SandboxResources.from_mapping(spec.resources)
        kwargs: dict[str, Any] = {
            "app": self._app,
            "image": self._image(spec.image),
            "block_network": self._block_network,
            "volumes": self._volumes(spec.provider_options.get("volumes")),
            "workdir": spec.workdir,
            "timeout": int(spec.ttl_s) if spec.ttl_s else 24 * 60 * 60,
        }
        if spec.env:
            kwargs["secrets"] = [self._modal.Secret.from_dict(dict(spec.env))]
        if resources.cpu:
            kwargs["cpu"] = float(resources.cpu)
        if resources.memory_mib:
            kwargs["memory"] = int(resources.memory_mib)
        sandbox = await self._modal.Sandbox.create.aio("sleep", "infinity", **kwargs)
        if spec.workdir:
            process = await sandbox.exec.aio("mkdir", "-p", spec.workdir)
            await process.wait.aio()
        return SandboxHandle(
            sandbox_id=sandbox.object_id or f"modal-{uuid.uuid4().hex[:12]}",
            provider_name=self.name,
            raw={"sandbox": sandbox, "workdir": spec.workdir},
        )

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        if user is not None:
            raise ValueError("The Modal sandbox provider cannot run commands as another user")
        sandbox = handle.raw["sandbox"]
        timeout = timeout_s if timeout_s is not None else self._default_timeout_s
        async with self._semaphore:
            try:
                process = await sandbox.exec.aio(
                    "bash", "-c", command, workdir=cwd or handle.raw["workdir"], env=env or None, timeout=int(timeout)
                )
                stdout, stderr = await asyncio.wait_for(
                    asyncio.gather(process.stdout.read.aio(), process.stderr.read.aio()), timeout=timeout + 30
                )
                return_code = await process.wait.aio()
            except (TimeoutError, asyncio.TimeoutError):
                return SandboxExecResult(
                    stdout=None,
                    stderr=f"modal command timed out after {timeout:g}s",
                    return_code=SANDBOX_RUNTIME_RETURN_CODE,
                    error_type="timeout",
                )
        if return_code is None or return_code < 0:
            return SandboxExecResult(
                stdout=stdout, stderr=stderr, return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="timeout"
            )
        return SandboxExecResult(stdout=stdout, stderr=stderr, return_code=return_code, error_type=None)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        await handle.raw["sandbox"].filesystem.copy_from_local.aio(source_path, target_path)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        await handle.raw["sandbox"].filesystem.copy_to_local.aio(source_path, target_path)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        code = await handle.raw["sandbox"].poll.aio()
        return SandboxStatus.RUNNING if code is None else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        await handle.raw["sandbox"].terminate.aio()

    async def aclose(self) -> None:
        return None
