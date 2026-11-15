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
"""Drug-target discovery in a simulated biobank world, one sandbox per episode.

Each episode runs in its own sandbox (NeMo Gym sandbox layer, Docker by default) built from the analysis toolkit
image, with no network, the worlds mounted read-only at /worlds and a writable trial folder at /trial. The agent works
through a single `bash` tool; the episode's paths are environment variables in the sandbox ($DBGAP, $REFERENCE,
$TRIAL, $SCRATCH). The agent commits by writing $TRIAL/committed/run_summary.json; `verify` reads it out of the
sandbox and scores it with the BioDiscoveryRL scorer (`tasks.scoring`) against the task's answer key, which travels
in the task row and never enters the sandbox.
"""

import json
import shlex
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxResources,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint

WORLDS = "/worlds"
TRIAL = "/trial"
ANSWER = f"{TRIAL}/committed/run_summary.json"
REMINDER_FRACTIONS = (0.6, 0.8, 0.9)


class BioDiscoveryResourcesServerConfig(BaseResourcesServerConfig):
    sandbox_provider: str = "sandbox"
    """Top-level provider block in the global config (Docker by default; see configs/biodiscovery.yaml)."""
    image: str = "bdrl-toolkit:latest"
    """Analysis toolkit image (toolkit/Dockerfile): python with pandas etc., plink2, bcftools, an unprivileged user."""
    worlds_root: str
    """Host folder with the dataset layout (worlds/<world_id>/{dbGaP,wgs}, shared/genes), mounted read-only."""
    sandbox_resources: dict[str, Any] = Field(default_factory=lambda: {"cpu": 4, "memory_mib": 8192})
    sandbox_ttl_seconds: int = 6 * 60 * 60
    command_timeout_seconds: int = 600
    max_output_chars: int = 8000
    step_budget: int = 40


class BashRequest(BaseModel):
    command: str


class BioDiscoveryVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    world_id: str
    answer_key: dict[str, Any]


class BioDiscoveryVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    score_components: dict[str, float]


@dataclass
class _Episode:
    sandbox: AsyncSandbox
    steps: int = 0


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head, tail = limit * 2 // 3, limit // 3
    return f"{text[:head]}\n...[{len(text) - head - tail} characters omitted]...\n{text[-tail:]}"


class BioDiscoveryResourcesServer(SimpleResourcesServer):
    config: BioDiscoveryResourcesServerConfig
    _episodes: dict[str, _Episode] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/bash")(self.bash)
        return app

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest | BaseSeedSessionRequest
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        if not isinstance(body, ResourcesSeedSessionRequest):
            raise HTTPException(422, "biodiscovery needs a task row with world_id; seed with a resources session")
        session_id = body.resources_session_id
        if session_id not in self._episodes:
            world_id = (body.task_data or {}).get("world_id")
            if not world_id:
                raise HTTPException(422, "task row has no world_id")
            self._episodes[session_id] = _Episode(sandbox=await self._start_sandbox(session_id, str(world_id)))
        request.session[SESSION_ID_KEY] = session_id
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def _start_sandbox(self, session_id: str, world_id: str) -> AsyncSandbox:
        global_config = get_global_config_dict()
        world = f"{WORLDS}/worlds/{world_id}"
        spec = SandboxSpec(
            image=self.config.image,
            ttl_s=self.config.sandbox_ttl_seconds,
            workdir=f"{TRIAL}/workspace",
            env={
                "DBGAP": f"{world}/dbGaP",
                "REFERENCE": f"{TRIAL}/.reference",
                "TRIAL": TRIAL,
                "SCRATCH": f"{TRIAL}/scratch",
                "TMPDIR": f"{TRIAL}/scratch",
                "HOME": TRIAL,
                "OMP_NUM_THREADS": "4",
                "POLARS_MAX_THREADS": "4",
            },
            resources=SandboxResources.from_mapping(self.config.sandbox_resources),
            metadata=resolve_provider_metadata(self.config.sandbox_provider, global_config)
            | {"nemo_gym_agent": self.config.name, "world_id": world_id, "session": session_id[:63]},
            provider_options={"volumes": [f"{self.config.worlds_root}:{WORLDS}:ro"]},
        )
        sandbox = AsyncSandbox(resolve_provider_config(self.config.sandbox_provider, global_config))
        await sandbox.start(spec)
        setup = (
            f"test -d {shlex.quote(world)}/dbGaP && test -d {shlex.quote(world)}/wgs"
            f" && mkdir -p {TRIAL}/workspace {TRIAL}/committed {TRIAL}/scratch {TRIAL}/.reference"
            f" && ln -sfn {shlex.quote(world)}/wgs {TRIAL}/.reference/wgs"
            f" && ln -sfn {WORLDS}/shared/genes {TRIAL}/.reference/genes"
        )
        result = await sandbox.exec(setup, cwd="/", timeout_s=60)
        if result.return_code != 0 or result.error_type is not None:
            await sandbox.stop()
            raise HTTPException(503, f"sandbox setup for world {world_id} failed: {result.stderr or result.error_type}")
        return sandbox

    def _episode(self, request: Request) -> _Episode:
        session_id = request.session.get(SESSION_ID_KEY)
        episode = self._episodes.get(session_id) if session_id else None
        if episode is None:
            raise HTTPException(409, "no seeded biodiscovery session for this request")
        return episode

    async def bash(self, request: Request, body: BashRequest) -> PlainTextResponse:
        # Plain text, because the agent shows the response body to the model as the tool output.
        episode = self._episode(request)
        episode.steps += 1
        result = await episode.sandbox.exec(body.command, timeout_s=self.config.command_timeout_seconds)
        if result.error_type is not None:
            output = f"[command did not complete: {result.error_type}]"
        else:
            text = (result.stdout or "") + (result.stderr or "")
            output = f"[exit {result.return_code}]\n{_clip(text, self.config.max_output_chars)}"
        return PlainTextResponse(output + await self._budget_note(episode))

    async def _budget_note(self, episode: _Episode) -> str:
        budget, steps = self.config.step_budget, episode.steps
        if steps < budget and not any(steps == int(budget * f) for f in REMINDER_FRACTIONS):
            return ""
        committed = (await episode.sandbox.exec(f"test -f {ANSWER}", timeout_s=30)).return_code == 0
        state = "Committed already; commit again if you improve the result." if committed else "Nothing is committed yet."
        if steps >= budget:
            return (f"\n[budget] Step {steps} of {budget}: the budget is used up. Write $TRIAL/committed/run_summary.json "
                    f"now from the results you have, then stop. {state}")
        return f"\n[budget] Step {steps} of {budget} ({budget - steps} left). {state}"

    async def verify(self, request: Request, body: BioDiscoveryVerifyRequest) -> BioDiscoveryVerifyResponse:
        from contracts import AnswerKey
        from tasks.scoring import parse_answer, score_answer

        key_fields = dict(body.answer_key)
        aliases = key_fields.pop("aliases", None) or {}
        key = AnswerKey(**key_fields)
        content: object = None
        episode = self._episodes.get(request.session.get(SESSION_ID_KEY, ""))
        if episode is not None:
            result = await episode.sandbox.exec(f"cat {ANSWER}", timeout_s=60)
            if result.return_code == 0 and result.error_type is None:
                try:
                    content = json.loads(result.stdout or "")
                except json.JSONDecodeError:
                    content = None
        report = score_answer(parse_answer(key.task_id, content), key, aliases)
        return BioDiscoveryVerifyResponse(
            **body.model_dump(), reward=report.score, score_components=dict(report.components)
        )

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        episode = self._episodes.pop(body.resources_session_id, None)
        if episode is not None:
            await episode.sandbox.stop()
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)


if __name__ == "__main__":
    BioDiscoveryResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = BioDiscoveryResourcesServer.run_webserver()  # noqa: F401
