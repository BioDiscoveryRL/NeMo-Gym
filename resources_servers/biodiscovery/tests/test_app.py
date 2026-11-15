# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import yaml

from resources_servers.biodiscovery.app import world_mounts

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "biodiscovery.yaml"
DOCKER_TEMPLATES = ["/data/bdrl/worlds/{world_id}:/worlds/worlds/{world_id}:ro", "/data/bdrl/shared/genes:/worlds/shared/genes:ro"]
MODAL_TEMPLATES = ["target-rl-world-{world_id}:/worlds/worlds/{world_id}:ro", "target-rl-genes:/worlds/shared/genes:ro"]


@pytest.mark.parametrize("templates", [DOCKER_TEMPLATES, MODAL_TEMPLATES])
def test_episode_mounts_only_its_own_world(templates: list[str]) -> None:
    mounts = world_mounts(templates, "w934000")
    worlds = {m.split(":")[1] for m in mounts if m.split(":")[1].startswith("/worlds/worlds/")}
    assert worlds == {"/worlds/worlds/w934000"}
    assert all(m.endswith(":ro") for m in mounts)
    assert not any(m.split(":")[1] in ("/worlds", "/worlds/worlds") for m in mounts)


@pytest.mark.parametrize(
    "bad",
    [
        "/data/bdrl:/worlds:rw",  # writable
        "/data/bdrl/worlds/{world_id}:/worlds/worlds/{world_id}",  # no mode
        "/data/bdrl:/data:ro",  # outside /worlds
    ],
)
def test_rejects_writable_or_misplaced_mounts(bad: str) -> None:
    with pytest.raises(ValueError):
        world_mounts([bad], "w934000")


def test_shipped_config_mounts_one_world() -> None:
    server = yaml.safe_load(CONFIG.read_text())["biodiscovery"]["resources_servers"]["biodiscovery"]
    templates = [t.replace("${oc.env:BDRL_WORLDS_ROOT}", "/data/bdrl") for t in server["world_mounts"]]
    test_episode_mounts_only_its_own_world(templates)
