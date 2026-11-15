# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert BioDiscoveryRL Harbor task folders into NeMo Gym rows for the biodiscovery resources server.

Each row carries the brief (the task's instruction.md, with the cluster paths replaced by the episode's environment
variables and the budget sentence set to the step budget), the bash tool, the agent to run it, and two hidden fields
for the resources server: world_id and answer_key (tests/truth/key.json). overfit-* tasks go to train.jsonl and
heldout-* tasks to validation.jsonl.

    python target_rl/scripts/harbor_to_jsonl.py ../../data/biodiscoveryrl/harbor/tasks target_rl/data/biodiscovery
"""

import argparse
import json
import re
import tomllib
from pathlib import Path

AGENT = "biodiscovery_simple_agent"

# Longest paths first, so a prefix never rewrites part of a longer path.
PATHS = (
    ("/projects/euan/main/sam/discovery-traces/reference/wgs", "$REFERENCE/wgs"),
    ("/projects/euan/main/sam/discovery-traces/reference/genes", "$REFERENCE/genes"),
    ("/projects/euan/main/sam/dbGaP", "$DBGAP"),
    ("/app/trial", "$TRIAL"),
    ("/tmp/scratch", "$SCRATCH"),
)

ENVIRONMENT_NOTE = (
    "Paths in this brief use environment variables that are set in your shell: $DBGAP (study data and its "
    "documentation, read-only), $REFERENCE (analysis-ready genotypes and gene annotation, read-only), $TRIAL (your "
    "trial folder) and $SCRATCH (scratch space). Use them exactly as written, in commands and in scripts "
    "(os.environ in Python). You work through the bash tool: each call runs one command in $TRIAL/workspace and "
    "returns its output; shell state such as cd or exported variables does not carry over between calls.\n\n"
)

BASH_TOOL = {
    "type": "function",
    "name": "bash",
    "description": "Run one bash command in your trial workspace and return its exit code and output.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The bash command to run."}},
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": True,
}


def brief(instruction: str, step_budget: int) -> str:
    text = instruction
    for old, new in PATHS:
        text = text.replace(old, new)
    text = re.sub(r"Budget: about [^.]*\.", f"Budget: {step_budget} tool calls.", text)
    text = text.replace("Use your shell to run commands.", "Use the bash tool to run commands.")
    return ENVIRONMENT_NOTE + text


def row(task_dir: Path, step_budget: int) -> dict:
    meta = tomllib.loads((task_dir / "task.toml").read_text())["metadata"]
    key = json.loads((task_dir / "tests" / "truth" / "key.json").read_text())
    if key["task_id"] != meta["task_id"]:
        raise ValueError(f"{task_dir}: key is for {key['task_id']}, task.toml says {meta['task_id']}")
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": brief((task_dir / "instruction.md").read_text(), step_budget)}],
            "tools": [BASH_TOOL],
        },
        "agent_ref": {"type": "responses_api_agents", "name": AGENT},
        "world_id": meta["world_id"],
        "answer_key": key,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("harbor_tasks", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--step-budget", type=int, default=40)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for prefix, split in (("overfit-", "train"), ("heldout-", "validation")):
        tasks = sorted(p for p in args.harbor_tasks.iterdir() if p.name.startswith(prefix))
        with (args.out_dir / f"{split}.jsonl").open("w") as f:
            for task_dir in tasks:
                f.write(json.dumps(row(task_dir, args.step_budget)) + "\n")
        print(f"{split}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
