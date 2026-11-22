# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that collected rollouts carry consistent, trainable token chains.

For every rollout in a Gym rollouts JSONL:
  - every model call has prompt ids, generated ids and one logprob per generated id;
  - each call's prompt starts with the previous call's prompt + generated ids (one unbroken on-policy chain);
And across repeats of the same task:
  - the first call's prompt ids are identical (the same input renders to the same tokens every time).

    python target_rl/scripts/check_token_chains.py results/<run>_rollouts.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def model_calls(rollout: dict) -> list[tuple[list[int], list[int], list[float]]]:
    """(prompt ids, generated ids, logprobs) per model call, in order. One call can produce several output items;
    its token information is attached to one of them."""
    calls = []
    for item in (rollout.get("response") or {}).get("output") or []:
        if item.get("prompt_token_ids") is not None:
            calls.append((item["prompt_token_ids"], item.get("generation_token_ids") or [],
                          item.get("generation_log_probs") or []))
    return calls


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    args = parser.parse_args()

    rollouts = [json.loads(line) for line in args.rollouts.read_text().splitlines() if line.strip()]
    problems: list[str] = []
    first_prompts: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    n_calls = n_multi = 0

    for r in rollouts:
        where = f"task {r.get('_ng_task_index')} repeat {r.get('_ng_rollout_index')}"
        if r.get("mask_sample"):
            problems.append(f"{where}: masked ({r.get('failure_kind')}: {r.get('failure_reason')})")
        calls = model_calls(r)
        if not calls:
            problems.append(f"{where}: no model call carries token ids")
            continue
        n_calls += len(calls)
        n_multi += len(calls) > 1
        first_prompts[r.get("_ng_task_index")].add(tuple(calls[0][0]))
        for k, (prompt, generated, logprobs) in enumerate(calls):
            if not prompt or not generated:
                problems.append(f"{where} call {k}: empty prompt or generated ids")
            if len(logprobs) != len(generated):
                problems.append(f"{where} call {k}: {len(generated)} generated ids but {len(logprobs)} logprobs")
            if k:
                prev_prompt, prev_generated, _ = calls[k - 1]
                expected = prev_prompt + prev_generated
                match = common_prefix(expected, prompt)
                if match != len(expected):
                    problems.append(f"{where} call {k}: prompt diverges from previous prompt+generation at id "
                                    f"{match} of {len(expected)}")

    for task, prompts in sorted(first_prompts.items(), key=lambda kv: str(kv[0])):
        if len(prompts) > 1:
            problems.append(f"task {task}: first prompt renders to {len(prompts)} different token sequences")

    print(f"{len(rollouts)} rollouts, {n_calls} model calls, {n_multi} rollouts with more than one call, "
          f"{len(first_prompts)} tasks")
    for p in problems:
        print(f"  FAIL {p}")
    print("all token chains consistent" if not problems else f"{len(problems)} problems")
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
