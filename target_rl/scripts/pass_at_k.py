"""pass@k and best-of-k F1 per task from `gym eval run` rollouts of the biodiscovery environment.

    python target_rl/scripts/pass_at_k.py results/passk/train_rollouts.jsonl [--thresholds 0.5 0.8 1.0]

A rollout passes at threshold t when its reward (target-discovery F1) is at least t. pass@k uses the unbiased
estimator 1 - C(n-c, k) / C(n, k) over the n rollouts of each task, averaged over tasks. Also reports each task's
mean and best F1 and the share of tasks whose rollouts all scored the same (groups GRPO would drop). Rows are
grouped by Gym's task index; answer keys are never printed.
"""

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path


def pass_at_k(n: int, c: int, k: int) -> float:
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    args = parser.parse_args()

    rewards: dict[int, list[float]] = defaultdict(list)
    worlds: dict[int, str] = {}
    for line in args.rollouts.read_text().splitlines():
        row = json.loads(line)
        task = int(row["_ng_task_index"])
        rewards[task].append(float(row.get("reward") or 0.0))
        worlds[task] = row.get("world_id", "?")

    n_min = min(len(r) for r in rewards.values())
    ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= n_min]
    print(f"{len(rewards)} tasks, {sum(len(r) for r in rewards.values())} rollouts (n per task >= {n_min})\n")
    print(f"{'task':>4} {'world':>8} {'n':>3} {'mean F1':>8} {'best F1':>8}  rewards")
    for task in sorted(rewards):
        r = rewards[task]
        print(f"{task:>4} {worlds[task]:>8} {len(r):>3} {sum(r) / len(r):>8.3f} {max(r):>8.3f}  "
              + " ".join(f"{x:.2f}" for x in sorted(r, reverse=True)))

    print("\npass@k (mean over tasks)")
    print(f"{'F1 >=':>6} " + " ".join(f"{'k=' + str(k):>7}" for k in ks))
    for t in args.thresholds:
        cells = []
        for k in ks:
            values = [pass_at_k(len(r), sum(x >= t for x in r), k) for r in rewards.values()]
            cells.append(f"{sum(values) / len(values):>7.3f}")
        print(f"{t:>6.2f} " + " ".join(cells))

    best_of_k = []
    for k in ks:
        # Expected best F1 among k rollouts drawn without replacement, exact over all subsets via order statistics.
        total = 0.0
        for r in rewards.values():
            s, n = sorted(r), len(r)
            total += sum(s[i] * comb(i, k - 1) for i in range(k - 1, n)) / comb(n, k)
        best_of_k.append(total / len(rewards))
    print("\nexpected best-of-k F1  " + " ".join(f"k={k}: {v:.3f}" for k, v in zip(ks, best_of_k)))
    uniform = sum(len(set(r)) == 1 for r in rewards.values())
    print(f"tasks whose rollouts all scored the same (dropped by dynamic sampling): {uniform}/{len(rewards)}")


if __name__ == "__main__":
    main()
