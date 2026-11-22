# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe an OpenAI-compatible policy endpoint for what RL training needs.

Checks, in order:
  1. chat        - a plain completion works (thinking off)
  2. thinking    - thinking on returns a reasoning block
  3. tools       - native tool calls come back as structured tool_calls
  4. tokens      - vLLM token information: prompt token ids, generated token ids, per-token logprobs
  5. chaining    - turn 2's prompt token ids start with turn 1's prompt + generated ids (on-policy chain)

Reads policy_base_url / policy_api_key / policy_model_name from env.yaml at the repo root.
Uses only the standard library so it runs before any environment is set up:

    python target_rl/scripts/probe_endpoint.py [--env-file env.yaml] [--thinking]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" in line:
            key, _, value = line.partition(":")
            values[key.strip()] = value.strip().strip("'\"")
    missing = [k for k in ("policy_base_url", "policy_api_key", "policy_model_name") if not values.get(k)]
    if missing:
        sys.exit(f"{path} is missing {missing}")
    if "PASTE_SECRET_HERE" in values["policy_api_key"]:
        sys.exit(f"Fill in the token secret in {path} first.")
    return values


class Endpoint:
    def __init__(self, env: dict[str, str]) -> None:
        self.url = env["policy_base_url"].rstrip("/") + "/chat/completions"
        self.key = env["policy_api_key"]
        self.model = env["policy_model_name"]

    def chat(self, messages: list[dict], *, thinking: bool, timeout: int = 900, **extra: object) -> tuple[dict, float]:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": extra.pop("max_tokens", 1024),
            "temperature": extra.pop("temperature", 0.6),
            # Qwen3 templates read enable_thinking; GLM-style servers read reasoning_effort.
            "chat_template_kwargs": {"enable_thinking": thinking},
            **({} if thinking else {"reasoning_effort": "none"}),
            **extra,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        start = time.time()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read()), time.time() - start
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:600]}") from e


def token_request() -> dict[str, object]:
    # The same fields NeMo Gym's vllm_model adds when return_token_id_information is on.
    return {"logprobs": True, "top_logprobs": 0, "return_tokens_as_token_ids": True, "return_token_ids": True}


def generated_ids(choice: dict) -> list[int]:
    # vLLM names them token_ids, SGLang response_token_ids.
    for key in ("token_ids", "response_token_ids"):
        if choice.get(key):
            return list(choice[key])
    content = (choice.get("logprobs") or {}).get("content") or []
    return [int(item["token"].removeprefix("token_id:")) for item in content]


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(Path(__file__).resolve().parents[2] / "env.yaml"))
    parser.add_argument("--thinking", action="store_true", help="run the token and chaining checks with thinking on")
    args = parser.parse_args()
    ep = Endpoint(read_env(Path(args.env_file)))
    print(f"endpoint {ep.url}  model {ep.model}\n")
    results = []

    # 1. chat
    r, dt = ep.chat([{"role": "user", "content": "Reply with exactly: ok"}], thinking=False, max_tokens=2048)
    text = r["choices"][0]["message"].get("content") or ""
    results.append(report("chat", bool(text.strip()), f"{text.strip()[:60]!r} in {dt:.1f}s"))

    # 2. thinking
    r, dt = ep.chat([{"role": "user", "content": "What is 17*23? Answer with the number."}], thinking=True,
                    max_tokens=2048)
    msg = r["choices"][0]["message"]
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    results.append(report("thinking", bool(reasoning), f"{len(reasoning)} reasoning chars, answer "
                          f"{(msg.get('content') or '').strip()[:40]!r}, {r['usage']['completion_tokens']} tokens, {dt:.1f}s"))

    # 3. tools
    r, dt = ep.chat([{"role": "user", "content": "What's the weather in Paris? Use the tool."}], thinking=True,
                    tools=[WEATHER_TOOL], tool_choice="auto", max_tokens=4096)
    calls = r["choices"][0]["message"].get("tool_calls") or []
    detail = (f"{calls[0]['function']['name']}({calls[0]['function']['arguments']})" if calls
              else f"no tool_calls; content={(r['choices'][0]['message'].get('content') or '')[:120]!r}")
    results.append(report("tools", bool(calls), detail))

    # 4. tokens
    turn1 = [{"role": "user", "content": "Name one prime number between 10 and 20. One word."}]
    r1, _ = ep.chat(turn1, thinking=args.thinking, max_tokens=1024, **token_request())
    c1 = r1["choices"][0]
    p1, g1 = list(r1.get("prompt_token_ids") or c1.get("prompt_token_ids") or []), generated_ids(c1)
    lps = (c1.get("logprobs") or {}).get("content") or []
    results.append(report("tokens", bool(p1) and bool(g1) and len(lps) == len(g1),
                          f"{len(p1)} prompt ids, {len(g1)} generated ids, {len(lps)} logprobs"))

    # 5. chaining: append turn 1's assistant message exactly as returned, then a user turn
    m1 = c1["message"]
    assistant = {"role": "assistant", "content": m1.get("content") or ""}
    if m1.get("reasoning_content") or m1.get("reasoning"):
        assistant["reasoning_content"] = m1.get("reasoning_content") or m1.get("reasoning")
    turn2 = [*turn1, assistant, {"role": "user", "content": "Now name another one. One word."}]
    r2, _ = ep.chat(turn2, thinking=args.thinking, max_tokens=1024, **token_request())
    p2 = list(r2.get("prompt_token_ids") or r2["choices"][0].get("prompt_token_ids") or [])
    prefix = p1 + g1
    match = 0
    while match < min(len(prefix), len(p2)) and prefix[match] == p2[match]:
        match += 1
    ok = bool(p2) and match == len(prefix)
    detail = (f"turn-2 prompt extends turn 1 exactly ({len(prefix)} ids)" if ok else
              f"diverges at id {match} of {len(prefix)} (turn-1 prompt {len(p1)}, generated {len(g1)}); "
              "the chat template re-renders the earlier turn differently")
    results.append(report("chaining", ok, detail))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
