# target_rl

Our adaptations on the `target-rl` branch: inference checks, policy configs and data for training drug-target discovery agents with NeMo Gym + NeMo RL.
The world generator, task builder and scorer live in the BioDiscoveryRL repos and are consumed as inputs, never edited here.

See [ARCHITECTURE.md](ARCHITECTURE.md) for what each repo owns, how the parts communicate during training, and what is built versus planned.

## Layout

| Path | What it is |
|---|---|
| `configs/policy_sglang.yaml` | Policy model served by SGLang (Modal endpoints). Endpoint, model and key come from `env.yaml`. |
| `scripts/probe_endpoint.py` | Probes an OpenAI-compatible endpoint: chat, thinking, tool calls, token ids + logprobs, multi-turn token chaining. Standard library only. |
| `scripts/check_token_chains.py` | Checks a rollouts JSONL: every model call has token ids and logprobs, each call's prompt extends the previous prompt + generation, and repeats of a task render the same first prompt. |
| `data/smoke/` | Five toy tool-calling tasks (`example_single_tool_call`) with `agent_ref`, for training smoke tests. |

## Inference check

```bash
# env.yaml (gitignored): policy_base_url, policy_model_name, policy_api_key
python target_rl/scripts/probe_endpoint.py            # thinking off
python target_rl/scripts/probe_endpoint.py --thinking # thinking on

gym env start --resources-server example_single_tool_call --config target_rl/configs/policy_sglang.yaml
gym eval run --no-serve --agent example_single_tool_call_simple_agent \
  --input resources_servers/example_single_tool_call/data/example.jsonl \
  --output results/probe_rollouts.jsonl --num-repeats 4
python target_rl/scripts/check_token_chains.py results/probe_rollouts.jsonl
```

Override the endpoint without editing `env.yaml`: append `++policy_base_url=... ++policy_model_name=...` to `gym env start`.

## SGLang endpoints

`vllm_model` reads vLLM's token fields. SGLang returns `choice.prompt_token_ids`, `choice.response_token_ids` and logprob tokens as text, so `policy_sglang.yaml` sets `sglang_token_fields: true`, which requests `return_token_ids` and rewrites each response into the vLLM shape before parsing (`responses_api_models/vllm_model/app.py`).

## Training

Training runs from the NeMo RL fork (`BioDiscoveryRL/nemo-RL`, branch `target-rl`), which checks this branch out as its `3rdparty/Gym-workspace/Gym` submodule. See `examples/nemo_gym/target_rl/README.md` there.
