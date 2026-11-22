# target-rl architecture

How the drug-target RL stack is split, what each part owns, and how the parts talk to each other.
This file lives in `BioDiscoveryRL/NeMo-Gym` (`target_rl/ARCHITECTURE.md`); `BioDiscoveryRL/nemo-RL` checks this branch out as `3rdparty/Gym-workspace/Gym`, so both repos read the same copy.

## Layers

| Layer | Repos | Owns | Hands over |
|---|---|---|---|
| 1. World factory | `BioDiscoveryRL/generator`, `BioDiscoveryRL/validator` (planned), `BioDiscoveryRL/contracts`, export half of `BioDiscoveryRL/bench-dev` | Simulated biobank worlds: released files and the sealed answer key; checks that worlds are solvable and leak-free | A world bundle per world id: `release/` (what an agent may read) and `sealed/` (`spec.json`, `key.json`, `indexes/`), published as a pinned Hugging Face dataset revision |
| 2. Task layer | `BioDiscoveryRL/tasks`, `BioDiscoveryRL/contracts`, brief half of `bench-dev` | Turning a world into tasks (`TaskSpec`) with per-task answer keys (`AnswerKey`); the scorer `tasks.scoring.score` | Task rows and keys; the scorer as an importable function |
| 3. Environment and harness | `BioDiscoveryRL/NeMo-Gym`, branch `target-rl` (`target_rl/`, `resources_servers/`, `responses_api_models/vllm_model`) | Running one episode: sandbox with the world mounted, tools, agent loop, calling the scorer in `verify()` | Rollouts: model calls with token ids and logprobs, plus a reward |
| 4. Training | `BioDiscoveryRL/nemo-RL`, branch `target-rl` (`examples/nemo_gym/target_rl/`) | GRPO updates, policy weights, the policy's own generation engine, launch on GPUs | Checkpoints and metrics |

Layers 1 and 2 are owned by the scientists and consumed as inputs; this work never edits them. A new generator or scorer version reaches training by bumping a pin (dataset revision, `tasks` commit), not by changing environment code.

## What talks to what during training

```
NeMo RL (GRPO driver, Ray)
 ├─ starts vLLM on the training GPUs, exposed as an OpenAI-compatible HTTP server
 ├─ starts the NeMo Gym servers from env.nemo_gym.config_paths, pointing policy_model at that vLLM server
 └─ for each batch: sends task rows to NeMo Gym, gets rollouts back, computes advantages, updates LoRA weights,
    then pushes the new weights into vLLM (refit)

NeMo Gym, one episode
 environment server (single_agent_turn)
   1. POST resources /seed_session    task_data (world id, task key) -> sandbox with the world mounted read-only
   2. POST agent /v1/responses        brief only; the task key never reaches the agent
        simple_agent loop: POST model /v1/chat/completions -> tool call -> POST resources /<tool> -> ...
   3. POST resources /verify          final answer + task key -> tasks.scoring.score -> reward
   4. close session, stop sandbox
```

All arrows are HTTP inside one machine (or one Modal container). Every model call goes through the Gym model server (`vllm_model`), which records prompt ids, generated ids and logprobs per call; NeMo RL trains on exactly those tokens.

## Interfaces

| Interface | Shape | Defined by |
|---|---|---|
| World bundle | `worlds/<id>/{dbGaP,wgs}` + `shared/genes` at a pinned dataset revision | generator + bench-dev; schemas in `contracts` |
| Task row (Gym dataset) | JSONL: `responses_create_params` (brief, tools), `agent_ref`, hidden fields (`world_id`, task key) | this branch; key fields follow `contracts.AnswerKey` |
| Scorer | `tasks.scoring.parse_answer(...)`, `tasks.scoring.score(...) -> ScoreReport` | `tasks` (imported at a pinned commit) |
| Model endpoint | OpenAI Chat Completions with token ids and logprobs | vLLM (training) or SGLang (Modal endpoints, via `sglang_token_fields`) |
| Rollout | Responses API output items carrying `prompt_token_ids`, `generation_token_ids`, `generation_log_probs`; `reward`, `mask_sample` | NeMo Gym |

## Inference engines

| Where | Engine | Why |
|---|---|---|
| Inference checks and evaluation | SGLang on Modal endpoints (`discovery-rl` = Qwen3.8-27B, GLM fallback) | How those endpoints are deployed. `vllm_model` with `sglang_token_fields: true` reads their token fields (`target_rl/configs/policy_sglang.yaml`). |
| Training | vLLM started by NeMo RL on the training GPUs | NeMo RL refits weights into its own engine every step, and its NeMo Gym integration takes the engine's OpenAI server URLs, which only its vLLM and TRT-LLM backends provide. |

Keep the model, chat template and sampling identical between the two so evaluation measures the trained policy.

## Compute

The same `train.sh` runs on any machine with 8 GPUs. Modal is a launcher, not a dependency:

| Script | Role |
|---|---|
| `examples/nemo_gym/target_rl/train.sh` (NeMo RL) | Runs GRPO on the current machine |
| `examples/nemo_gym/target_rl/modal_app.py` | Runs `train.sh` in one Modal container with 8 H100s; copies both forks into the NeMo RL release image |
| `examples/nemo_gym/target_rl/modal_worlds.py` | Stages world bundles from Hugging Face into the `target-rl-worlds` Modal volume |

On a dedicated box the worlds live on local disk (`data/biodiscoveryrl/`); on Modal in the `target-rl-worlds` volume. The environment reads the location from config.

## Status

| Piece | State |
|---|---|
| Inference checks (`probe_endpoint.py`, `check_token_chains.py`) | Done: token chains consistent through NeMo Gym on the GLM endpoint; direct probe passes on the 27B |
| SGLang token fields in `vllm_model` | Done |
| GRPO recipe, `train.sh`, Modal launchers | Written; 3-step smoke run on the toy tool-calling environment in progress |
| World staging on Modal | Done: all 17 worlds in the `target-rl-worlds` volume at dataset revision `9c3c4eb` |
| Drug-target resources server (sandbox, `bash` and `submit` tools, scorer in `verify`) | Planned |
| Task rows from the Harbor tasks / bench-dev output | Planned |
| Condenser, sub-agents, LLM judge | Deferred: condensation breaks the token chain training needs |
