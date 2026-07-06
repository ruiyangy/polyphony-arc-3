# Polyphony Agent - ARC

Polyphony Agent is our agent harness line for deployable open-weight models. The
bias is simple: put task knowledge where people can see and audit it - tools,
execution control, memory, traces, tests, and policy code - instead of burying it
inside a remote model's weights or transient activations. A self-hosted model can
do more serious work when the surrounding system carries real structure.

This repository applies that idea to ARC-AGI-3. The agent observes interactive
grid games through the returned frames, keeps a small Python policy system in a
per-game workspace, and plans through that system before it spends real actions.

For ARC, we use the Heuristic Learning frame described in Jiayi Weng's
[Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/):
the object being improved is not a neural weight vector, but a Heuristic System
made of code, traces, tests, planner, verifier, and memory. In this repo that
system is the solver the agent grows while playing.

## How The Harness Works

A run starts from `arc_hs/run_swarm.py`, which gives each game its own workspace
and a plain file-and-shell workbench, then runs one coding agent per game. Inside
that workspace the agent sees frames, metadata, the available actions, and its own action history. 
It never reads the game source, hidden state, external memory, or a solution database.

![Polyphony Agent — ARC architecture: an open-model backend drives the harness, which grows and runs a Heuristic System (state / engine / planner / verifier) against the ARC game through an Observe → Edit → Plan → Act loop, with all traces and memory kept as readable files.](docs/assets/2026-07-06_arc_pipeline.png)

The agent treats a game the way a programmer treats an unfamiliar system: it
turns playing into writing and debugging a small program. That program is the
**Heuristic System** — a Python policy that captures how the game behaves, along
with a state representation, a planner, and a verifier — kept as ordinary files
the agent edits. The workspace ships only empty scaffolding; the policy that
actually solves a given game is written from scratch as the agent plays and
watches what happens, not shipped with the repo.

Inside the loop, after each edit the agent runs its policy against the frames the
game really returned. The policy is accepted only when it reproduces those
transitions exactly; any frame it gets wrong is a concrete bug to fix in the next
round. A policy that holds up is then used to plan: the agent searches it for an
action sequence, plays that sequence against the live game, and checks the frames
again. Real moves are spent on plans the policy already backs, not on trial and
error.

Because the policy is code, what the agent learns stays out in the open. The
state representation, the rules it tried and dropped, the planner, and the traces
behind them are all readable files — you can see why the agent acts as it does,
correct it, and audit it, instead of trusting behavior sealed inside weights.
File memory and context compaction carry this working state across a long game,
so it does not have to live inside the model's conversation window.

The model backend is intentionally ordinary — an OpenAI-compatible Chat
Completions endpoint with tool calling. Vision models can read PNG frames;
text-only models run the same loop from ASCII grids.

## Repository Layout

```text
polyphony-arc-3/
├── arc_hs/               harness orchestration and coding-agent loop
│   ├── run_swarm.py      CLI entry point
│   ├── swarm.py          concurrent multi-game scheduler
│   ├── agent.py          per-game protocol driver
│   ├── coding_session.py LLM tool loop for editing and testing the workspace
│   ├── hs_compaction.py  context compaction for long runs
│   ├── prompts/          protocol prompts
│   └── workspace/        per-game Heuristic System template
├── compat/               model adapter and execution sandbox
│   ├── qwen_policy.py    OpenAI-compatible chat adapter used by the local path
│   ├── policy.py         small model-interface contract
│   └── sandbox.py        supervised shell execution
├── vendor/arc/           vendored ARC SDK and public environment files
│   ├── arc_agi/
│   ├── arcengine/
│   └── environment_files/
├── scripts/
│   └── preflight_release_check.sh
└── requirements.txt
```

The ARC SDK and public offline game files are vendored under `vendor/arc/`, so a
fresh clone does not need a machine-level `.pth` file or a separate ARC SDK
install. See `vendor/arc/NOTICE.md` for the vendored ARC license notice.

## Running

The runner expects an OpenAI-compatible Chat Completions endpoint with tool
calling. The default examples assume local vLLM.

Serve a model:

```bash
vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 \
  --max-model-len 262144 --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --limit-mm-per-prompt '{"image":30}' \
  --enable-prefix-caching
```

Offline smoke test:

```bash
cd arc_hs
python3 run_swarm.py --games ft09 --parallel-nums 1 --mode offline \
  --per-game-deadline-s 1800 --vllm-base http://127.0.0.1:8000/v1
```

Text-only smoke test:

```bash
cd arc_hs
python3 run_swarm.py --games ft09 --parallel-nums 1 --mode offline --text-only \
  --per-game-deadline-s 1800 --vllm-base http://127.0.0.1:8000/v1 \
  --model <your-text-only-served-model>
```

Competition scorecard run:

```bash
cd arc_hs
WM_NO_SANDBOX=1 WM_TOKENIZER_DIR=<PATH_TO_Qwen3.6-27B> \
HS_COMPRESSION_EVERY=3 HS_LIGHT_MAX_CALLS=15 \
python3 run_swarm.py --mode competition --arc-key <YOUR_ARC_KEY> \
  --parallel-nums 5 --per-game-deadline-s 14400 \
  --global-deadline-epoch $(python3 -c "import time;print('%.0f'%(time.time()+24*3600))") \
  --dispatch-min-s 10800 --max-tool-calls-per-send 40 \
  --vllm-base http://127.0.0.1:8000/v1 --model Qwen/Qwen3.6-27B \
  --run-root swarm_runs/run1
```

Notes:

- `--model` must match the model name served by the endpoint.
- `--vllm-base` points to the OpenAI-compatible base URL.
- `--text-only` prevents PNG image blocks and uses ASCII frames only.
- `--parallel-nums` controls the number of games in flight. Tune it to the
  serving backend's KV-cache capacity.
- With no `--games`, offline mode discovers vendored public games. Competition
  and online modes discover games from the authoritative ARC server or gateway.
