# Polyphony Agent — ARC

**Polyphony Agent — ARC** is the ARC-AGI-3 member of the **Polyphony Agent**
series: agents built to run on **small, open-weight models** rather than a
frontier closed API. This repository is Polyphony's instance for
[ARC-AGI-3](https://arcprize.org/) — a **training-free** agent that solves the
interactive grid games by **growing a Heuristic System**, not by driving a neural
policy move-by-move.

The intelligence is meant to live in the harness, so a modest open model can
carry it: an open-weight coding agent (validated with **Qwen3.6-27B served
locally via vLLM**) sits behind a controlled experiment bench and incrementally
writes, tests, and simplifies a policy in code that *is* the solver. The game is
the environment this policy learns from; passing levels efficiently is the
verifiable reward that tells the agent whether its code is right.

Polyphony Agent — ARC follows an **executable heuristic-system loop**: infer
dynamics from frames → write/update a small Python policy → verify it against
recorded transitions → plan through it before spending real actions. That
infer-model / plan / verify pattern is a known lineage; what this project focuses
on is **running it on a small, self-hosted open model** — see §2.

> **Naming.** *Polyphony Agent* is our product/series brand; *Heuristic System
> (HS)* is the method — the code policy (engine + planner + verifier + memory)
> the agent grows and maintains. The two layers are independent.

---

## 1. The idea: Heuristic Learning, not gradient learning

This project is a concrete instance of **Heuristic Learning (HL)** — the paradigm
described in Jiayi Weng's *Learning Beyond Gradients*
([blog post](https://trinkle23897.github.io/learning-beyond-gradients/)). The shift:

| | Deep RL | This system (HL) |
|---|---|---|
| Policy | neural-net weights | **code** (engine + planner) |
| State | learned features | explicit variables / detectors |
| Action | one forward pass | run the planner over the model |
| Feedback | scalar reward | env result + verifier + logs, read by the coding agent |
| Update | backprop | **the agent rewrites the code** |
| Memory | replay buffer | explicit files: model doc, reasoning log, failed-idea notes |

The object being improved is not a weight vector but a **Heuristic System (HS)**:
a living software system with a program strategy, a state representation, a
feedback entry point, regression checks, memory, and an update mechanism carried
out by the coding agent. A healthy HS does two things as it grows — **absorb
feedback** (fold each failure back into code) and **compress history** (fold
accumulated special-cases back into a simpler representation). Both are built into
the loop.

## 2. What makes this different from other ARC-AGI-3 agents

Many ARC-AGI-3 agents are, at their core, **a neural policy playing the game**: a
(V)LM looks at each frame and emits the next action(s). Polyphony Agent is
deliberately the other shape — the LLM **writes and repairs a policy in code**,
and that code (not a model forward pass) is what chooses actions. The LLM's job is
to grow and maintain the policy from real frames as evidence.

In line with the Polyphony series' bet on **small, open-weight models**, what this
project focuses on is running that shape on an **open, self-hosted model** rather
than a frontier closed API:

- **Self-hosted Qwen/vLLM first.** The backend is an OpenAI-compatible Chat
  Completions endpoint with tool-calling, plus sampling-compat gates so the same
  harness drives open models served locally.
- **Multimodal *or* ASCII text-only frames.** Non-vision open models can drive
  the whole loop on ASCII frames alone; vision models get PNGs. Same harness.
- **Context compaction built for long unattended runs.** A dedicated compaction
  component keeps the coding agent's working memory coherent across a long game
  so a modest open model doesn't lose the thread as context grows.
- **Object-centric observation toolkit** (`object_tools.py`) the agent can build
  its state representation on.
- **Concurrent swarm runner.** Many games are played in parallel, sized to the
  serving backend, for high-throughput scorecard runs.

The method framing (Heuristic Learning / HS) emphasizes what the loop
*maintains*: absorb each failure back into code, and periodically compress
accumulated special-cases into a simpler representation. The harness externalizes
much of the problem-solving state into code, tests, traces, and planners, which is
what lets a modest open model carry it.

## 3. What the agent is *not* given (honest test-time play)

The agent plays the way a person sitting down to a new game would — it earns
everything from the frames it observes. Concretely, it does **not**:

- **read or reverse-engineer game source code**, hidden state, or engine
  internals — it only ever sees the frames the client returns;
- use any **pre-built memory, solution database, hints, or per-game
  meta-strategy** — nothing about specific games is baked in ahead of time;
- get any **human-authored per-game knowledge** or offline tuning between games.

It is a single, ordinary **test-time** run over the whole game set: the agent
meets each game cold, builds its Heuristic System from live observations, and
moves on. The only thing carried across games is the *method* (this harness and
its prompts), never game-specific answers.

## 4. How the Heuristic System works

The Heuristic System the agent grows is a small, living policy made of four
cooperating roles:

- **Predictor** — predicts how the game reacts to an action, the core the
  rest of the system plans over;
- **Planner** — searches for the shortest reliable route to the goal;
- **Verifier** — verifies the policy against real frames, so any mismatch becomes
  a verifiable reward that guides the coding agent to fix and update the policy;
- **Memory** — explicit, human-readable files that survive across levels.

The agent drives this per game as a loop: read where it is → decide what kind of
step is needed → let the coding agent probe, edit the code, and execute a planned action sequence →
repeat until the game is solved or the budget runs out.

## 5. Tools the agent may use (its lab equipment)

The workspace ships helpers the agent can import *and improve* — lab equipment,
not answers fed to it. The headline one is an **object-centric observation layer**
(`object_tools.py`): connected-component segmentation plus object relations (shape
hashes, adjacency, containment, boundaries) that let the agent reason about the
board in terms of objects and their structure when it writes its state
representation and engine — a real capability added on top of the raw
frame/model/planner loop. Alongside it are state-reconstruction, plotting, and
session-reading helpers. The agent decides whether and how to use any of them;
nothing is force-injected into the model's input.

## 6. Repository layout

```
polyphony-arc-3/
├── arc_hs/               orchestration + the coding agent
│   ├── run_swarm.py      CLI entry point
│   ├── swarm.py          concurrent multi-game scheduler
│   ├── agent.py          per-game protocol driver (HSAgent)
│   ├── coding_session.py the coding agent (talks to the LLM; grows the HS)
│   ├── hs_compaction.py  context auto-compaction
│   ├── prompts/          protocol prompts (the method, in words)
│   └── workspace/        the HS template copied into each game's workspace
├── compat/               model client + execution sandbox
│   └── qwen_policy.py, sandbox.py, policy.py
├── vendor/arc/           vendored ARC SDK — self-contained, no external install
│   └── arc_agi/, arcengine/, environment_files/
└── requirements.txt
```

Everything needed to run is in-repo: the ARC SDK and the public offline game
files (**25 public game ids; 26 vendored environment versions — one game has two
versions, deduplicated by short id at discovery**) are vendored under
`vendor/arc/`, so a fresh clone runs without any machine-level `.pth` or separate
SDK install. The vendored SDK and game files are the ARC Prize Foundation's,
redistributed under MIT — see `vendor/arc/NOTICE.md`.

## 7. Running it

The LLM is any **OpenAI-compatible chat-completions endpoint with tool-calling**.
The default is a local vLLM serving an open Qwen model — bring your own endpoint
by pointing `--vllm-base` / `--model` elsewhere.

Serve the model (example, 8×GPU):

```bash
vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 \
  --max-model-len 262144 --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --limit-mm-per-prompt '{"image":30}' \
  --enable-prefix-caching
```

Small offline smoke (no scorecard, no ARC key — the way to validate the harness):

```bash
cd arc_hs
python3 run_swarm.py --games ft09 --parallel-nums 1 --mode offline \
    --per-game-deadline-s 1800 --vllm-base http://127.0.0.1:8000/v1
```

Text-only smoke (drive a non-vision open model — ASCII frames only, never PNGs):

```bash
cd arc_hs
python3 run_swarm.py --games ft09 --parallel-nums 1 --mode offline --text-only \
    --per-game-deadline-s 1800 --vllm-base http://127.0.0.1:8000/v1 \
    --model <your-text-only-served-model>
```

Full public competition scorecard (consumes one real scorecard):

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

**Game discovery differs by mode.** With no `--games`, `offline` mode discovers
the local vendored public games; `competition`/`online` mode discovers games from
the authoritative ARC server / gateway and scores against it — offline vendored
files are never used for scoring.

> `--model` must match the name your endpoint serves. `Qwen/Qwen3.6-27B` is the
> default placeholder; point `--model` (and `vllm serve <path> --served-model-name`)
> at whatever your vLLM actually serves. Set `MPLCONFIGDIR=/tmp/matplotlib` if the
> vendored SDK's renderer warns about a non-writable matplotlib cache.

### Key flags

- `--vllm-base` — base URL of the OpenAI-compatible endpoint (`EMPTY` key ok for a
  bare local vLLM via `--vllm-api-key`).
- `--model` — served model name / path (default `Qwen/Qwen3.6-27B`).
- `--text-only` — for text-only models: feed ASCII frames, never PNGs (lets a
  non-vision open model drive the same harness).
- `--parallel-nums` — games in flight (KV-cache bound; calibrate).
- `--max-tool-calls-per-send` — per-send tool-call ceiling.

### Env knobs

- `HS_COMPRESSION_EVERY` (default `3`) — fire the heavy compression pass every Nth
  iteration; `1` restores always-on.
- `HS_LIGHT_MAX_CALLS` (default `15`) — tool-call budget for light nudges.
- `HS_PER_GAME_ACTION_CAP` (default `750`) — hard cap on real actions per game.
- `WM_NO_SANDBOX=1` — disable the strace sandbox (already-isolated containers).
- `WM_TOKENIZER_DIR` — path to the model's tokenizer for exact context accounting.
