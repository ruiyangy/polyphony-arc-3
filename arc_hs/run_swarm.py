#!/usr/bin/env python3
"""
run_swarm.py — CLI launcher for the parallel multi-game swarm.

Examples:
  # offline (zero scorecard cost), 2 games concurrent, continuous batching:
  python3 run_swarm.py --games ft09 sb26 ar25 --parallel-nums 2 --mode offline \
      --environments-dir ../vendor/arc/environment_files

  # competition (consumes ONE real scorecard; needs --arc-key, user-authorized):
  python3 run_swarm.py --games ft09 sb26 --parallel-nums 2 --mode competition \
      --arc-key <YOUR_ARC_KEY>

N (--parallel-nums) is bounded by vLLM KV-cache, NOT by CPU. Calibrate: run
N=1 baseline, then increase while watching (a) aggregate tokens/s (stop when it
stops rising), (b) vLLM 'preemption'/waiting-queue logs (>0 == overloaded),
(c) wall-clock / n_games. Take the knee. Default 2 is a safe start.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import paths  # side-effect: puts the vendored ARC SDK on sys.path
from swarm import Swarm

# Offline game files ship vendored under vendor/arc/environment_files.
DEFAULT_ENVDIR = str(paths.environments_dir())


def _all_offline_games(environments_dir: str) -> list[str]:
    from arc_agi import Arcade, OperationMode
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                 environments_dir=environments_dir)
    seen, out = set(), []
    for env in arc.get_environments():
        base = env.game_id.split("-", 1)[0]
        if base not in seen:
            seen.add(base); out.append(base)
    return out


def main():
    ap = argparse.ArgumentParser(description="Heuristic-System multi-game swarm for ARC-AGI-3")
    ap.add_argument("--games", nargs="+", default=None,
                    help="short game ids (e.g. ft09 sb26 ar25); omit = all 25 public")
    ap.add_argument("--parallel-nums", type=int, default=2,
                    help="max games in flight (KV-cache bound; calibrate). default 2")
    ap.add_argument("--mode", choices=["offline", "competition", "online"],
                    default="offline")
    ap.add_argument("--environments-dir", default=DEFAULT_ENVDIR)
    ap.add_argument("--arc-base-url", default=None,
                    help="online mode: the gateway URL (e.g. http://gateway:8001 "
                         "/ edit http://127.0.0.1:8001). client connects ONLINE.")
    ap.add_argument("--vllm-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--vllm-api-key", default="EMPTY",
                    help="API key for an OpenAI-compatible chat endpoint "
                         "('EMPTY' for a local vLLM server). "
                         "")
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--text-only", action="store_true",
                    help="text-only model (e.g. qwen3.7-max): never send PNG "
                         "frames; read_file on a .png points to the ASCII .txt "
                         "frame instead (same grid, no info lost). Default: "
                         "multimodal (PNG enabled).")
    ap.add_argument("--run-root", default=str(ROOT / "swarm_runs"))
    ap.add_argument("--max-tool-calls-per-send", type=int, default=60)
    ap.add_argument("--arc-key", default=None,
                    help="ARC API key (competition only). main "
                         "<YOUR_ARC_KEY>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-close-scorecard", action="store_true",
                    help="competition: skip close_scorecard so the official site "
                         "shows the partial (open) score.")
    # ── TIME is the primary stop authority (bounded wall-clock budget) ────────────
    ap.add_argument("--global-deadline-epoch", type=float, default=None,
                    help="absolute wall-clock cutoff (time.time() epoch). Anchor it "
                         "anchor at notebook launch BEFORE startup. run() returns "
                         "by it (bounded join), guaranteeing no host hard-kill.")
    ap.add_argument("--per-game-deadline-s", type=float, default=1800.0,
                    help="each game's own time cap in seconds (anti-starvation). "
                         "default 1800 (30min).")
    ap.add_argument("--dispatch-min-s", type=float, default=300.0,
                    help="don't dispatch a NEW game unless this many seconds of "
                         "solve time remain before the global deadline.")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="keep game order as given (default: shuffle, since 110 "
                         "a large game set may exceed the wall-clock budget).")
    args = ap.parse_args()

    games = args.games
    if not games and args.mode == "offline":
        # offline: discover from local disk (the only source — no server).
        games = _all_offline_games(args.environments_dir)
        print(f"[run_swarm] no --games given; using all {len(games)} offline games")
    # competition/online: leave games empty -> Swarm auto-discovers from
    # the LIVE arcade (public server / gateway), so play == scoring == discovery
    # all come from the authoritative server, not local disk.

    if args.mode == "competition" and not args.arc_key:
        raise SystemExit("--mode competition requires --arc-key (consumes a real "
                         "scorecard; must be user-authorized)")
    if args.mode == "online" and not args.arc_base_url:
        raise SystemExit("--mode online requires --arc-base-url (the gateway URL)")

    swarm = Swarm(
        games=games, parallel_nums=args.parallel_nums, mode=args.mode,
        environments_dir=args.environments_dir, vllm_base=args.vllm_base,
        model=args.model, run_root=args.run_root,
        max_tool_calls_per_send=args.max_tool_calls_per_send,
        arc_key=args.arc_key, seed=args.seed,
        no_close=args.no_close_scorecard,
        global_deadline_epoch=args.global_deadline_epoch,
        per_game_deadline_s=args.per_game_deadline_s,
        dispatch_min_s=args.dispatch_min_s,
        shuffle=not args.no_shuffle,
        arc_base_url=args.arc_base_url,
        vllm_api_key=args.vllm_api_key,
        multimodal=not args.text_only)
    report = swarm.run()
    return 0 if report.get("n_games", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
