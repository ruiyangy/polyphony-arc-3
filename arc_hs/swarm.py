#!/usr/bin/env python3
"""
swarm.py — run MANY ARC games concurrently on the Heuristic-System
harness, with continuous batching (a finished game is immediately replaced by
the next from the queue, keeping <= N games in flight).

Why: a single game leaves the vLLM server (8xA800, TP=8) badly under-utilised.
vLLM does token-level continuous batching internally, so N concurrent request
streams fill the batch and aggregate throughput rises ~linearly until KV-cache
saturates. Competition is also time-limited, and the server accepts one
scorecard shared across many concurrently-played games (verified live).

Design (see plan encapsulated-herding-falcon.md):
  - threads, single shared Arcade for competition (per the split
    proved a split session => 400 "game not found"; the GAMESESSION + ALB
    cookies must persist on ONE Arcade for the whole run).
  - pull queue: N daemon workers each loop `q.get_nowait(); run_one_game()`.
  - per-game isolation: unique run_dir (copytree workspace), per-game local
    server (OfflineARCServer offline / CompetitionProxyServer competition, both
    port=0), per-run agent.log, thread-local runner context for server_url.
  - each game runs the HSAgent loop to a terminal state.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from pathlib import Path

import agent as agent_mod
from coding_session import CodingSession
from offline_arc_server import OfflineARCServer
from session_inspector import inspect_sessions


# ── runner factory: reads per-worker context from the thread-local ───────────
def _qwen_factory(work_dir, log_file, model, reasoning_effort, error_handling_manual):
    ctx = agent_mod._RUNNER_CTX
    return CodingSession(
        work_dir=work_dir, log_file=log_file,
        model=getattr(ctx, "model", None) or model,
        server_url=ctx.server_url,                       # set per worker, same thread
        vllm_base=getattr(ctx, "vllm_base", "http://127.0.0.1:8000/v1"),
        vllm_api_key=getattr(ctx, "vllm_api_key", "EMPTY"),
        max_tool_calls_per_send=getattr(ctx, "max_tool_calls_per_send", 60),
        deadline=getattr(ctx, "deadline", None),
        multimodal=getattr(ctx, "multimodal", True))


def _ts() -> str:
    # caller passes a base timestamp; avoid Date.now-style nondeterminism issues
    return time.strftime("%Y%m%d_%H%M%S")


def _discover_full_ids(arcade, wanted: list[str]) -> dict[str, str]:
    """short id (ft09) -> full id (ft09-<ver>) from the arcade's env list."""
    full: dict[str, str] = {}
    for env in arcade.get_environments():
        gid = env.game_id
        base = gid.split("-", 1)[0]
        if base in set(wanted) and base not in full:
            full[base] = gid
    return full


def _stop_reason(run_dir: Path, insp) -> str:
    """Best-effort terminal-reason from artifacts (agent.run() only returns 0/1).
    inspection first, then the per-run agent.log 'stop condition met' reason."""
    if insp.is_solved:
        return "solved"
    if insp.is_game_over:
        return "game_over"
    if insp.n_steps_current_level >= 750:
        return "level_step_cap_750"
    # fall back to the logged stop reason if present
    log = run_dir / "agent.log"
    try:
        last = ""
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "stop condition met" in line:
                try:
                    last = json.loads(line).get("reason", "") or last
                except Exception:
                    pass
        if last:
            return f"stop:{last}"
    except Exception:
        pass
    return "ended"


class Swarm:
    def __init__(self, games, parallel_nums, mode, environments_dir,
                 vllm_base, model, run_root, max_tool_calls_per_send=60,
                 arc_key=None, seed=0, no_close=False,
                 global_deadline_epoch=None, per_game_deadline_s=1800.0,
                 dispatch_min_s=300.0, shuffle=True, arc_base_url=None,
                 vllm_api_key="EMPTY", multimodal=True):
        self.games = list(games) if games else []
        self.N = max(1, int(parallel_nums))
        self.mode = mode                       # "offline" | "competition" | "online"
        self.environments_dir = environments_dir
        # online mode: arc_base_url = the gateway (e.g. http://gateway:8001 /
        # edit: local listen_and_serve). The agent side connects ONLINE to it.
        self.arc_base_url = arc_base_url
        self.vllm_base = vllm_base
        self.vllm_api_key = vllm_api_key       # API key for an OpenAI-compatible
                                               # LLM gateway (e.g. qwen3.7-plus);
                                               # "EMPTY" for a local vLLM server.
        self.multimodal = multimodal           # False -> text-only model (e.g.
                                               # qwen3.7-max): no PNG sent, ASCII only.
        self.model = model
        self.run_root = Path(run_root)
        self.max_tool_calls_per_send = max_tool_calls_per_send
        self.arc_key = arc_key
        self.seed = seed
        # ── TIME is the primary authority under a bounded wall-clock budget ──────────
        # global_deadline_epoch: absolute wall-clock cutoff (anchored at notebook
        #   launch, BEFORE startup, so startup cost is inside budget). run() is
        #   guaranteed to return by it (bounded join) — no host hard-kill.
        # per_game_deadline_s: each game's own time cap (anti-starvation: one
        #   slow game must not eat the budget of the other ~110).
        # dispatch_min_s: don't dispatch a NEW game unless at least this much
        #   solve time remains before the global deadline.
        # shuffle: a large game set may exceed the wall-clock budget, so randomise order each
        #   run to avoid always covering the same prefix.
        self.global_deadline_epoch = global_deadline_epoch
        self.per_game_deadline_s = per_game_deadline_s
        self.dispatch_min_s = dispatch_min_s
        self.shuffle = shuffle
        # skip close_scorecard so the official site shows the
        # partial (open) score (useful for debugging / hosted reruns; a formal
        # result should close the scorecard). Local results are
        # still saved. NOT an officially-closed score.
        self.no_close = no_close
        self.results: list[dict] = []
        self._results_lock = threading.Lock()
        # monotonic run counter -> unique run_dir even when the SAME game runs
        # multiple times (repeated --games ft09 ft09 ...) on the same worker via
        # continuous batching ({game}__w{wid}__{ts} alone would collide).
        self._run_seq = 0
        self._seq_lock = threading.Lock()
        self.shared_arcade = None
        self.card_id = None
        self.full_ids: dict[str, str] = {}
        self._batch_ts = f"{_ts()}_{os.getpid()}"

    # ── setup / teardown ────────────────────────────────────────────────────
    def _setup_competition(self):
        from arc_agi import Arcade, OperationMode
        if not self.arc_key:
            raise SystemExit("competition mode requires --arc-key")
        self.shared_arcade = Arcade(arc_api_key=self.arc_key,
                                    operation_mode=OperationMode.COMPETITION)
        self.card_id = self.shared_arcade.open_scorecard(tags=["polyphony"])
        # Auto-discover ALL games from the COMPETITION arcade (public server) when
        # none were given — same as _setup_online. The authoritative public game
        # set lives on the server, not on local disk; discovering from the live
        # arcade keeps play == scoring == discovery all on the same source (no
        # local/server mismatch in ids or versions).
        if not self.games:
            self.games = self._discover_online_games()
            print(f"[swarm] competition auto-discovered {len(self.games)} games: "
                  f"{self.games[:8]}{'...' if len(self.games) > 8 else ''}")
        self.full_ids = _discover_full_ids(self.shared_arcade, self.games)
        print(f"[swarm] competition card_id={self.card_id} "
              f"games={len(self.games)} resolved={list(self.full_ids)}")

    def _setup_online(self):
        """ONLINE mode: ONE shared Arcade(ONLINE, arc_base_url=gateway).
        Do NOT open/close a scorecard — the gateway uses its default card and the
        the hosting framework settles it at the end of a rerun (rerun
        plays only). Scoring is per-action via each env.step() POST. Mirrors
        the online probe path.
        """
        from arc_agi import Arcade, OperationMode
        self.shared_arcade = Arcade(
            arc_api_key=self.arc_key or "test-key-123",
            operation_mode=OperationMode.ONLINE,
            arc_base_url=self.arc_base_url)
        self.card_id = None                       # gateway default card; never close
        # auto-discover ALL games from the gateway if none were given (REQUIRED
        # for rerun: the 110 hidden game ids are unknown ahead of time).
        if not self.games:
            self.games = self._discover_online_games()
            print(f"[swarm] online auto-discovered {len(self.games)} games: "
                  f"{self.games[:8]}{'...' if len(self.games) > 8 else ''}")
        self.full_ids = _discover_full_ids(self.shared_arcade, self.games)
        print(f"[swarm] online arc_base_url={self.arc_base_url} "
              f"games={len(self.games)} resolved={len(self.full_ids)}")

    def _discover_online_games(self) -> list[str]:
        """All game short-ids visible on the shared online arcade (= gateway's
        /api/games). Same as probe_swarm._discover_games."""
        seen, out = set(), []
        for env in self.shared_arcade.get_environments():
            base = env.game_id.split("-", 1)[0]
            if base not in seen:
                seen.add(base); out.append(base)
        return out

    def _close_competition(self):
        if self.shared_arcade is None or self.card_id is None:
            return None
        if self.no_close:
            # leave the scorecard OPEN so the official site shows
            # the partial score. Do NOT call close_scorecard.
            print(f"[swarm] --no-close: leaving scorecard OPEN. "
                  f"View partial score: https://three.arcprize.org/scorecards/{self.card_id}")
            return None
        try:
            # a shared jar's stale cookies would otherwise overwrite the
            # session's fresh ones and close 404s.
            self.shared_arcade._master_cookie_jar.update(
                self.shared_arcade._session.cookies)
        except Exception:
            pass
        try:
            return self.shared_arcade.close_scorecard(self.card_id)
        except Exception as e:  # noqa: BLE001
            print(f"[swarm] close_scorecard failed: {type(e).__name__}: {e}")
            return None

    def _make_server(self, game: str):
        if self.mode == "competition":
            from competition_proxy_server import CompetitionProxyServer
            full = self.full_ids.get(game, game)
            return CompetitionProxyServer(self.shared_arcade, self.card_id, full,
                                          port=0, seed=self.seed)
        if self.mode == "online":
            from gateway_proxy_server import GatewayProxyServer
            full = self.full_ids.get(game, game)
            return GatewayProxyServer(self.shared_arcade, full,
                                      port=0, seed=self.seed)
        return OfflineARCServer(self.environments_dir, port=0)

    # ── per-game run (worker thread, synchronous) ───────────────────────────
    def _run_one_game(self, wid: int, game: str) -> dict:
        t0 = time.time()
        with self._seq_lock:
            self._run_seq += 1
            seq = self._run_seq
        run_dir = self.run_root / f"{game}__r{seq:03d}__w{wid}__{self._batch_ts}"
        srv = None
        try:
            srv = self._make_server(game)
            base = srv.start()
            agent_mod.ensure_fresh_run_paths(run_dir)
            agent_mod.prepare_agent_run(run_dir)
            # per-game deadline = min(per-game budget from now, global cutoff).
            # TIME is the stop authority: HSAgent.stop_condition checks it
            # between sends, AND (since the send-internal fix) CodingSession
            # checks it inside send() so one long send can't overrun it.
            per_game_end = time.time() + self.per_game_deadline_s
            if self.global_deadline_epoch is not None:
                deadline = min(per_game_end, self.global_deadline_epoch)
            else:
                deadline = per_game_end
            # set per-worker context BEFORE constructing HSAgent (same thread
            # -> factory reads these exact values; race-free). deadline goes in too
            # so the runner (CodingSession) enforces it inside send().
            agent_mod.set_runner_thread_context(
                server_url=base, vllm_base=self.vllm_base, model=self.model,
                vllm_api_key=self.vllm_api_key,
                max_tool_calls_per_send=self.max_tool_calls_per_send,
                deadline=deadline, multimodal=self.multimodal)
            agent = agent_mod.HSAgent(
                run_dir=run_dir, game_name=game, model=self.model,
                log_file=run_dir / "agent.log", deadline=deadline)
            rc = agent.run()
            insp = inspect_sessions(run_dir / "client" / "session")
            levels_completed = max(0, (insp.current_level_index or 1) - 1)
            if insp.is_solved and insp.current_level_index:
                levels_completed = insp.current_level_index
            return {"game": game, "worker": wid, "rc": rc,
                    "levels_completed": levels_completed,
                    "steps": insp.n_steps_total,
                    "current_level": insp.current_level_index,
                    "reason": _stop_reason(run_dir, insp),
                    "run_dir": str(run_dir), "seconds": round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001
            return {"game": game, "worker": wid, "rc": 1,
                    "error": f"{type(e).__name__}: {e}",
                    "tb": traceback.format_exc()[-1500:],
                    "run_dir": str(run_dir), "seconds": round(time.time() - t0, 1)}
        finally:
            if srv is not None:
                try:
                    srv.stop()
                except Exception:
                    pass

    def _worker(self, wid: int, q: "queue.Queue[str]"):
        while True:
            # Dispatch gate: don't start a NEW game unless at least dispatch_min_s
            # of solve time remains before the global deadline (else we'd make a
            # scorecard guid for a game that can't make progress).
            if self.global_deadline_epoch is not None:
                if time.time() >= self.global_deadline_epoch - self.dispatch_min_s:
                    print(f"[swarm] worker {wid} stop: <{self.dispatch_min_s}s "
                          f"before global deadline, not dispatching new games")
                    return
            try:
                game = q.get_nowait()
            except queue.Empty:
                return
            print(f"[swarm] worker {wid} -> {game}")
            rec = self._run_one_game(wid, game)
            with self._results_lock:
                self.results.append(rec)
            tag = rec.get("error") or f"levels={rec['levels_completed']} " \
                f"steps={rec['steps']} reason={rec['reason']}"
            print(f"[swarm] worker {wid} done {game}: {tag} ({rec['seconds']}s)")
            q.task_done()

    # ── orchestration ───────────────────────────────────────────────────────
    def run(self) -> dict:
        self.run_root.mkdir(parents=True, exist_ok=True)
        agent_mod.set_runner_factory(_qwen_factory)   # global, once
        if self.mode == "competition":
            self._setup_competition()
        elif self.mode == "online":
            self._setup_online()   # may auto-discover self.games from the gateway

        games = list(self.games)
        if self.shuffle:
            # a large game set may exceed the wall-clock budget — randomise order so we don't
            # always cover the same prefix. seed from time for run-to-run variety.
            import random
            random.Random(time.time()).shuffle(games)
            print(f"[swarm] shuffled game order: {games[:8]}"
                  f"{'...' if len(games) > 8 else ''}")
        q: "queue.Queue[str]" = queue.Queue()
        for g in games:
            q.put(g)
        n_workers = min(self.N, len(games))
        threads = [threading.Thread(target=self._worker, args=(i, q), daemon=True)
                   for i in range(n_workers)]
        t0 = time.time()
        for t in threads:
            t.start()
        # ── HARD GLOBAL-DEADLINE GUARANTEE ───────────────────────────────────
        # run() MUST return before the global deadline so the notebook can write
        # submission + exit before the environment's hard wall-clock kill (which fails the whole
        # run). Workers are daemon threads; any still running at the deadline are
        # ABANDONED — levels already scored on the gateway are safe, and a worker
        # stuck inside a long LLM generate can't be trusted to stop itself.
        if self.global_deadline_epoch is not None:
            for t in threads:
                remaining = self.global_deadline_epoch - time.time()
                if remaining <= 0:
                    break
                t.join(timeout=remaining)
            alive = [t for t in threads if t.is_alive()]
            if alive:
                print(f"[swarm] GLOBAL DEADLINE hit: abandoning {len(alive)} "
                      f"still-running worker(s) to return before the host wall-clock kill.")
        else:
            for t in threads:
                t.join()
        wall = round(time.time() - t0, 1)

        # competition: close + merge server-authoritative scores
        comp_scores = {}
        if self.mode == "competition":
            sc = self._close_competition()
            if sc is not None:
                for env in getattr(sc, "environments", []):
                    base = env.id.split("-", 1)[0]
                    comp_scores[base] = {
                        "levels_completed": env.levels_completed,
                        "actions": env.actions, "score": env.score}
            for rec in self.results:
                if rec["game"] in comp_scores:
                    rec["competition"] = comp_scores[rec["game"]]

        report = self._write_report(wall, comp_scores)
        return report

    def _write_report(self, wall: float, comp_scores: dict) -> dict:
        solved = sum(1 for r in self.results if r.get("reason") == "solved")
        total_levels = sum(r.get("levels_completed", 0) for r in self.results)
        report = {
            "mode": self.mode, "parallel_nums": self.N,
            "n_games": len(self.games), "wall_clock_s": wall,
            "solved": solved, "total_levels_completed": total_levels,
            "card_id": self.card_id,
            "results": sorted(self.results, key=lambda r: r["game"]),
        }
        if self.mode == "competition":
            report["competition_total_score"] = sum(
                v.get("score", 0) for v in comp_scores.values())
        out = self.run_root / f"swarm_report_{self._batch_ts}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[swarm] === report -> {out} ===")
        print(f"[swarm] mode={self.mode} N={self.N} games={len(self.games)} "
              f"wall={wall}s solved={solved} total_levels={total_levels}")
        for r in report["results"]:
            line = f"   {r['game']:12s} levels={r.get('levels_completed','?')} " \
                   f"steps={r.get('steps','?')} reason={r.get('reason', r.get('error',''))}"
            if "competition" in r:
                line += f" score={r['competition'].get('score')}"
            print(line)
        return report
