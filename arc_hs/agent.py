from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

from agent_funs import (
    load_pgroup,
    load_prompt,
    run_git_snapshot,
    run_program,
    send_pgroup,
    send_prompt,
    setup_logger,
)
from coding_session import CodingSession
from session_inspector import SessionInspection, inspect_sessions


# The coding agent that grows the Heuristic System. A runner_factory(work_dir,
# log_file, model, reasoning_effort, error_handling_manual) -> runner exposing
# .send()/.new_session() can be injected (the swarm does this to pass per-game
# model/endpoint context); when none is injected the default CodingSession is
# used, driving an open OpenAI-compatible chat model (Qwen via vLLM by default).
_RUNNER_FACTORY = None


class ClientCommandError(RuntimeError):
    """Raised by run_client when all retries are exhausted. Callers treat it as
    'this game cannot continue' and stop cleanly (rc=0, partial score kept on the
    gateway) rather than crashing the whole run (rc=1)."""


# Per-worker runner context for the parallel swarm. The runner factory is a
# single global, but concurrent workers each need a DIFFERENT server_url (their
# own per-game local server). We stash the per-worker values in a thread-local;
# the factory reads them. This is race-free because the whole chain
#   set_runner_thread_context(...) -> HSAgent(...) -> __init__ builds runner
# runs SYNCHRONOUSLY on the worker's own thread, so the factory always reads the
# values that same thread just set. After construction, server_url is frozen
# into the runner instance and the thread-local is never read again.
_RUNNER_CTX = threading.local()


def set_runner_thread_context(**kwargs) -> None:
    for k, v in kwargs.items():
        setattr(_RUNNER_CTX, k, v)


def set_runner_factory(factory) -> None:
    global _RUNNER_FACTORY
    _RUNNER_FACTORY = factory


ROOT = Path(__file__).resolve().parent
DEFAULT_AGENT_RUN_DIR = ROOT / "agent_run"
WORKSPACE_INIT_DIR = ROOT / "workspace"
PROMPT_DIR = ROOT / "prompts"
LOG_FILE = ROOT / "agent.log"
# Max live actions on a SINGLE level before we stop the game (anti-thrash on dead
# /unsolvable states; m0r0 fired 1937, s5i5 2007 ACTIONs on one level = waste).
# Per-level, so clearing levels resets the budget; high enough not to cut genuine
# deep levels short.
LEVEL_STEP_CAP = 750

# Heuristic-System loosening (HL: compression should be ON-DEMAND / periodic, not
# every iteration). The 6-send hard-refactoring burst used to fire EVERY level>=2
# iteration -> ~7x LLM cost per iteration, starving real interaction (4h/game,
# only ~100-600 real steps). We now fire the heavy compression only every
# HEURISTIC_COMPRESSION_EVERY iterations; the off-cadence iterations do the light
# single-prompt refinement instead. Env-overridable for tuning (1 == old always-on
# behaviour). See the repository README.
import os as _os
HEURISTIC_COMPRESSION_EVERY = int(_os.getenv("HS_COMPRESSION_EVERY", "3"))
# Per-send tool-call budgets. Heavy code-rewrite sends (compression) legitimately
# need many tool calls; light continuation/stuck/reset nudges do NOT and used to be
# allowed the same 60, letting a single send churn to the deadline. Cap the light
# ones low, keep the heavy ones generous. Env-overridable.
LIGHT_SEND_MAX_CALLS = int(_os.getenv("HS_LIGHT_MAX_CALLS", "15"))


def resolve_log_file() -> Path:
    return LOG_FILE


def ensure_fresh_run_paths(run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(f"Agent run directory already exists: {run_dir}")


def prepare_agent_run(run_dir: Path) -> None:
    shutil.copytree(WORKSPACE_INIT_DIR, run_dir)


class HSAgent:
    def __init__(
        self,
        run_dir: str | Path,
        game_name: str | None = None,
        model: str = "Qwen/Qwen3.6-27B",
        reasoning_effort: str = "medium",
        retry_error_handling_manual: bool = False,
        log_file: str | Path | None = None,
        deadline: float | None = None,
    ):
        self.run_dir = Path(run_dir)
        # Per-game wall-clock deadline (epoch seconds). TIME is the primary stop
        # authority under a bounded wall-clock budget. None = no deadline (single-game
        # launcher / local debug). Set by the swarm.
        self.deadline = deadline
        self.client_path = self.run_dir / "client" / "client.py"
        self.session_dir = self.run_dir / "client" / "session"
        # Parallel swarm passes a per-run_dir log_file so concurrent agents do
        # not share one file; single-game launcher passes nothing -> global.
        self.log_file = Path(log_file) if log_file is not None else resolve_log_file()
        self.game_name = game_name
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.retry_error_handling_manual = retry_error_handling_manual
        # Per-worker server_url for the swarm: client.py subprocesses (start/move)
        # read ARC_SERVER_URL from env, but run_client doesn't go through the
        # runner's env_overrides. Capture the thread-local server_url HERE (same
        # thread that set it) so run_client can inject it per-worker without a
        # global os.environ (which concurrent workers would clobber). None ->
        # single-game launcher behaviour (inherit process env) unchanged.
        self.server_url = getattr(_RUNNER_CTX, "server_url", None)
        # Unique logger name per run_dir so concurrent workers don't clobber each
        # other's handlers (see agent_funs.setup_logger). Default run_dir name for
        # the single-game launcher is "agent_run", a stable unique name.
        self.logger = setup_logger(self.log_file, name=f"polyphony.{self.run_dir.name}")
        self.logger.info(
            "agent parameters",
            extra={
                "run_dir": str(self.run_dir.resolve()),
                "game_name": self.game_name,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "retry_error_handling_manual": self.retry_error_handling_manual,
            },
        )
        self.runner = (_RUNNER_FACTORY or CodingSession)(
            work_dir=self.run_dir,
            log_file=self.log_file,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            error_handling_manual=self.retry_error_handling_manual,
        )
        self.iteration_id = 0
        self.previous_loop_level_index: int | None = None
        self.previous_loop_n_steps_total: int | None = None
        self.last_trouble1: defaultdict[int, int] = defaultdict(int)
        self.last_trouble2: defaultdict[int, int] = defaultdict(int)

        self.main_prompt = load_prompt(PROMPT_DIR, "main_prompt.md", self.logger)
        self.continuation_string = load_prompt(
            PROMPT_DIR,
            "protocols/continuation_string.txt",
            self.logger,
        )
        self.hard_refactoring_pgroup = load_pgroup(
            PROMPT_DIR,
            self.logger,
            "refinement/hs_engine_simplification*",
            "refinement/hs_state_io_simplification*",
            "refinement/hs_planner.txt",
        )
        self.level1_light_simplification_prompt = load_prompt(
            PROMPT_DIR,
            "refinement/light_simplification_level1.txt",
            self.logger,
        )
        # light (single-prompt) refinement for level>=2 off-cadence iterations,
        # used when we skip the heavy compression pgroup (HL: on-demand compression)
        self.light_simplification_prompt = load_prompt(
            PROMPT_DIR,
            "refinement/light_simplification.txt",
            self.logger,
        )
        self.level1_normal_continuation_prompt = load_prompt(
            PROMPT_DIR,
            "protocols/continuation_level1.txt",
            self.logger,
        )
        self.normal_continuation_prompt = load_prompt(
            PROMPT_DIR,
            "protocols/continuation_l2.txt",
            self.logger,
        )
        self.new_level_prompt = load_prompt(PROMPT_DIR, "protocols/on_new_level_v1.txt", self.logger)
        self.level1_trouble1_prompt = load_prompt(
            PROMPT_DIR,
            "protocols/trouble1_prompt_level1.txt",
            self.logger,
        )
        self.level1_trouble2_prompt = load_prompt(
            PROMPT_DIR,
            "protocols/trouble2_prompt_level1.txt",
            self.logger,
        )
        self.trouble1_prompt = load_prompt(PROMPT_DIR, "protocols/trouble1_prompt.txt", self.logger)
        self.trouble2_prompt = load_prompt(PROMPT_DIR, "protocols/trouble2_prompt.txt", self.logger)
        self.stuck_reminder_prompt = load_prompt(
            PROMPT_DIR,
            "protocols/stuck_reminder_prompt.txt",
            self.logger,
        )
        self.death_prompt = load_prompt(PROMPT_DIR, "protocols/death_prompt.txt", self.logger)

    def init_iterations(self) -> None:
        if self.game_name is None:
            raise RuntimeError("game_name is required to initialize a game.")

        game_init_screenout = self.run_client(f"start {self.game_name}")
        initial_prompt = (
            self.main_prompt
            + "\n\n"
            + self.continuation_string
            + "\n\nThe initial output of the game client:\n"
            + game_init_screenout
        )
        self.send_prompt(initial_prompt)

    def run_git(self, inspection: SessionInspection) -> None:
        # Per-iteration git snapshot of the workspace is an optional debugging
        # aid (it lets you inspect how the HS evolved). It is OFF by default
        # because it creates a .git in every run dir and would otherwise touch
        # git config; enable with HS_GIT_SNAPSHOT=1.
        if _os.getenv("HS_GIT_SNAPSHOT", "") != "1":
            return
        run_git_snapshot(
            agent_run_dir=self.run_dir,
            level_index=inspection.current_level_index,
            global_step_count=inspection.n_steps_total,
            iteration_count=self.iteration_id,
            logger=self.logger,
        )

    def run_client(self, command: str, retries: int = 5,
                   backoff: float = 3.0) -> str:
        """Run client.py with retry + backoff (Bug#2 fix). A single transient
        failure (e.g. a None frame from the gateway, a momentary hiccup) used to
        raise straight up and crash the WHOLE game (rc=1). With an online gateway
        is a local sidecar so failures are rare, but a one-off must not zero a
        whole game. We retry a few times; only if ALL attempts fail do we raise,
        and callers (init / reset) treat that as "this game is done", not a crash.

        Backoff is intentionally generous: on the PUBLIC competition server
        (three.arcprize.org) the opening RESET occasionally returns None for
        several seconds (observed: lp85/bp35 each hit 3 Nones within 6s and the
        whole game was lost at 0 steps). retries=5 with backoff 3,6,9,12 (=30s
        total) rides out those transient reset hiccups instead of abandoning it.
        """
        env = None
        if self.server_url:
            import os
            env = dict(os.environ)
            env["ARC_SERVER_URL"] = self.server_url
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return run_program(f"python3 {self.client_path} {command}", cwd=ROOT,
                                   logger=self.logger, env=env)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self.logger.warning(
                    "run_client attempt failed",
                    extra={"command": command, "attempt": attempt,
                           "retries": retries, "error": f"{type(exc).__name__}: {exc}"},
                )
                if attempt < retries:
                    time.sleep(backoff * attempt)
        # all retries exhausted — surface a typed error so callers can degrade
        raise ClientCommandError(
            f"client command failed after {retries} attempts: {command}") from last_exc

    def send_prompt(self, prompt: str, max_calls: int | None = None) -> None:
        send_prompt(self.runner, prompt, self.logger, max_calls=max_calls)

    def send_pgroup(self, prompts: list[str], max_calls: int | None = None) -> None:
        send_pgroup(self.runner, prompts, self.logger, max_calls=max_calls)

    def stop_condition(self, inspection: SessionInspection) -> bool:
        # TIME is the primary stop authority, not step/stuck rules.
        # We removed the old step-budget (n_steps_current_level>=600, never once
        # hit) and the stuck rule (last_stuck_step == n_steps_total, which killed
        # productive games like ar25 after a single offline-reasoning round —
        # that's normal in the Heuristic-System paradigm). Fewer overlapping limiters,
        # no more cross-fighting. What remains: solved, lost-session, and the
        # wall-clock deadline owned by the swarm (self.deadline, epoch seconds).
        # HS per-game action hard-cap: the runner sets _hard_stop from INSIDE
        # a send when cumulative real actions exceed the cap (a single bash can
        # batch-fire actions past the iteration-level caps). Honor it here so the
        # run loop returns cleanly (scored progress already banked on the gateway).
        hs = getattr(self.runner, "_hard_stop", None)
        if hs is not None:
            self.logger.info("stop condition met", extra={"reason": f"hard_stop:{hs}"})
            return True
        if self.deadline is not None and time.time() >= self.deadline:
            self.logger.info("stop condition met", extra={"reason": "per-game deadline"})
            return True
        if inspection.current_level_index is None:
            self.logger.info("stop condition met", extra={"reason": "no current level found"})
            return True
        if inspection.is_solved:
            self.logger.info("stop condition met", extra={"reason": "game solved"})
            return True
        # Per-level step cap: if the agent has burned this many live actions on the
        # SAME level without clearing it, it's almost certainly thrashing on a dead
        # /unsolvable state (observed: m0r0 fired 1937 ACTIONs on level 1, s5i5
        # 2007 — pure waste of the per-game budget). Stop the game so the worker
        # frees up. This is a high cap (genuine deep levels rarely exceed it) and
        # is per-LEVEL, so a game that keeps clearing levels is never penalised.
        if (inspection.n_steps_current_level is not None
                and inspection.n_steps_current_level >= LEVEL_STEP_CAP):
            self.logger.info("stop condition met",
                             extra={"reason": "level step cap",
                                    "n_steps_current_level": inspection.n_steps_current_level})
            return True
        return False

    def get_simple_contination_prompt(self, inspection: SessionInspection) -> str:
        if inspection.current_level_index == 1:
            return self.level1_normal_continuation_prompt + "\n" + self.continuation_string
        return self.normal_continuation_prompt + "\n" + self.continuation_string

    def send_simplification_prompts(self, inspection: SessionInspection) -> None:
        if inspection.current_level_index == 1:
            self.send_prompt(self.level1_light_simplification_prompt,
                             max_calls=LIGHT_SEND_MAX_CALLS)
            return
        # HL on-demand compression: fire the heavy multi-step refactoring pgroup only
        # every HEURISTIC_COMPRESSION_EVERY iterations. Off-cadence iterations do a
        # single light-refinement prompt instead, so we stop paying the ~7x per-
        # iteration cost that starved real interaction. Set HS_COMPRESSION_EVERY=1 to
        # restore the old always-on behaviour.
        if (HEURISTIC_COMPRESSION_EVERY <= 1
                or self.iteration_id % HEURISTIC_COMPRESSION_EVERY == 0):
            self.send_pgroup(self.hard_refactoring_pgroup)
        else:
            self.send_prompt(self.light_simplification_prompt,
                             max_calls=LIGHT_SEND_MAX_CALLS)

    def normal_continuation_protocol(self, inspection: SessionInspection) -> None:
        self.logger.info("selected protocol", extra={"protocol": "normal_continuation_protocol"})
        self.send_simplification_prompts(inspection)
        self.send_prompt(self.get_simple_contination_prompt(inspection))

    def new_level_protocol(self, inspection: SessionInspection) -> None:
        self.logger.info("selected protocol", extra={"protocol": "new_level_protocol"})
        self.send_prompt(self.new_level_prompt)
        self.normal_continuation_protocol(inspection)

    def normal_reset_protocol(self, inspection: SessionInspection, reset_string: str) -> None:
        self.logger.info("selected protocol", extra={"protocol": "normal_reset_protocol"})
        contination_prompt = self.get_simple_contination_prompt(inspection)
        prompt = (
            contination_prompt
            + "\n\nThe level has been reset. You have another attempt. Output from the client:\n"
            + reset_string
        )
        self.send_prompt(prompt)

    def trouble_protocol1(self, inspection: SessionInspection, reset_string: str) -> None:
        self.logger.info("selected protocol", extra={"protocol": "trouble_protocol1"})
        if inspection.current_level_index == 1:
            prompt_prefix = self.level1_trouble1_prompt
        else:
            prompt_prefix = self.trouble1_prompt

        prompt = (
            prompt_prefix
            + "\n"
            + self.continuation_string
            + "\n\nThe level has been reset. You have another attempt. Output from the client:\n"
            + reset_string
        )
        self.send_prompt(prompt)

    def trouble_protocol2(self, inspection: SessionInspection, reset_string: str) -> None:
        self.logger.info("selected protocol", extra={"protocol": "trouble_protocol2"})
        self.runner.new_session()

        if inspection.current_level_index == 1:
            middle_prompt = self.level1_trouble2_prompt
        else:
            middle_prompt = self.trouble2_prompt

        prompt = (
            self.main_prompt
            + "\n\n"
            + self.continuation_string
            + "\n\n"
            + middle_prompt
            + "\n"
            + self.continuation_string
            + "\n\nThe level has been reset. You have another attempt. Output from the client:\n"
            + reset_string
        )
        self.send_prompt(prompt)

    def reset_protocol(self, inspection: SessionInspection) -> None:
        self.logger.info("selected protocol", extra={"protocol": "reset_protocol"})
        if inspection.is_game_over:
            self.send_prompt(self.death_prompt)
        self.send_simplification_prompts(inspection)
        reset_string = self.run_client("move RESET")

        steps = inspection.n_steps_current_level
        level = inspection.current_level_index
        if level is None:
            raise RuntimeError("Current level is unknown during reset protocol.")

        if steps > self.last_trouble2[level] + 200:
            self.last_trouble2[level] = steps
            self.last_trouble1[level] = steps
            self.trouble_protocol2(inspection, reset_string)
        elif steps > self.last_trouble1[level] + 100:
            self.last_trouble1[level] = steps
            self.trouble_protocol1(inspection, reset_string)
        else:
            self.normal_reset_protocol(inspection, reset_string)

    def stuck_protocol(self, inspection: SessionInspection) -> None:
        self.logger.info("selected protocol", extra={"protocol": "stuck_protocol"})
        self.send_simplification_prompts(inspection)
        prompt = self.get_simple_contination_prompt(inspection) + "\n" + self.stuck_reminder_prompt
        # Stuck = no live-step progress since last iteration (agent spinning on
        # offline reasoning). Cap the tool budget so a spinning nudge can't churn
        # to the deadline; real solving happens in normal_continuation (full budget).
        self.send_prompt(prompt, max_calls=LIGHT_SEND_MAX_CALLS)

    def iteration_loop(self, inspection: SessionInspection) -> None:
        self.run_git(inspection)

        if inspection.is_game_over:
            self.reset_protocol(inspection)
        elif inspection.current_level_index != 1 and inspection.current_level_index != self.previous_loop_level_index:
            self.new_level_protocol(inspection)
        elif self.previous_loop_n_steps_total == inspection.n_steps_total:
            # No live-step progress since last iteration. We no longer TREAT this
            # as terminal (that killed productive games — offline reasoning rounds
            # legitimately make no live move). Just nudge with the stuck reminder
            # and let TIME (the deadline) decide when to stop.
            self.stuck_protocol(inspection)
        else:
            self.normal_continuation_protocol(inspection)

        self.previous_loop_level_index = inspection.current_level_index
        self.previous_loop_n_steps_total = inspection.n_steps_total

    def run(self) -> int:
        self.logger.info(
            "starting agent",
            extra={
                "game_name": self.game_name,
                "run_dir": str(self.run_dir.resolve()),
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
            },
        )

        try:
            self.init_iterations()
        except Exception:
            self.logger.exception("initialization failed")
            return 1

        while True:
            self.iteration_id += 1
            try:
                inspection = inspect_sessions(self.session_dir)
            except Exception:
                self.logger.exception("failed to inspect sessions", extra={"iteration_id": self.iteration_id})
                return 1

            self.logger.info(
                "iteration inspection",
                extra={
                    "iteration_id": self.iteration_id,
                    **inspection.to_dict(),
                },
            )

            if self.stop_condition(inspection):
                self.logger.info("stopping main loop", extra={"iteration_id": self.iteration_id})
                return 0

            try:
                self.iteration_loop(inspection)
            except ClientCommandError:
                # Bug#2: a client command (e.g. reset) failed all its retries.
                # Stop this game CLEANLY — levels already cleared are already
                # scored on the gateway. Do NOT rc=1 crash (that wastes the whole
                # game and, in a swarm, looks like a hard failure).
                self.logger.warning(
                    "stopping cleanly after client command failure",
                    extra={"iteration_id": self.iteration_id})
                return 0
            except Exception:
                self.logger.exception("iteration loop failed", extra={"iteration_id": self.iteration_id})
                return 1

            time.sleep(1.0)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ARC Heuristic-System agent on one game.")
    parser.add_argument("--game", metavar="GAME_NAME", required=True,
                        help="short game id to play (e.g. ft09)")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument(
        "--retry-error-handling-manual",
        action="store_true",
        help="Prompt for Enter and retry the exact same command when the coding model exits with an error.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    run_dir = DEFAULT_AGENT_RUN_DIR
    ensure_fresh_run_paths(run_dir)
    prepare_agent_run(run_dir)
    agent = HSAgent(
        run_dir=run_dir,
        game_name=args.game,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        retry_error_handling_manual=args.retry_error_handling_manual,
    )
    return agent.run()


if __name__ == "__main__":
    raise SystemExit(main())
