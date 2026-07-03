"""
session_inspector.py — read the Polyphony Agent per-game session directory and summarise
progress for the agent's control loop.

The client writes one directory per attempt (``level_<L>_attempt_<A>/``) holding
``initial_metadata.json`` and ``step_<NNNN>_metadata.json`` files. This module
turns that on-disk trace into two views the driver consumes each iteration:

    read_session_attempts(dir) -> {level_index: [AttemptInfo, ...]}   (per-level)
    inspect_sessions(dir)      -> SessionInspection                    (rollup)

Both are pure readers: they never mutate the session and tolerate a missing or
still-empty session directory by returning an empty summary.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# Per-attempt terminal/non-terminal status vocabulary (matches the engine's).
RUNNING = "RUNNING"
LEVEL_COMPLETED = "LEVEL_COMPLETED"
GAME_OVER = "GAME_OVER"

DEFAULT_SESSION_DIR = Path(__file__).resolve().parent / "agent_run" / "client" / "session"

_ATTEMPT_DIR = re.compile(r"^level_(\d+)_attempt_(\d+)$")
_STEP_FILE = re.compile(r"^step_(\d{4})_metadata\.json$")


@dataclass(frozen=True)
class AttemptInfo:
    level_index: int
    attempt_index: int
    win_levels: int
    status: str
    n_steps: int
    path: Path


@dataclass(frozen=True)
class SessionInspection:
    is_solved: bool
    is_game_over: bool
    is_level_completed: bool
    n_steps_total: int
    n_steps_current_level: int
    n_steps_current_attempt: int
    n_game_over_attempts_current_level: int
    current_level_index: int | None
    attempts_per_level: dict[int, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _classify(level_index: int, last_step: dict) -> str:
    """Terminal status of an attempt from its final recorded step frame."""
    if last_step.get("state") == GAME_OVER:
        return GAME_OVER
    if int(last_step.get("levels_completed", 0)) >= level_index:
        return LEVEL_COMPLETED
    return RUNNING


def _inspect_attempt(attempt_dir: Path, level_index: int, attempt_index: int) -> AttemptInfo:
    """Summarise one attempt directory into an AttemptInfo."""
    initial = _load_json(attempt_dir / "initial_metadata.json")

    # Collect step indices from step_<NNNN>_metadata.json filenames; the highest
    # one carries the attempt's final state.
    steps = sorted(
        int(m.group(1))
        for p in attempt_dir.glob("step_*_metadata.json")
        if (m := _STEP_FILE.match(p.name))
    )

    if steps:
        final_step = _load_json(attempt_dir / f"step_{steps[-1]:04d}_metadata.json")
        status = _classify(level_index, final_step)
    else:
        status = RUNNING

    return AttemptInfo(
        level_index=level_index,
        attempt_index=attempt_index,
        win_levels=int(initial.get("win_levels", 0)),
        status=status,
        n_steps=len(steps),
        path=attempt_dir,
    )


def read_session_attempts(session_dir: str | Path | None = None) -> dict[int, list[AttemptInfo]]:
    """Map each level index to its attempts (sorted by attempt index).

    Directories that don't match ``level_<L>_attempt_<A>`` are ignored, so
    stray files never break inspection.
    """
    root = DEFAULT_SESSION_DIR if session_dir is None else Path(session_dir)

    by_level: dict[int, list[AttemptInfo]] = {}
    if not root.is_dir():
        return by_level

    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        m = _ATTEMPT_DIR.match(entry.name)
        if not m:
            continue
        level_index, attempt_index = int(m.group(1)), int(m.group(2))
        by_level.setdefault(level_index, []).append(
            _inspect_attempt(entry, level_index, attempt_index)
        )

    for attempts in by_level.values():
        attempts.sort(key=lambda a: a.attempt_index)
    return by_level


def _empty() -> SessionInspection:
    return SessionInspection(
        is_solved=False,
        is_game_over=False,
        is_level_completed=False,
        n_steps_total=0,
        n_steps_current_level=0,
        n_steps_current_attempt=0,
        n_game_over_attempts_current_level=0,
        current_level_index=None,
        attempts_per_level={},
    )


def inspect_sessions(session_dir: str | Path = DEFAULT_SESSION_DIR) -> SessionInspection:
    """Roll the per-attempt view up into the summary the driver acts on.

    "Current level" is the highest level index seen on disk; its latest attempt
    decides the terminal flags. Returns an empty summary if nothing is recorded.
    """
    by_level = read_session_attempts(Path(session_dir))
    if not by_level:
        return _empty()

    # Aggregate step counts in one pass; track the current (highest) level.
    total_steps = 0
    per_level_counts: dict[int, int] = {}
    for level_index, attempts in by_level.items():
        per_level_counts[level_index] = len(attempts)
        total_steps += sum(a.n_steps for a in attempts)

    current_level = max(by_level)
    current_attempts = by_level[current_level]
    latest = current_attempts[-1]

    return SessionInspection(
        is_solved=(latest.status == LEVEL_COMPLETED and latest.level_index == latest.win_levels),
        is_game_over=(latest.status == GAME_OVER),
        is_level_completed=(latest.status == LEVEL_COMPLETED),
        n_steps_total=total_steps,
        n_steps_current_level=sum(a.n_steps for a in current_attempts),
        n_steps_current_attempt=latest.n_steps,
        n_game_over_attempts_current_level=sum(1 for a in current_attempts if a.status == GAME_OVER),
        current_level_index=current_level,
        attempts_per_level=per_level_counts,
    )
