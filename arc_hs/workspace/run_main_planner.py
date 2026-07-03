from __future__ import annotations

import argparse

from script_tools import (
    add_start_source_arguments,
    format_action,
    main_planner,
    state_from_source_args,
)
from state_reconstruction_tools import simulate_actions
from plan_executor import execute_actions
from timeout_tools import fail_after_timeout
from game_status import LEVEL_COMPLETED


TIMEOUT_SECONDS = 180
PLANNER_TIMEOUT_MESSAGE = "Planner too slow, please consider using a faster planner."


def main() -> int:
    parser = argparse.ArgumentParser()
    add_start_source_arguments(parser)
    args = parser.parse_args()

    with fail_after_timeout(TIMEOUT_SECONDS, PLANNER_TIMEOUT_MESSAGE):
        state, description = state_from_source_args(args)

        plan = main_planner(state)
        if not plan:
            print(f"run_main_planner.py: no plan found from {description}")
            return 1

        _, game_status = simulate_actions(state, plan)

        if game_status != LEVEL_COMPLETED:
            raise AssertionError(
                f"run_main_planner.py: planner returned a plan from {description}, "
                f"but Heuristic System execution ended with {game_status}."
            )

    print(f"run_main_planner.py: plan from {description}")
    for action in plan:
        print(format_action(action))

    # Auto-execute a validated plan when planning FROM THE CURRENT real state.
    # Rationale: agents reliably reach "model verified ->
    # planner found a LEVEL_COMPLETED plan" and then run out of the per-game time
    # budget (gen_timeout / tool-call cap / deadline) on the SAME turn, before
    # they get a next turn to call plan_executor — so a winning plan is found but
    # never played to the gateway (0 score, e.g. tu93). Since the plan was just
    # validated above (simulate_actions == LEVEL_COMPLETED) and plan_executor runs
    # from the current real state, we play it now, in this same process.
    #   - Only for --from-current: executing a --from-initial/--from-attempt plan
    #     against a possibly-divergent current state would be wrong, so those stay
    #     print-only (inspection), exactly as before.
    #   - Agent feedback is unchanged: execute_actions prints the same per-step
    #     progress and, on any mismatch, the same mismatch block to this stdout;
    #     the agent reads it next turn and repairs the model as usual. The real
    #     moves we make are visible to inspect_sessions on the next iteration, so
    #     the agent's state view stays consistent.
    if args.from_current:
        print("run_main_planner.py: validated plan from current state; "
              "executing it on the real game now (--from-current auto-exec).")
        return execute_actions(plan)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
