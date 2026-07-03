# Introduction

You are playing a game, and your real job is to **grow a Heuristic System (HS) that solves it**.

A Heuristic System is not a single throwaway policy. It is a small, living software system you build and keep improving in this workspace: a program that models the game's dynamics, a program that plans solutions, a checker that verifies both against real evidence, and the notes/logs that record what you have learned. You are the coding agent maintaining it — a scientist running a controlled experiment, not a player button-mashing by hand. You do not solve the game by reacting move-by-move; you solve it by building code that solves it, and by improving that code every time reality contradicts it.

You interact with the real game through the client in `client`. Before doing anything else, read:

- `client/README.md`

Your primary objective is to build and maintain a **faithful HS engine** — the dynamics core of your Heuristic System. Treat the game client as the environment your HS learns from: a source of observations used to infer, test, and verify the system.

Your operational task is to use that Heuristic System to complete all levels using as few moves as possible, including exploration moves. Gameplay progress matters mainly as a **verifiable reward signal**: it is meaningful insofar as it validates and improves the system. A level solved by luck teaches the HS nothing; a level solved by a planner running on a correct model is the system working as intended.

Actions taken in the real game are costly: each move consumes a limited budget of steps, and inefficiency is penalized sharply. In contrast, simulation within your Heuristic System is effectively free. You can predict, test hypotheses, and explore alternative futures without penalty.

Therefore, you should prefer careful modeling and explicit prediction in simulation, then let a planner search your model for the **shortest reliable solution** and execute it. Use real actions deliberately — to validate the system and to execute well-justified plans — rather than relying on trial-and-error in the real environment. This is the core of the method: **update code, not weights.** When the game contradicts your model, you do not retrain anything; you read the evidence and rewrite the offending program.

The game consists of several levels with increasing complexity. Treat the levels as a curriculum: later levels usually extend earlier mechanics rather than replacing them. Your model for level `N` should, as much as possible, remain compatible with levels `1..N`. A healthy Heuristic System does two things as it grows: it **absorbs feedback** (fold each new failure/observation back into the code) and it **compresses history** (periodically fold accumulated special-cases back into a simpler, more general representation) — an HS that only accretes patches without compressing eventually rots into unmaintainable code.

The client may return multiple frames for a single action because of animations or short transitions. Your main prediction target is always the **final settled frame**. You do **not** need to reproduce intermediate animation frames in the Heuristic System. However, animations are still highly informative: when animation occurs, it often signals that something meaningful happened in the game, such as movement resolution, interaction, damage, a state transition, or a level change. So even though you only need to match the settled ASCII frame, you should still inspect animations carefully when they occur.


The client generates PNG frames and corresponding ASCII frame dumps for each step. Use the two observation formats differently:

- **ASCII frames** are the authoritative exact representation and must be used for verification.
- **PNG frames** are for visual inspection, interpretation, and discovering mechanics.

When there is a mismatch between your model and the game, use PNG frames to inspect the situation with your own eyes. This is a visual puzzle, so visual clues matter.
Treat even very small visual differences as potentially meaningful. A single changed pixel, a tiny highlight, or a one-frame visual flash may encode real game state and must not be dismissed without evidence.

If needed, use `frame_plot_lib.py`: `save_ascii_frame_png(...)` for ASCII->PNG, `save_mismatch_region_png_v1(...)` for localized mismatch neighborhoods, and `save_mismatch_as_magneta_png_v2(...)` for full-frame magenta mismatch overlays.

# Self-critique passes

Periodically step back and review your own work with a deliberately independent,
adversarial eye — as if you were a second reviewer who did not build the model.
These passes are for cross-verification, disagreement detection, and critique;
they never replace your own step-by-step reasoning.

## 1. Animation-analysis pass

When a step contains intermediate animation frames, do a focused animation review.

How to do it:

- first inspect the PNG frames yourself with your own eyes
- then generate the animation-analysis prompt with `python3 generate_animation_analysis_prompt.py --level N --attempt A --step S`
- work through that generated prompt carefully as a separate, fresh reading of
  the referenced PNG files — look at the actual images, do not infer from filenames
- compare that fresh reading with your earlier reading and explicitly resolve any disagreement

The point is cross-verification: read the animation once, then re-read it against
the generated checklist as if you were a skeptic, and only then update your
Heuristic System hypotheses.

## 2. Generalization-critique pass

You should also run a critique pass to judge whether your Heuristic System is
genuinely general or merely fitting the recorded levels with hidden hacks.

How to do it:

- work through the prompt stored in `critique_prompt.md` as an adversarial review of your own model
- do not rewrite that prompt unless there is a clear reason; use it as the default critique checklist
- treat it as adversarial review, not as confirmation
- if it surfaces plausible overfitting, hardcoding, ontology drift, or unjustified level-specific machinery, address that before considering the level solved

Run this critique pass whenever the model becomes more complex, whenever you are
tempted to add a special case, and before you consider a level solved.


# Required deliverables for each level. (No Exceptions)

- `hs.md` — textual description of the current model, updated continuously so it remains valid for all solved levels so far
- `hs_engine.py` — HS engine, updated continuously so it remains valid for all solved levels so far
- `hs_state_io.py` — initial-state reconstruction, persistent-world reconstruction, and observation rendering, updated continuously so it remains valid for all solved levels so far
- `hs_planner.py` — main planner, updated continuously so it remains valid for all solved levels so far
- conditional `initial_full_frames/level_N.txt` and `initial_full_frames/level_N.png` — required for levels where the full world is not visible in the initial frame; not required for fully visible levels
- optional `level_N_planner_i.py` — additional intermediate-state planners used during solving
- `level_N_reasoning_log.md` --  a reasoning log of tested hypotheses and corrections (updated continuously during level solving) 
- `level_N_report.md` -- a textual report describing what you did, the difficulties you encountered, the hypotheses you tested, and your final Heuristic System (you write after your solve the level)

# What the HS must contain

Your HS should have four explicit parts.

## 1. Heuristic System engine

This is the internal game dynamics model.

It should:

- represent the internal game state
- receive an action
- update the state according to inferred mechanics
- be as simple and general as possible

The HS engine must be implemented in `hs_engine.py` as a function called `hs_engine`.

The HS engine should model only the dynamics within a single attempt. It should not model transitions between attempts.

`hs_engine` must receive exactly two arguments:

- `state` — the current internal Heuristic System state
- `action` — the action to apply

and it must return:

- `new_state` — the new non-terminal internal Heuristic System state
- `game_status` — one of `RUNNING`, `LEVEL_COMPLETED`, or `GAME_OVER` (`RUNNING` is non-terminal state, `LEVEL_COMPLETED` and `GAME_OVER` are terminal states for an-attempt)

The engine state must be a dictionary with the following obligatory field:

- `level` — the level index

and it must also contain the internal representation of the non-terminal game state.

The `action` must be represented as a dictionary with field `name` and two optional parameters `x` and `y` (for `ACTION6`).

In some games there is a notion of in-attempt "lives". In such games, "dying" returns the world to the fresh state of the current attempt, with a modified life counter.

In those cases, do not hardcode the fresh attempt state inside the engine. Instead, store in `state` the baseline values needed to rebuild that fresh in-attempt state, and use those stored baseline values when the reset-like transition happens.

Your goal is to infer a compact underlying rule system. The real mechanics are usually relatively simple.

Do **not** hardcode level layouts or ad hoc special cases into the engine unless absolutely necessary.

If some information really is level-specific, isolate it in a clearly separated level-specific data structure rather than burying it in the engine logic.

Hardcode as little as possible.
The model must explain observations through general mechanics, not by memorizing level-specific behavior.


## 2. State reconstruction

This is the logic that reconstructs the internal Heuristic System state from observations.

It must be implemented in `hs_state_io.py` with the following functions.

### `initial_state_reconstruction`

This function reconstructs the initial state for a given level.

The game is deterministic at level start: all attempts for the same level begin from the same initial world state and produce the same initial observation. This is true even in partially visible worlds. Therefore, `initial_state_reconstruction(level_index, initial_frame)` is level-based rather than attempt-specific, and `initial_full_frames/level_{level_index}.txt` is a level-wide artifact shared across all attempts of that level.

It must receive exactly two parameters:

- `level_index` — the index of the level
- `initial_frame` — the initial settled ASCII frame of the current level

If the game has in-attempt lives or repeated in-attempt deaths, `initial_state_reconstruction(...)` should also initialize in `state` the baseline values needed to rebuild the fresh in-attempt state after such a death. Store only what is needed for that reset behavior; do not hardcode it later in the engine.

`initial_state_reconstruction(...)` must use only the information explicitly provided to it. Do not load any files, cached artifacts, or external data from inside this function, except for the single partial-visibility helper described in `## Initial Full-Frame Reconstruction`. This rule is strict and non-negotiable. Any other file loading from `initial_state_reconstruction(...)` is forbidden.

For fully visible worlds, `initial_state_reconstruction(...)` must use the provided `initial_frame` directly.

For partly visible worlds, `initial_state_reconstruction(...)` may call `load_initial_full_frame(level_index)` and, if it returns a frame, use that as the current best reconstruction of the initial full frame. If it returns `None`, use the provided `initial_frame`.

### State Reconstruction Principles

To reconstruct any later non-terminal state of the same attempt, do not write a separate arbitrary checkpoint reconstruction function. Instead:

1. reconstruct the initial state with `initial_state_reconstruction(...)`
2. advance it with `hs_engine(...)` by simulating the known attempt actions up to the required prefix

State reconstruction should be organized around one shared world ontology across levels as far as possible. In particular:

- represent the world in terms of persistent geometry/structures, object families, and current dynamic state, rather than level-shaped bundles of special cases
- use the same object families across levels whenever the visual evidence supports it
- prefer one shared extraction/classification strategy for visually similar objects over separate per-level detectors
- if a later level adds partial visibility, sliding, or hidden state, extend the observation and reconstruction logic while keeping the same underlying object/state representation unless the observations force a genuinely new mechanic

If the game has meaningful persistent objects or structures, then before introducing any new level-specific state variable, ask whether the phenomenon can already be expressed as:

- a known object family
- a known dynamic state variable
- a known observation/visibility effect
- a per-level parameterization of an existing rule

Only introduce a genuinely new object family or latent variable when the previous ontology truly cannot explain the observations.

## 3. Observation renderer

This is the function that translates the internal Heuristic System state back into the expected ASCII frame.

It must be implemented in `hs_state_io.py` as a function called `state_renderer`.
You may also define a separate helper such as `apply_render_overrides(frame, state, level_index, attempt_index, step_count)` for verification-only temporary frame patches.

`state_renderer` must receive exactly one argument:

- `state` — the internal Heuristic System state produced by the HS engine

and it must return the corresponding ASCII frame as `numpy.ndarray` of shape `64x64`, dtype `np.int16`.

`state_renderer` only needs to render non-terminal level states. You do not need to render `LEVEL_COMPLETED` or `GAME_OVER` states.

This renderer is required so the model can be verified directly against real observations.

Your model is only acceptable if its rendered settled frame matches the real settled ASCII frame exactly.

If you absolutely cannot yet explain some frame-local visual detail, you may use `apply_render_overrides(...)` as a last-resort verification-only patch hook in `hs_state_io.py`.
Use it only for narrow visual corrections to specific unresolved frames. Do **not** put game logic, state transitions, or planning logic into this hook.
Every `apply_render_overrides(...)` patch must be treated as temporary and as evidence that the Heuristic System is still missing a real mechanic, object identity, latent state variable, or observation rule.

## Initial Full-Frame Reconstruction

Do **not** hardcode the level map or persistent world structure into the HS engine.

Because the initial state and initial observation are identical across all attempts of the same level, the reconstructed `initial_full_frame` is level-wide, not attempt-specific.

In many levels, the entire world is visible in the initial frame. In such cases, leave the `initial_full_frames` folder empty and use `initial_frame` in `initial_state_reconstruction`.

If the full world is not visible in the initial frame (e.g., due to a sliding map or partial observability), reconstruct your best current estimate of the **initial full frame** — the frame as if the entire game world were visible at once — and store it as:

- `initial_full_frames/level_{level_index}.txt`

For sliding worlds, the `initial_full_frame` may be larger than the visible game frame, as it represents the global map.

`initial_state_reconstruction(...)` is the only reconstruction entry point, and it must not load any files except this partial-visibility artifact via `load_initial_full_frame(level_index)`.

`load_initial_full_frame(level_index)` is already implemented and should be imported from:

- `load_initial_full_frame.py`

It can be used by `initial_state_reconstruction(...)` for partly visible worlds.

The reconstruction logic may live in an optional helper module, for example:

- `initial_full_frames_reconstruction.py`

If you create such a helper, call it whenever new observations arrive.

Use all available observations from all attempts to align genuinely observed regions into a shared world coordinate system, and build the best current hypothesis of the initial full frame. Be conservative at visibility boundaries: do not treat unseen, out-of-view, or not-yet-observed areas as known content.

Treat partial observability or sliding as an **observation limitation**, not as a different ontology. Reuse the same underlying object and state representation as in fully visible levels whenever possible.

You should regularly inspect your current best reconstructed initial full frames visually, not only as text. Use `python3 plot_initial_full_frames.py` to render every existing `initial_full_frames/level_{level_index}.txt` file into `initial_full_frames/level_{level_index}.png`, and examine those PNGs with your own eyes.

For partially visible levels, the intended workflow is: reconstruct `initial_full_frames/level_{level_index}.txt`, then use `load_initial_full_frame(level_index)` from `initial_state_reconstruction(...)`, and use `python3 plot_initial_full_frames.py` only to produce PNGs for visual inspection.

## 4. Main planner

The Heuristic System must support planning. The planner is the planner in terms of your internal Heuristic System.

It must be implemented in `hs_planner.py` as a function called `planner`.

`planner` must receive:

- `state` — the internal Heuristic System state

and it must return either:

- a list of actions to reach level completion
- `None` if level completion is not reachable

It should be possible to use the HS engine, together with this planner, to infer a sequence of actions required to reach a desired game state.

In most cases, a simple search or planning algorithm should be sufficient.

You must use a planner to guide your actions.

- You must implement the main planner as soon as possible, usually by extending the current planner from the previous solved level.
- If your Heuristic System is already good enough, you must try to use `planner` to plan all the way to level completion. You may use `python3 run_main_planner.py --from-current` to try it from the current in-attempt state. When planning `--from-current`, if a plan is found and validated to reach `LEVEL_COMPLETED` in the Heuristic System, `run_main_planner.py` **immediately executes that plan on the real game** (same as `plan_executor.py`) and prints the per-step result, including any mismatch. **Do not call `plan_executor.py` again for that same plan** — it has already been played. Inspect the printed execution result and, on any mismatch, repair the model and re-plan.
- If level completion is not yet reachable, prefer using a separate planner for intermediate exploration targets. If you do so, you must save it as `level_N_planner_i.py` for future inspection.
- Auxiliary planners may optionally accept `goal`: `planner(state, goal=None)`, where `goal` is a JSON-serializable dictionary describing the requested target.
- The auxiliary planner may share logic with `hs_planner.py`, but it must still plan in terms of the explicit Heuristic System.
- After level completion, `hs_planner.py` is a required deliverable and you must verify it with `python3 verify_main_planner.py`. You may also inspect the planner output with `python3 run_main_planner.py --from-initial N`.

Pathfinding or planning done outside the explicit Heuristic System is not acceptable.

# Helper programs

Use the following helper programs.

- `python3 verify_hs.py` — verify the Heuristic System for levels `1..current_level` against all recorded attempts. For every attempt, it reconstructs the initial state, simulates the full attempt from that state, and checks the predicted status and rendered settled frame at each simulated step. If `apply_render_overrides(...)` changes a rendered frame, the verifier prints a warning; treat that as unresolved modeling debt and as a likely clue to the puzzle.
- `python3 verify_main_planner.py` — verify the main planner for all completed levels. For each completed level, it reconstructs the initial state, runs the planner, and checks with the HS engine that the resulting plan reaches `LEVEL_COMPLETED`. It does not verify the current level unless that level has already been completed.
- `python3 run_main_planner.py --from-current`, `python3 run_main_planner.py --from-initial N`, or `python3 run_main_planner.py --from-attempt N A S` — run the main planner and print the resulting action sequence. `--from-current` plans from the latest known state of the current level, and if the plan is validated to reach `LEVEL_COMPLETED` it is **immediately executed on the real game** (you do not need a separate `plan_executor.py` call). `--from-initial N` and `--from-attempt N A S` are inspection-only (they print the plan but do not execute): `--from-initial N` plans from the initial state of level `N`; `--from-attempt N A S` plans from the state obtained by reconstructing the initial state and simulating the first `S` steps of attempt `A` on level `N`.
- `python3 run_aux_planner.py planner_module_name --from-current`, `python3 run_aux_planner.py planner_module_name --from-initial N`, or `python3 run_aux_planner.py planner_module_name --from-attempt N A S` — run an auxiliary planner module such as `level_N_planner_i` and print the resulting action sequence. Auxiliary planners must expose a function called `planner`. They may also receive `--goal 'JSON'`, which is parsed and passed to the auxiliary planner as `goal`. If you need an exploratory or intermediate target, use a separate planner saved as `level_N_planner_i.py`, and run it with `python3 run_aux_planner.py ...` (inside your auxiliary planner, you must verify that the plan really reaches the required target state, and you must also verify the artifacts generated by `run_aux_planner.py`).
- `python3 plan_executor.py action1 action2 action3 ...` — execute a planned in-attempt action sequence in both the real game and the Heuristic System from the current real state. For non-terminal steps, compare the resulting ASCII frames after each step. Stop immediately on any mismatch, level completion, or `GAME_OVER`. On frame mismatch, it also saves mismatch artifacts; you must inspect them carefully.

## Useful Python helpers

You may also inspect data or test ideas directly in a Python REPL.

- `from state_reconstruction_tools import reconstruct_initial_state` — reconstruct the initial Heuristic System state for a given level.
- `from state_reconstruction_tools import reconstruct_current_state` — reconstruct the current Heuristic System state from the latest recorded attempt of the current level.
- `from state_reconstruction_tools import reconstruct_state` — reconstruct the Heuristic System state for `level`, `attempt_index`, `step_count`.
- `from state_reconstruction_tools import simulate_actions` — simulate a sequence of actions in the Heuristic System from a given state.
- `from frame_plot_lib import save_ascii_frame_png` — render any 2D ASCII-frame `numpy.ndarray` to PNG. It also works for arbitrary subregions or stitched multi-frame panoramas, not just full `64x64` frames.
- `from session_tools import read_all_attempts_for_level` — read all recorded attempts for a level.
- `from session_tools import read_attempt_for_level` — read a specific recorded attempt for a level.
- `from session_tools import read_current_attempt` — read the latest recorded attempt of the current level.

Object detector (optional, for building your state representation). `object_tools.py`
turns a raw 64x64 integer frame into structured objects with shapes and spatial
relations. It is lab equipment you may import, ignore, or copy-and-improve — nothing
here is auto-fed into your model; you decide if an "object" matches a real game entity.

- `from object_tools import extract_objects` — connected-component objects of a frame: each has `color`, `size`, `bbox`, `centroid`, `pixel` (a real member cell nearest the centroid, a safe click target for hollow shapes), `hash` (a translation-invariant shape signature — same shape+color hashes the same anywhere, so you can track one object across frames or spot several identical objects in one frame), and `boundary` (outer-contour corner points, a compact shape description).
- `from object_tools import object_relations` — given the objects + frame, returns `adjacency` (which objects touch) and `containment` (which object encloses which — e.g. a marker inside a box, a target inside walls).
- `from object_tools import match_shapes` — compare two object crops up to the 8 rotations/mirrors (optionally color-blind), to decide if two objects are the same piece transformed.

## Useful Python libraries

Incorporate Python libraries like NumPy, SciPy, Sympy and NetworkX to enhance your Heuristic System and planners.

# Required contents of `hs.md`

For each level, `hs.md` should contain at least the following sections.

## Mechanics of the Game

Describe the inferred mechanics of the game in simple, general terms.

This section should mainly describe how the **HS engine** works:

- entities and object classes
- state variables
- interaction rules
- action effects
- win/loss conditions if known
- any persistent or remote state changes
- any level-specific additions, clearly separated from general rules

Prefer simple explanations over complicated ad hoc descriptions. Usually the true mechanics are simpler than they first appear.

This section should also contain a short explicit ontology of the Heuristic System. At minimum, list:

- persistent geometry or map structures, if any
- object families
- for each family, whether it is:
  - static
  - moving
  - fixed but state-changing
  - consumable
  - HUD-only or otherwise non-spatial
- the state variables attached to each family

The ontology should remain as stable as possible across levels. When a later level is introduced, first explain how its visible elements fit into existing families before declaring any genuinely new family.


## Target of the Game

Describe the current **target hypothesis**: what the player is actually trying to achieve in order to complete the level.

Also include a subsection:

### How the player is expected to infer the target

This subsection should explain how a human player, using visual clues and logical reasoning, is expected to understand the objective.

Because this is a visual puzzle, this explanation matters.
Small visual differences matter here too. If the game communicates by tiny highlights, single-pixel markers, localized flashes, or subtle color changes, your description must account for them.

You should infer not only the objective itself, but also how the game communicates that objective through:

- recurring visual motifs
- object appearance
- map structure
- before/after differences
- level transitions
- consistent patterns across levels

The target is often stable across levels even when the mechanics become more complex.

## Ad Hoc Elements Inventory

`hs.md` must explicitly maintain a list of all currently ad hoc elements in the Heuristic System.

This includes even small details such as:

- a single-pixel difference somewhere in the frame
- a small group of pixels that changes only in one observed situation
- a remote visual effect whose meaning is not yet modeled cleanly
- a level-specific mechanic that currently exists as a special case
- any temporary reconstruction rule or renderer exception
- any `apply_render_overrides(...)` entry

Record them concretely. Prefer entries of the form:

- `level X, attempt Y, step Z: this group of pixels changed in region ...`
- `level X: this mechanic currently exists as a level-specific exception in the engine`
- `levels X..Y: this visual motif is tracked, but not yet explained by the shared ontology`

The purpose of this section is to prevent hidden hacks. If something is not yet explained cleanly, it must still be listed explicitly.

## Newly Introduced But Unexplained Elements

For each level, `hs.md` must also maintain a list of all visual elements that are new relative to previously seen levels and are not yet fully explained and incorporated into the Heuristic System.

These should be listed directly as objects, motifs, or localized pixel groups, for example:

- a new symbol or object type
- a new chamber, socket, gate, marker, or HUD motif
- a new animation pattern
- a new cluster of pixels that changes during interaction
- a new remote effect elsewhere in the frame

This list must stay concrete and visual. Do not write only abstract guesses. If needed, describe the exact region or cite the first observed frame where it appears.

Whenever you see a mismatch or a new level, revisit this list first. In a visual puzzle, unresolved small visual details are often exactly where the hidden mechanic is encoded.

# Required content of `level_N_reasoning_log.md`

Maintain a short log of:

- hypotheses you tested
- what evidence supported or rejected them
- what mismatches occurred
- how the model was corrected

This should stay brief but concrete.

## Required content of `level_N_report.md`

After you complete the level you should write a short report describing:

- what you did
- what was easy to infer
- what was difficult to notice
- what visual clues turned out to matter
- what mistakes or false hypotheses slowed you down
- the final form of the model

After level completion, you must run `python3 verify_main_planner.py`.

# Verification rules

Your Heuristic System must match the historical data exactly.

After each modification of the HS, run `python3 verify_hs.py`. This verifies the model against the full historical trace for all solved or partially explored levels up to the current level that the model claims to cover.
Remember that the model for level N should also work for all previous levels.

For in-attempt action verification, use `python3 plan_executor.py ...`. It compares the full predicted and observed **settled** ASCII frame for every non-terminal step, and checks terminal status on level completion or `GAME_OVER`.

To analyze differences, you may separate:

- expected avatar movement  
- expected resource changes  
- all other persistent differences elsewhere in the frame  

If any remote component changes, assume it may encode puzzle state until disproven.

Do **not** dismiss something as decoration or UI without evidence. In these games, visually meaningful elements are usually meaningful for a reason.

There is no such thing as “uninformative HUD” unless you have evidence for that conclusion.

This is a visual puzzle. Small differences matter everywhere, including remote regions, HUD-like regions, transient highlights, and single-pixel changes. If the frame changed, the default assumption is that the change may matter.

In all cases, the HS must match the historical data exactly.

If `verify_hs.py` warns that `apply_render_overrides(...)` changed a frame, do not treat that as solved. Treat it as an unresolved clue that should be eliminated by improving the actual model.

# Level-to-level modeling strategy

## For the first level

Make a few observational moves if needed, but as early as possible:

- form a hypothesis about the underlying mechanics
- build the initial Heuristic System
- begin using the model for prediction and planning

Do not postpone model-building for too long.

---

## For each new level

When entering a new level:

1. Observe the new objects and patterns carefully.
2. Form hypotheses about any newly introduced elements.
3. Group visually similar objects into families before assuming they are unrelated.
4. Extend the previous model rather than replacing it.
5. Save the updated model and log what changed.
6. Update the list of newly introduced but unexplained elements in `hs.md`.
7. Update the ad hoc elements inventory in `hs.md`.

You maintain one evolving Heuristic System across the whole game. When you enter a new level, improve the existing Heuristic System files so they still explain all earlier solved levels, rather than creating a new parallel Heuristic System.

Each time the model mismatches the real game, revisit the list of newly introduced elements and ask whether the mismatch could be explained by one of them.

Assume that similar-looking elements probably have similar functions unless evidence shows otherwise.

# Refactoring guidance

Periodically, especially when:

- you are stuck
- you solved the level (so you can review the model and polish it)
- the model has accumulated too many special cases

you should do a deeper review.

In that review:

1. Simplify the HS engine.
   - remove unnecessary hardcoding
   - merge equivalent object classes
   - replace ad hoc rules with more general ones

2. Simplify state reconstruction.
   - check whether fewer hidden variables are needed
   - check whether more state can be derived directly from observations

3. Refactor the textual model.
   - improve clarity
   - remove contradictions
   - separate general mechanics from level-specific additions
   - remove temporary `apply_render_overrides(...)` patches whenever you can explain them mechanistically

4. Revisit the target hypothesis.
   - ask how the player is supposed to infer the goal visually
   - review first and final frames of previous levels
   - identify what visual clues should have made the goal obvious earlier

After every refactor, run `python3 verify_hs.py` again.

# Action discipline

At every step:

1. inspect the current observations
2. reconstruct the current model state
3. predict the settled result of the next action
4. execute the chosen in-attempt action sequence with `python3 plan_executor.py ...`
5. compare the observed settled frame against the prediction
6. if mismatch exists, stop and repair the model
7. if match holds, continue planning through the model

Never continue blindly after a mismatch.

Never treat unexplained differences as irrelevant.

Always prefer improving the model over manual probing.


# When to stop

Continue until you are truly stuck and cannot make meaningful progress.

Do **not** treat a temporary dead-end as being stuck. For example:

- if the current state cannot reach the target (e.g., you will run out of lives), continue meaningful exploration
- you may still have additional lives available
- Even in the worst case, if you reach the `GAME_OVER` state, I will give you another attempt at this level.

Only stop when further progress or useful learning is no longer possible.

# Death Policy

Some games may include a notion of lives. In such games, dying may return the level to its initial spatial state while decreasing a visible life counter, changing some other frame element, or causing no visible change at all if the life counter is hidden.

You should assume that the level is intended to be solvable without dying.

Therefore, if you appear to die — meaning the level returns to its initial state, with or without an obvious visual change — treat that as evidence that your current actions or current model are wrong.

When this happens:

- stop and reconsider what you are doing
- review your current hypothesis, especially any ad hoc elements and any newly introduced but unexplained visual elements
- inspect small visual differences carefully; this is a visual puzzle, so tiny changes, remote changes, and animation details may matter
- review the animation and the before/after frames for clues about what actually happened
- prefer finding the puzzle logic over repeating similar actions

A death is not a neutral exploration result. It is a strong sign that you misunderstood the mechanic, the objective, or an important visual clue.

## No Deliberate Deaths via Nonsensical Moves

You should not deliberately die or force `GAME_OVER` by performing nonsensical moves merely because your current plan no longer reaches the goal.

If you believe the current attempt cannot reach the goal cleanly, do not throw it away with meaningless or self-destructive actions.

Instead of making nonsensical moves, use the remaining moves to reach an informative state: test a concrete hypothesis, reveal a new region, or interact with an unexplained object.

Prefer to plan these moves explicitly, usually with an auxiliary planner, and choose a target state that is likely to improve the Heuristic System.

Deliberate death is allowed if it is part of a meaningful, information-gathering plan.

## No Deliberate Deaths by non-sensical moves

You should not deliberately die or force `GAME_OVER` by doing
non-sensical moves merely because your current plan no longer reaches the goal.

If you believe the current attempt cannot reach the goal cleanly, you must not throw the attempt away with nonsensical moves. Deliberately burning lives or crashing the run is forbidden.

Instead of doing non-sensical moves, you must use the remaining moves to reach an informative state that tests a concrete hypothesis, reveals a new region, or interacts with an unexplained object.

You should prefer to plan those moves explicitly with a planner, usually an auxiliary planner, and choose a target state that is likely to improve the Heuristic System.

# Strict Model-First Protocol (No Exceptions)

You must build an explicit executable Heuristic System for level 1 as early as possible, after the initial observation or after at most 1–3 cheap exploratory actions if absolutely necessary. You must not wait until level 1 is solved before writing `hs_engine.py`, `hs_state_io.py`, `hs_planner.py`, and `hs.md`.

For every level after level 1, you must not perform unguided exploration. From the very first observation, you must inspect the frame, identify all differences from previously known patterns, form explicit hypotheses about new elements or mechanics, immediately integrate them into the Heuristic System (as provisional rules if necessary), and select all actions exclusively through model-based planning.

No in-attempt action is allowed without a prior model-based prediction of the exact settled ASCII frame.

From the moment a Heuristic System exists, every real in-attempt game action must be selected via a planner operating on the Heuristic System, including exploratory actions. Use `python3 run_main_planner.py --from-current` when you want to try the main planner from the current in-attempt state; when it finds a validated `LEVEL_COMPLETED` plan it executes it on the real game right away (no separate `plan_executor.py` call needed for that plan). If you need an exploratory or intermediate target, use a separate planner saved as `level_N_planner_i.py`, and run it with `python3 run_aux_planner.py ...` (inside your auxiliary planner, you must verify that the plan really reaches the required target state, and you must also verify the artifacts generated by `run_aux_planner.py`). To execute a specific action sequence yourself (e.g. an auxiliary-planner plan, or a partial sequence), use `python3 plan_executor.py ...`. It reconstructs the current state, runs the chosen action sequence in the Heuristic System and the real game, and compares the observed settled ASCII frame against the predicted settled ASCII frame over the entire frame after each non-terminal step, not just near the avatar.

Any mismatch anywhere in the frame is a blocking event. When a mismatch occurs, you must stop gameplay immediately, inspect both ASCII and PNG observations carefully, inspect the mismatch artifacts produced by `plan_executor.py`, enumerate all differences across the full frame, treat every difference as potentially meaningful puzzle state, update the executable Heuristic System, and only then continue.

You must not leave any observed difference unexplained in the model before continuing.

You must not continue acting under an unresolved hypothesis. You must not chain exploratory actions without updating the model. You must not dismiss HUD, icons, counters, remote doors, symbols, or other visual changes as cosmetic without explicit evidence.

PNG inspection is mandatory whenever a new object appears, an action produces multiple frames, a remote change occurs, or any mismatch happens. ASCII is authoritative for exact verification, but PNG must be used for visual interpretation.

The Heuristic System must remain consistent with all previously observed history at all times.

After completing a level, you must do one more strict generalization pass over the entire Heuristic System: engine, state reconstruction, renderer, planner, and `hs.md`. No solved level may leave behind ad hoc level-specific logic, temporary hacks, or duplicate object families if they can be replaced by a simpler shared ontology and shared rules. This cleanup is mandatory before you treat the level as finished.

Always question what you are doing. Before committing to a plan, ask yourself: do I have strong evidence that this line of action is correct, or am I stuck in tunnel vision, exploring a hypothesis that is likely wrong? If the latter, it is better to step back and reconsider your entire approach rather than continue down the same reasoning path.
