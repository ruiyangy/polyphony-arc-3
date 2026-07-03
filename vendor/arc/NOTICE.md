# Vendored ARC-AGI SDK — NOTICE

This directory vendors the upstream **ARC-AGI SDK** so that Polyphony Agent — ARC
runs self-contained (no separate SDK install, no machine-level `.pth`), and so that
`--mode offline` works out of the box for local validation.

## Contents & provenance

| Path | Upstream | License |
|------|----------|---------|
| `arc_agi/` | ARC-AGI SDK (`arc_agi`) by the ARC Prize Foundation | MIT (see `LICENSE`) |
| `arcengine/` | ARC game engine (`arcengine`) by the ARC Prize Foundation | MIT (see `LICENSE`) |
| `environment_files/` | Public ARC-AGI-3 game environment files (25 public game ids) | MIT (see `LICENSE`) |

Copyright of the vendored SDK and public environment files belongs to the ARC
Prize Foundation. They are redistributed here under the MIT License (`LICENSE`
in this directory). Polyphony Agent's own code (everything under `arc_hs/` and
`compat/`) is licensed separately — see the repository-root `LICENSE`.

## Local modifications to `arc_agi/`

We apply a small number of compatibility/robustness patches to the SDK. They do
**not** change scoring or environment dynamics; they only harden the client for
long unattended runs:

- `remote_wrapper.py` / `api.py` — network robustness: longer, explicit HTTP
  timeouts and clearer error surfacing on transient failures.
- `base.py` / `local_wrapper.py` / `scorecard.py` — minor compatibility glue for
  the offline/self-hosted run path.

`arcengine/` and `environment_files/` are vendored **unmodified** (aside from the
removal of `__pycache__` / VCS metadata).

If you prefer to depend on the upstream SDK directly instead of this vendored
copy, point `ARC_VENDOR_DIR` at your own checkout.
