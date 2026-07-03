#!/usr/bin/env python3
"""
local_gateway.py — run a LOCAL arc gateway over our environment_files, identical
to a hosted `gateway:8001` sidecar (both use arc_agi.server.create_app via
Arcade.listen_and_serve). This lets us run the WHOLE probe in `online` mode
locally — the same shape as a hosted rerun — so what we test here is what runs
there. The only difference from rerun is the URL (localhost vs gateway).

Usage:
  python3 local_gateway.py --environments-dir <env_dir> [--port 8001] [--competition]

Then the agent side connects with:
  Arcade(operation_mode=ONLINE, arc_base_url="http://localhost:8001")
which routes every action over HTTP to this gateway (RemoteEnvironmentWrapper),
exactly like the rerun scoring channel.
"""
from __future__ import annotations
# Make the vendored ARC SDK importable from within this repo (no machine .pth).
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths  # noqa: F401  (side-effect: puts vendor/arc on sys.path)

import argparse
import arc_agi
from arc_agi import OperationMode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--environments-dir", required=True)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--competition", action="store_true",
                    help="competition_mode scoring (like the real rerun gateway)")
    args = ap.parse_args()

    # The gateway's backend is a white-box OFFLINE arcade over our env files —
    # i.e. it owns the real games + scorecard, exactly as a hosted gateway does.
    arcade = arc_agi.Arcade(arc_api_key="",
                            operation_mode=OperationMode.OFFLINE,
                            environments_dir=args.environments_dir)
    print(f"[local-gateway] serving {len(arcade.available_environments)} games "
          f"on http://{args.host}:{args.port}  competition={args.competition}")
    arcade.listen_and_serve(host=args.host, port=args.port,
                            competition_mode=args.competition,
                            include_frame_data=True)


if __name__ == "__main__":
    main()
