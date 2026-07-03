#!/usr/bin/env python3
"""
paths.py — resolve the self-contained dependency dirs shipped in this repo.

This repo is self-contained: the model/sandbox helpers live in ``compat/`` and a
vendored copy of the ARC SDK (arc_agi + arcengine) plus the offline game files
live in ``vendor/arc/``. This module puts both on ``sys.path`` so that
``import sandbox`` / ``import qwen_policy`` and ``import arc_agi`` / ``import
arcengine`` work without any machine-level ``.pth`` or external directory.

Layout (relative to the repo root, one level above this file's ``arc_hs/``):

    <repo>/arc_hs/        <- this file
    <repo>/compat/        <- qwen_policy.py, sandbox.py, policy.py
    <repo>/vendor/arc/    <- arc_agi/, arcengine/, environment_files/

``ARC_COMPAT_DIR`` / ``ARC_VENDOR_DIR`` env vars override the defaults if you
relocate those directories.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # <repo>/arc_hs
_REPO = _HERE.parent                              # <repo>
_COMPAT_DEFAULT = _REPO / "compat"
_VENDOR_DEFAULT = _REPO / "vendor" / "arc"


def compat_dir() -> Path:
    env = os.getenv("ARC_COMPAT_DIR", "").strip()
    if env:
        return Path(env)
    return _COMPAT_DEFAULT


def vendor_dir() -> Path:
    env = os.getenv("ARC_VENDOR_DIR", "").strip()
    if env:
        return Path(env)
    return _VENDOR_DEFAULT


def environments_dir() -> Path:
    """Offline game files bundled under the vendored SDK."""
    return vendor_dir() / "environment_files"


def _prepend(path: Path) -> None:
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)


def ensure_vendor_on_path() -> Path:
    """Put the vendored ARC SDK (arc_agi + arcengine) on sys.path so plain
    ``import arc_agi`` / ``import arcengine`` resolve to the in-repo copy,
    replacing any machine-level .pth."""
    d = vendor_dir()
    _prepend(d)
    return d


def ensure_compat_on_path() -> Path:
    """Put the compat dir (sandbox.py / qwen_policy.py / policy.py) on sys.path."""
    d = compat_dir()
    _prepend(d)
    return d


# Importing this module makes the repo self-contained: the vendored ARC SDK is
# put on sys.path immediately, so any subsequent `import arc_agi` / `import
# arcengine` resolves in-repo without a machine-level .pth. (compat is injected
# on demand by ensure_compat_on_path, since it also depends on WM_NO_SANDBOX.)
ensure_vendor_on_path()

