"""object_tools.py — object detector for the Heuristic System (HS).

Lab equipment for the coding agent, NOT a framework injection. Import it (or copy
/ improve it) while writing your `hs_state_io.py` state reconstruction and
your `hs_engine.py`: it turns a raw 64x64 integer frame into a small set
of structured objects with their shapes, symmetries, and spatial relations, so you
can reason about the board as objects instead of pixels.

Two layers, both pure numpy/scipy (no LLM, no network, no game-specific hacks):

  extract_objects(frame)   -> connected-component objects (color, size, bbox,
                              centroid, representative pixel, translation-invariant
                              shape hash, boundary corner points)
  object_relations(objects, frame)
                           -> adjacency (which objects touch) + containment
                              (which object encloses which)
  match_shapes(a, b)       -> compare two object crops up to the 8 dihedral
                              symmetries (rotation/mirror), optionally color-blind

Everything is advisory: you decide whether an "object" here matches a real game
entity. Segmentation is per-color 4/8-connected components; the background
(most common color) is excluded by default.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

try:
    import scipy.ndimage as _ndi
    _HAVE_NDI = True
except Exception:  # pragma: no cover - scipy is available in this env
    _HAVE_NDI = False


# --- shape signature -------------------------------------------------------

def shape_hash(cells: list[tuple[int, int]], color: int) -> str:
    """Translation-invariant signature of an object: color + cell shape normalized
    so the bounding box's top-left is the origin. Same shape+color -> same hash
    regardless of position, so you can track one object across frames or spot
    several identical objects in one frame (ported from Tufa's segmentation)."""
    if not cells:
        return "empty"
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    payload = repr((int(color), norm)).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CW = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
_CW_INDEX = {off: i for i, off in enumerate(_CW)}


def _trace_outer_contour(cells: set, start: tuple[int, int]) -> list[tuple[int, int]]:
    """Moore-neighbour trace of a component's outer perimeter, clockwise."""
    if len(cells) == 1:
        return [start]
    contour = [start]
    b = start
    prev = (start[0], start[1] - 1)
    second = None
    for _ in range(8 * len(cells) + 16):
        idx = _CW_INDEX[(prev[0] - b[0], prev[1] - b[1])]
        nxt = None
        new_prev = None
        for k in range(1, 9):
            off = _CW[(idx + k) % 8]
            cand = (b[0] + off[0], b[1] + off[1])
            if cand in cells:
                nxt = cand
                back = _CW[(idx + k - 1) % 8]
                new_prev = (b[0] + back[0], b[1] + back[1])
                break
        if nxt is None:
            break
        if second is None:
            second = nxt
        elif b == start and nxt == second:
            break
        contour.append(nxt)
        prev, b = new_prev, nxt
    if len(contour) > 1 and contour[-1] == contour[0]:
        contour.pop()
    return contour


def _corner_points(contour: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reduce a traced contour to only the points where direction changes — a
    compact way to describe a shape (few corners instead of many pixels)."""
    if len(contour) <= 2:
        return list(contour)
    m = len(contour)
    corners = []
    for i in range(m):
        prev, cur, nxt = contour[i - 1], contour[i], contour[(i + 1) % m]
        d_in = (cur[0] - prev[0], cur[1] - prev[1])
        d_out = (nxt[0] - cur[0], nxt[1] - cur[1])
        if d_in != d_out:
            corners.append(cur)
    return corners


def _background_color(frame: np.ndarray) -> int:
    return int(np.bincount(frame.ravel(), minlength=16).argmax())


# --- segmentation ----------------------------------------------------------

def extract_objects(
    frame,
    bg_colors: Optional[list[int]] = None,
    min_size: int = 1,
    connectivity: int = 2,
    with_boundary: bool = True,
) -> list[dict]:
    """Connected-component objects of a single 64x64 integer frame.

    Args:
        frame: 2D array-like of ints 0-15 (a single settled frame layer).
        bg_colors: colors to skip; None -> auto (most common color = background).
        min_size: drop components smaller than this many cells.
        connectivity: 1 = 4-connected (orthogonal), 2 = 8-connected (default;
            fewer fragments). Falls back to a pure-python BFS if scipy is absent.
        with_boundary: also compute the outer-contour corner points (small cost).

    Returns list of dicts sorted by size desc, each:
        id, color, size, bbox=(x,y,w,h), centroid=(cx,cy) float,
        pixel=(y,x) a real member nearest the centroid (safe click target for
              hollow shapes), hash=translation-invariant shape signature,
        boundary=[(y,x),...] corner points (if with_boundary), cells=[(y,x),...].
    """
    frame = np.asarray(frame, dtype=np.int64)
    if frame.ndim != 2:
        raise ValueError(f"extract_objects expects a 2D frame, got shape {frame.shape}")
    if bg_colors is None:
        bg_colors = [_background_color(frame)]
    bg_set = set(int(c) for c in bg_colors)

    comps: list[dict] = []
    if _HAVE_NDI:
        struct = np.ones((3, 3), dtype=int) if connectivity == 2 else np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        for color in range(16):
            if color in bg_set:
                continue
            mask = frame == color
            if not mask.any():
                continue
            lab, n = _ndi.label(mask, structure=struct)
            for i in range(1, n + 1):
                ys, xs = np.where(lab == i)
                if len(ys) < min_size:
                    continue
                comps.append(_mk_obj(color, ys, xs, with_boundary))
    else:
        comps = _extract_objects_bfs(frame, bg_set, min_size, connectivity, with_boundary)

    comps.sort(key=lambda o: o["size"], reverse=True)
    for i, o in enumerate(comps):
        o["id"] = i
    return comps


def _mk_obj(color, ys, xs, with_boundary) -> dict:
    ys = np.asarray(ys); xs = np.asarray(xs)
    cyc, cxc = float(ys.mean()), float(xs.mean())
    k = int(np.argmin(np.abs(ys - cyc) + np.abs(xs - cxc)))
    cells = list(zip(ys.tolist(), xs.tolist()))
    obj = {
        "color": int(color),
        "size": int(len(ys)),
        "bbox": (int(xs.min()), int(ys.min()),
                 int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
        "centroid": (round(cxc, 1), round(cyc, 1)),
        "pixel": (int(ys[k]), int(xs[k])),
        "hash": shape_hash(cells, color),
        "cells": cells,
    }
    if with_boundary:
        cset = set(cells)
        start = min(cells)  # top-most, then left-most
        obj["boundary"] = _corner_points(_trace_outer_contour(cset, start))
    return obj


def _extract_objects_bfs(frame, bg_set, min_size, connectivity, with_boundary):
    from collections import deque
    h, w = frame.shape
    seen = np.zeros((h, w), dtype=bool)
    nbrs = _CW if connectivity == 2 else _ORTH
    out = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0, x0]:
                continue
            c = int(frame[y0, x0])
            if c in bg_set:
                seen[y0, x0] = True
                continue
            q = deque([(y0, x0)]); seen[y0, x0] = True; pix = []
            while q:
                cy, cx = q.popleft(); pix.append((cy, cx))
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(frame[ny, nx]) == c:
                        seen[ny, nx] = True; q.append((ny, nx))
            if len(pix) < min_size:
                continue
            ys = np.array([p[0] for p in pix]); xs = np.array([p[1] for p in pix])
            out.append(_mk_obj(c, ys, xs, with_boundary))
    return out


# --- relations -------------------------------------------------------------

def object_relations(objects: list[dict], frame) -> dict:
    """Spatial relations between the given objects (ported from Tufa's approach).

    Returns:
        adjacency: sorted list of [id_i, id_j] pairs whose cells are 4-adjacent
                   (they physically touch — collisions, connections).
        containment: dict child_id -> parent_id, where parent is the innermost
                   object fully enclosing the child (e.g. a marker inside a box,
                   a target inside walls). Computed by flood-filling each object's
                   complement inward from the frame border.
    """
    frame = np.asarray(frame, dtype=np.int64)
    h, w = frame.shape
    n = len(objects)
    owner = -np.ones((h, w), dtype=np.int64)
    for o in objects:
        for (r, c) in o["cells"]:
            owner[r, c] = o["id"]

    # adjacency: 4-neighbour cells owned by two different objects
    adj = set()
    for r in range(h):
        for c in range(w):
            a = owner[r, c]
            if a < 0:
                continue
            if r + 1 < h and owner[r + 1, c] >= 0 and owner[r + 1, c] != a:
                adj.add((min(a, owner[r + 1, c]), max(a, owner[r + 1, c])))
            if c + 1 < w and owner[r, c + 1] >= 0 and owner[r, c + 1] != a:
                adj.add((min(a, owner[r, c + 1]), max(a, owner[r, c + 1])))
    adjacency = sorted([int(i), int(j)] for i, j in adj)

    # containment: for each object b, flood its complement from the border; any
    # object never reached is enclosed by b. Parent = innermost such encloser.
    id_to_obj = {o["id"]: o for o in objects}
    enclosers = {o["id"]: set() for o in objects}
    for b in objects:
        bid = b["id"]
        reached = np.zeros((h, w), dtype=bool)
        stack = []
        for r in range(h):
            for c in (0, w - 1):
                if owner[r, c] != bid and not reached[r, c]:
                    reached[r, c] = True; stack.append((r, c))
        for c in range(w):
            for r in (0, h - 1):
                if owner[r, c] != bid and not reached[r, c]:
                    reached[r, c] = True; stack.append((r, c))
        while stack:
            r, c = stack.pop()
            for dr, dc in _ORTH:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not reached[nr, nc] and owner[nr, nc] != bid:
                    reached[nr, nc] = True; stack.append((nr, nc))
        for a in objects:
            if a["id"] == bid:
                continue
            ar, ac = a["cells"][0]
            if not reached[ar, ac]:
                enclosers[a["id"]].add(bid)

    containment = {}
    for cid, encs in enclosers.items():
        if encs:
            # innermost encloser = the one itself most deeply enclosed
            parent = max(encs, key=lambda e: (len(enclosers[e]), -e))
            containment[int(cid)] = int(parent)
    return {"adjacency": adjacency, "containment": containment}


# --- symmetry-aware shape matching ----------------------------------------

def _crop(frame, obj) -> np.ndarray:
    x, y, w, h = obj["bbox"]
    return np.asarray(frame)[y:y + h, x:x + w]


def all_transforms(pattern: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """The 8 dihedral transforms (4 rotations x optional mirror) of a 2D pattern."""
    pattern = np.asarray(pattern)
    out = []
    for k, name in enumerate(("identity", "rot90", "rot180", "rot270")):
        out.append((np.rot90(pattern, k), name))
    m = np.fliplr(pattern)
    for k, name in enumerate(("mirror", "mirror+rot90", "mirror+rot180", "mirror+rot270")):
        out.append((np.rot90(m, k), name))
    return out


def match_shapes(a, b, allow_rotation: bool = True, allow_mirror: bool = True,
                 ignore_colors: bool = False) -> tuple[bool, Optional[str]]:
    """Do two 2D patterns match under the allowed symmetries?

    a, b: 2D arrays (e.g. object crops). Returns (match, transform_name). With
    ignore_colors=True, compares shape only (nonzero -> 1). Useful to decide if
    two objects are the same piece rotated/mirrored (common ARC mechanic)."""
    a = np.asarray(a); b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return (False, None)
    if ignore_colors:
        a = (a != 0).astype(np.int8); b = (b != 0).astype(np.int8)
    transforms = all_transforms(a)
    if not allow_rotation and not allow_mirror:
        transforms = transforms[:1]
    elif not allow_mirror:
        transforms = transforms[:4]
    elif not allow_rotation:
        transforms = [transforms[0], transforms[4]]
    for t, name in transforms:
        if t.shape == b.shape and np.array_equal(t, b):
            return (True, name)
    return (False, None)


def crop_object(frame, obj) -> np.ndarray:
    """Return the bounding-box crop of one object from the frame."""
    return _crop(frame, obj)


__all__ = [
    "extract_objects", "object_relations", "match_shapes",
    "shape_hash", "all_transforms", "crop_object",
]
