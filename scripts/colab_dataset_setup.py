"""
Colab dataset setup: extract AnyScript archive, set DATA_ROOT and CANDIDATES.

Loaded by colab_quickstart.ipynb via exec so `git pull` updates behavior even when
the open Colab copy of the .ipynb is stale (old cells still importing colab_* helpers).
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections import deque

os.chdir("/content")
EXTRACT_PARENT = "/content/drive/MyDrive"
SCRIPTS_ROOT = "/content/ai-hw/scripts"
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

from data_anyscript import build_records, group_by_author


def _triplet_usable(p: str) -> bool:
    if not os.path.isdir(p):
        return False
    try:
        return len(group_by_author(build_records(p))) >= 2
    except OSError:
        return False


def _anyscriptish(p: str) -> bool:
    x = p.replace("\\", "/").lower()
    return any(k in x for k in ("anyscript", "binarized", "icdar", "handwrit"))


ARCHIVE_PATHS = [
    f"{EXTRACT_PARENT}/AnyScriptFiltered/AnyScriptFiltered.tar.gz",
    f"{EXTRACT_PARENT}/AnyScriptFiltered.tar.gz",
]
FIXED_CANDIDATES = [
    f"{EXTRACT_PARENT}/AnyScriptFiltered",
    f"{EXTRACT_PARENT}/data/datasets/AnyScriptFiltered/binarized",
    f"{EXTRACT_PARENT}/AnyScriptFiltered/train",
    f"{EXTRACT_PARENT}/AnyScriptFiltered/binarized",
    f"{EXTRACT_PARENT}/datasets/AnyScriptFiltered/binarized",
]


def _extra_paths_from_drive() -> list[str]:
    out: list[str] = []
    if not os.path.isdir(EXTRACT_PARENT):
        return out
    q = deque([(EXTRACT_PARENT, 0)])
    seen = 0
    while q and seen < 4000 and len(out) < 80:
        cur, depth = q.popleft()
        seen += 1
        if depth > 14:
            continue
        try:
            names = os.listdir(cur)
        except OSError:
            continue
        for name in names:
            full = os.path.join(cur, name)
            try:
                if not os.path.isdir(full):
                    continue
            except OSError:
                continue
            q.append((full, depth + 1))
            if name == "binarized" and full not in out:
                out.append(full)
            if name in ("train", "Train") and _anyscriptish(full) and full not in out:
                out.append(full)
    return out


def _ordered_candidates() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for p in FIXED_CANDIDATES + _extra_paths_from_drive():
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


CANDIDATES = _ordered_candidates()
DATA_ROOT = FIXED_CANDIDATES[0]

ARCHIVE = next((p for p in ARCHIVE_PATHS if os.path.isfile(p)), None)
if ARCHIVE:
    print("Extracting from", ARCHIVE, "(long run; needs Drive space)...")
    os.makedirs(EXTRACT_PARENT, exist_ok=True)
    subprocess.run(["tar", "-xzf", ARCHIVE, "-C", EXTRACT_PARENT], check=False)
else:
    print("No archive at:", ARCHIVE_PATHS)

CANDIDATES = _ordered_candidates()
picked = next((p for p in CANDIDATES if _triplet_usable(p)), None)
if picked:
    DATA_ROOT = picked
    print("OK: DATA_ROOT (auto) ->", DATA_ROOT)
else:
    print("WARNING: no folder with >=2 authors and 2+ pages each. Tried (sample):")
    for c in CANDIDATES[:12]:
        print(" ", c, "exists=" + str(os.path.isdir(c)))
    print("Upload/extract data, git pull /content/ai-hw, or run inspect_anyscript_layout.py.")
    DATA_ROOT = FIXED_CANDIDATES[0]
